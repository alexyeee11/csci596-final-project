#include "comm/pipeline.h"
#include <stdlib.h>
#include <time.h>
#include "cuda_check.h"
#include "kernels/layers.h"

void pp_split(int nlayers, int P, int stage, int* first, int* last) {
  int base = nlayers / P, extra = nlayers % P;
  *first = stage * base + (stage < extra ? stage : extra);
  *last = *first + base + (stage < extra);
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Transfer buffers, grown on demand and kept across steps. Every micro-batch has its own slot in the host buffers,
   so all receives can be posted up front and a send stays untouched until the end of the step. */
typedef struct { void* p; size_t cap; int host; } Buf;
static Buf d_in, d_dx, d_dout, d_dlog;  /* device: stage inputs (all M), dX of one micro-batch, dOut, head dlogits */
/* pinned host: forward recv/send, backward recv/send, head logits for pp_correct */
static Buf h_fin = {.host = 1}, h_fout = {.host = 1}, h_bin = {.host = 1}, h_bout = {.host = 1}, h_logits = {.host = 1};

static float* grow(Buf* b, size_t floats) {
  size_t bytes = floats * sizeof(float);
  if (bytes > b->cap) {
    if (b->p) { if (b->host) cudaFreeHost(b->p); else cudaFree(b->p); }
    if (b->host) CUDA_CHECK(cudaMallocHost(&b->p, bytes));
    else CUDA_CHECK(cudaMalloc(&b->p, bytes));
    b->cap = bytes;
  }
  return (float*)b->p;
}

void pp_free(void) {
  Buf* all[] = {&d_in, &d_dx, &d_dout, &d_dlog, &h_fin, &h_fout, &h_bin, &h_bout, &h_logits};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) {
    if (all[i]->p) { if (all[i]->host) cudaFreeHost(all[i]->p); else cudaFree(all[i]->p); }
    all[i]->p = NULL;
    all[i]->cap = 0;
  }
}

float pp_step(MLP* m, const float* X, const int* y, int b_total, int M, MPI_Comm pipe, double* busy_s) {
  int s, P;
  MPI_Comm_rank(pipe, &s);
  MPI_Comm_size(pipe, &P);
  const int first = s == 0, head = s == P - 1, b = b_total / M;
  const int in = m->L[0].in, out = m->L[m->nlayers - 1].out;
  const size_t nin = (size_t)b * in, nout = (size_t)b * out;  /* floats per micro-batch at each end */
  float *din = NULL, *ddx = NULL, *ddout = NULL, *ddlog = NULL, *hfin = NULL, *hfout = NULL, *hbin = NULL,
        *hbout = NULL;
  if (!first) {
    din = grow(&d_in, nin * M);  ddx = grow(&d_dx, nin);
    hfin = grow(&h_fin, nin * M);  hbout = grow(&h_bout, nin * M);
  }
  if (!head) {
    ddout = grow(&d_dout, nout);
    hfout = grow(&h_fout, nout * M);  hbin = grow(&h_bin, nout * M);
  } else {
    ddlog = grow(&d_dlog, nout * M);
  }
  /* request slots: [0, M) forward recv, [M, 2M) forward send, [2M, 3M) backward recv, [3M, 4M) backward send */
  MPI_Request* rq = malloc(4 * (size_t)M * sizeof(MPI_Request));
  for (int i = 0; i < 4 * M; i++) rq[i] = MPI_REQUEST_NULL;
  for (int k = 0; k < M; k++) {  /* tags: k forward, M + k backward */
    if (!first) MPI_Irecv(hfin + k * nin, (int)nin, MPI_FLOAT, s - 1, k, pipe, &rq[k]);
    if (!head) MPI_Irecv(hbin + k * nout, (int)nout, MPI_FLOAT, s + 1, M + k, pipe, &rq[2 * M + k]);
  }

  double busy = 0, t;
  float loss = 0;
  for (int k = 0; k < M; k++) {
    const float* xk = X + k * nin;
    if (!first) {
      MPI_Wait(&rq[k], MPI_STATUS_IGNORE);
      CUDA_CHECK(cudaMemcpy(din + k * nin, hfin + k * nin, nin * sizeof(float), cudaMemcpyHostToDevice));
      xk = din + k * nin;
    }
    t = now();
    float* yk = mlp_stage_forward(m, xk, b, k);
    if (head) loss += softmax_xent(yk, y + k * b, ddlog + k * nout, m->loss_buf, b, out);
    CUDA_CHECK(cudaStreamSynchronize(0));
    busy += now() - t;
    if (!head) {
      CUDA_CHECK(cudaMemcpy(hfout + k * nout, yk, nout * sizeof(float), cudaMemcpyDeviceToHost));
      MPI_Isend(hfout + k * nout, (int)nout, MPI_FLOAT, s + 1, k, pipe, &rq[M + k]);
    }
  }

  t = now();
  mlp_zero_grads(m);
  CUDA_CHECK(cudaStreamSynchronize(0));
  busy += now() - t;
  for (int k = M - 1; k >= 0; k--) {
    const float* dk = ddlog + (head ? k * nout : 0);
    if (!head) {
      MPI_Wait(&rq[2 * M + k], MPI_STATUS_IGNORE);
      CUDA_CHECK(cudaMemcpy(ddout, hbin + k * nout, nout * sizeof(float), cudaMemcpyHostToDevice));
      dk = ddout;
    }
    t = now();
    mlp_stage_backward(m, first ? X + k * nin : din + k * nin, dk, ddx, b, k);
    CUDA_CHECK(cudaStreamSynchronize(0));
    busy += now() - t;
    if (!first) {
      CUDA_CHECK(cudaMemcpy(hbout + k * nin, ddx, nin * sizeof(float), cudaMemcpyDeviceToHost));
      MPI_Isend(hbout + k * nin, (int)nin, MPI_FLOAT, s - 1, M + k, pipe, &rq[3 * M + k]);
    }
  }
  MPI_Waitall(4 * M, rq, MPI_STATUSES_IGNORE);
  free(rq);

  loss /= M;  /* equal micro-batches: mean of the micro-batch means = batch mean */
  MPI_Bcast(&loss, 1, MPI_FLOAT, P - 1, pipe);
  *busy_s = busy;
  return loss;
}

int pp_correct(MLP* m, const float* X, const int* y, int b, MPI_Comm pipe) {
  int s, P;
  MPI_Comm_rank(pipe, &s);
  MPI_Comm_size(pipe, &P);
  const int first = s == 0, head = s == P - 1, tag = 1 << 20;
  const int in = m->L[0].in, out = m->L[m->nlayers - 1].out;
  const size_t nin = (size_t)b * in, nout = (size_t)b * out;
  const float* x = X;
  if (!first) {
    float* h = grow(&h_fin, nin);
    float* d = grow(&d_in, nin);
    MPI_Recv(h, (int)nin, MPI_FLOAT, s - 1, tag, pipe, MPI_STATUS_IGNORE);
    CUDA_CHECK(cudaMemcpy(d, h, nin * sizeof(float), cudaMemcpyHostToDevice));
    x = d;
  }
  const float* yk = mlp_stage_forward(m, x, b, 0);
  float* h = grow(head ? &h_logits : &h_fout, nout);
  CUDA_CHECK(cudaMemcpy(h, yk, nout * sizeof(float), cudaMemcpyDeviceToHost));
  int correct = 0;
  if (!head) {
    MPI_Send(h, (int)nout, MPI_FLOAT, s + 1, tag, pipe);
  } else {
    for (int i = 0; i < b; i++) {
      int best = 0;
      for (int c = 1; c < out; c++) if (h[(size_t)i * out + c] > h[(size_t)i * out + best]) best = c;
      correct += best == y[i];
    }
  }
  MPI_Bcast(&correct, 1, MPI_INT, P - 1, pipe);
  return correct;
}
