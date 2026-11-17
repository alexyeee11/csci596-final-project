#include <math.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/layers.h"
#include "model/mlp.h"

static unsigned long long rng;
static double uniform01(void) {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((rng >> 11) + 0.5) / 9007199254740992.0;
}
static float normal(void) { return (float)(sqrt(-2.0 * log(uniform01())) * cos(6.283185307179586 * uniform01())); }

MLP* mlp_create_range(const int* sizes, int nsizes, int first, int last, int max_batch, unsigned seed,
                      gemm_impl impl) {
  MLP* m = (MLP*)calloc(1, sizeof(MLP));
  m->nlayers = last - first;
  m->first = first;
  m->head = last == nsizes - 1;
  m->max_batch = max_batch;
  m->impl = impl;
  m->L = (Layer*)calloc(m->nlayers, sizeof(Layer));
  for (int l = first; l < last; l++) {
    m->nparams += (size_t)sizes[l] * sizes[l + 1] + sizes[l + 1];
    if (sizes[l + 1] > m->max_width) m->max_width = sizes[l + 1];
  }
  CUDA_CHECK(cudaMalloc(&m->params, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m->grads, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMemset(m->grads, 0, m->nparams * sizeof(float)));
  float* h = (float*)calloc(m->nparams, sizeof(float));
  rng = seed;
  size_t off = 0;
  for (int g = 0; g < last; g++) {  /* draw every earlier layer's weights too, keeping only the range */
    float sd = sqrtf(2.0f / sizes[g]);  /* He init; biases stay zero */
    size_t nw = (size_t)sizes[g] * sizes[g + 1];
    if (g < first) { for (size_t i = 0; i < nw; i++) normal(); continue; }
    Layer* L = &m->L[g - first];
    L->in = sizes[g];
    L->out = sizes[g + 1];
    for (size_t i = 0; i < nw; i++) h[off + i] = sd * normal();
    L->W = m->params + off;  L->b = L->W + nw;
    L->dW = m->grads + off;  L->db = L->dW + nw;
    CUDA_CHECK(cudaMalloc(&L->Y, (size_t)max_batch * L->out * sizeof(float)));
    off += nw + L->out;
  }
  CUDA_CHECK(cudaMemcpy(m->params, h, m->nparams * sizeof(float), cudaMemcpyHostToDevice));
  free(h);
  for (int i = 0; i < 2; i++)
    CUDA_CHECK(cudaMalloc(&m->dbuf[i], (size_t)max_batch * m->max_width * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m->loss_buf, max_batch * sizeof(float)));
  return m;
}

MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl) {
  return mlp_create_range(sizes, nsizes, 0, nsizes - 1, max_batch, seed, impl);
}

void mlp_free(MLP* m) {
  for (int l = 0; l < m->nlayers; l++) cudaFree(m->L[l].Y);
  cudaFree(m->params); cudaFree(m->grads);
  cudaFree(m->dbuf[0]); cudaFree(m->dbuf[1]); cudaFree(m->loss_buf);
  free(m->L);
  free(m);
}

float* mlp_stage_forward(MLP* m, const float* X, int b, int k) {
  const float* A = X;
  float* Y = NULL;
  for (int l = 0; l < m->nlayers; l++) {
    Layer* L = &m->L[l];
    Y = L->Y + (size_t)k * b * L->out;
    gemm(m->impl, 0, 0, b, L->out, L->in, 1.f, A, L->W, 0.f, Y);       /* Y = A W */
    bias_act_forward(Y, L->b, b, L->out, !(m->head && l == m->nlayers - 1));
    A = Y;
  }
  return Y;
}

const float* mlp_forward(MLP* m, const float* X, int B) { return mlp_stage_forward(m, X, B, 0); }

/* dZ of the stage's last layer lives in dOut (head) or dbuf[1] (dOut times the ReLU mask), so the first dA goes to
   dbuf[0] and the two then alternate. hook (may be NULL) sees each layer's grads once they are final. */
static void stage_backward(MLP* m, const float* X, const float* dOut, float* dX, int b, int k,
                           grad_hook hook, void* ctx) {
  int last = m->nlayers - 1, nxt = 0;
  size_t r0 = (size_t)k * b;
  float beta = k ? 1.f : 0.f;
  const float* dZ = dOut;
  if (!m->head) {
    /* ponytail: one device copy so dOut stays const; fold the mask into the caller if it shows up in profiles */
    size_t n = (size_t)b * m->L[last].out;
    CUDA_CHECK(cudaMemcpy(m->dbuf[1], dOut, n * sizeof(float), cudaMemcpyDeviceToDevice));
    relu_backward(m->dbuf[1], m->L[last].Y + r0 * m->L[last].out, n);
    dZ = m->dbuf[1];
  }
  for (int l = last; l >= 0; l--) {
    Layer* L = &m->L[l];
    const float* Ain = l > 0 ? m->L[l - 1].Y + r0 * L->in : X;
    gemm(m->impl, 1, 0, L->in, L->out, b, 1.f, Ain, dZ, beta, L->dW);  /* dW (+)= Ain^T dZ */
    bias_grad(L->db, dZ, b, L->out, k > 0);
    if (hook) hook(l, L->dW, (size_t)L->in * L->out + L->out, ctx);
    if (l > 0) {
      float* dA = m->dbuf[nxt];
      gemm(m->impl, 0, 1, b, L->in, L->out, 1.f, dZ, L->W, 0.f, dA);    /* dA = dZ W^T */
      relu_backward(dA, Ain, (size_t)b * L->in);
      dZ = dA;
      nxt ^= 1;
    } else if (dX) {
      gemm(m->impl, 0, 1, b, L->in, L->out, 1.f, dZ, L->W, 0.f, dX);    /* stage input is post-ReLU: no mask */
    }
  }
}

void mlp_stage_backward(MLP* m, const float* X, const float* dOut, float* dX, int b, int k) {
  stage_backward(m, X, dOut, dX, b, k, NULL, NULL);
}

float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx) {
  const float* logits = mlp_forward(m, X, B);
  float loss = softmax_xent(logits, y, m->dbuf[1], m->loss_buf, B, m->L[m->nlayers - 1].out);
  stage_backward(m, X, m->dbuf[1], NULL, B, 0, hook, ctx);
  return loss;
}

void mlp_sgd(MLP* m, float lr, float grad_scale) {
  sgd_step(m->params, m->grads, lr * grad_scale, m->nparams);
}
