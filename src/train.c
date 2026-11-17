#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "comm/allreduce.h"
#include "comm/nccl.h"
#include "comm/overlap.h"
#include "comm/pipeline.h"
#include "cuda_check.h"
#include "data/cifar.h"
#include "model/mlp.h"

typedef struct {
  const char* data;
  int synthetic, epochs, steps, batch, hidden, layers, gemm, allreduce, nccl, overlap, pp, mb;
  float lr, bucket_mb;
  unsigned seed;
} Opts;

static void usage(void) {
  fprintf(stderr,
          "usage: train (--data DIR | --synthetic) [--epochs N] [--steps N] [--batch N] [--hidden N]\n"
          "             [--layers N] [--lr X] [--seed N] [--gemm naive|tiled|regtile|cublas]\n"
          "             [--allreduce naive|tree|ring|mpi|nccl] [--overlap [--bucket-mb X]]\n"
          "             [--pp P [--microbatches M]]\n");
  exit(2);
}

static Opts parse(int argc, char** argv) {
  Opts o = {NULL, 0, 1, 0, 256, 4096, 3, GEMM_TILED, AR_RING, 0, 0, 1, 1, 0.01f, 25.f, 42};
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (!strcmp(a, "--synthetic")) { o.synthetic = 1; continue; }
    if (!strcmp(a, "--overlap")) { o.overlap = 1; continue; }
    if (i + 1 >= argc) usage();
    const char* v = argv[++i];
    if (!strcmp(a, "--data")) o.data = v;
    else if (!strcmp(a, "--epochs")) o.epochs = atoi(v);
    else if (!strcmp(a, "--steps")) o.steps = atoi(v);
    else if (!strcmp(a, "--batch")) o.batch = atoi(v);
    else if (!strcmp(a, "--hidden")) o.hidden = atoi(v);
    else if (!strcmp(a, "--layers")) o.layers = atoi(v);
    else if (!strcmp(a, "--lr")) o.lr = (float)atof(v);
    else if (!strcmp(a, "--bucket-mb")) { if ((o.bucket_mb = (float)atof(v)) <= 0) usage(); }
    else if (!strcmp(a, "--seed")) o.seed = (unsigned)atoi(v);
    else if (!strcmp(a, "--pp")) { if ((o.pp = atoi(v)) < 1) usage(); }
    else if (!strcmp(a, "--microbatches")) { if ((o.mb = atoi(v)) < 1) usage(); }
    else if (!strcmp(a, "--gemm")) { if ((o.gemm = gemm_parse(v)) < 0) usage(); }
    else if (!strcmp(a, "--allreduce") && !strcmp(v, "nccl")) o.nccl = 1;
    else if (!strcmp(a, "--allreduce")) { if ((o.allreduce = ar_parse(v)) < 0) usage(); }
    else usage();
  }
  if ((!o.data && !o.synthetic) || o.layers < 1 || o.layers > 14 || o.batch < 1) usage();
  return o;
}

/* Wall-clock seconds. Not MPI_Wtime: with --overlap the comm thread may be inside MPI, and the library only
   promises MPI_THREAD_SERIALIZED. */
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Same seed on every rank -> same permutation everywhere. */
static unsigned long long rng;
static unsigned next_u32(void) { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; return (unsigned)(rng >> 33); }
static void shuffle(int* perm, int n, unsigned seed) {
  rng = seed;
  for (int i = 0; i < n; i++) perm[i] = i;
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(next_u32() % (unsigned)(i + 1)), t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
}

static float evaluate(MLP* m, const Dataset* d, int b, float* dX, float* hlogits) {
  const int classes = 10;
  int correct = 0, total = 0;
  for (int s = 0; s + b <= d->n; s += b) {
    CUDA_CHECK(cudaMemcpy(dX, d->X + (size_t)s * d->dim, (size_t)b * d->dim * sizeof(float), cudaMemcpyHostToDevice));
    const float* logits = mlp_forward(m, dX, b);
    CUDA_CHECK(cudaMemcpy(hlogits, logits, (size_t)b * classes * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < b; i++) {
      int best = 0;
      for (int c = 1; c < classes; c++) if (hlogits[i * classes + c] > hlogits[i * classes + best]) best = c;
      correct += best == d->y[s + i];
      total++;
    }
  }
  return total ? (float)correct / total : 0.f;
}

/* Pipeline version: every rank walks the test set; the first stage feeds inputs, the head counts. */
static float evaluate_pp(MLP* m, const Dataset* d, int b, float* dX) {
  int correct = 0, total = 0;
  for (int s = 0; s + b <= d->n; s += b) {
    if (m->first == 0)
      CUDA_CHECK(cudaMemcpy(dX, d->X + (size_t)s * d->dim, (size_t)b * d->dim * sizeof(float), cudaMemcpyHostToDevice));
    correct += pp_correct(m, dX, d->y + s, b, MPI_COMM_WORLD);
    total += b;
  }
  return total ? (float)correct / total : 0.f;
}

/* Reject flag combinations no test covers. Called on every rank; rank 0 prints. */
static void check_opts(const Opts* o, int rank, int p) {
  const char* err = NULL;
  char msg[160];
  if (o->nccl && o->overlap) err = "--overlap stages buckets through host memory; it does not combine with --allreduce nccl";
  else if (o->pp > 1 && (o->overlap || o->nccl)) err = "--pp does not combine with --overlap or --allreduce nccl";
  else if (o->mb > 1 && o->pp == 1) err = "--microbatches needs --pp > 1";
  else if (o->pp > o->layers + 1) {
    snprintf(msg, sizeof msg, "--pp %d is more than the %d layers of the model", o->pp, o->layers + 1);
    err = msg;
  } else if (o->pp > 1 && p != o->pp) {
    snprintf(msg, sizeof msg, "--pp %d needs exactly %d ranks (got %d)", o->pp, o->pp, p);
    err = msg;
  } else if (o->batch % (p / o->pp)) {
    snprintf(msg, sizeof msg, "--batch %d is not divisible by %d ranks", o->batch, p / o->pp);
    err = msg;
  } else if ((o->batch / (p / o->pp)) % o->mb) {
    snprintf(msg, sizeof msg, "per-replica batch %d is not divisible by --microbatches %d", o->batch / (p / o->pp), o->mb);
    err = msg;
  }
  if (!err) return;
  if (!rank) fprintf(stderr, "%s\n", err);
  MPI_Abort(MPI_COMM_WORLD, 2);
}

int main(int argc, char** argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  int rank, p, lrank, ndev;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &lrank);
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(lrank % ndev));

  Opts o = parse(argc, argv);
  if (o.overlap && provided < MPI_THREAD_SERIALIZED) {
    if (!rank) fprintf(stderr, "--overlap needs MPI_THREAD_SERIALIZED; this MPI provides level %d\n", provided);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  check_opts(&o, rank, p);
  if (o.nccl) nccl_init(MPI_COMM_WORLD);  /* exits with a message when built without NCCL */
  /* With --pp the world is one pipeline (rank = stage) and each rank sees the whole batch; otherwise dp = p. */
  const int pp = o.pp, dp = p / pp, replica = rank % dp, stage = rank / dp;
  int b = o.batch / dp;

  Dataset train, test;
  if (o.synthetic) {
    dataset_synthetic(&train, 4096, 3072, 10, 1);
    dataset_synthetic(&test, 1024, 3072, 10, 2);
  } else if (cifar_load(o.data, 1, &train) || cifar_load(o.data, 0, &test)) {
    fprintf(stderr, "cannot load CIFAR-10 from %s\n", o.data);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int sizes[16], ns = 0;
  sizes[ns++] = train.dim;
  for (int l = 0; l < o.layers; l++) sizes[ns++] = o.hidden;
  sizes[ns++] = 10;
  int lfirst = 0, llast = ns - 1;
  if (pp > 1) pp_split(ns - 1, pp, stage, &lfirst, &llast);
  MLP* m = mlp_create_range(sizes, ns, lfirst, llast, b, o.seed, (gemm_impl)o.gemm);

  float* hX = malloc((size_t)b * train.dim * sizeof(float));
  float* hlogits = malloc((size_t)b * 10 * sizeof(float));
  int* hy = malloc(b * sizeof(int));
  float* dX;
  int* dy;
  CUDA_CHECK(cudaMalloc((void**)&dX, (size_t)b * train.dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dy, b * sizeof(int)));
  float* hgrad = NULL;  /* pinned staging buffer for the host all-reduce */
  Overlap* ov = NULL;     /* with --overlap, buckets are all-reduced on a comm thread during backward */
  if (pp > 1) {
    /* the pipeline all-reduces nothing */
  } else if (p > 1 && o.overlap) {
    size_t bucket = (size_t)(o.bucket_mb * (1 << 20) / sizeof(float));
    ov = overlap_create(m->grads, m->nparams, bucket ? bucket : 1, (ar_algo)o.allreduce, MPI_COMM_WORLD, lrank % ndev);
  } else if (p > 1 && !o.nccl)
    CUDA_CHECK(cudaMallocHost((void**)&hgrad, m->nparams * sizeof(float)));
  int* perm = malloc(train.n * sizeof(int));

  int steps_per_epoch = train.n / o.batch, step = 0, done = 0;
  for (int e = 0; e < o.epochs && !done; e++) {
    shuffle(perm, train.n, o.seed + e);
    for (int s = 0; s < steps_per_epoch && !done; s++) {
      /* replica r takes samples r, r+dp, r+2dp, ... of the global batch */
      for (int k = 0; k < b; k++) {
        int idx = perm[s * o.batch + replica + k * dp];
        memcpy(hX + (size_t)k * train.dim, train.X + (size_t)idx * train.dim, train.dim * sizeof(float));
        hy[k] = train.y[idx];
      }
      CUDA_CHECK(cudaMemcpy(dX, hX, (size_t)b * train.dim * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dy, hy, b * sizeof(int), cudaMemcpyHostToDevice));

      if (pp > 1) {
        /* compute_s: mean over stages of each stage's kernel time (incl. SGD); comm_s: the rest of the slowest
           stage's step, i.e. transfers plus bubble */
        double busy, t0 = now();
        float loss = pp_step(m, dX, dy, b, o.mb, MPI_COMM_WORLD, &busy);
        double t1 = now();
        mlp_sgd(m, o.lr, 1.0f / o.mb);  /* sum of M micro-batch means / M = batch mean */
        CUDA_CHECK(cudaStreamSynchronize(0));
        double t2 = now(), st[2] = {busy + (t2 - t1), t2 - t0}, sum, mx;
        MPI_Reduce(&st[0], &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st[1], &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank) printf("step %d loss %.6f compute_s %.6f comm_s %.6f step_s %.6f\n", step, loss, sum / p, mx - sum / p, mx);
        step++;
        if (o.steps && step >= o.steps) done = 1;
        continue;
      }
      /* compute_s: forward + backward until the default stream is idle (not cudaDeviceSynchronize, which would also
         wait for the overlap copies); comm_s: communication still outstanding after that; step_s: whole step */
      double t0 = now();
      if (ov) overlap_begin(ov);
      float loss = mlp_forward_backward(m, dX, dy, b, ov ? overlap_on_grad : NULL, ov);
      CUDA_CHECK(cudaStreamSynchronize(0));
      double t1 = now();
      if (ov) {
        overlap_wait(ov);
      } else if (p > 1 && o.nccl) {
        nccl_allreduce_sum(m->grads, m->nparams);
      } else if (p > 1) {
        CUDA_CHECK(cudaMemcpy(hgrad, m->grads, m->nparams * sizeof(float), cudaMemcpyDeviceToHost));
        allreduce_sum(hgrad, m->nparams, (ar_algo)o.allreduce, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(m->grads, hgrad, m->nparams * sizeof(float), cudaMemcpyHostToDevice));
      }
      double t2 = now();
      mlp_sgd(m, o.lr, 1.0f / p);  /* sum of per-rank means / p = global-batch mean */
      CUDA_CHECK(cudaStreamSynchronize(0));
      double t3 = now();

      float gl;
      MPI_Allreduce(&loss, &gl, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      if (!rank) printf("step %d loss %.6f compute_s %.6f comm_s %.6f step_s %.6f\n", step, gl / p, t1 - t0, t2 - t1, t3 - t0);
      step++;
      if (o.steps && step >= o.steps) done = 1;
    }
    if (pp > 1) {
      float acc = evaluate_pp(m, &test, b, dX);
      if (!rank) printf("epoch %d test_acc %.4f\n", e, acc);
      fflush(stdout);
    } else if (!rank) {
      printf("epoch %d test_acc %.4f\n", e, evaluate(m, &test, b, dX, hlogits));
      fflush(stdout);
    }
  }

  free(perm); free(hX); free(hlogits); free(hy);
  cudaFree(dX); cudaFree(dy);
  if (hgrad) cudaFreeHost(hgrad);
  if (ov) overlap_free(ov);
  if (o.nccl) nccl_finalize();
  if (pp > 1) pp_free();
  mlp_free(m);
  dataset_free(&train);
  dataset_free(&test);
  MPI_Finalize();
  return 0;
}
