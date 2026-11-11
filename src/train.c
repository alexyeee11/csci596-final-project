#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "comm/allreduce.h"
#include "cuda_check.h"
#include "data/cifar.h"
#include "model/mlp.h"

typedef struct {
  const char* data;
  int synthetic, epochs, steps, batch, hidden, layers, gemm, allreduce;
  float lr;
  unsigned seed;
} Opts;

static void usage(void) {
  fprintf(stderr,
          "usage: train (--data DIR | --synthetic) [--epochs N] [--steps N] [--batch N] [--hidden N]\n"
          "             [--layers N] [--lr X] [--seed N] [--gemm naive|tiled|regtile|cublas]\n"
          "             [--allreduce naive|tree|ring|mpi]\n");
  exit(2);
}

static Opts parse(int argc, char** argv) {
  Opts o = {NULL, 0, 1, 0, 256, 4096, 3, GEMM_TILED, AR_RING, 0.01f, 42};
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (!strcmp(a, "--synthetic")) { o.synthetic = 1; continue; }
    if (i + 1 >= argc) usage();
    const char* v = argv[++i];
    if (!strcmp(a, "--data")) o.data = v;
    else if (!strcmp(a, "--epochs")) o.epochs = atoi(v);
    else if (!strcmp(a, "--steps")) o.steps = atoi(v);
    else if (!strcmp(a, "--batch")) o.batch = atoi(v);
    else if (!strcmp(a, "--hidden")) o.hidden = atoi(v);
    else if (!strcmp(a, "--layers")) o.layers = atoi(v);
    else if (!strcmp(a, "--lr")) o.lr = (float)atof(v);
    else if (!strcmp(a, "--seed")) o.seed = (unsigned)atoi(v);
    else if (!strcmp(a, "--gemm")) { if ((o.gemm = gemm_parse(v)) < 0) usage(); }
    else if (!strcmp(a, "--allreduce")) { if ((o.allreduce = ar_parse(v)) < 0) usage(); }
    else usage();
  }
  if ((!o.data && !o.synthetic) || o.layers < 1 || o.layers > 14 || o.batch < 1) usage();
  return o;
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

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, p, lrank, ndev;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &lrank);
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(lrank % ndev));

  Opts o = parse(argc, argv);
  if (o.batch % p) {
    if (!rank) fprintf(stderr, "--batch %d is not divisible by %d ranks\n", o.batch, p);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  int b = o.batch / p;

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
  MLP* m = mlp_create(sizes, ns, b, o.seed, (gemm_impl)o.gemm);

  float* hX = malloc((size_t)b * train.dim * sizeof(float));
  float* hlogits = malloc((size_t)b * 10 * sizeof(float));
  int* hy = malloc(b * sizeof(int));
  float* dX;
  int* dy;
  CUDA_CHECK(cudaMalloc((void**)&dX, (size_t)b * train.dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dy, b * sizeof(int)));
  float* hgrad = NULL;  /* pinned staging buffer for the host all-reduce */
  if (p > 1) CUDA_CHECK(cudaMallocHost((void**)&hgrad, m->nparams * sizeof(float)));
  int* perm = malloc(train.n * sizeof(int));

  int steps_per_epoch = train.n / o.batch, step = 0, done = 0;
  for (int e = 0; e < o.epochs && !done; e++) {
    shuffle(perm, train.n, o.seed + e);
    for (int s = 0; s < steps_per_epoch && !done; s++) {
      /* rank r takes samples r, r+p, r+2p, ... of the global batch */
      for (int k = 0; k < b; k++) {
        int idx = perm[s * o.batch + rank + k * p];
        memcpy(hX + (size_t)k * train.dim, train.X + (size_t)idx * train.dim, train.dim * sizeof(float));
        hy[k] = train.y[idx];
      }
      CUDA_CHECK(cudaMemcpy(dX, hX, (size_t)b * train.dim * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dy, hy, b * sizeof(int), cudaMemcpyHostToDevice));

      double t0 = MPI_Wtime();
      float loss = mlp_forward_backward(m, dX, dy, b, NULL, NULL);
      CUDA_CHECK(cudaDeviceSynchronize());
      double t1 = MPI_Wtime();
      if (p > 1) {
        CUDA_CHECK(cudaMemcpy(hgrad, m->grads, m->nparams * sizeof(float), cudaMemcpyDeviceToHost));
        allreduce_sum(hgrad, m->nparams, (ar_algo)o.allreduce, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(m->grads, hgrad, m->nparams * sizeof(float), cudaMemcpyHostToDevice));
      }
      double t2 = MPI_Wtime();
      mlp_sgd(m, o.lr, 1.0f / p);  /* sum of per-rank means / p = global-batch mean */

      float gl;
      MPI_Allreduce(&loss, &gl, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      if (!rank) printf("step %d loss %.6f compute_s %.6f comm_s %.6f\n", step, gl / p, t1 - t0, t2 - t1);
      step++;
      if (o.steps && step >= o.steps) done = 1;
    }
    if (!rank) {
      printf("epoch %d test_acc %.4f\n", e, evaluate(m, &test, b, dX, hlogits));
      fflush(stdout);
    }
  }

  free(perm); free(hX); free(hlogits); free(hy);
  cudaFree(dX); cudaFree(dy);
  if (hgrad) cudaFreeHost(hgrad);
  mlp_free(m);
  dataset_free(&train);
  dataset_free(&test);
  MPI_Finalize();
  return 0;
}
