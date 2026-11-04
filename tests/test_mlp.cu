#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "model/mlp.h"

static unsigned seed = 7;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

int main(void) {
  const int sizes[] = {8, 16, 4};
  enum { B = 3 };
  float X[B * 8];
  for (int i = 0; i < B * 8; i++) X[i] = rnd();
  int y[B] = {0, 3, 1};
  float* dX; int* dy;
  CUDA_CHECK(cudaMalloc(&dX, sizeof X));
  CUDA_CHECK(cudaMalloc(&dy, sizeof y));
  CUDA_CHECK(cudaMemcpy(dX, X, sizeof X, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, y, sizeof y, cudaMemcpyHostToDevice));
  int fails = 0;

  /* 1. analytic gradients match central differences */
  MLP* m = mlp_create(sizes, 3, B, 123, GEMM_TILED);
  size_t n = m->nparams;
  float *ana = (float*)malloc(n * 4), *par = (float*)malloc(n * 4), *g2 = (float*)malloc(n * 4);
  mlp_forward_backward(m, dX, dy, B, NULL, NULL);
  CUDA_CHECK(cudaMemcpy(ana, m->grads, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(par, m->params, n * 4, cudaMemcpyDeviceToHost));
  const float eps = 1e-3f;
  for (size_t i = 0; i < n; i++) {
    float v = par[i] + eps;
    CUDA_CHECK(cudaMemcpy(m->params + i, &v, 4, cudaMemcpyHostToDevice));
    float lp = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
    v = par[i] - eps;
    CUDA_CHECK(cudaMemcpy(m->params + i, &v, 4, cudaMemcpyHostToDevice));
    float lm = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
    CUDA_CHECK(cudaMemcpy(m->params + i, &par[i], 4, cudaMemcpyHostToDevice));
    float num = (lp - lm) / (2 * eps);
    if (fabsf(num - ana[i]) > 2e-3f + 2e-2f * fabsf(ana[i])) {
      if (fails < 5) printf("FAIL grad[%zu]: numeric %g analytic %g\n", i, num, ana[i]);
      fails++;
    }
  }

  /* 2. every GEMM implementation gives the same gradients */
  for (int impl = GEMM_NAIVE; impl <= GEMM_CUBLAS; impl++) {
    MLP* m2 = mlp_create(sizes, 3, B, 123, (gemm_impl)impl);
    mlp_forward_backward(m2, dX, dy, B, NULL, NULL);
    CUDA_CHECK(cudaMemcpy(g2, m2->grads, n * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++)
      if (fabsf(g2[i] - ana[i]) > 1e-4f * (1.f + fabsf(ana[i]))) {
        printf("FAIL impl %d grad[%zu]: %g vs %g\n", impl, i, g2[i], ana[i]);
        fails++;
        break;
      }
    mlp_free(m2);
  }

  /* 3. SGD overfits a fixed batch */
  float first = mlp_forward_backward(m, dX, dy, B, NULL, NULL), last = first;
  for (int it = 0; it < 500; it++) {
    mlp_sgd(m, 0.2f, 1.0f);
    last = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
  }
  if (!(last < 0.1f * first)) { printf("FAIL overfit: loss %g -> %g\n", first, last); fails++; }

  mlp_free(m);
  free(ana); free(par); free(g2);
  printf(fails ? "test_mlp: FAIL (%d)\n" : "test_mlp: PASS\n", fails);
  return fails != 0;
}
