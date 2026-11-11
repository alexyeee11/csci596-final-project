#include <stdio.h>
#include "cuda_check.h"
#include "kernels/gemm.h"

typedef struct { int M, N, K, ta, tb; } shape;

int main(void) {
  const char* names[] = {"naive", "tiled", "regtile", "cublas"};
  const shape shapes[] = {
    {256, 256, 256, 0, 0}, {512, 512, 512, 0, 0}, {1024, 1024, 1024, 0, 0},
    {2048, 2048, 2048, 0, 0}, {4096, 4096, 4096, 0, 0},
    /* training GEMMs at batch 512, hidden 4096 */
    {512, 4096, 3072, 0, 0},  /* forward, first layer */
    {512, 4096, 4096, 0, 0},  /* forward, hidden layer */
    {4096, 4096, 512, 1, 0},  /* dW = A^T dZ */
    {512, 4096, 4096, 0, 1},  /* dA = dZ W^T */
  };
  /* largest operand is 4096x4096 floats; contents don't affect timing */
  size_t n = (size_t)4096 * 4096;
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&C, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(A, 0, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(B, 0, n * sizeof(float)));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  printf("impl,M,N,K,transA,transB,seconds,gflops\n");
  for (int i = 0; i < 4; i++)
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++) {
      const shape* h = &shapes[s];
      gemm((gemm_impl)i, h->ta, h->tb, h->M, h->N, h->K, 1.f, A, B, 0.f, C);  /* warm-up */
      CUDA_CHECK(cudaDeviceSynchronize());
      /* average R back-to-back calls so launch/event overhead doesn't dominate small shapes;
         naive on big shapes takes seconds per call, so it runs once */
      int R = (i == GEMM_NAIVE && (double)h->M * h->N * h->K > 1e9) ? 1 : 10;
      CUDA_CHECK(cudaEventRecord(e0));
      for (int r = 0; r < R; r++)
        gemm((gemm_impl)i, h->ta, h->tb, h->M, h->N, h->K, 1.f, A, B, 0.f, C);
      CUDA_CHECK(cudaEventRecord(e1));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      double t = ms / 1e3 / R;
      printf("%s,%d,%d,%d,%d,%d,%.6f,%.2f\n", names[i], h->M, h->N, h->K, h->ta, h->tb, t,
             2.0 * h->M * h->N * h->K / t / 1e9);
    }
  return 0;
}
