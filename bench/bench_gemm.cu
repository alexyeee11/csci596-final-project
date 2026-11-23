#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_profiler_api.h>
#include "cuda_check.h"
#include "kernels/gemm.h"
#include "kernels/gemm_h.h"

typedef struct { int M, N, K, ta, tb; } shape;

__global__ void to_half_k(__half* d, const float* s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = __float2half(s[i]);
}

/* cheap deterministic hash of the index, mapped to [-1, 1) exactly (24 bits) */
__global__ void fill_k(float* d, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = (unsigned)i * 2654435761u;
  h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
  d[i] = (h >> 8) * (2.f / 16777216.f) - 1.f;
}

/* i: 0-3 FP32 gemm(); 4 = wmma, 5 = cublas_fp16 via gemm_h() */
static void run(int i, const shape* h, const float* A, const float* B, const __half* Ah, const __half* Bh, float* C) {
  if (i < 4) gemm((gemm_impl)i, h->ta, h->tb, h->M, h->N, h->K, 1.f, A, B, 0.f, C);
  else gemm_h(i == 4 ? GEMMH_WMMA : GEMMH_CUBLAS, h->ta, h->tb, h->M, h->N, h->K, 1.f, Ah, Bh, 0.f, C);
}

static const char* names[] = {"naive", "tiled", "regtile", "cublas", "wmma", "cublas_fp16"};
static cudaEvent_t e0, e1;

/* time one (already warmed-up) case and print its CSV row */
static void time_case(int i, const shape* h, const float* A, const float* B, const __half* Ah, const __half* Bh, float* C) {
  /* average R back-to-back calls so launch/event overhead doesn't dominate small shapes;
     naive on big shapes takes seconds per call, so it runs once */
  int R = (i == GEMM_NAIVE && (double)h->M * h->N * h->K > 1e9) ? 1 : 10;
  CUDA_CHECK(cudaEventRecord(e0));
  for (int r = 0; r < R; r++)
    run(i, h, A, B, Ah, Bh, C);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  double t = ms / 1e3 / R;
  printf("%s,%d,%d,%d,%d,%d,%.6f,%.2f\n", names[i], h->M, h->N, h->K, h->ta, h->tb, t,
         2.0 * h->M * h->N * h->K / t / 1e9);
}

static int usage(void) {
  fprintf(stderr, "usage: bench_gemm [naive|tiled|regtile|cublas|wmma|cublas_fp16 M N K ta tb]\n"
                  "  (wmma, cublas_fp16 need sm_70+; operands must fit 4096x4096)\n");
  return 2;
}

int main(int argc, char** argv) {
  const shape shapes[] = {
    {256, 256, 256, 0, 0}, {512, 512, 512, 0, 0}, {1024, 1024, 1024, 0, 0},
    {2048, 2048, 2048, 0, 0}, {4096, 4096, 4096, 0, 0},
    /* training GEMMs at batch 512, hidden 4096 */
    {512, 4096, 3072, 0, 0},  /* forward, first layer */
    {512, 4096, 4096, 0, 0},  /* forward, hidden layer */
    {4096, 4096, 512, 1, 0},  /* dW = A^T dZ */
    {512, 4096, 4096, 0, 1},  /* dA = dZ W^T */
  };
  /* largest operand is 4096x4096 floats */
  size_t n = (size_t)4096 * 4096;
  int half_ok = gemm_h_supported();
  /* single-case mode: bench_gemm impl M N K ta tb */
  int one = argc > 1, oi = 0;
  shape os = {0, 0, 0, 0, 0};
  if (one) {
    if (argc != 7) return usage();
    oi = -1;
    for (int i = 0; i < 6; i++) if (!strcmp(argv[1], names[i])) oi = i;
    os.M = atoi(argv[2]); os.N = atoi(argv[3]); os.K = atoi(argv[4]); os.ta = atoi(argv[5]); os.tb = atoi(argv[6]);
    if (oi < 0 || (oi >= 4 && !half_ok) || os.M < 1 || os.N < 1 || os.K < 1 || (unsigned)os.ta > 1 ||
        (unsigned)os.tb > 1 || (size_t)os.M * os.K > n || (size_t)os.K * os.N > n || (size_t)os.M * os.N > n)
      return usage();
  }
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&C, n * sizeof(float)));
  /* not zeros: zero operands lower switching power and inflate sustained clocks. Filled once, before any timing */
  fill_k<<<(unsigned)((n + 255) / 256), 256>>>(A, n);
  fill_k<<<(unsigned)((n + 255) / 256), 256>>>(B, n);
  CUDA_CHECK(cudaGetLastError());
  /* FP16 rows (wmma, cublas_fp16) only on sm_70+; inputs converted once, outside the timed loops */
  __half *Ah = NULL, *Bh = NULL;
  if (half_ok) {
    CUDA_CHECK(cudaMalloc(&Ah, n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&Bh, n * sizeof(__half)));
    to_half_k<<<(unsigned)((n + 255) / 256), 256>>>(Ah, A, n);
    to_half_k<<<(unsigned)((n + 255) / 256), 256>>>(Bh, B, n);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  if (one) {
    run(oi, &os, A, B, Ah, Bh, C);  /* warm-up */
    CUDA_CHECK(cudaDeviceSynchronize());
    /* exactly one call in the profiled region, for ncu --profile-from-start off */
    CUDA_CHECK(cudaProfilerStart());
    run(oi, &os, A, B, Ah, Bh, C);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaProfilerStop());
    time_case(oi, &os, A, B, Ah, Bh, C);
    return 0;
  }
  printf("impl,M,N,K,transA,transB,seconds,gflops\n");
  for (int i = 0; i < (half_ok ? 6 : 4); i++)
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++) {
      run(i, &shapes[s], A, B, Ah, Bh, C);  /* warm-up */
      CUDA_CHECK(cudaDeviceSynchronize());
      time_case(i, &shapes[s], A, B, Ah, Bh, C);
    }
  return 0;
}
