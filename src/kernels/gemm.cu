#include <cublas_v2.h>
#include <string.h>
#include "cuda_check.h"
#include "kernels/gemm.h"

#define TILE 32

__device__ __forceinline__ float ldA(const float* A, int ta, int M, int K, int i, int k) {
  return ta ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
}
__device__ __forceinline__ float ldB(const float* B, int tb, int N, int K, int k, int j) {
  return tb ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
}

__global__ void gemm_naive_k(int ta, int tb, int M, int N, int K, float alpha,
                             const float* A, const float* B, float beta, float* C) {
  int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; k++) acc += ldA(A, ta, M, K, i, k) * ldB(B, tb, N, K, k, j);
  size_t c = (size_t)i * N + j;
  C[c] = alpha * acc + (beta == 0.f ? 0.f : beta * C[c]);
}

/* Each 32x32 block stages one tile of op(A) and op(B) in shared memory per k-step. */
__global__ void gemm_tiled_k(int ta, int tb, int M, int N, int K, float alpha,
                             const float* A, const float* B, float beta, float* C) {
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];
  int ty = threadIdx.y, tx = threadIdx.x;
  int i = blockIdx.y * TILE + ty, j = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    As[ty][tx] = (i < M && k0 + tx < K) ? ldA(A, ta, M, K, i, k0 + tx) : 0.f;
    Bs[ty][tx] = (k0 + ty < K && j < N) ? ldB(B, tb, N, K, k0 + ty, j) : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (i < M && j < N) {
    size_t c = (size_t)i * N + j;
    C[c] = alpha * acc + (beta == 0.f ? 0.f : beta * C[c]);
  }
}

static cublasHandle_t handle(void) {
  static cublasHandle_t h = NULL;
  if (!h) cublasCreate(&h);
  return h;
}

int gemm_parse(const char* s) {
  const char* names[] = {"naive", "tiled", "cublas"};
  for (int i = 0; i < 3; i++) if (!strcmp(s, names[i])) return i;
  return -1;
}

void gemm(gemm_impl impl, int transA, int transB, int M, int N, int K,
          float alpha, const float* A, const float* B, float beta, float* C) {
  if (impl == GEMM_CUBLAS) {
    /* cuBLAS is column-major: compute C^T = op(B)^T * op(A)^T, which is row-major C. */
    cublasSgemm(handle(), transB ? CUBLAS_OP_T : CUBLAS_OP_N, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                N, M, K, &alpha, B, transB ? K : N, A, transA ? M : K, &beta, C, N);
    return;
  }
  dim3 blk(TILE, TILE), grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  if (impl == GEMM_NAIVE) gemm_naive_k<<<grd, blk>>>(transA, transB, M, N, K, alpha, A, B, beta, C);
  else                    gemm_tiled_k<<<grd, blk>>>(transA, transB, M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}
