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

/* 128x128 block tile, K-step 8, 256 threads; each thread keeps an 8x8 sub-tile of C in registers.
   op(A) and op(B) tiles are staged k-major, so a thread's 8 A and 8 B values per k are two float4 loads. */
#define RT 128
#define RK 8
__global__ void __launch_bounds__(256) gemm_regtile_k(int ta, int tb, int M, int N, int K, float alpha,
                                                       const float* A, const float* B, float beta, float* C) {
  __shared__ __align__(16) float As[RK][RT], Bs[RK][RT];
  int t = threadIdx.x, tx = t % 16, ty = t / 16;
  int i0 = blockIdx.y * RT, j0 = blockIdx.x * RT;
  float acc[8][8] = {};
  for (int k0 = 0; k0 < K; k0 += RK) {
    /* 1024 values per tile, 4 per thread; the contiguous index of each operand varies fastest.
       ponytail: k-major stores bank-conflict (8-way) when k is contiguous (A plain, B transposed);
       pad or transpose through registers only if profiling shows it matters. */
    for (int r = 0; r < 4; r++) {
      int idx = t + r * 256;
      int ka = ta ? idx / RT : idx % RK, ia = ta ? idx % RT : idx / RK;
      int kb = tb ? idx % RK : idx / RT, jb = tb ? idx / RK : idx % RT;
      As[ka][ia] = (i0 + ia < M && k0 + ka < K) ? ldA(A, ta, M, K, i0 + ia, k0 + ka) : 0.f;
      Bs[kb][jb] = (k0 + kb < K && j0 + jb < N) ? ldB(B, tb, N, K, k0 + kb, j0 + jb) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < RK; k++) {
      float4 a0 = *(const float4*)&As[k][ty * 8], a1 = *(const float4*)&As[k][ty * 8 + 4];
      float4 b0 = *(const float4*)&Bs[k][tx * 8], b1 = *(const float4*)&Bs[k][tx * 8 + 4];
      float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int r = 0; r < 8; r++)
#pragma unroll
        for (int c = 0; c < 8; c++) acc[r][c] += a[r] * b[c];
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 8; r++) {
    int i = i0 + ty * 8 + r;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      int j = j0 + tx * 8 + c;
      if (i < M && j < N) {
        size_t o = (size_t)i * N + j;
        C[o] = alpha * acc[r][c] + (beta == 0.f ? 0.f : beta * C[o]);
      }
    }
  }
}

static cublasHandle_t handle(void) {
  static cublasHandle_t h = NULL;
  if (!h) cublasCreate(&h);
  return h;
}

int gemm_parse(const char* s) {
  const char* names[] = {"naive", "tiled", "regtile", "cublas"};
  for (int i = 0; i < 4; i++) if (!strcmp(s, names[i])) return i;
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
  if (impl == GEMM_REGTILE) {
    gemm_regtile_k<<<dim3((N + RT - 1) / RT, (M + RT - 1) / RT), 256>>>(transA, transB, M, N, K,
                                                                     alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  dim3 blk(TILE, TILE), grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  if (impl == GEMM_NAIVE) gemm_naive_k<<<grd, blk>>>(transA, transB, M, N, K, alpha, A, B, beta, C);
  else                    gemm_tiled_k<<<grd, blk>>>(transA, transB, M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}
