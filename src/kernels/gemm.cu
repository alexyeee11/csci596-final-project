#include <cublas_v2.h>
#include <mma.h>
#include <string.h>
#include <type_traits>
#include "cuda_check.h"
#include "kernels/gemm.h"
#include "kernels/gemm_h.h"

#define TILE 32

#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); \
    exit(1); } } while (0)

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

/* FP16 tensor-core GEMM: 64x64 block tile, K-step 32, 128 threads = 4 warps, each warp owning a 32x32 output
   (2x2 16x16x16 fragments). Tiles are staged in shared memory in their stored orientation (coalesced loads),
   so a transposed operand is just a col_major fragment view. Rows are padded by 8 halves, which keeps ldm a
   multiple of 8 and every fragment pointer 32-byte aligned (all offsets are multiples of 16 halves), whatever
   the global shape or leading dimension.
   The next K tile is prefetched into registers while the MMAs run on the current one (smem single-buffered).
   ponytail: larger block tiles (128x128, 8 warps) are what is left before chasing cuBLAS. */
#define WT 64
#define WK 32
#define WPAD 8  /* smem row strides (WK or WT) + WPAD = 40 and 72 halves = 80 and 144 bytes: 16-byte multiples */
#define WLDC (WT + 4)  /* FP32 epilogue tile row stride: multiple of 4 floats, fragment offsets multiples of 8 */

/* 16-byte path: each thread moves R*C/1024 uint4s (8 halves each). The tile must lie inside the matrix, and g
   and cols must keep every row start 16-byte aligned (c0 is a multiple of WK, so of 8). Load and store are
   separate so the load can be issued a K step early. */
template <int R, int C>
__device__ __forceinline__ void wmma_ld(uint4* v, const __half* g, int cols, int r0, int c0) {
#pragma unroll
  for (int t = 0; t < R * C / 1024; t++) {
    int idx = threadIdx.x + t * 128, r = idx / (C / 8), c = idx % (C / 8) * 8;
    v[t] = *(const uint4*)(g + (size_t)(r0 + r) * cols + c0 + c);
  }
}
template <int R, int C>
__device__ __forceinline__ void wmma_st(__half* s, const uint4* v) {
#pragma unroll
  for (int t = 0; t < R * C / 1024; t++) {
    int idx = threadIdx.x + t * 128, r = idx / (C / 8), c = idx % (C / 8) * 8;
    *(uint4*)(s + r * (C + WPAD) + c) = v[t];
  }
}

template <int R, int C>
__device__ __forceinline__ bool wmma_fast(const __half* g, int rows, int cols, int r0, int c0) {
  return r0 + R <= rows && c0 + C <= cols && cols % 8 == 0 && ((size_t)g & 15) == 0;
}

/* Copy the R x C tile at (r0, c0) of a rows x cols row-major matrix into s (row stride C + WPAD), zero-filling
   outside the matrix. The path choice depends only on kernel arguments and the tile, so it is block-uniform. */
template <int R, int C>
__device__ __forceinline__ void wmma_stage(__half* s, const __half* g, int rows, int cols, int r0, int c0) {
  if (wmma_fast<R, C>(g, rows, cols, r0, c0)) {
    uint4 v[R * C / 1024];
    wmma_ld<R, C>(v, g, cols, r0, c0);
    wmma_st<R, C>(s, v);
    return;
  }
#pragma unroll
  for (int t = 0; t < R * C / 128; t++) {
    int idx = threadIdx.x + t * 128, r = idx / C, c = idx % C;
    s[r * (C + WPAD) + c] = (r0 + r < rows && c0 + c < cols) ? g[(size_t)(r0 + r) * cols + c0 + c] : __float2half(0.f);
  }
}

template <bool TA, bool TB>
__global__ void __launch_bounds__(128) gemm_wmma_k(int M, int N, int K, float alpha, const __half* A,
                                                   const __half* B, float beta, float* C) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  typedef typename std::conditional<TA, wmma::col_major, wmma::row_major>::type LA;
  typedef typename std::conditional<TB, wmma::col_major, wmma::row_major>::type LB;
  /* A^T is staged k-major (WK x WT), A as WT x WK; likewise B^T as WT x WK, B as WK x WT */
  const int lda = TA ? WT + WPAD : WK + WPAD, ldb = TB ? WK + WPAD : WT + WPAD;
  /* the A and B tiles during the K loop, then the FP32 output tile for the epilogue */
  __shared__ __align__(32) unsigned char smem[WT * WLDC * sizeof(float)];
  __half *As = (__half*)smem, *Bs = As + WT * (WK + WPAD);
  int warp = threadIdx.x / 32, wi = (warp / 2) * 32, wj = (warp % 2) * 32;
  int i0 = blockIdx.y * WT, j0 = blockIdx.x * WT;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int c = 0; c < 2; c++) wmma::fill_fragment(acc[r][c], 0.f);
  /* stored tile shapes (rows x cols) and matrix shapes of A and B */
  constexpr int AR = TA ? WK : WT, AC = TA ? WT : WK, BR = TB ? WT : WK, BC = TB ? WK : WT;
  const int ar = TA ? K : M, ac = TA ? M : K, br = TB ? N : K, bc = TB ? K : N;
  wmma_stage<AR, AC>(As, A, ar, ac, TA ? 0 : i0, TA ? i0 : 0);
  wmma_stage<BR, BC>(Bs, B, br, bc, TB ? j0 : 0, TB ? 0 : j0);
  __syncthreads();
  for (int k0 = 0; k0 < K; k0 += WK) {
    /* issue the next tile's 16-byte loads now; they land in registers while the MMAs below run */
    int k1 = k0 + WK, ra = TA ? k1 : i0, ca = TA ? i0 : k1, rb = TB ? j0 : k1, cb = TB ? k1 : j0;
    bool more = k1 < K, fa = more && wmma_fast<AR, AC>(A, ar, ac, ra, ca),
         fb = more && wmma_fast<BR, BC>(B, br, bc, rb, cb);
    uint4 va[AR * AC / 1024], vb[BR * BC / 1024];
    if (fa) wmma_ld<AR, AC>(va, A, ac, ra, ca);
    if (fb) wmma_ld<BR, BC>(vb, B, bc, rb, cb);
#pragma unroll
    for (int kk = 0; kk < WK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, LA> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, LB> b[2];
#pragma unroll
      for (int f = 0; f < 2; f++) {
        int i = wi + f * 16, j = wj + f * 16;
        wmma::load_matrix_sync(a[f], As + (TA ? kk * lda + i : i * lda + kk), lda);
        wmma::load_matrix_sync(b[f], Bs + (TB ? j * ldb + kk : kk * ldb + j), ldb);
      }
#pragma unroll
      for (int r = 0; r < 2; r++)
#pragma unroll
        for (int c = 0; c < 2; c++) wmma::mma_sync(acc[r][c], a[r], b[c], acc[r][c]);
    }
    __syncthreads();  /* all MMA reads of this tile are done: As/Bs may be overwritten */
    if (more) {  /* block-uniform, so the barrier inside is safe */
      /* ponytail: tiles off the 16-byte path (edges, K tail, misaligned) are still loaded synchronously here */
      if (fa) wmma_st<AR, AC>(As, va); else wmma_stage<AR, AC>(As, A, ar, ac, ra, ca);
      if (fb) wmma_st<BR, BC>(Bs, vb); else wmma_stage<BR, BC>(Bs, B, br, bc, rb, cb);
      __syncthreads();
    }
  }
  /* fragments can't be stored to ragged global edges: go through shared memory. Reusing smem is safe because
     the barrier after the last tile's MMAs (or after the first staging, if K == 0) ends every use of As/Bs. */
  float* Cs = (float*)smem;
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int c = 0; c < 2; c++)
      wmma::store_matrix_sync(Cs + (wi + r * 16) * WLDC + wj + c * 16, acc[r][c], WLDC, wmma::mem_row_major);
  __syncthreads();
#pragma unroll
  for (int t = 0; t < WT * WT / 128; t++) {
    int idx = threadIdx.x + t * 128, r = idx / WT, c = idx % WT, i = i0 + r, j = j0 + c;
    if (i < M && j < N) {
      size_t o = (size_t)i * N + j;
      C[o] = alpha * Cs[r * WLDC + c] + (beta == 0.f ? 0.f : beta * C[o]);
    }
  }
#endif
}

static cublasHandle_t handle(void) {
  static cublasHandle_t h = NULL;
  if (!h) CUBLAS_CHECK(cublasCreate(&h));
  return h;
}

int gemm_parse(const char* s) {
  const char* names[] = {"naive", "tiled", "regtile", "cublas", "wmma"};
  for (int i = 0; i < 5; i++) if (!strcmp(s, names[i])) return i;
  return -1;
}

void gemm(gemm_impl impl, int transA, int transB, int M, int N, int K,
          float alpha, const float* A, const float* B, float beta, float* C) {
  if (impl == GEMM_WMMA) { fprintf(stderr, "wmma needs --fp16\n"); exit(1); }
  if (impl == GEMM_CUBLAS) {
    /* cuBLAS is column-major: compute C^T = op(B)^T * op(A)^T, which is row-major C. */
    CUBLAS_CHECK(cublasSgemm(handle(), transB ? CUBLAS_OP_T : CUBLAS_OP_N, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                N, M, K, &alpha, B, transB ? K : N, A, transA ? M : K, &beta, C, N));
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

int gemm_h_supported(void) {
  int dev, major;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  return major >= 7;
}

void gemm_h(gemmh_impl impl, int transA, int transB, int M, int N, int K,
            float alpha, const __half* A, const __half* B, float beta, float* C) {
  if (impl == GEMMH_CUBLAS) {
    /* same row-major trick as the FP32 call: C^T = op(B)^T * op(A)^T */
    CUBLAS_CHECK(cublasGemmEx(handle(), transB ? CUBLAS_OP_T : CUBLAS_OP_N, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                 N, M, K, &alpha, B, CUDA_R_16F, transB ? K : N, A, CUDA_R_16F, transA ? M : K,
                 &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return;
  }
  if (!gemm_h_supported()) { fprintf(stderr, "gemm_h: wmma needs an sm_70+ GPU (tensor cores)\n"); exit(1); }
  dim3 grd((N + WT - 1) / WT, (M + WT - 1) / WT);
  if (transA && transB) gemm_wmma_k<true, true><<<grd, 128>>>(M, N, K, alpha, A, B, beta, C);
  else if (transA)      gemm_wmma_k<true, false><<<grd, 128>>>(M, N, K, alpha, A, B, beta, C);
  else if (transB)      gemm_wmma_k<false, true><<<grd, 128>>>(M, N, K, alpha, A, B, beta, C);
  else                  gemm_wmma_k<false, false><<<grd, 128>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}
