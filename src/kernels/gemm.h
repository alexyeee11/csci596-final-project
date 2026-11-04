#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { GEMM_NAIVE, GEMM_TILED, GEMM_CUBLAS } gemm_impl;

int gemm_parse(const char* s);  /* -1 if unknown */
/* Row-major, device pointers: C[MxN] = alpha * op(A)[MxK] * op(B)[KxN] + beta * C.
   transA: A stored KxM, else MxK. transB: B stored NxK, else KxN. beta == 0: C not read. */
void gemm(gemm_impl impl, int transA, int transB, int M, int N, int K,
          float alpha, const float* A, const float* B, float beta, float* C);

#ifdef __cplusplus
}
#endif
