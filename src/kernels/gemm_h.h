#pragma once
/* FP16-input GEMM. Included from .cu files only (uses __half), so no extern "C" guard. */
#include <cuda_fp16.h>

typedef enum { GEMMH_WMMA, GEMMH_CUBLAS } gemmh_impl;

/* 1 when the current device is sm_70+ (has tensor cores for GEMMH_WMMA) */
int gemm_h_supported(void);
/* Same layout and transpose contract as gemm(): row-major, device pointers,
   C[MxN] = alpha * op(A)[MxK] * op(B)[KxN] + beta * C. FP16 inputs, FP32 accumulate and output.
   beta == 0: C not read. GEMMH_WMMA exits with a message on a pre-sm_70 device. */
void gemm_h(gemmh_impl impl, int transA, int transB, int M, int N, int K,
            float alpha, const __half* A, const __half* B, float beta, float* C);
