#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/gemm.h"
#include "kernels/gemm_h.h"

static unsigned seed = 1;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

/* scale gets alpha * sum|a||b| + |beta * C| per element: the size FP32 accumulation error scales with */
static void ref(int ta, int tb, int M, int N, int K, float alpha, const float* A, const float* B, float beta,
                float* C, float* scale) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double acc = 0, s = 0;
      for (int k = 0; k < K; k++) {
        float a = ta ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
        float b = tb ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
        acc += (double)a * b;
        s += fabs((double)a * b);
      }
      size_t c = (size_t)i * N + j;
      scale[c] = (float)(alpha * s + fabs(beta * C[c]));
      C[c] = (float)(alpha * acc + beta * C[c]);
    }
}

/* fp16: impl is a gemmh_impl, and A and B are rounded to FP16 before both the GPU call and the reference,
   so only FP32 accumulation order separates the two. Returns 1 on mismatch. */
static int run_case(int fp16, int impl, const char* name, int M, int N, int K, int ta, int tb) {
  size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N, es = fp16 ? sizeof(__half) : sizeof(float);
  float *hA = (float*)malloc(na * 4), *hB = (float*)malloc(nb * 4), *hC = (float*)malloc(nc * 4);
  float *want = (float*)malloc(nc * 4), *scale = (float*)malloc(nc * 4);
  __half *hAh = (__half*)malloc(na * 2), *hBh = (__half*)malloc(nb * 2);
  for (size_t i = 0; i < na; i++) { hA[i] = rnd(); hAh[i] = __float2half(hA[i]); if (fp16) hA[i] = __half2float(hAh[i]); }
  for (size_t i = 0; i < nb; i++) { hB[i] = rnd(); hBh[i] = __float2half(hB[i]); if (fp16) hB[i] = __half2float(hBh[i]); }
  for (size_t i = 0; i < nc; i++) want[i] = hC[i] = rnd();
  ref(ta, tb, M, N, K, 1.5f, hA, hB, 0.5f, want, scale);
  void *dA, *dB;
  float* dC;
  CUDA_CHECK(cudaMalloc(&dA, na * es));
  CUDA_CHECK(cudaMalloc(&dB, nb * es));
  CUDA_CHECK(cudaMalloc(&dC, nc * 4));
  CUDA_CHECK(cudaMemcpy(dA, fp16 ? (void*)hAh : (void*)hA, na * es, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, fp16 ? (void*)hBh : (void*)hB, nb * es, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC, nc * 4, cudaMemcpyHostToDevice));
  if (fp16) gemm_h((gemmh_impl)impl, ta, tb, M, N, K, 1.5f, (const __half*)dA, (const __half*)dB, 0.5f, dC);
  else      gemm((gemm_impl)impl, ta, tb, M, N, K, 1.5f, (const float*)dA, (const float*)dB, 0.5f, dC);
  CUDA_CHECK(cudaMemcpy(hC, dC, nc * 4, cudaMemcpyDeviceToHost));
  int fail = 0;
  for (size_t i = 0; i < nc && !fail; i++) {
    float tol = fp16 ? 1e-3f * scale[i] : 1e-3f * (1.f + fabsf(want[i]));
    if (fabsf(hC[i] - want[i]) > tol) {
      printf("FAIL %s %dx%dx%d ta=%d tb=%d at %zu: got %g want %g\n", name, M, N, K, ta, tb, i, hC[i], want[i]);
      fail = 1;
    }
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(hA); free(hB); free(hC); free(want); free(scale); free(hAh); free(hBh);
  return fail;
}

/* 5 shapes x 4 transpose combos. 300x400x129 spans several 128x128 regtile blocks (and 64x64 wmma blocks)
   with ragged edges in M, N and K */
static int run_all(int fp16, int impl, const char* name) {
  const int shapes[][3] = {{1, 1, 1}, {33, 17, 65}, {128, 256, 64}, {257, 129, 300}, {300, 400, 129}};
  int fails = 0;
  for (int s = 0; s < 5; s++)
    for (int ta = 0; ta < 2; ta++)
      for (int tb = 0; tb < 2; tb++)
        fails += run_case(fp16, impl, name, shapes[s][0], shapes[s][1], shapes[s][2], ta, tb);
  return fails;
}

int main(void) {
  const char* names[] = {"naive", "tiled", "regtile", "cublas", "wmma"};
  int fails = 0;
  /* an unparsed name would silently fall through to the tiled kernel and still pass below */
  for (int impl = 0; impl <= GEMM_WMMA; impl++)
    if (gemm_parse(names[impl]) != impl) {
      printf("FAIL gemm_parse(\"%s\") = %d, want %d\n", names[impl], gemm_parse(names[impl]), impl);
      fails++;
    }
  /* FP32 gemm() rejects GEMM_WMMA (it exits), so the FP32 loop stops at cuBLAS */
  for (int impl = 0; impl <= GEMM_CUBLAS; impl++) fails += run_all(0, impl, names[impl]);
  if (gemm_h_supported()) {
    fails += run_all(1, GEMMH_WMMA, "wmma");
    fails += run_all(1, GEMMH_CUBLAS, "cublas_fp16");
  } else {
    printf("test_gemm: fp16 cases skipped (needs sm_70+)\n");
  }
  printf(fails ? "test_gemm: FAIL (%d)\n" : "test_gemm: PASS\n", fails);
  return fails != 0;
}
