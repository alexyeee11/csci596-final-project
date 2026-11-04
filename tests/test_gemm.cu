#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/gemm.h"

static unsigned seed = 1;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

static void ref(int ta, int tb, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double acc = 0;
      for (int k = 0; k < K; k++) {
        float a = ta ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
        float b = tb ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
        acc += (double)a * b;
      }
      C[(size_t)i * N + j] = (float)(alpha * acc + beta * C[(size_t)i * N + j]);
    }
}

int main(void) {
  const int shapes[][3] = {{1, 1, 1}, {33, 17, 65}, {128, 256, 64}, {257, 129, 300}};
  const char* names[] = {"naive", "tiled", "cublas"};
  int fails = 0;
  for (int impl = 0; impl < 3; impl++)
    for (int s = 0; s < 4; s++)
      for (int ta = 0; ta < 2; ta++)
        for (int tb = 0; tb < 2; tb++) {
          int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
          size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
          float *hA = (float*)malloc(na * 4), *hB = (float*)malloc(nb * 4);
          float *hC = (float*)malloc(nc * 4), *want = (float*)malloc(nc * 4);
          for (size_t i = 0; i < na; i++) hA[i] = rnd();
          for (size_t i = 0; i < nb; i++) hB[i] = rnd();
          for (size_t i = 0; i < nc; i++) want[i] = hC[i] = rnd();
          ref(ta, tb, M, N, K, 1.5f, hA, hB, 0.5f, want);
          float *dA, *dB, *dC;
          CUDA_CHECK(cudaMalloc(&dA, na * 4));
          CUDA_CHECK(cudaMalloc(&dB, nb * 4));
          CUDA_CHECK(cudaMalloc(&dC, nc * 4));
          CUDA_CHECK(cudaMemcpy(dA, hA, na * 4, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(dB, hB, nb * 4, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(dC, hC, nc * 4, cudaMemcpyHostToDevice));
          gemm((gemm_impl)impl, ta, tb, M, N, K, 1.5f, dA, dB, 0.5f, dC);
          CUDA_CHECK(cudaMemcpy(hC, dC, nc * 4, cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < nc; i++)
            if (fabsf(hC[i] - want[i]) > 1e-3f * (1.f + fabsf(want[i]))) {
              printf("FAIL %s %dx%dx%d ta=%d tb=%d at %zu: got %g want %g\n",
                     names[impl], M, N, K, ta, tb, i, hC[i], want[i]);
              fails++;
              break;
            }
          cudaFree(dA); cudaFree(dB); cudaFree(dC);
          free(hA); free(hB); free(hC); free(want);
        }
  printf(fails ? "test_gemm: FAIL (%d)\n" : "test_gemm: PASS\n", fails);
  return fails != 0;
}
