#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/layers.h"

enum { R = 37, C = 10 };
static int fails = 0;
static unsigned seed = 1;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

static void check(const char* name, const float* got, const float* want, size_t n, float tol) {
  for (size_t i = 0; i < n; i++)
    if (fabsf(got[i] - want[i]) > tol * (1.f + fabsf(want[i]))) {
      printf("FAIL %s at %zu: got %g want %g\n", name, i, got[i], want[i]);
      fails++;
      return;
    }
}
static float* up(const float* h, size_t n) {
  float* d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}
static void down(float* h, const float* d, size_t n) {
  CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

int main(void) {
  float x[R * C], g[R * C], b[C], got[R * C], want[R * C];
  for (int i = 0; i < R * C; i++) { x[i] = rnd(); g[i] = rnd(); }
  for (int j = 0; j < C; j++) b[j] = rnd();

  for (int relu = 0; relu < 2; relu++) {
    float *dY = up(x, R * C), *db = up(b, C);
    bias_act_forward(dY, db, R, C, relu);
    down(got, dY, R * C);
    for (int i = 0; i < R * C; i++) { float v = x[i] + b[i % C]; want[i] = relu && v < 0 ? 0.f : v; }
    check(relu ? "bias_relu" : "bias", got, want, R * C, 1e-6f);
    cudaFree(dY); cudaFree(db);
  }

  {
    float *dG = up(g, R * C), *dY = up(x, R * C);
    relu_backward(dG, dY, R * C);
    down(got, dG, R * C);
    for (int i = 0; i < R * C; i++) want[i] = x[i] > 0 ? g[i] : 0.f;
    check("relu_backward", got, want, R * C, 0.f);
    cudaFree(dG); cudaFree(dY);
  }

  {
    float *dG = up(x, R * C), *dB;
    CUDA_CHECK(cudaMalloc(&dB, C * sizeof(float)));
    bias_grad(dB, dG, R, C);
    down(got, dB, C);
    for (int j = 0; j < C; j++) { want[j] = 0; for (int r = 0; r < R; r++) want[j] += x[r * C + j]; }
    check("bias_grad", got, want, C, 1e-5f);
    cudaFree(dG); cudaFree(dB);
  }

  {
    int y[R];
    for (int r = 0; r < R; r++) y[r] = r % C;
    int* dy;
    CUDA_CHECK(cudaMalloc(&dy, R * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dy, y, R * sizeof(int), cudaMemcpyHostToDevice));
    float *dx = up(x, R * C), *dd, *dl;
    CUDA_CHECK(cudaMalloc(&dd, R * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dl, R * sizeof(float)));
    float loss = softmax_xent(dx, dy, dd, dl, R, C);
    down(got, dd, R * C);
    double total = 0;
    for (int r = 0; r < R; r++) {
      const float* row = x + r * C;
      float m = row[0];
      for (int c = 1; c < C; c++) m = fmaxf(m, row[c]);
      double s = 0;
      for (int c = 0; c < C; c++) s += exp(row[c] - m);
      for (int c = 0; c < C; c++) want[r * C + c] = (float)((exp(row[c] - m) / s - (c == y[r])) / R);
      total += log(s) - (row[y[r]] - m);
    }
    check("softmax_xent_grad", got, want, R * C, 1e-5f);
    float wl = (float)(total / R);
    check("softmax_xent_loss", &loss, &wl, 1, 1e-5f);
    cudaFree(dy); cudaFree(dx); cudaFree(dd); cudaFree(dl);
  }

  {
    float *dw = up(x, R * C), *dg = up(g, R * C);
    sgd_step(dw, dg, 0.1f, R * C);
    down(got, dw, R * C);
    for (int i = 0; i < R * C; i++) want[i] = x[i] - 0.1f * g[i];
    check("sgd_step", got, want, R * C, 1e-6f);
    cudaFree(dw); cudaFree(dg);
  }

  printf(fails ? "test_layers: FAIL\n" : "test_layers: PASS\n");
  return fails != 0;
}
