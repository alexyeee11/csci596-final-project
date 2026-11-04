#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/layers.h"

static unsigned blocks(size_t n) { return (unsigned)((n + 255) / 256); }

__global__ void bias_act_k(float* Y, const float* b, int rows, int cols, int relu) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * cols) return;
  float v = Y[i] + b[i % cols];
  Y[i] = relu && v < 0.f ? 0.f : v;
}

__global__ void relu_bwd_k(float* dY, const float* Y, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && Y[i] <= 0.f) dY[i] = 0.f;
}

__global__ void bias_grad_k(float* db, const float* dY, int rows, int cols) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  float s = 0.f;
  for (int r = 0; r < rows; r++) s += dY[(size_t)r * cols + j];
  db[j] = s;
}

/* One thread per row; fine for 10 classes. */
__global__ void softmax_xent_k(const float* x, const int* y, float* dx, float* loss, int rows, int C) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* row = x + (size_t)r * C;
  float m = row[0];
  for (int c = 1; c < C; c++) m = fmaxf(m, row[c]);
  float s = 0.f;
  for (int c = 0; c < C; c++) s += expf(row[c] - m);
  float inv = 1.f / rows;
  for (int c = 0; c < C; c++) dx[(size_t)r * C + c] = (expf(row[c] - m) / s - (c == y[r])) * inv;
  loss[r] = logf(s) - (row[y[r]] - m);
}

__global__ void sgd_k(float* w, const float* g, float lr, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) w[i] -= lr * g[i];
}

void bias_act_forward(float* Y, const float* b, int rows, int cols, int relu) {
  bias_act_k<<<blocks((size_t)rows * cols), 256>>>(Y, b, rows, cols, relu);
  CUDA_CHECK(cudaGetLastError());
}

void relu_backward(float* dY, const float* Y, size_t n) {
  relu_bwd_k<<<blocks(n), 256>>>(dY, Y, n);
  CUDA_CHECK(cudaGetLastError());
}

void bias_grad(float* db, const float* dY, int rows, int cols) {
  bias_grad_k<<<blocks(cols), 256>>>(db, dY, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

float softmax_xent(const float* logits, const int* labels, float* dlogits, float* loss_buf,
                   int rows, int classes) {
  softmax_xent_k<<<blocks(rows), 256>>>(logits, labels, dlogits, loss_buf, rows, classes);
  CUDA_CHECK(cudaGetLastError());
  /* ponytail: per-row losses summed on the host; a device reduction only if this shows up in profiles */
  float* h = (float*)malloc(rows * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h, loss_buf, rows * sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0;
  for (int r = 0; r < rows; r++) s += h[r];
  free(h);
  return (float)(s / rows);
}

void sgd_step(float* w, const float* g, float lr, size_t n) {
  sgd_k<<<blocks(n), 256>>>(w, g, lr, n);
  CUDA_CHECK(cudaGetLastError());
}
