#include <math.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/layers.h"
#include "model/mlp.h"

static unsigned long long rng;
static double uniform01(void) {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((rng >> 11) + 0.5) / 9007199254740992.0;
}
static float normal(void) { return (float)(sqrt(-2.0 * log(uniform01())) * cos(6.283185307179586 * uniform01())); }

MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl) {
  MLP* m = (MLP*)calloc(1, sizeof(MLP));
  m->nlayers = nsizes - 1;
  m->max_batch = max_batch;
  m->impl = impl;
  m->L = (Layer*)calloc(m->nlayers, sizeof(Layer));
  for (int l = 0; l < m->nlayers; l++) {
    m->nparams += (size_t)sizes[l] * sizes[l + 1] + sizes[l + 1];
    if (sizes[l + 1] > m->max_width) m->max_width = sizes[l + 1];
  }
  CUDA_CHECK(cudaMalloc(&m->params, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m->grads, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMemset(m->grads, 0, m->nparams * sizeof(float)));
  float* h = (float*)calloc(m->nparams, sizeof(float));
  rng = seed;
  size_t off = 0;
  for (int l = 0; l < m->nlayers; l++) {
    Layer* L = &m->L[l];
    L->in = sizes[l];
    L->out = sizes[l + 1];
    size_t nw = (size_t)L->in * L->out;
    float sd = sqrtf(2.0f / L->in);  /* He init; biases stay zero */
    for (size_t i = 0; i < nw; i++) h[off + i] = sd * normal();
    L->W = m->params + off;  L->b = L->W + nw;
    L->dW = m->grads + off;  L->db = L->dW + nw;
    CUDA_CHECK(cudaMalloc(&L->Y, (size_t)max_batch * L->out * sizeof(float)));
    off += nw + L->out;
  }
  CUDA_CHECK(cudaMemcpy(m->params, h, m->nparams * sizeof(float), cudaMemcpyHostToDevice));
  free(h);
  for (int i = 0; i < 2; i++)
    CUDA_CHECK(cudaMalloc(&m->dbuf[i], (size_t)max_batch * m->max_width * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m->loss_buf, max_batch * sizeof(float)));
  return m;
}

void mlp_free(MLP* m) {
  for (int l = 0; l < m->nlayers; l++) cudaFree(m->L[l].Y);
  cudaFree(m->params); cudaFree(m->grads);
  cudaFree(m->dbuf[0]); cudaFree(m->dbuf[1]); cudaFree(m->loss_buf);
  free(m->L);
  free(m);
}

const float* mlp_forward(MLP* m, const float* X, int B) {
  const float* A = X;
  for (int l = 0; l < m->nlayers; l++) {
    Layer* L = &m->L[l];
    gemm(m->impl, 0, 0, B, L->out, L->in, 1.f, A, L->W, 0.f, L->Y);     /* Y = A W */
    bias_act_forward(L->Y, L->b, B, L->out, l < m->nlayers - 1);
    A = L->Y;
  }
  return A;
}

float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx) {
  const float* logits = mlp_forward(m, X, B);
  int last = m->nlayers - 1, cur = 0;
  float loss = softmax_xent(logits, y, m->dbuf[0], m->loss_buf, B, m->L[last].out);
  for (int l = last; l >= 0; l--) {
    Layer* L = &m->L[l];
    const float* Ain = l > 0 ? m->L[l - 1].Y : X;
    float* dZ = m->dbuf[cur];
    gemm(m->impl, 1, 0, L->in, L->out, B, 1.f, Ain, dZ, 0.f, L->dW);    /* dW = Ain^T dZ */
    bias_grad(L->db, dZ, B, L->out);
    if (hook) hook(l, L->dW, (size_t)L->in * L->out + L->out, ctx);
    if (l > 0) {
      float* dA = m->dbuf[cur ^ 1];
      gemm(m->impl, 0, 1, B, L->in, L->out, 1.f, dZ, L->W, 0.f, dA);    /* dA = dZ W^T */
      relu_backward(dA, Ain, (size_t)B * L->in);
      cur ^= 1;
    }
  }
  return loss;
}

void mlp_sgd(MLP* m, float lr, float grad_scale) {
  sgd_step(m->params, m->grads, lr * grad_scale, m->nparams);
}
