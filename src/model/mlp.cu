#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/gemm_h.h"
#include "kernels/layers.h"
#include "model/mlp.h"

static unsigned long long rng;
static double uniform01(void) {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((rng >> 11) + 0.5) / 9007199254740992.0;
}
static float normal(void) { return (float)(sqrt(-2.0 * log(uniform01())) * cos(6.283185307179586 * uniform01())); }

MLP* mlp_create_range(const int* sizes, int nsizes, int first, int last, int max_batch, unsigned seed,
                      gemm_impl impl) {
  MLP* m = (MLP*)calloc(1, sizeof(MLP));
  m->nlayers = last - first;
  m->first = first;
  m->head = last == nsizes - 1;
  m->max_batch = max_batch;
  m->impl = impl;
  m->loss_scale = 1.f;
  m->L = (Layer*)calloc(m->nlayers, sizeof(Layer));
  for (int l = first; l < last; l++) {
    m->nparams += (size_t)sizes[l] * sizes[l + 1] + sizes[l + 1];
    if (sizes[l + 1] > m->max_width) m->max_width = sizes[l + 1];
  }
  CUDA_CHECK(cudaMalloc(&m->params, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m->grads, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMemset(m->grads, 0, m->nparams * sizeof(float)));
  float* h = (float*)calloc(m->nparams, sizeof(float));
  rng = seed;
  size_t off = 0;
  for (int g = 0; g < last; g++) {  /* draw every earlier layer's weights too, keeping only the range */
    float sd = sqrtf(2.0f / sizes[g]);  /* He init; biases stay zero */
    size_t nw = (size_t)sizes[g] * sizes[g + 1];
    if (g < first) { for (size_t i = 0; i < nw; i++) normal(); continue; }
    Layer* L = &m->L[g - first];
    L->in = sizes[g];
    L->out = sizes[g + 1];
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

MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl) {
  return mlp_create_range(sizes, nsizes, 0, nsizes - 1, max_batch, seed, impl);
}

void mlp_enable_fp16(MLP* m) {
  if (!gemm_h_supported()) { fprintf(stderr, "--fp16 needs an sm_70+ GPU (tensor cores)\n"); exit(1); }
  CUDA_CHECK(cudaMalloc(&m->params16, m->nparams * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&m->X16, (size_t)m->max_batch * m->L[0].in * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&m->dz16, (size_t)m->max_batch * m->max_width * sizeof(__half)));
  for (int l = 0; l < m->nlayers; l++) {
    Layer* L = &m->L[l];
    L->W16 = (__half*)m->params16 + (L->W - m->params);
    CUDA_CHECK(cudaMalloc(&L->Y16, (size_t)m->max_batch * L->out * sizeof(__half)));
  }
  f32_to_f16((__half*)m->params16, m->params, m->nparams);
  /* ponytail: static loss scale. A scaled dZ above 65504 (unscaled |dZ| > 64) becomes inf in its FP16 copy and
     silently poisons the step; one below ~6e-8 / 1024 flushes to 0. Upgrade: dynamic scaling (check grads for
     inf/nan, skip the step and halve the scale, double it after N clean steps). */
  m->loss_scale = 1024.f;
}

void mlp_free(MLP* m) {
  for (int l = 0; l < m->nlayers; l++) { cudaFree(m->L[l].Y); cudaFree(m->L[l].Y16); }
  cudaFree(m->params16); cudaFree(m->X16); cudaFree(m->dz16);
  cudaFree(m->params); cudaFree(m->grads);
  cudaFree(m->dbuf[0]); cudaFree(m->dbuf[1]); cudaFree(m->loss_buf);
  free(m->L);
  free(m);
}

/* C (+)= op(A) op(B), through the FP16 copies A16, B16 once mlp_enable_fp16 ran, else in FP32 */
static void mm(const MLP* m, int ta, int tb, int M, int N, int K, const float* A, const void* A16, const float* B,
               const void* B16, float beta, float* C) {
  if (m->params16)
    gemm_h(m->impl == GEMM_CUBLAS ? GEMMH_CUBLAS : GEMMH_WMMA, ta, tb, M, N, K, 1.f, (const __half*)A16,
           (const __half*)B16, beta, C);
  else
    gemm(m->impl, ta, tb, M, N, K, 1.f, A, B, beta, C);
}

float* mlp_stage_forward(MLP* m, const float* X, int b, int k) {
  const float* A = X;
  const __half* A16 = (const __half*)m->X16;
  if (m->params16) f32_to_f16((__half*)m->X16, X, (size_t)b * m->L[0].in);
  float* Y = NULL;
  for (int l = 0; l < m->nlayers; l++) {
    Layer* L = &m->L[l];
    Y = L->Y + (size_t)k * b * L->out;
    __half* Y16 = m->params16 ? (__half*)L->Y16 + (size_t)k * b * L->out : NULL;
    mm(m, 0, 0, b, L->out, L->in, A, A16, L->W, L->W16, 0.f, Y);       /* Y = A W */
    bias_act_forward(Y, L->b, b, L->out, !(m->head && l == m->nlayers - 1));
    if (Y16) f32_to_f16(Y16, Y, (size_t)b * L->out);
    A = Y;
    A16 = Y16;
  }
  return Y;
}

const float* mlp_forward(MLP* m, const float* X, int B) { return mlp_stage_forward(m, X, B, 0); }

/* dZ of the stage's last layer lives in dOut (head) or dbuf[1] (dOut times the ReLU mask), so the first dA goes to
   dbuf[0] and the two then alternate. acc: add to grads instead of overwriting them. hook (may be NULL) sees each
   layer's grads once they are final. */
static void stage_backward(MLP* m, const float* X, const float* dOut, float* dX, int b, int k, int acc,
                           grad_hook hook, void* ctx) {
  int last = m->nlayers - 1, nxt = 0;
  size_t r0 = (size_t)k * b;
  float beta = acc ? 1.f : 0.f;
  const float* dZ = dOut;
  __half* dz16 = (__half*)m->dz16;  /* FP16 copy of dZ for the two GEMMs of each layer; NULL in FP32 */
  if (!m->head) {
    /* ponytail: one device copy so dOut stays const; fold the mask into the caller if it shows up in profiles */
    size_t n = (size_t)b * m->L[last].out;
    CUDA_CHECK(cudaMemcpy(m->dbuf[1], dOut, n * sizeof(float), cudaMemcpyDeviceToDevice));
    relu_backward(m->dbuf[1], m->L[last].Y + r0 * m->L[last].out, n);
    dZ = m->dbuf[1];
  }
  for (int l = last; l >= 0; l--) {
    Layer* L = &m->L[l];
    const float* Ain = l > 0 ? m->L[l - 1].Y + r0 * L->in : X;
    const __half* Ain16 = !dz16 ? NULL : l > 0 ? (const __half*)m->L[l - 1].Y16 + r0 * L->in : (const __half*)m->X16;
    if (dz16) f32_to_f16(dz16, dZ, (size_t)b * L->out);
    mm(m, 1, 0, L->in, L->out, b, Ain, Ain16, dZ, dz16, beta, L->dW);  /* dW (+)= Ain^T dZ */
    bias_grad(L->db, dZ, b, L->out, acc);
    if (hook) hook(l, L->dW, (size_t)L->in * L->out + L->out, ctx);
    if (l > 0) {
      float* dA = m->dbuf[nxt];
      mm(m, 0, 1, b, L->in, L->out, dZ, dz16, L->W, L->W16, 0.f, dA);    /* dA = dZ W^T */
      relu_backward(dA, Ain, (size_t)b * L->in);
      dZ = dA;
      nxt ^= 1;
    } else if (dX) {
      mm(m, 0, 1, b, L->in, L->out, dZ, dz16, L->W, L->W16, 0.f, dX);    /* stage input is post-ReLU: no mask */
    }
  }
}

void mlp_stage_backward(MLP* m, const float* X, const float* dOut, float* dX, int b, int k) {
  /* X16 holds only the last forward's input and dOut is not loss-scaled */
  if (m->params16) { fprintf(stderr, "mlp_stage_backward: FP16 is not supported by the stage API\n"); exit(1); }
  stage_backward(m, X, dOut, dX, b, k, 1, NULL, NULL);
}

float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx) {
  const float* logits = mlp_forward(m, X, B);
  int classes = m->L[m->nlayers - 1].out;
  float loss = softmax_xent(logits, y, m->dbuf[1], m->loss_buf, B, classes);
  if (m->params16) scale_f32(m->dbuf[1], m->loss_scale, (size_t)B * classes);  /* every grad is now scaled */
  stage_backward(m, X, m->dbuf[1], NULL, B, 0, 0, hook, ctx);
  return loss;
}

void mlp_zero_grads(MLP* m) { CUDA_CHECK(cudaMemset(m->grads, 0, m->nparams * sizeof(float))); }

void mlp_sgd(MLP* m, float lr, float grad_scale) {
  sgd_step(m->params, m->grads, lr * grad_scale / m->loss_scale, m->nparams);
  if (m->params16) f32_to_f16((__half*)m->params16, m->params, m->nparams);
}
