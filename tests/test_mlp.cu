#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_check.h"
#include "kernels/gemm_h.h"
#include "kernels/layers.h"
#include "model/mlp.h"

static unsigned seed = 7;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

int main(void) {
  const int sizes[] = {8, 16, 4};
  enum { B = 3 };
  float X[B * 8];
  for (int i = 0; i < B * 8; i++) X[i] = rnd();
  int y[B] = {0, 3, 1};
  float* dX; int* dy;
  CUDA_CHECK(cudaMalloc(&dX, sizeof X));
  CUDA_CHECK(cudaMalloc(&dy, sizeof y));
  CUDA_CHECK(cudaMemcpy(dX, X, sizeof X, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, y, sizeof y, cudaMemcpyHostToDevice));
  int fails = 0;

  /* 1. analytic gradients match central differences */
  MLP* m = mlp_create(sizes, 3, B, 123, GEMM_TILED);
  size_t n = m->nparams;
  float *ana = (float*)malloc(n * 4), *par = (float*)malloc(n * 4), *g2 = (float*)malloc(n * 4);
  mlp_forward_backward(m, dX, dy, B, NULL, NULL);
  CUDA_CHECK(cudaMemcpy(ana, m->grads, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(par, m->params, n * 4, cudaMemcpyDeviceToHost));
  const float eps = 1e-3f;
  for (size_t i = 0; i < n; i++) {
    float v = par[i] + eps;
    CUDA_CHECK(cudaMemcpy(m->params + i, &v, 4, cudaMemcpyHostToDevice));
    float lp = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
    v = par[i] - eps;
    CUDA_CHECK(cudaMemcpy(m->params + i, &v, 4, cudaMemcpyHostToDevice));
    float lm = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
    CUDA_CHECK(cudaMemcpy(m->params + i, &par[i], 4, cudaMemcpyHostToDevice));
    float num = (lp - lm) / (2 * eps);
    if (fabsf(num - ana[i]) > 2e-3f + 2e-2f * fabsf(ana[i])) {
      if (fails < 5) printf("FAIL grad[%zu]: numeric %g analytic %g\n", i, num, ana[i]);
      fails++;
    }
  }

  /* 2. every GEMM implementation gives the same gradients */
  for (int impl = GEMM_NAIVE; impl <= GEMM_CUBLAS; impl++) {
    MLP* m2 = mlp_create(sizes, 3, B, 123, (gemm_impl)impl);
    mlp_forward_backward(m2, dX, dy, B, NULL, NULL);
    CUDA_CHECK(cudaMemcpy(g2, m2->grads, n * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++)
      if (fabsf(g2[i] - ana[i]) > 1e-4f * (1.f + fabsf(ana[i]))) {
        printf("FAIL impl %d grad[%zu]: %g vs %g\n", impl, i, g2[i], ana[i]);
        fails++;
        break;
      }
    mlp_free(m2);
  }

  /* 3. SGD overfits a fixed batch */
  float first = mlp_forward_backward(m, dX, dy, B, NULL, NULL), last = first;
  for (int it = 0; it < 500; it++) {
    mlp_sgd(m, 0.2f, 1.0f);
    last = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
  }
  if (!(last < 0.1f * first)) { printf("FAIL overfit: loss %g -> %g\n", first, last); fails++; }

  mlp_free(m);
  free(ana); free(par); free(g2);

  /* Stages of an 8->16->12->4 network: stage 0 = layer [0,1), stage 1 = layers [1,3). */
  const int s4[] = {8, 16, 12, 4};
  enum { BB = 8, MB = 2, b = BB / MB };
  float X8[BB * 8];
  for (int i = 0; i < BB * 8; i++) X8[i] = rnd();
  int y8[BB] = {0, 3, 1, 2, 2, 0, 3, 1};
  float *dX8, *dH, *dlog; int* dy8;
  CUDA_CHECK(cudaMalloc(&dX8, sizeof X8));
  CUDA_CHECK(cudaMalloc(&dy8, sizeof y8));
  CUDA_CHECK(cudaMalloc(&dH, BB * 16 * sizeof(float)));    /* grad of stage 0 output, all micro-batches */
  CUDA_CHECK(cudaMalloc(&dlog, BB * 4 * sizeof(float)));
  float* lbuf;
  CUDA_CHECK(cudaMalloc(&lbuf, b * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dX8, X8, sizeof X8, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy8, y8, sizeof y8, cudaMemcpyHostToDevice));

  /* 4. a range holds exactly the matching slice of the full model's initial weights */
  {
    MLP* full = mlp_create(s4, 4, BB, 123, GEMM_TILED);
    MLP* part = mlp_create_range(s4, 4, 1, 3, BB, 123, GEMM_TILED);
    size_t off = 8 * 16 + 16, np = part->nparams;  /* layer 1 starts after [W0|b0] */
    float *hf = (float*)malloc(full->nparams * 4), *hp = (float*)malloc(np * 4);
    CUDA_CHECK(cudaMemcpy(hf, full->params, full->nparams * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp, part->params, np * 4, cudaMemcpyDeviceToHost));
    if (np != full->nparams - off || memcmp(hp, hf + off, np * 4) || part->first != 1 || !part->head ||
        part->nlayers != 2 || full->first != 0 || !full->head) {
      printf("FAIL range init: nparams %zu (want %zu), first %d, head %d\n", np, full->nparams - off,
             part->first, part->head);
      fails++;
    }
    mlp_free(full); mlp_free(part);
    free(hf); free(hp);
  }

  /* 5. two stages driven by hand with 2 micro-batches give the full model's gradients and loss */
  for (int impl = GEMM_NAIVE; impl <= GEMM_CUBLAS; impl++) {
    MLP* full = mlp_create(s4, 4, BB, 123, (gemm_impl)impl);
    MLP* st[2] = {mlp_create_range(s4, 4, 0, 1, BB, 123, (gemm_impl)impl),
                  mlp_create_range(s4, 4, 1, 3, BB, 123, (gemm_impl)impl)};
    float lfull = mlp_forward_backward(full, dX8, dy8, BB, NULL, NULL), lsum = 0;
    const float* h[MB];
    for (int k = 0; k < MB; k++) {  /* GPipe: all forwards, then all backwards */
      h[k] = mlp_stage_forward(st[0], dX8 + k * b * 8, b, k);
      const float* logits = mlp_stage_forward(st[1], h[k], b, k);
      lsum += softmax_xent(logits, dy8 + k * b, dlog + k * b * 4, lbuf, b, 4);
    }
    /* stage grads start dirty (from a throwaway backward) so mlp_zero_grads is exercised */
    mlp_stage_backward(st[1], h[0], dlog, dH, b, 0);
    mlp_stage_backward(st[0], dX8, dH, NULL, b, 0);
    mlp_zero_grads(st[0]); mlp_zero_grads(st[1]);
    for (int k = MB - 1; k >= 0; k--) {  /* GPipe backward order; grads accumulate for every k */
      mlp_stage_backward(st[1], h[k], dlog + k * b * 4, dH + k * b * 16, b, k);
      mlp_stage_backward(st[0], dX8 + k * b * 8, dH + k * b * 16, NULL, b, k);
    }
    size_t n0 = st[0]->nparams, nf = full->nparams;
    float *gf = (float*)malloc(nf * 4), *gs = (float*)malloc(nf * 4);
    CUDA_CHECK(cudaMemcpy(gf, full->grads, nf * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gs, st[0]->grads, n0 * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gs + n0, st[1]->grads, st[1]->nparams * 4, cudaMemcpyDeviceToHost));
    float gmax = 0, err = 0;
    for (size_t i = 0; i < nf; i++) gmax = fmaxf(gmax, fabsf(gf[i]));
    for (size_t i = 0; i < nf; i++) err = fmaxf(err, fabsf(gs[i] / MB - gf[i]));
    if (n0 + st[1]->nparams != nf || !(err <= 1e-4f * gmax) || !(fabsf(lsum / MB - lfull) <= 1e-5f * lfull)) {
      printf("FAIL split impl %d: grad err %g (max |g| %g), loss %g vs %g\n", impl, err, gmax, lsum / MB, lfull);
      fails++;
    }
    mlp_free(full); mlp_free(st[0]); mlp_free(st[1]);
    free(gf); free(gs);
  }
  cudaFree(dX8); cudaFree(dy8); cudaFree(dH); cudaFree(dlog); cudaFree(lbuf);

  /* 6. FP16 (both FP16 GEMMs) vs FP32 on a small network: per-layer gradients (unscaled) and loss within 1e-2
     relative, before and after one SGD step (checks the loss-scale division and the FP16 weight refresh) */
  if (!gemm_h_supported()) {
    printf("test_mlp: fp16 check skipped (needs sm_70+)\n");
  } else {
    const int s16[] = {64, 128, 64, 10};
    enum { HB = 32 };
    float Xh[HB * 64];
    int yh[HB];
    for (int i = 0; i < HB * 64; i++) Xh[i] = rnd();
    for (int i = 0; i < HB; i++) yh[i] = i % 10;
    float* dXh; int* dyh;
    CUDA_CHECK(cudaMalloc(&dXh, sizeof Xh));
    CUDA_CHECK(cudaMalloc(&dyh, sizeof yh));
    CUDA_CHECK(cudaMemcpy(dXh, Xh, sizeof Xh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dyh, yh, sizeof yh, cudaMemcpyHostToDevice));
    const gemm_impl hi[2] = {GEMM_WMMA, GEMM_CUBLAS};
    for (int t = 0; t < 2; t++) {
      MLP* r = mlp_create(s16, 4, HB, 123, GEMM_CUBLAS);
      MLP* h = mlp_create(s16, 4, HB, 123, hi[t]);
      mlp_enable_fp16(h);
      size_t nh = r->nparams;
      float *g32 = (float*)malloc(nh * 4), *g16 = (float*)malloc(nh * 4);
      for (int step = 0; step < 2; step++) {
        float l32 = mlp_forward_backward(r, dXh, dyh, HB, NULL, NULL);
        float l16 = mlp_forward_backward(h, dXh, dyh, HB, NULL, NULL);
        CUDA_CHECK(cudaMemcpy(g32, r->grads, nh * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(g16, h->grads, nh * 4, cudaMemcpyDeviceToHost));
        if (!(fabsf(l16 - l32) <= 1e-2f * fabsf(l32))) {
          printf("FAIL fp16 impl %d step %d: loss %g vs fp32 %g\n", hi[t], step, l16, l32);
          fails++;
        }
        for (int l = 0; l < r->nlayers; l++) {
          size_t o = r->L[l].dW - r->grads, len = (size_t)r->L[l].in * r->L[l].out + r->L[l].out;
          double num = 0, den = 0;
          for (size_t i = o; i < o + len; i++) {
            double d = g16[i] / h->loss_scale - g32[i];
            num += d * d;
            den += (double)g32[i] * g32[i];
          }
          if (!(sqrt(num) <= 1e-2 * sqrt(den))) {
            printf("FAIL fp16 impl %d step %d layer %d: grad rel err %g\n", hi[t], step, l, sqrt(num / den));
            fails++;
          }
        }
        mlp_sgd(r, 0.5f, 1.0f);
        mlp_sgd(h, 0.5f, 1.0f);
      }
      mlp_free(r); mlp_free(h);
      free(g32); free(g16);
    }
    cudaFree(dXh); cudaFree(dyh);
  }

  printf(fails ? "test_mlp: FAIL (%d)\n" : "test_mlp: PASS\n", fails);
  return fails != 0;
}
