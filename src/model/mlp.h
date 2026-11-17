#pragma once
#include <stddef.h>
#include "kernels/gemm.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Called when a layer's gradients are final (last layer first). grad = that layer's [dW | db]. */
typedef void (*grad_hook)(int layer, float* grad, size_t n, void* ctx);

typedef struct {
  int in, out;
  float *W, *b;    /* views into MLP.params; W is in x out */
  float *dW, *db;  /* views into MLP.grads, same layout */
  float* Y;        /* post-activation output, max_batch x out */
  void *W16, *Y16; /* __half copies of W (view into MLP.params16) and Y; NULL unless mlp_enable_fp16 */
} Layer;

typedef struct {
  int nlayers, max_batch, max_width;
  int first;              /* global index of L[0] in the full network */
  int head;               /* 1 if this holds the network's last layer (no ReLU, outputs logits) */
  Layer* L;
  float *params, *grads;  /* flat device buffers: [W0|b0|W1|b1|...] */
  size_t nparams;
  float* dbuf[2];         /* backward ping-pong scratch, max_batch x max_width */
  float* loss_buf;        /* max_batch */
  gemm_impl impl;
  /* mlp_enable_fp16: __half copies of params, of the input batch (max_batch x input) and of the backward dZ
     (max_batch x max_width); NULL otherwise. grads are scaled by loss_scale (1 without FP16). */
  void *params16, *X16, *dz16;
  float loss_scale;
} MLP;

/* sizes = {input, hidden..., classes}. ReLU after every layer except the last. */
MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl);
/* Only layers [first, last) of that network, with weights bit-identical to the same layers of mlp_create. */
MLP* mlp_create_range(const int* sizes, int nsizes, int first, int last, int max_batch, unsigned seed,
                      gemm_impl impl);
void mlp_free(MLP* m);
/* Switch to FP16-input GEMMs (gemm_h: cuBLAS if impl == GEMM_CUBLAS, else WMMA) with FP32 master weights and
   grads and a static loss scale. Exits with a message on a pre-sm_70 GPU. Not supported by the stage API. */
void mlp_enable_fp16(MLP* m);
const float* mlp_forward(MLP* m, const float* X, int B);  /* device logits, B x classes */
float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx);
/* Pipeline stage, micro-batch k of b rows (max_batch = all micro-batches). Forward writes rows [k*b, (k+1)*b) of
   every layer's Y and returns the stage output rows. Backward: X = this micro-batch's stage input, dOut = grad of
   the stage output (dlogits on the head), dX = grad of X (NULL to skip). Grads accumulate for every k, in any
   order: call mlp_zero_grads once per step before the backwards, and scale by 1/M in mlp_sgd. */
float* mlp_stage_forward(MLP* m, const float* X, int b, int k);
void mlp_stage_backward(MLP* m, const float* X, const float* dOut, float* dX, int b, int k);
void mlp_zero_grads(MLP* m);
void mlp_sgd(MLP* m, float lr, float grad_scale);         /* params -= lr * grad_scale / loss_scale * grads */

#ifdef __cplusplus
}
#endif
