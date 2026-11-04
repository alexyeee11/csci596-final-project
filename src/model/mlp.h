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
} Layer;

typedef struct {
  int nlayers, max_batch, max_width;
  Layer* L;
  float *params, *grads;  /* flat device buffers: [W0|b0|W1|b1|...] */
  size_t nparams;
  float* dbuf[2];         /* backward ping-pong scratch, max_batch x max_width */
  float* loss_buf;        /* max_batch */
  gemm_impl impl;
} MLP;

/* sizes = {input, hidden..., classes}. ReLU after every layer except the last. */
MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl);
void mlp_free(MLP* m);
const float* mlp_forward(MLP* m, const float* X, int B);  /* device logits, B x classes */
float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx);
void mlp_sgd(MLP* m, float lr, float grad_scale);         /* params -= lr * grad_scale * grads */

#ifdef __cplusplus
}
#endif
