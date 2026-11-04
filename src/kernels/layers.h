#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* All pointers are device pointers; matrices are row-major rows x cols. */
void bias_act_forward(float* Y, const float* b, int rows, int cols, int relu);  /* Y += b; ReLU if relu */
void relu_backward(float* dY, const float* Y, size_t n);                        /* dY = 0 where Y <= 0 */
void bias_grad(float* db, const float* dY, int rows, int cols);                 /* db = column sums */
/* Stable softmax + cross-entropy. Writes dlogits = (p - onehot) / rows, returns mean loss.
   loss_buf: device scratch, rows floats. */
float softmax_xent(const float* logits, const int* labels, float* dlogits, float* loss_buf,
                   int rows, int classes);
void sgd_step(float* w, const float* g, float lr, size_t n);                     /* w -= lr * g */

#ifdef __cplusplus
}
#endif
