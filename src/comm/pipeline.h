#pragma once
#include <mpi.h>
#include "model/mlp.h"

/* GPipe over the ranks of `pipe`, rank i of pipe holding stage i (see mlp_create_range). Activations and their
   gradients go device -> pinned host -> MPI -> pinned host -> device, like the host-staged all-reduce. */

/* Contiguous layer range [first, last) of stage `stage` out of P, equal in layer count up to one, earlier stages
   taking the extra layer. */
void pp_split(int nlayers, int P, int stage, int* first, int* last);

/* One GPipe step on b_total rows split into M micro-batches: all forwards, then all backwards (last micro-batch
   first). X: device input rows, read by the first stage only; y: device labels, read by the head only. Leaves the
   sum of the M micro-batch gradients in stage->grads (call mlp_sgd with grad_scale 1/M). Returns the mean loss on
   every rank of pipe; *busy_s = this rank's kernel time (synchronized), everything else is transfer or bubble.
   stage must be created with max_batch >= b_total. */
float pp_step(MLP* stage, const float* X, const int* y, int b_total, int M, MPI_Comm pipe, double* busy_s);

/* Forward only, b rows (b <= max_batch): X device input (first stage), y host labels (head). Returns the number of
   correct predictions on every rank of pipe. */
int pp_correct(MLP* stage, const float* X, const int* y, int b, MPI_Comm pipe);

void pp_free(void);  /* releases the transfer buffers pp_step and pp_correct allocate on first use */
