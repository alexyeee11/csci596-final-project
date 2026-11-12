#pragma once
#include <mpi.h>
#include <stddef.h>
#include "comm/allreduce.h"

/* Bucketed gradient all-reduce that overlaps backward. Buckets are fixed-size slices of the flat device gradient
   buffer counted from its end, since backward finishes the last layer first. A comm thread copies each finished
   bucket to pinned host memory, all-reduces it (sum) and copies it back, while backward keeps running. */
typedef struct Overlap Overlap;

/* dgrads: device buffer of n floats. MPI must be initialized with at least MPI_THREAD_SERIALIZED. */
Overlap* overlap_create(float* dgrads, size_t n, size_t bucket_floats, ar_algo algo, MPI_Comm comm, int device);
void overlap_begin(Overlap* o);                                     /* once per step, before backward */
void overlap_on_grad(int layer, float* grad, size_t n, void* ctx);  /* a grad_hook; ctx = Overlap* */
void overlap_wait(Overlap* o);                                      /* all buckets reduced and back on the GPU */
void overlap_free(Overlap* o);
