#pragma once
#include <mpi.h>
#include <stddef.h>

/* NCCL baseline: all-reduce on the device buffer, no host staging. Compiled in only when the build defines
   HAVE_NCCL (the Makefile does when NCCL_HOME is set); otherwise nccl_init exits with a message. */
int nccl_available(void);
void nccl_init(MPI_Comm comm);                   /* collective over comm; one GPU per rank, already selected */
void nccl_allreduce_sum(float* dbuf, size_t n);  /* in place, on the default stream, then synchronizes it */
void nccl_finalize(void);
