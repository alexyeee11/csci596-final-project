#include "comm/nccl.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_NCCL
#include <nccl.h>
#include "cuda_check.h"

#define NCCL_CHECK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r_)); exit(1); } } while (0)

static ncclComm_t nc;

int nccl_available(void) { return 1; }

/* Rank 0 makes the unique id and MPI broadcasts it; every rank then joins the NCCL communicator. */
void nccl_init(MPI_Comm comm) {
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);
  ncclUniqueId id;
  if (!rank) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, comm);
  NCCL_CHECK(ncclCommInitRank(&nc, p, id, rank));
}

/* Synchronizing keeps comm_s comparable with the host algorithms, which block until the sum is back. */
void nccl_allreduce_sum(float* dbuf, size_t n) {
  NCCL_CHECK(ncclAllReduce(dbuf, dbuf, n, ncclFloat, ncclSum, nc, 0));
  CUDA_CHECK(cudaStreamSynchronize(0));
}

void nccl_finalize(void) { ncclCommDestroy(nc); }

#else

int nccl_available(void) { return 0; }

void nccl_init(MPI_Comm comm) {
  (void)comm;
  fprintf(stderr, "--allreduce nccl: this build has no NCCL (set NCCL_HOME, then make clean && make gpu)\n");
  exit(2);
}

void nccl_allreduce_sum(float* dbuf, size_t n) { (void)dbuf; (void)n; }
void nccl_finalize(void) {}

#endif
