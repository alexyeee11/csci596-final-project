#include "comm/nccl.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_NCCL
#include <dlfcn.h>
#include <nccl.h>
#include "cuda_check.h"

/* NCCL is loaded with dlopen at nccl_init, not linked: Discovery's only usable NCCL (ver/2506 stack) has debug
   sections compressed in a format the system linker cannot read, and runs without --allreduce nccl never load it.
   NCCL_LIB is the library's absolute path, set by the Makefile. */
static struct {
  ncclResult_t (*getUniqueId)(ncclUniqueId*);
  ncclResult_t (*commInitRank)(ncclComm_t*, int, ncclUniqueId, int);
  ncclResult_t (*allReduce)(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
  ncclResult_t (*commDestroy)(ncclComm_t);
  const char* (*getErrorString)(ncclResult_t);
} f;

#define NCCL_CHECK(x) do { ncclResult_t r_ = (x); if (r_ != ncclSuccess) { \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, f.getErrorString(r_)); exit(1); } } while (0)

static ncclComm_t nc;

int nccl_available(void) { return 1; }

static void* sym(void* lib, const char* name) {
  void* p = dlsym(lib, name);
  if (!p) { fprintf(stderr, "NCCL: %s\n", dlerror()); exit(1); }
  return p;
}

static void load(void) {
  void* lib = dlopen(NCCL_LIB, RTLD_NOW | RTLD_LOCAL);
  if (!lib) { fprintf(stderr, "cannot load NCCL: %s\n", dlerror()); exit(1); }
  f.getUniqueId = sym(lib, "ncclGetUniqueId");
  f.commInitRank = sym(lib, "ncclCommInitRank");
  f.allReduce = sym(lib, "ncclAllReduce");
  f.commDestroy = sym(lib, "ncclCommDestroy");
  f.getErrorString = sym(lib, "ncclGetErrorString");
}

/* Rank 0 makes the unique id and MPI broadcasts it; every rank then joins the NCCL communicator. */
void nccl_init(MPI_Comm comm) {
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);
  load();
  ncclUniqueId id;
  if (!rank) NCCL_CHECK(f.getUniqueId(&id));
  MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, comm);
  NCCL_CHECK(f.commInitRank(&nc, p, id, rank));
}

/* Synchronizing keeps comm_s comparable with the host algorithms, which block until the sum is back. */
void nccl_allreduce_sum(float* dbuf, size_t n) {
  NCCL_CHECK(f.allReduce(dbuf, dbuf, n, ncclFloat, ncclSum, nc, 0));
  CUDA_CHECK(cudaStreamSynchronize(0));
}

void nccl_finalize(void) { f.commDestroy(nc); }

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
