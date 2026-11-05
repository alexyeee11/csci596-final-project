#include "comm/allreduce.h"
#include <stdlib.h>
#include <string.h>

/* ponytail: MPI counts are int, so one call handles < 2^31 floats; chunk if a model ever exceeds that */
static const char* NAMES[] = {"naive", "tree", "ring", "mpi"};

int ar_parse(const char* s) {
  for (int i = 0; i < 4; i++) if (!strcmp(s, NAMES[i])) return i;
  return -1;
}

const char* ar_name(ar_algo a) { return NAMES[a]; }

static void add(float* dst, const float* src, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] += src[i];
}

/* Everyone sends to rank 0, rank 0 sums and sends the result back. */
static void naive(float* buf, size_t n, MPI_Comm comm, int rank, int p) {
  if (rank == 0) {
    float* tmp = malloc(n * sizeof(float));
    for (int r = 1; r < p; r++) {
      MPI_Recv(tmp, (int)n, MPI_FLOAT, r, 0, comm, MPI_STATUS_IGNORE);
      add(buf, tmp, n);
    }
    free(tmp);
    for (int r = 1; r < p; r++) MPI_Send(buf, (int)n, MPI_FLOAT, r, 1, comm);
  } else {
    MPI_Send(buf, (int)n, MPI_FLOAT, 0, 0, comm);
    MPI_Recv(buf, (int)n, MPI_FLOAT, 0, 1, comm, MPI_STATUS_IGNORE);
  }
}

/* Recursive doubling. Ranks beyond the largest power of two fold into a partner first.
   Float addition is commutative, so both partners compute identical sums. */
static void tree(float* buf, size_t n, MPI_Comm comm, int rank, int p) {
  int p2 = 1;
  while (p2 * 2 <= p) p2 *= 2;
  if (rank >= p2) {
    MPI_Send(buf, (int)n, MPI_FLOAT, rank - p2, 0, comm);
    MPI_Recv(buf, (int)n, MPI_FLOAT, rank - p2, 1, comm, MPI_STATUS_IGNORE);
    return;
  }
  float* tmp = malloc(n * sizeof(float));
  if (rank + p2 < p) {
    MPI_Recv(tmp, (int)n, MPI_FLOAT, rank + p2, 0, comm, MPI_STATUS_IGNORE);
    add(buf, tmp, n);
  }
  for (int mask = 1; mask < p2; mask <<= 1) {
    int peer = rank ^ mask;
    MPI_Sendrecv(buf, (int)n, MPI_FLOAT, peer, 2, tmp, (int)n, MPI_FLOAT, peer, 2, comm, MPI_STATUS_IGNORE);
    add(buf, tmp, n);
  }
  if (rank + p2 < p) MPI_Send(buf, (int)n, MPI_FLOAT, rank + p2, 1, comm);
  free(tmp);
}

static size_t cstart(int i, size_t n, int p) { return (size_t)i * n / p; }

/* Reduce-scatter then all-gather around a ring: 2(p-1) steps of ~n/p floats each. */
static void ring(float* buf, size_t n, MPI_Comm comm, int rank, int p) {
  int right = (rank + 1) % p, left = (rank - 1 + p) % p;
  float* tmp = malloc((n / p + 1) * sizeof(float));
  for (int s = 0; s < p - 1; s++) {
    int si = (rank - s + p) % p, ri = (rank - s - 1 + p) % p;
    size_t so = cstart(si, n, p), sn = cstart(si + 1, n, p) - so;
    size_t ro = cstart(ri, n, p), rn = cstart(ri + 1, n, p) - ro;
    MPI_Sendrecv(buf + so, (int)sn, MPI_FLOAT, right, 3, tmp, (int)rn, MPI_FLOAT, left, 3, comm, MPI_STATUS_IGNORE);
    add(buf + ro, tmp, rn);
  }
  /* now rank r holds the fully reduced chunk (r + 1) % p */
  for (int s = 0; s < p - 1; s++) {
    int si = (rank - s + 1 + p) % p, ri = (rank - s + p) % p;
    size_t so = cstart(si, n, p), sn = cstart(si + 1, n, p) - so;
    size_t ro = cstart(ri, n, p), rn = cstart(ri + 1, n, p) - ro;
    MPI_Sendrecv(buf + so, (int)sn, MPI_FLOAT, right, 4, buf + ro, (int)rn, MPI_FLOAT, left, 4, comm, MPI_STATUS_IGNORE);
  }
  free(tmp);
}

void allreduce_sum(float* buf, size_t n, ar_algo algo, MPI_Comm comm) {
  int rank, p;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);
  if (p == 1 || n == 0) return;
  switch (algo) {
    case AR_NAIVE: naive(buf, n, comm, rank, p); break;
    case AR_TREE:  tree(buf, n, comm, rank, p); break;
    case AR_RING:  ring(buf, n, comm, rank, p); break;
    case AR_MPI:   MPI_Allreduce(MPI_IN_PLACE, buf, (int)n, MPI_FLOAT, MPI_SUM, comm); break;
  }
}
