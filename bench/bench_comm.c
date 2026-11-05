#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "comm/allreduce.h"

static int iters_for(size_t bytes) { return bytes < (1u << 20) ? 100 : 10; }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  size_t maxbytes = argc > 1 ? (size_t)atol(argv[1]) : (size_t)256 << 20;
  float* buf = calloc(maxbytes / sizeof(float) + 1, sizeof(float));
  if (!rank) printf("kind,algo,np,bytes,seconds\n");

  /* one-way latency/bandwidth between rank 0 and the last rank (crosses nodes when spread) */
  if (p >= 2) {
    for (size_t bytes = 4; bytes <= maxbytes; bytes *= 4) {
      int iters = iters_for(bytes), cnt = (int)(bytes / sizeof(float));
      MPI_Barrier(MPI_COMM_WORLD);
      double t0 = MPI_Wtime();
      for (int it = 0; it < iters; it++) {
        if (rank == 0) {
          MPI_Send(buf, cnt, MPI_FLOAT, p - 1, 0, MPI_COMM_WORLD);
          MPI_Recv(buf, cnt, MPI_FLOAT, p - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == p - 1) {
          MPI_Recv(buf, cnt, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Send(buf, cnt, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
      }
      double t = (MPI_Wtime() - t0) / (2.0 * iters);
      if (!rank) printf("pingpong,-,%d,%zu,%.9f\n", p, bytes, t);
    }
  }

  for (int a = AR_NAIVE; a <= AR_MPI; a++) {
    for (size_t bytes = 4; bytes <= maxbytes; bytes *= 4) {
      size_t n = bytes / sizeof(float);
      int iters = iters_for(bytes);
      memset(buf, 0, bytes);
      allreduce_sum(buf, n, (ar_algo)a, MPI_COMM_WORLD);  /* warm-up */
      MPI_Barrier(MPI_COMM_WORLD);
      double t0 = MPI_Wtime();
      for (int it = 0; it < iters; it++) allreduce_sum(buf, n, (ar_algo)a, MPI_COMM_WORLD);
      double local = (MPI_Wtime() - t0) / iters, t;
      MPI_Reduce(&local, &t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
      if (!rank) printf("allreduce,%s,%d,%zu,%.9f\n", ar_name((ar_algo)a), p, bytes, t);
    }
  }
  free(buf);
  MPI_Finalize();
  return 0;
}
