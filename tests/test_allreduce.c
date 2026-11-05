#include <stdio.h>
#include <stdlib.h>
#include "comm/allreduce.h"

static float val(int rank, size_t i) { return (float)(rank * 1000 + (int)(i % 97)); }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  const size_t sizes[] = {1, 7, 1000, (1u << 20) + 3};
  const ar_algo algos[] = {AR_NAIVE, AR_TREE, AR_RING, AR_MPI};
  int fails = 0;
  for (int a = 0; a < 4; a++) {
    for (int s = 0; s < 4; s++) {
      size_t n = sizes[s];
      float* buf = malloc(n * sizeof(float));
      for (size_t i = 0; i < n; i++) buf[i] = val(rank, i);
      allreduce_sum(buf, n, algos[a], MPI_COMM_WORLD);
      for (size_t i = 0; i < n; i++) {
        float want = 1000.0f * p * (p - 1) / 2 + (float)p * (int)(i % 97);
        if (buf[i] != want) {
          printf("FAIL %s n=%zu rank %d i=%zu: got %g want %g\n", ar_name(algos[a]), n, rank, i, buf[i], want);
          fails++;
          break;
        }
      }
      free(buf);
    }
  }
  int total;
  MPI_Allreduce(&fails, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) printf("test_allreduce np=%d: %s\n", p, total ? "FAIL" : "PASS");
  MPI_Finalize();
  return total != 0;
}
