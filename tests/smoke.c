#include <mpi.h>
#include <stdio.h>
#include "cuda_check.h"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size, lrank, ndev = 0, len;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &lrank);
  char host[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(host, &len);
  cudaGetDeviceCount(&ndev);
  if (ndev > 0) {
    struct cudaDeviceProp prop;
    CUDA_CHECK(cudaSetDevice(lrank % ndev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, lrank % ndev));
    printf("rank %d/%d host %s local %d gpu %s\n", rank, size, host, lrank, prop.name);
  } else {
    printf("rank %d/%d host %s: FAIL no GPU visible\n", rank, size, host);
  }
  MPI_Finalize();
  return ndev == 0;
}
