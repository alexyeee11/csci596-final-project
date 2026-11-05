#pragma once
#include <mpi.h>
#include <stddef.h>

typedef enum { AR_NAIVE, AR_TREE, AR_RING, AR_MPI } ar_algo;

int ar_parse(const char* s);        /* -1 if unknown */
const char* ar_name(ar_algo a);
/* In-place sum of host buffer across comm. All ranks end with identical bits. */
void allreduce_sum(float* buf, size_t n, ar_algo algo, MPI_Comm comm);
