#pragma once

typedef struct { int n, dim; float* X; int* y; } Dataset;  /* X is n x dim, row-major */

/* CIFAR-10 binary format from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz.
   train=1 reads data_batch_1..5.bin, train=0 reads test_batch.bin. Pixels map to [-0.5, 0.5].
   Returns 0, or -1 with d left empty. */
int cifar_load(const char* dir, int train, Dataset* d);
/* Learnable fake data: label i % classes, class means fixed, noise drawn from seed. */
void dataset_synthetic(Dataset* d, int n, int dim, int classes, unsigned seed);
void dataset_free(Dataset* d);
