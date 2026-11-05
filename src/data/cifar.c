#include "data/cifar.h"
#include <stdio.h>
#include <stdlib.h>

#define REC 3073
#define DIM 3072

void dataset_free(Dataset* d) {
  free(d->X);
  free(d->y);
  d->X = NULL; d->y = NULL; d->n = 0;
}

static int load_file(const char* path, Dataset* d) {
  FILE* f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0 || sz % REC) { fclose(f); return -1; }
  int n = (int)(sz / REC);
  d->X = realloc(d->X, (size_t)(d->n + n) * DIM * sizeof(float));
  d->y = realloc(d->y, (size_t)(d->n + n) * sizeof(int));
  unsigned char rec[REC];
  for (int i = 0; i < n; i++) {
    if (fread(rec, 1, REC, f) != REC) { fclose(f); return -1; }
    d->y[d->n + i] = rec[0];
    float* x = d->X + (size_t)(d->n + i) * DIM;
    for (int j = 0; j < DIM; j++) x[j] = rec[1 + j] / 255.f - 0.5f;
  }
  d->n += n;
  fclose(f);
  return 0;
}

int cifar_load(const char* dir, int train, Dataset* d) {
  d->n = 0; d->dim = DIM; d->X = NULL; d->y = NULL;
  char path[1024];
  for (int i = 1; i <= (train ? 5 : 1); i++) {
    if (train) snprintf(path, sizeof path, "%s/data_batch_%d.bin", dir, i);
    else       snprintf(path, sizeof path, "%s/test_batch.bin", dir);
    if (load_file(path, d)) { dataset_free(d); return -1; }
  }
  return 0;
}

static float uni(unsigned* s) { *s = *s * 1664525u + 1013904223u; return (*s >> 8) / 16777216.0f - 0.5f; }

void dataset_synthetic(Dataset* d, int n, int dim, int classes, unsigned seed) {
  d->n = n; d->dim = dim;
  d->X = malloc((size_t)n * dim * sizeof(float));
  d->y = malloc((size_t)n * sizeof(int));
  float* means = malloc((size_t)classes * dim * sizeof(float));
  unsigned s = 12345;  /* fixed so train and test sets share class means */
  for (size_t i = 0; i < (size_t)classes * dim; i++) means[i] = uni(&s);
  s = seed;
  for (int i = 0; i < n; i++) {
    int c = i % classes;
    d->y[i] = c;
    for (int j = 0; j < dim; j++) d->X[(size_t)i * dim + j] = means[(size_t)c * dim + j] + 0.5f * uni(&s);
  }
  free(means);
}
