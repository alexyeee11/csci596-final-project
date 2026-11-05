#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "data/cifar.h"

static int fails = 0;
#define EXPECT(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

/* Two records with ids first, first+1: label id % 10, pixel j = (id*7 + j) % 256. */
static void write_batch(const char* path, int first) {
  FILE* f = fopen(path, "wb");
  unsigned char rec[3073];
  for (int r = 0; r < 2; r++) {
    int id = first + r;
    rec[0] = (unsigned char)(id % 10);
    for (int j = 0; j < 3072; j++) rec[1 + j] = (unsigned char)((id * 7 + j) % 256);
    fwrite(rec, 1, sizeof rec, f);
  }
  fclose(f);
}

int main(void) {
  char dir[] = "/tmp/cifar_test_XXXXXX";
  if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
  char path[6][512];
  for (int i = 1; i <= 5; i++) {
    snprintf(path[i - 1], sizeof path[0], "%s/data_batch_%d.bin", dir, i);
    write_batch(path[i - 1], (i - 1) * 2);
  }
  snprintf(path[5], sizeof path[0], "%s/test_batch.bin", dir);
  write_batch(path[5], 100);

  Dataset tr, te, bad, sy;
  EXPECT(cifar_load(dir, 1, &tr) == 0);
  EXPECT(tr.n == 10 && tr.dim == 3072);
  EXPECT(tr.y[3] == 3);
  EXPECT(fabsf(tr.X[3 * 3072 + 5] - (((3 * 7 + 5) % 256) / 255.f - 0.5f)) < 1e-6f);
  EXPECT(cifar_load(dir, 0, &te) == 0);
  EXPECT(te.n == 2 && te.y[1] == 101 % 10);
  EXPECT(cifar_load("/nonexistent", 1, &bad) == -1 && bad.n == 0 && bad.X == NULL);

  dataset_synthetic(&sy, 50, 3072, 10, 1);
  EXPECT(sy.n == 50 && sy.dim == 3072 && sy.y[13] == 3);
  Dataset sy2;
  dataset_synthetic(&sy2, 50, 3072, 10, 2);
  EXPECT(sy.X[0] != sy2.X[0]);  /* different noise, same class means */

  dataset_free(&tr); dataset_free(&te); dataset_free(&sy); dataset_free(&sy2);
  for (int i = 0; i < 6; i++) remove(path[i]);
  rmdir(dir);
  printf(fails ? "test_cifar: FAIL\n" : "test_cifar: PASS\n");
  return fails != 0;
}
