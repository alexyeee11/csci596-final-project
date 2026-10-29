# Mini-DDP Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch C/CUDA/MPI trainer that trains an MLP on CIFAR-10 on one GPU with our own GEMM and layer kernels, and on N GPUs with our own naive/tree/ring all-reduce, where the N-GPU loss curve matches the 1-GPU curve.

**Architecture:** One MPI rank drives one GPU. CUDA kernels (`src/kernels/`, `src/model/`) expose `extern "C"` functions over device pointers; the communication layer (`src/comm/`) is pure C + MPI over host buffers, so it builds and tests on a laptop without a GPU. In this milestone, gradients move GPU → pinned host → all-reduce → GPU (the "staging" path from the spec); GPU-aware MPI, overlap and NCCL come in Plan 2.

**Tech Stack:** C (gcc/clang, GNU dialect), CUDA C++ (nvcc), cuBLAS (reference only), OpenMPI, Make, Bash/awk, Slurm on USC CARC Discovery.

**Spec:** `docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md`

## Global Constraints

- One MPI rank per GPU; each rank uses device `local_rank % device_count`, where `local_rank` comes from `MPI_Comm_split_type(..., MPI_COMM_TYPE_SHARED, ...)`.
- All matrices are row-major. `W` of a layer is `in x out`; activations are `batch x width`.
- GPU arch targets: `sm_60` (P100), `sm_70` (V100), `sm_80` (A100), `sm_86` (A40), `sm_89` (L40S).
- Dependencies: MPI, CUDA runtime, cuBLAS (reference/baseline only). No others.
- `.cu` files contain no MPI; `.c` files contain no kernels. Headers shared between them use `extern "C"` guards.
- Tests are plain executables: print `<name>: PASS` or `FAIL ...`, exit non-zero on failure. No test framework.
- CARC charges per reserved minute: P100/V100 = 4 SU per GPU, A100/A40/L40S = 8 SU per GPU, plus 1 SU per CPU and per 4 GB. Develop on 1–2 P100s in the `debug` partition. Team budget is ~12,000 SU.
- Git commit messages: no attribution lines (no `Co-Authored-By`, no "Generated with").
- Final deadline Dec 11, 2026. This milestone targets Nov 10.

## Ownership and order

| Task | Owner | Needs GPU? | Depends on |
|---|---|---|---|
| 1. Build skeleton + CARC smoke test | both | yes | — |
| 2. All-reduce: naive / tree / ring | T2 (communication) | no | 1 |
| 3. GEMM: naive / tiled / cuBLAS | T1 (kernels) | yes | 1 |
| 4. CIFAR-10 loader + synthetic data | T2 | no | 1 |
| 5. Layer kernels | T1 | yes | 1 |
| 6. MLP forward/backward + gradient check | T1 | yes | 3, 5 |
| 7. Single-GPU trainer | T1 | yes | 4, 6 |
| 8. Data-parallel training + equality test | T2 | yes | 2, 7 |
| 9. Communication microbenchmark | T2 | no | 2 |

Tasks 2–5 can run in parallel right after Task 1.

## File structure

```
Makefile                      all build + local check rules (written once, Task 1)
.gitignore
scripts/env.sh                CARC module setup
scripts/slurm/run.sh          generic GPU job: sbatch ... scripts/slurm/run.sh <cmd...>
scripts/test_dp.sh            N-rank vs 1-rank loss-curve equality test
src/cuda_check.h              CUDA_CHECK macro (usable from C and CUDA)
src/comm/allreduce.{h,c}      naive / tree / ring / MPI all-reduce on host float buffers
src/data/cifar.{h,c}          CIFAR-10 binary loader + synthetic dataset
src/kernels/gemm.{h,cu}       gemm(): naive, tiled, cuBLAS, with transpose flags
src/kernels/layers.{h,cu}     bias+ReLU, ReLU backward, bias grad, softmax+xent, SGD
src/model/mlp.{h,cu}          MLP with flat params/grads buffers, forward/backward, grad hook
src/train.c                   CLI trainer: data, model, (all-reduce), SGD, eval
tests/smoke.c                 MPI + GPU visibility check
tests/test_allreduce.c
tests/test_cifar.c
tests/test_gemm.cu
tests/test_layers.cu
tests/test_mlp.cu
bench/bench_comm.c            ping-pong + all-reduce timing, CSV out
```

---

### Task 1: Build skeleton and CARC smoke test (both teammates)

**Files:**
- Create: `Makefile`, `.gitignore`, `src/cuda_check.h`, `scripts/env.sh`, `scripts/slurm/run.sh`, `tests/smoke.c`

**Interfaces:**
- Produces: `CUDA_CHECK(expr)` macro in `src/cuda_check.h`; Make targets `bin/<name>` for every binary in this plan; `make check-cpu`; `scripts/slurm/run.sh <cmd...>`.

- [ ] **Step 1: Install MPI locally (laptop, for CPU-only tasks)**

```bash
brew install open-mpi
mpicc --version
```
Expected: prints the underlying compiler version (Apple clang).

- [ ] **Step 2: Write `.gitignore`**

```
bin/
build/
logs/
```

- [ ] **Step 3: Write `src/cuda_check.h`**

```c
#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)
```

- [ ] **Step 4: Write the `Makefile` (complete, used by every later task)**

Recipe lines must start with a real tab.

```make
# Laptop: `make check-cpu` (MPI only). CARC: `source scripts/env.sh && make gpu`.
# ponytail: no header dependency tracking; run `make clean` after editing a header.
MPICC     ?= mpicc
NVCC      ?= nvcc
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell which nvcc 2>/dev/null))
ARCH      ?= -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89
CFLAGS    = -O2 -Wall -Isrc
NVFLAGS   = -O2 $(ARCH) -Isrc
GPU_INC   = -I$(CUDA_HOME)/include
GPU_LIBS  = -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lstdc++ -lm
KERNELS   = build/gemm.o build/layers.o build/mlp.o

.PHONY: cpu gpu check-cpu clean
cpu: bin/test_allreduce bin/test_cifar bin/bench_comm
gpu: cpu bin/smoke bin/test_gemm bin/test_layers bin/test_mlp bin/train

bin build:
	mkdir -p $@

build/%.o: src/kernels/%.cu | build
	$(NVCC) $(NVFLAGS) -c $< -o $@
build/mlp.o: src/model/mlp.cu | build
	$(NVCC) $(NVFLAGS) -c $< -o $@

bin/test_allreduce: tests/test_allreduce.c src/comm/allreduce.c | bin
	$(MPICC) $(CFLAGS) $^ -o $@
bin/bench_comm: bench/bench_comm.c src/comm/allreduce.c | bin
	$(MPICC) $(CFLAGS) $^ -o $@
bin/test_cifar: tests/test_cifar.c src/data/cifar.c | bin
	$(MPICC) $(CFLAGS) $^ -o $@ -lm
bin/smoke: tests/smoke.c | bin
	$(MPICC) $(CFLAGS) $(GPU_INC) $^ -o $@ $(GPU_LIBS)
bin/test_gemm: tests/test_gemm.cu build/gemm.o | bin
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcublas
bin/test_layers: tests/test_layers.cu build/layers.o | bin
	$(NVCC) $(NVFLAGS) $^ -o $@
bin/test_mlp: tests/test_mlp.cu $(KERNELS) | bin
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcublas
bin/train: src/train.c src/comm/allreduce.c src/data/cifar.c $(KERNELS) | bin
	$(MPICC) $(CFLAGS) $(GPU_INC) $^ -o $@ $(GPU_LIBS)

check-cpu: cpu
	./bin/test_cifar
	for n in 1 2 3 4; do mpirun --oversubscribe -np $$n ./bin/test_allreduce || exit 1; done
	test "$$(mpirun --oversubscribe -np 2 ./bin/bench_comm 65536 | wc -l | tr -d ' ')" -eq 41

clean:
	rm -rf bin build
```

- [ ] **Step 5: Write `scripts/env.sh`**

```bash
# source this on CARC Discovery before building or running
module purge
module load gcc openmpi cuda
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
# MPI launcher inside Slurm jobs; check `srun --mpi=list` if pmix is not listed
export SRUN_MPI="${SRUN_MPI:-pmix}"
```

- [ ] **Step 6: Write `scripts/slurm/run.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=mddp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/run.sh <command...>
# cost: ~16 SU per minute while running (2 P100 + 4 CPU + 16 GB)
set -euo pipefail
source scripts/env.sh
"$@"
```

```bash
chmod +x scripts/slurm/run.sh
```

- [ ] **Step 7: Write the smoke test `tests/smoke.c`**

```c
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
```

- [ ] **Step 8: Build on a CARC login node**

```bash
ssh <netid>@discovery.usc.edu
git clone <repo url> && cd csci596-final-project
source scripts/env.sh
make bin/smoke
```
Expected: `bin/smoke` exists. If `module load gcc openmpi cuda` fails because there is no default version, run `module spider openmpi` and `module spider cuda`, then pin the versions it lists in `scripts/env.sh` (for example `module load gcc/<v> openmpi/<v> cuda/<v>`).

- [ ] **Step 9: Run the smoke test on 2 GPUs**

```bash
myaccount            # note the class account name
mkdir -p logs
sbatch --account=<class account> scripts/slurm/run.sh srun --mpi=$SRUN_MPI ./bin/smoke
cat logs/mddp-<jobid>.out
```
Expected: two lines like `rank 0/2 host d23-10 local 0 gpu Tesla P100-PCIE-16GB` and `rank 1/2 ... local 1 ...`, with different local ranks. If `srun` reports a PMI error, run `srun --mpi=list` on the login node and set `SRUN_MPI` in `scripts/env.sh` to an entry it lists.

- [ ] **Step 10: Commit**

```bash
git add Makefile .gitignore src/cuda_check.h scripts/env.sh scripts/slurm/run.sh tests/smoke.c
git commit -m "Add build skeleton and CARC MPI+GPU smoke test"
```

---

### Task 2: All-reduce — naive, tree, ring (T2)

**Files:**
- Create: `src/comm/allreduce.h`, `src/comm/allreduce.c`
- Test: `tests/test_allreduce.c`

**Interfaces:**
- Produces:
  - `typedef enum { AR_NAIVE, AR_TREE, AR_RING, AR_MPI } ar_algo;`
  - `int ar_parse(const char* s);` returns the enum value for `"naive"|"tree"|"ring"|"mpi"`, else `-1`
  - `const char* ar_name(ar_algo a);`
  - `void allreduce_sum(float* buf, size_t n, ar_algo algo, MPI_Comm comm);` in-place sum of host buffer across all ranks of `comm`; every rank ends with bit-identical results

- [ ] **Step 1: Write the header `src/comm/allreduce.h`**

```c
#pragma once
#include <mpi.h>
#include <stddef.h>

typedef enum { AR_NAIVE, AR_TREE, AR_RING, AR_MPI } ar_algo;

int ar_parse(const char* s);        /* -1 if unknown */
const char* ar_name(ar_algo a);
/* In-place sum of host buffer across comm. All ranks end with identical bits. */
void allreduce_sum(float* buf, size_t n, ar_algo algo, MPI_Comm comm);
```

- [ ] **Step 2: Write the failing test `tests/test_allreduce.c`**

Values are small integers, so float sums are exact regardless of summation order.

```c
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
```

- [ ] **Step 3: Run it to verify it fails**

Run: `make bin/test_allreduce`
Expected: link error, `undefined symbol: _allreduce_sum` (or `_ar_name`).

- [ ] **Step 4: Implement `src/comm/allreduce.c`**

```c
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
```

- [ ] **Step 5: Run the test on 1–4 ranks**

```bash
make bin/test_allreduce
for n in 1 2 3 4; do mpirun --oversubscribe -np $n ./bin/test_allreduce; done
```
Expected: four lines, `test_allreduce np=1: PASS` through `np=4: PASS`. The np=3 case checks the tree fold-in path and uneven ring chunks.

- [ ] **Step 6: Commit**

```bash
git add src/comm/allreduce.h src/comm/allreduce.c tests/test_allreduce.c
git commit -m "Add naive, tree and ring all-reduce with correctness test"
```

---

### Task 3: GEMM — naive, tiled, cuBLAS (T1)

**Files:**
- Create: `src/kernels/gemm.h`, `src/kernels/gemm.cu`
- Test: `tests/test_gemm.cu`

**Interfaces:**
- Produces:
  - `typedef enum { GEMM_NAIVE, GEMM_TILED, GEMM_CUBLAS } gemm_impl;`
  - `int gemm_parse(const char* s);` for `"naive"|"tiled"|"cublas"`, else `-1`
  - `void gemm(gemm_impl impl, int transA, int transB, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C);`
    Row-major, device pointers: `C[MxN] = alpha * op(A)[MxK] * op(B)[KxN] + beta * C`. With `transA`, `A` is stored `KxM`, else `MxK`. With `transB`, `B` is stored `NxK`, else `KxN`. When `beta == 0`, `C` is not read.

- [ ] **Step 1: Write the header `src/kernels/gemm.h`**

```c
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { GEMM_NAIVE, GEMM_TILED, GEMM_CUBLAS } gemm_impl;

int gemm_parse(const char* s);  /* -1 if unknown */
/* Row-major, device pointers: C[MxN] = alpha * op(A)[MxK] * op(B)[KxN] + beta * C.
   transA: A stored KxM, else MxK. transB: B stored NxK, else KxN. beta == 0: C not read. */
void gemm(gemm_impl impl, int transA, int transB, int M, int N, int K,
          float alpha, const float* A, const float* B, float beta, float* C);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Write the failing test `tests/test_gemm.cu`**

```cuda
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/gemm.h"

static unsigned seed = 1;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

static void ref(int ta, int tb, int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      double acc = 0;
      for (int k = 0; k < K; k++) {
        float a = ta ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
        float b = tb ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
        acc += (double)a * b;
      }
      C[(size_t)i * N + j] = (float)(alpha * acc + beta * C[(size_t)i * N + j]);
    }
}

int main(void) {
  const int shapes[][3] = {{1, 1, 1}, {33, 17, 65}, {128, 256, 64}, {257, 129, 300}};
  const char* names[] = {"naive", "tiled", "cublas"};
  int fails = 0;
  for (int impl = 0; impl < 3; impl++)
    for (int s = 0; s < 4; s++)
      for (int ta = 0; ta < 2; ta++)
        for (int tb = 0; tb < 2; tb++) {
          int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
          size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
          float *hA = (float*)malloc(na * 4), *hB = (float*)malloc(nb * 4);
          float *hC = (float*)malloc(nc * 4), *want = (float*)malloc(nc * 4);
          for (size_t i = 0; i < na; i++) hA[i] = rnd();
          for (size_t i = 0; i < nb; i++) hB[i] = rnd();
          for (size_t i = 0; i < nc; i++) want[i] = hC[i] = rnd();
          ref(ta, tb, M, N, K, 1.5f, hA, hB, 0.5f, want);
          float *dA, *dB, *dC;
          CUDA_CHECK(cudaMalloc(&dA, na * 4));
          CUDA_CHECK(cudaMalloc(&dB, nb * 4));
          CUDA_CHECK(cudaMalloc(&dC, nc * 4));
          CUDA_CHECK(cudaMemcpy(dA, hA, na * 4, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(dB, hB, nb * 4, cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(dC, hC, nc * 4, cudaMemcpyHostToDevice));
          gemm((gemm_impl)impl, ta, tb, M, N, K, 1.5f, dA, dB, 0.5f, dC);
          CUDA_CHECK(cudaMemcpy(hC, dC, nc * 4, cudaMemcpyDeviceToHost));
          for (size_t i = 0; i < nc; i++)
            if (fabsf(hC[i] - want[i]) > 1e-3f * (1.f + fabsf(want[i]))) {
              printf("FAIL %s %dx%dx%d ta=%d tb=%d at %zu: got %g want %g\n",
                     names[impl], M, N, K, ta, tb, i, hC[i], want[i]);
              fails++;
              break;
            }
          cudaFree(dA); cudaFree(dB); cudaFree(dC);
          free(hA); free(hB); free(hC); free(want);
        }
  printf(fails ? "test_gemm: FAIL (%d)\n" : "test_gemm: PASS\n", fails);
  return fails != 0;
}
```

- [ ] **Step 3: Run it to verify it fails**

Run (CARC login node, after `source scripts/env.sh`): `make bin/test_gemm`
Expected: `No rule to make target 'src/kernels/gemm.cu'`.

- [ ] **Step 4: Implement `src/kernels/gemm.cu`**

```cuda
#include <cublas_v2.h>
#include <string.h>
#include "cuda_check.h"
#include "kernels/gemm.h"

#define TILE 32

__device__ __forceinline__ float ldA(const float* A, int ta, int M, int K, int i, int k) {
  return ta ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
}
__device__ __forceinline__ float ldB(const float* B, int tb, int N, int K, int k, int j) {
  return tb ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
}

__global__ void gemm_naive_k(int ta, int tb, int M, int N, int K, float alpha,
                             const float* A, const float* B, float beta, float* C) {
  int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; k++) acc += ldA(A, ta, M, K, i, k) * ldB(B, tb, N, K, k, j);
  size_t c = (size_t)i * N + j;
  C[c] = alpha * acc + (beta == 0.f ? 0.f : beta * C[c]);
}

/* Each 32x32 block stages one tile of op(A) and op(B) in shared memory per k-step. */
__global__ void gemm_tiled_k(int ta, int tb, int M, int N, int K, float alpha,
                             const float* A, const float* B, float beta, float* C) {
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];
  int ty = threadIdx.y, tx = threadIdx.x;
  int i = blockIdx.y * TILE + ty, j = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    As[ty][tx] = (i < M && k0 + tx < K) ? ldA(A, ta, M, K, i, k0 + tx) : 0.f;
    Bs[ty][tx] = (k0 + ty < K && j < N) ? ldB(B, tb, N, K, k0 + ty, j) : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; k++) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (i < M && j < N) {
    size_t c = (size_t)i * N + j;
    C[c] = alpha * acc + (beta == 0.f ? 0.f : beta * C[c]);
  }
}

static cublasHandle_t handle(void) {
  static cublasHandle_t h = NULL;
  if (!h) cublasCreate(&h);
  return h;
}

int gemm_parse(const char* s) {
  const char* names[] = {"naive", "tiled", "cublas"};
  for (int i = 0; i < 3; i++) if (!strcmp(s, names[i])) return i;
  return -1;
}

void gemm(gemm_impl impl, int transA, int transB, int M, int N, int K,
          float alpha, const float* A, const float* B, float beta, float* C) {
  if (impl == GEMM_CUBLAS) {
    /* cuBLAS is column-major: compute C^T = op(B)^T * op(A)^T, which is row-major C. */
    cublasSgemm(handle(), transB ? CUBLAS_OP_T : CUBLAS_OP_N, transA ? CUBLAS_OP_T : CUBLAS_OP_N,
                N, M, K, &alpha, B, transB ? K : N, A, transA ? M : K, &beta, C, N);
    return;
  }
  dim3 blk(TILE, TILE), grd((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  if (impl == GEMM_NAIVE) gemm_naive_k<<<grd, blk>>>(transA, transB, M, N, K, alpha, A, B, beta, C);
  else                    gemm_tiled_k<<<grd, blk>>>(transA, transB, M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 5: Run the test on a GPU**

```bash
make bin/test_gemm
sbatch --account=<class account> scripts/slurm/run.sh ./bin/test_gemm
cat logs/mddp-<jobid>.out
```
Expected: `test_gemm: PASS` (48 cases: 3 implementations × 4 shapes × 4 transpose combinations).

- [ ] **Step 6: Commit**

```bash
git add src/kernels/gemm.h src/kernels/gemm.cu tests/test_gemm.cu
git commit -m "Add naive, tiled and cuBLAS GEMM with transpose flags"
```

---

### Task 4: CIFAR-10 loader and synthetic dataset (T2)

**Files:**
- Create: `src/data/cifar.h`, `src/data/cifar.c`
- Test: `tests/test_cifar.c`

**Interfaces:**
- Produces:
  - `typedef struct { int n, dim; float* X; int* y; } Dataset;` with `X` row-major `n x dim`, pixels scaled to `[-0.5, 0.5]`
  - `int cifar_load(const char* dir, int train, Dataset* d);` reads `data_batch_1..5.bin` (train) or `test_batch.bin`; returns 0, or -1 with `d` empty
  - `void dataset_synthetic(Dataset* d, int n, int dim, int classes, unsigned seed);` with label `i % classes`, fixed class means shared across seeds
  - `void dataset_free(Dataset* d);`

- [ ] **Step 1: Write the header `src/data/cifar.h`**

```c
#pragma once

typedef struct { int n, dim; float* X; int* y; } Dataset;  /* X is n x dim, row-major */

/* CIFAR-10 binary format from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz.
   train=1 reads data_batch_1..5.bin, train=0 reads test_batch.bin. Pixels map to [-0.5, 0.5].
   Returns 0, or -1 with d left empty. */
int cifar_load(const char* dir, int train, Dataset* d);
/* Learnable fake data: label i % classes, class means fixed, noise drawn from seed. */
void dataset_synthetic(Dataset* d, int n, int dim, int classes, unsigned seed);
void dataset_free(Dataset* d);
```

- [ ] **Step 2: Write the failing test `tests/test_cifar.c`**

```c
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
```

- [ ] **Step 3: Run it to verify it fails**

Run: `make bin/test_cifar`
Expected: `No rule to make target 'src/data/cifar.c'`.

- [ ] **Step 4: Implement `src/data/cifar.c`**

```c
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
```

- [ ] **Step 5: Run the test**

```bash
make bin/test_cifar && ./bin/test_cifar
```
Expected: `test_cifar: PASS`

- [ ] **Step 6: Download CIFAR-10 once into the class project storage**

```bash
ls /project                               # find the class project directory
mkdir -p /project/<class dir>/$USER && cd /project/<class dir>/$USER
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xzf cifar-10-binary.tar.gz && rm cifar-10-binary.tar.gz
ls cifar-10-batches-bin                   # data_batch_1.bin ... test_batch.bin
```
Expected: about 170 MB. Both teammates use this one copy. Record the absolute path; Task 7 calls it `$CIFAR_DIR`.

- [ ] **Step 7: Commit**

```bash
git add src/data/cifar.h src/data/cifar.c tests/test_cifar.c
git commit -m "Add CIFAR-10 binary loader and synthetic dataset"
```

---

### Task 5: Layer kernels (T1)

**Files:**
- Create: `src/kernels/layers.h`, `src/kernels/layers.cu`
- Test: `tests/test_layers.cu`

**Interfaces:**
- Produces (all device pointers, row-major):
  - `void bias_act_forward(float* Y, const float* b, int rows, int cols, int relu);` computes `Y[r][c] += b[c]`, then ReLU if `relu`
  - `void relu_backward(float* dY, const float* Y, size_t n);` zeroes `dY[i]` where post-activation `Y[i] <= 0`
  - `void bias_grad(float* db, const float* dY, int rows, int cols);` computes `db[c] = sum_r dY[r][c]`
  - `float softmax_xent(const float* logits, const int* labels, float* dlogits, float* loss_buf, int rows, int classes);` returns the mean loss and writes `dlogits = (softmax - onehot) / rows`; `loss_buf` is device scratch of `rows` floats
  - `void sgd_step(float* w, const float* g, float lr, size_t n);` computes `w -= lr * g`

- [ ] **Step 1: Write the header `src/kernels/layers.h`**

```c
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

/* All pointers are device pointers; matrices are row-major rows x cols. */
void bias_act_forward(float* Y, const float* b, int rows, int cols, int relu);  /* Y += b; ReLU if relu */
void relu_backward(float* dY, const float* Y, size_t n);                        /* dY = 0 where Y <= 0 */
void bias_grad(float* db, const float* dY, int rows, int cols);                 /* db = column sums */
/* Stable softmax + cross-entropy. Writes dlogits = (p - onehot) / rows, returns mean loss.
   loss_buf: device scratch, rows floats. */
float softmax_xent(const float* logits, const int* labels, float* dlogits, float* loss_buf,
                   int rows, int classes);
void sgd_step(float* w, const float* g, float lr, size_t n);                     /* w -= lr * g */

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Write the failing test `tests/test_layers.cu`**

```cuda
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/layers.h"

enum { R = 37, C = 10 };
static int fails = 0;
static unsigned seed = 1;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

static void check(const char* name, const float* got, const float* want, size_t n, float tol) {
  for (size_t i = 0; i < n; i++)
    if (fabsf(got[i] - want[i]) > tol * (1.f + fabsf(want[i]))) {
      printf("FAIL %s at %zu: got %g want %g\n", name, i, got[i], want[i]);
      fails++;
      return;
    }
}
static float* up(const float* h, size_t n) {
  float* d;
  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}
static void down(float* h, const float* d, size_t n) {
  CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
}

int main(void) {
  float x[R * C], g[R * C], b[C], got[R * C], want[R * C];
  for (int i = 0; i < R * C; i++) { x[i] = rnd(); g[i] = rnd(); }
  for (int j = 0; j < C; j++) b[j] = rnd();

  for (int relu = 0; relu < 2; relu++) {
    float *dY = up(x, R * C), *db = up(b, C);
    bias_act_forward(dY, db, R, C, relu);
    down(got, dY, R * C);
    for (int i = 0; i < R * C; i++) { float v = x[i] + b[i % C]; want[i] = relu && v < 0 ? 0.f : v; }
    check(relu ? "bias_relu" : "bias", got, want, R * C, 1e-6f);
    cudaFree(dY); cudaFree(db);
  }

  {
    float *dG = up(g, R * C), *dY = up(x, R * C);
    relu_backward(dG, dY, R * C);
    down(got, dG, R * C);
    for (int i = 0; i < R * C; i++) want[i] = x[i] > 0 ? g[i] : 0.f;
    check("relu_backward", got, want, R * C, 0.f);
    cudaFree(dG); cudaFree(dY);
  }

  {
    float *dG = up(x, R * C), *dB;
    CUDA_CHECK(cudaMalloc(&dB, C * sizeof(float)));
    bias_grad(dB, dG, R, C);
    down(got, dB, C);
    for (int j = 0; j < C; j++) { want[j] = 0; for (int r = 0; r < R; r++) want[j] += x[r * C + j]; }
    check("bias_grad", got, want, C, 1e-5f);
    cudaFree(dG); cudaFree(dB);
  }

  {
    int y[R];
    for (int r = 0; r < R; r++) y[r] = r % C;
    int* dy;
    CUDA_CHECK(cudaMalloc(&dy, R * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dy, y, R * sizeof(int), cudaMemcpyHostToDevice));
    float *dx = up(x, R * C), *dd, *dl;
    CUDA_CHECK(cudaMalloc(&dd, R * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dl, R * sizeof(float)));
    float loss = softmax_xent(dx, dy, dd, dl, R, C);
    down(got, dd, R * C);
    double total = 0;
    for (int r = 0; r < R; r++) {
      const float* row = x + r * C;
      float m = row[0];
      for (int c = 1; c < C; c++) m = fmaxf(m, row[c]);
      double s = 0;
      for (int c = 0; c < C; c++) s += exp(row[c] - m);
      for (int c = 0; c < C; c++) want[r * C + c] = (float)((exp(row[c] - m) / s - (c == y[r])) / R);
      total += log(s) - (row[y[r]] - m);
    }
    check("softmax_xent_grad", got, want, R * C, 1e-5f);
    float wl = (float)(total / R);
    check("softmax_xent_loss", &loss, &wl, 1, 1e-5f);
    cudaFree(dy); cudaFree(dx); cudaFree(dd); cudaFree(dl);
  }

  {
    float *dw = up(x, R * C), *dg = up(g, R * C);
    sgd_step(dw, dg, 0.1f, R * C);
    down(got, dw, R * C);
    for (int i = 0; i < R * C; i++) want[i] = x[i] - 0.1f * g[i];
    check("sgd_step", got, want, R * C, 1e-6f);
    cudaFree(dw); cudaFree(dg);
  }

  printf(fails ? "test_layers: FAIL\n" : "test_layers: PASS\n");
  return fails != 0;
}
```

- [ ] **Step 3: Run it to verify it fails**

Run: `make bin/test_layers`
Expected: `No rule to make target 'src/kernels/layers.cu'` (or `build/layers.o`).

- [ ] **Step 4: Implement `src/kernels/layers.cu`**

```cuda
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/layers.h"

static unsigned blocks(size_t n) { return (unsigned)((n + 255) / 256); }

__global__ void bias_act_k(float* Y, const float* b, int rows, int cols, int relu) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * cols) return;
  float v = Y[i] + b[i % cols];
  Y[i] = relu && v < 0.f ? 0.f : v;
}

__global__ void relu_bwd_k(float* dY, const float* Y, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && Y[i] <= 0.f) dY[i] = 0.f;
}

__global__ void bias_grad_k(float* db, const float* dY, int rows, int cols) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  float s = 0.f;
  for (int r = 0; r < rows; r++) s += dY[(size_t)r * cols + j];
  db[j] = s;
}

/* One thread per row; fine for 10 classes. */
__global__ void softmax_xent_k(const float* x, const int* y, float* dx, float* loss, int rows, int C) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const float* row = x + (size_t)r * C;
  float m = row[0];
  for (int c = 1; c < C; c++) m = fmaxf(m, row[c]);
  float s = 0.f;
  for (int c = 0; c < C; c++) s += expf(row[c] - m);
  float inv = 1.f / rows;
  for (int c = 0; c < C; c++) dx[(size_t)r * C + c] = (expf(row[c] - m) / s - (c == y[r])) * inv;
  loss[r] = logf(s) - (row[y[r]] - m);
}

__global__ void sgd_k(float* w, const float* g, float lr, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) w[i] -= lr * g[i];
}

void bias_act_forward(float* Y, const float* b, int rows, int cols, int relu) {
  bias_act_k<<<blocks((size_t)rows * cols), 256>>>(Y, b, rows, cols, relu);
  CUDA_CHECK(cudaGetLastError());
}

void relu_backward(float* dY, const float* Y, size_t n) {
  relu_bwd_k<<<blocks(n), 256>>>(dY, Y, n);
  CUDA_CHECK(cudaGetLastError());
}

void bias_grad(float* db, const float* dY, int rows, int cols) {
  bias_grad_k<<<blocks(cols), 256>>>(db, dY, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

float softmax_xent(const float* logits, const int* labels, float* dlogits, float* loss_buf,
                   int rows, int classes) {
  softmax_xent_k<<<blocks(rows), 256>>>(logits, labels, dlogits, loss_buf, rows, classes);
  CUDA_CHECK(cudaGetLastError());
  /* ponytail: per-row losses summed on the host; a device reduction only if this shows up in profiles */
  float* h = (float*)malloc(rows * sizeof(float));
  CUDA_CHECK(cudaMemcpy(h, loss_buf, rows * sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0;
  for (int r = 0; r < rows; r++) s += h[r];
  free(h);
  return (float)(s / rows);
}

void sgd_step(float* w, const float* g, float lr, size_t n) {
  sgd_k<<<blocks(n), 256>>>(w, g, lr, n);
  CUDA_CHECK(cudaGetLastError());
}
```

- [ ] **Step 5: Run the test on a GPU**

```bash
make bin/test_layers
sbatch --account=<class account> scripts/slurm/run.sh ./bin/test_layers
cat logs/mddp-<jobid>.out
```
Expected: `test_layers: PASS`

- [ ] **Step 6: Commit**

```bash
git add src/kernels/layers.h src/kernels/layers.cu tests/test_layers.cu
git commit -m "Add bias/ReLU, softmax cross-entropy and SGD kernels"
```

---

### Task 6: MLP forward/backward with gradient check (T1)

**Files:**
- Create: `src/model/mlp.h`, `src/model/mlp.cu`
- Test: `tests/test_mlp.cu`

**Interfaces:**
- Consumes: `gemm`, `gemm_impl` (Task 3); `bias_act_forward`, `relu_backward`, `bias_grad`, `softmax_xent`, `sgd_step` (Task 5).
- Produces:
  - `typedef void (*grad_hook)(int layer, float* grad, size_t n, void* ctx);` called right after layer `layer`'s gradients are final, last layer first. `grad` points at that layer's contiguous `[dW | db]` region of `n` floats. This is the spec's `on_grad_ready`; Plan 2 uses it for overlap.
  - `typedef struct { int in, out; float *W, *b, *dW, *db, *Y; } Layer;`
  - `typedef struct { int nlayers, max_batch, max_width; Layer* L; float *params, *grads; size_t nparams; float *dbuf[2]; float* loss_buf; gemm_impl impl; } MLP;` where `params` and `grads` are single flat device buffers (layer l = `[W_l | b_l]`, in order)
  - `MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl);` with He-normal weights, zero biases, and identical output for identical `seed`
  - `void mlp_free(MLP* m);`
  - `const float* mlp_forward(MLP* m, const float* X, int B);` returns device logits `B x sizes[nsizes-1]`
  - `float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx);` returns mean loss and fills `grads` (`hook` may be NULL)
  - `void mlp_sgd(MLP* m, float lr, float grad_scale);` computes `params -= lr * grad_scale * grads`

- [ ] **Step 1: Write the header `src/model/mlp.h`**

```c
#pragma once
#include <stddef.h>
#include "kernels/gemm.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Called when a layer's gradients are final (last layer first). grad = that layer's [dW | db]. */
typedef void (*grad_hook)(int layer, float* grad, size_t n, void* ctx);

typedef struct {
  int in, out;
  float *W, *b;    /* views into MLP.params; W is in x out */
  float *dW, *db;  /* views into MLP.grads, same layout */
  float* Y;        /* post-activation output, max_batch x out */
} Layer;

typedef struct {
  int nlayers, max_batch, max_width;
  Layer* L;
  float *params, *grads;  /* flat device buffers: [W0|b0|W1|b1|...] */
  size_t nparams;
  float* dbuf[2];         /* backward ping-pong scratch, max_batch x max_width */
  float* loss_buf;        /* max_batch */
  gemm_impl impl;
} MLP;

/* sizes = {input, hidden..., classes}. ReLU after every layer except the last. */
MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl);
void mlp_free(MLP* m);
const float* mlp_forward(MLP* m, const float* X, int B);  /* device logits, B x classes */
float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx);
void mlp_sgd(MLP* m, float lr, float grad_scale);         /* params -= lr * grad_scale * grads */

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Write the failing test `tests/test_mlp.cu`**

```cuda
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "model/mlp.h"

static unsigned seed = 7;
static float rnd(void) { seed = seed * 1664525u + 1013904223u; return ((seed >> 8) / 16777216.0f) * 2.f - 1.f; }

int main(void) {
  const int sizes[] = {8, 16, 4};
  enum { B = 3 };
  float X[B * 8];
  for (int i = 0; i < B * 8; i++) X[i] = rnd();
  int y[B] = {0, 3, 1};
  float* dX; int* dy;
  CUDA_CHECK(cudaMalloc(&dX, sizeof X));
  CUDA_CHECK(cudaMalloc(&dy, sizeof y));
  CUDA_CHECK(cudaMemcpy(dX, X, sizeof X, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dy, y, sizeof y, cudaMemcpyHostToDevice));
  int fails = 0;

  /* 1. analytic gradients match central differences */
  MLP* m = mlp_create(sizes, 3, B, 123, GEMM_TILED);
  size_t n = m->nparams;
  float *ana = (float*)malloc(n * 4), *par = (float*)malloc(n * 4), *g2 = (float*)malloc(n * 4);
  mlp_forward_backward(m, dX, dy, B, NULL, NULL);
  CUDA_CHECK(cudaMemcpy(ana, m->grads, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(par, m->params, n * 4, cudaMemcpyDeviceToHost));
  const float eps = 1e-3f;
  for (size_t i = 0; i < n; i++) {
    float v = par[i] + eps;
    CUDA_CHECK(cudaMemcpy(m->params + i, &v, 4, cudaMemcpyHostToDevice));
    float lp = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
    v = par[i] - eps;
    CUDA_CHECK(cudaMemcpy(m->params + i, &v, 4, cudaMemcpyHostToDevice));
    float lm = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
    CUDA_CHECK(cudaMemcpy(m->params + i, &par[i], 4, cudaMemcpyHostToDevice));
    float num = (lp - lm) / (2 * eps);
    if (fabsf(num - ana[i]) > 2e-3f + 2e-2f * fabsf(ana[i])) {
      if (fails < 5) printf("FAIL grad[%zu]: numeric %g analytic %g\n", i, num, ana[i]);
      fails++;
    }
  }

  /* 2. every GEMM implementation gives the same gradients */
  for (int impl = GEMM_NAIVE; impl <= GEMM_CUBLAS; impl++) {
    MLP* m2 = mlp_create(sizes, 3, B, 123, (gemm_impl)impl);
    mlp_forward_backward(m2, dX, dy, B, NULL, NULL);
    CUDA_CHECK(cudaMemcpy(g2, m2->grads, n * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; i++)
      if (fabsf(g2[i] - ana[i]) > 1e-4f * (1.f + fabsf(ana[i]))) {
        printf("FAIL impl %d grad[%zu]: %g vs %g\n", impl, i, g2[i], ana[i]);
        fails++;
        break;
      }
    mlp_free(m2);
  }

  /* 3. SGD overfits a fixed batch */
  float first = mlp_forward_backward(m, dX, dy, B, NULL, NULL), last = first;
  for (int it = 0; it < 500; it++) {
    mlp_sgd(m, 0.2f, 1.0f);
    last = mlp_forward_backward(m, dX, dy, B, NULL, NULL);
  }
  if (!(last < 0.1f * first)) { printf("FAIL overfit: loss %g -> %g\n", first, last); fails++; }

  mlp_free(m);
  free(ana); free(par); free(g2);
  printf(fails ? "test_mlp: FAIL (%d)\n" : "test_mlp: PASS\n", fails);
  return fails != 0;
}
```

- [ ] **Step 3: Run it to verify it fails**

Run: `make bin/test_mlp`
Expected: `No rule to make target 'src/model/mlp.cu'` (or `build/mlp.o`).

- [ ] **Step 4: Implement `src/model/mlp.cu`**

```cuda
#include <math.h>
#include <stdlib.h>
#include "cuda_check.h"
#include "kernels/layers.h"
#include "model/mlp.h"

static unsigned long long rng;
static double uniform01(void) {
  rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
  return ((rng >> 11) + 0.5) / 9007199254740992.0;
}
static float normal(void) { return (float)(sqrt(-2.0 * log(uniform01())) * cos(6.283185307179586 * uniform01())); }

MLP* mlp_create(const int* sizes, int nsizes, int max_batch, unsigned seed, gemm_impl impl) {
  MLP* m = (MLP*)calloc(1, sizeof(MLP));
  m->nlayers = nsizes - 1;
  m->max_batch = max_batch;
  m->impl = impl;
  m->L = (Layer*)calloc(m->nlayers, sizeof(Layer));
  for (int l = 0; l < m->nlayers; l++) {
    m->nparams += (size_t)sizes[l] * sizes[l + 1] + sizes[l + 1];
    if (sizes[l + 1] > m->max_width) m->max_width = sizes[l + 1];
  }
  CUDA_CHECK(cudaMalloc(&m->params, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m->grads, m->nparams * sizeof(float)));
  CUDA_CHECK(cudaMemset(m->grads, 0, m->nparams * sizeof(float)));
  float* h = (float*)calloc(m->nparams, sizeof(float));
  rng = seed;
  size_t off = 0;
  for (int l = 0; l < m->nlayers; l++) {
    Layer* L = &m->L[l];
    L->in = sizes[l];
    L->out = sizes[l + 1];
    size_t nw = (size_t)L->in * L->out;
    float sd = sqrtf(2.0f / L->in);  /* He init; biases stay zero */
    for (size_t i = 0; i < nw; i++) h[off + i] = sd * normal();
    L->W = m->params + off;  L->b = L->W + nw;
    L->dW = m->grads + off;  L->db = L->dW + nw;
    CUDA_CHECK(cudaMalloc(&L->Y, (size_t)max_batch * L->out * sizeof(float)));
    off += nw + L->out;
  }
  CUDA_CHECK(cudaMemcpy(m->params, h, m->nparams * sizeof(float), cudaMemcpyHostToDevice));
  free(h);
  for (int i = 0; i < 2; i++)
    CUDA_CHECK(cudaMalloc(&m->dbuf[i], (size_t)max_batch * m->max_width * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&m->loss_buf, max_batch * sizeof(float)));
  return m;
}

void mlp_free(MLP* m) {
  for (int l = 0; l < m->nlayers; l++) cudaFree(m->L[l].Y);
  cudaFree(m->params); cudaFree(m->grads);
  cudaFree(m->dbuf[0]); cudaFree(m->dbuf[1]); cudaFree(m->loss_buf);
  free(m->L);
  free(m);
}

const float* mlp_forward(MLP* m, const float* X, int B) {
  const float* A = X;
  for (int l = 0; l < m->nlayers; l++) {
    Layer* L = &m->L[l];
    gemm(m->impl, 0, 0, B, L->out, L->in, 1.f, A, L->W, 0.f, L->Y);     /* Y = A W */
    bias_act_forward(L->Y, L->b, B, L->out, l < m->nlayers - 1);
    A = L->Y;
  }
  return A;
}

float mlp_forward_backward(MLP* m, const float* X, const int* y, int B, grad_hook hook, void* ctx) {
  const float* logits = mlp_forward(m, X, B);
  int last = m->nlayers - 1, cur = 0;
  float loss = softmax_xent(logits, y, m->dbuf[0], m->loss_buf, B, m->L[last].out);
  for (int l = last; l >= 0; l--) {
    Layer* L = &m->L[l];
    const float* Ain = l > 0 ? m->L[l - 1].Y : X;
    float* dZ = m->dbuf[cur];
    gemm(m->impl, 1, 0, L->in, L->out, B, 1.f, Ain, dZ, 0.f, L->dW);    /* dW = Ain^T dZ */
    bias_grad(L->db, dZ, B, L->out);
    if (hook) hook(l, L->dW, (size_t)L->in * L->out + L->out, ctx);
    if (l > 0) {
      float* dA = m->dbuf[cur ^ 1];
      gemm(m->impl, 0, 1, B, L->in, L->out, 1.f, dZ, L->W, 0.f, dA);    /* dA = dZ W^T */
      relu_backward(dA, Ain, (size_t)B * L->in);
      cur ^= 1;
    }
  }
  return loss;
}

void mlp_sgd(MLP* m, float lr, float grad_scale) {
  sgd_step(m->params, m->grads, lr * grad_scale, m->nparams);
}
```

- [ ] **Step 5: Run the test on a GPU**

```bash
make bin/test_mlp
sbatch --account=<class account> scripts/slurm/run.sh ./bin/test_mlp
cat logs/mddp-<jobid>.out
```
Expected: `test_mlp: PASS`. If a single gradient-check entry fails with numeric and analytic values of opposite sign, a ReLU input sits within `eps` of zero; change `seed = 7` in the test to another value and rerun before debugging the backward pass.

- [ ] **Step 6: Commit**

```bash
git add src/model/mlp.h src/model/mlp.cu tests/test_mlp.cu
git commit -m "Add MLP forward/backward with flat gradient buffer and gradient check"
```

---

### Task 7: Single-GPU trainer (T1)

**Files:**
- Create: `src/train.c`

**Interfaces:**
- Consumes: `Dataset`, `cifar_load`, `dataset_synthetic`, `dataset_free` (Task 4); `MLP` API (Task 6); `gemm_parse` (Task 3); `ar_parse`, `allreduce_sum` (Task 2, only used when ranks > 1, which Task 8 tests).
- Produces: `bin/train` CLI:
  `train (--data DIR | --synthetic) [--epochs N] [--steps N] [--batch N] [--hidden N] [--layers N] [--lr X] [--seed N] [--gemm naive|tiled|cublas] [--allreduce naive|tree|ring|mpi]`
  stdout lines on rank 0: `step <i> loss <mean> compute_s <s> comm_s <s>` per step, and `epoch <e> test_acc <frac>` per epoch. `--batch` is the global batch; each rank gets `batch / ranks`.

- [ ] **Step 1: Define the acceptance check (the "test") before writing code**

Two runs on one GPU must succeed:

```bash
./bin/train --synthetic --steps 50 --batch 64 --hidden 256 --layers 2 --lr 0.05
```
Expected: the step-49 loss is below 20% of the step-0 loss, and the last line is `epoch 0 test_acc` ≥ 0.90 (synthetic classes are well separated).

```bash
./bin/train --data $CIFAR_DIR --epochs 5 --batch 256 --hidden 1024 --layers 2 --lr 0.02
```
Expected: `epoch 4 test_acc` ≥ 0.35. An MLP trained with plain SGD on CIFAR-10 typically reaches 0.40–0.50 here. If it stays near 0.10, the model isn't learning: check the learning rate and the data path.

- [ ] **Step 2: Run to verify it fails**

Run: `make bin/train`
Expected: `No rule to make target 'src/train.c'`.

- [ ] **Step 3: Implement `src/train.c`**

```c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "comm/allreduce.h"
#include "cuda_check.h"
#include "data/cifar.h"
#include "model/mlp.h"

typedef struct {
  const char* data;
  int synthetic, epochs, steps, batch, hidden, layers, gemm, allreduce;
  float lr;
  unsigned seed;
} Opts;

static void usage(void) {
  fprintf(stderr,
          "usage: train (--data DIR | --synthetic) [--epochs N] [--steps N] [--batch N] [--hidden N]\n"
          "             [--layers N] [--lr X] [--seed N] [--gemm naive|tiled|cublas]\n"
          "             [--allreduce naive|tree|ring|mpi]\n");
  exit(2);
}

static Opts parse(int argc, char** argv) {
  Opts o = {NULL, 0, 1, 0, 256, 4096, 3, GEMM_TILED, AR_RING, 0.01f, 42};
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    if (!strcmp(a, "--synthetic")) { o.synthetic = 1; continue; }
    if (i + 1 >= argc) usage();
    const char* v = argv[++i];
    if (!strcmp(a, "--data")) o.data = v;
    else if (!strcmp(a, "--epochs")) o.epochs = atoi(v);
    else if (!strcmp(a, "--steps")) o.steps = atoi(v);
    else if (!strcmp(a, "--batch")) o.batch = atoi(v);
    else if (!strcmp(a, "--hidden")) o.hidden = atoi(v);
    else if (!strcmp(a, "--layers")) o.layers = atoi(v);
    else if (!strcmp(a, "--lr")) o.lr = (float)atof(v);
    else if (!strcmp(a, "--seed")) o.seed = (unsigned)atoi(v);
    else if (!strcmp(a, "--gemm")) { if ((o.gemm = gemm_parse(v)) < 0) usage(); }
    else if (!strcmp(a, "--allreduce")) { if ((o.allreduce = ar_parse(v)) < 0) usage(); }
    else usage();
  }
  if ((!o.data && !o.synthetic) || o.layers < 1 || o.layers > 14 || o.batch < 1) usage();
  return o;
}

/* Same seed on every rank -> same permutation everywhere. */
static unsigned long long rng;
static unsigned next_u32(void) { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL; return (unsigned)(rng >> 33); }
static void shuffle(int* perm, int n, unsigned seed) {
  rng = seed;
  for (int i = 0; i < n; i++) perm[i] = i;
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(next_u32() % (unsigned)(i + 1)), t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
}

static float evaluate(MLP* m, const Dataset* d, int b, float* dX, float* hlogits) {
  const int classes = 10;
  int correct = 0, total = 0;
  for (int s = 0; s + b <= d->n; s += b) {
    CUDA_CHECK(cudaMemcpy(dX, d->X + (size_t)s * d->dim, (size_t)b * d->dim * sizeof(float), cudaMemcpyHostToDevice));
    const float* logits = mlp_forward(m, dX, b);
    CUDA_CHECK(cudaMemcpy(hlogits, logits, (size_t)b * classes * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < b; i++) {
      int best = 0;
      for (int c = 1; c < classes; c++) if (hlogits[i * classes + c] > hlogits[i * classes + best]) best = c;
      correct += best == d->y[s + i];
      total++;
    }
  }
  return total ? (float)correct / total : 0.f;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, p, lrank, ndev;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &lrank);
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(lrank % ndev));

  Opts o = parse(argc, argv);
  if (o.batch % p) {
    if (!rank) fprintf(stderr, "--batch %d is not divisible by %d ranks\n", o.batch, p);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  int b = o.batch / p;

  Dataset train, test;
  if (o.synthetic) {
    dataset_synthetic(&train, 4096, 3072, 10, 1);
    dataset_synthetic(&test, 1024, 3072, 10, 2);
  } else if (cifar_load(o.data, 1, &train) || cifar_load(o.data, 0, &test)) {
    fprintf(stderr, "cannot load CIFAR-10 from %s\n", o.data);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int sizes[16], ns = 0;
  sizes[ns++] = train.dim;
  for (int l = 0; l < o.layers; l++) sizes[ns++] = o.hidden;
  sizes[ns++] = 10;
  MLP* m = mlp_create(sizes, ns, b, o.seed, (gemm_impl)o.gemm);

  float* hX = malloc((size_t)b * train.dim * sizeof(float));
  float* hlogits = malloc((size_t)b * 10 * sizeof(float));
  int* hy = malloc(b * sizeof(int));
  float* dX;
  int* dy;
  CUDA_CHECK(cudaMalloc(&dX, (size_t)b * train.dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dy, b * sizeof(int)));
  float* hgrad = NULL;  /* pinned staging buffer for the host all-reduce */
  if (p > 1) CUDA_CHECK(cudaMallocHost((void**)&hgrad, m->nparams * sizeof(float)));
  int* perm = malloc(train.n * sizeof(int));

  int steps_per_epoch = train.n / o.batch, step = 0, done = 0;
  for (int e = 0; e < o.epochs && !done; e++) {
    shuffle(perm, train.n, o.seed + e);
    for (int s = 0; s < steps_per_epoch && !done; s++) {
      /* rank r takes samples r, r+p, r+2p, ... of the global batch */
      for (int k = 0; k < b; k++) {
        int idx = perm[s * o.batch + rank + k * p];
        memcpy(hX + (size_t)k * train.dim, train.X + (size_t)idx * train.dim, train.dim * sizeof(float));
        hy[k] = train.y[idx];
      }
      CUDA_CHECK(cudaMemcpy(dX, hX, (size_t)b * train.dim * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dy, hy, b * sizeof(int), cudaMemcpyHostToDevice));

      double t0 = MPI_Wtime();
      float loss = mlp_forward_backward(m, dX, dy, b, NULL, NULL);
      CUDA_CHECK(cudaDeviceSynchronize());
      double t1 = MPI_Wtime();
      if (p > 1) {
        CUDA_CHECK(cudaMemcpy(hgrad, m->grads, m->nparams * sizeof(float), cudaMemcpyDeviceToHost));
        allreduce_sum(hgrad, m->nparams, (ar_algo)o.allreduce, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(m->grads, hgrad, m->nparams * sizeof(float), cudaMemcpyHostToDevice));
      }
      double t2 = MPI_Wtime();
      mlp_sgd(m, o.lr, 1.0f / p);  /* sum of per-rank means / p = global-batch mean */

      float gl;
      MPI_Allreduce(&loss, &gl, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      if (!rank) printf("step %d loss %.6f compute_s %.6f comm_s %.6f\n", step, gl / p, t1 - t0, t2 - t1);
      step++;
      if (o.steps && step >= o.steps) done = 1;
    }
    if (!rank) {
      printf("epoch %d test_acc %.4f\n", e, evaluate(m, &test, b, dX, hlogits));
      fflush(stdout);
    }
  }

  free(perm); free(hX); free(hlogits); free(hy);
  cudaFree(dX); cudaFree(dy);
  if (hgrad) cudaFreeHost(hgrad);
  mlp_free(m);
  dataset_free(&train);
  dataset_free(&test);
  MPI_Finalize();
  return 0;
}
```

- [ ] **Step 4: Run both acceptance checks on one GPU**

```bash
make bin/train
export CIFAR_DIR=/project/<class dir>/<netid>/cifar-10-batches-bin
sbatch --account=<class account> --ntasks=1 --gres=gpu:p100:1 scripts/slurm/run.sh \
  bash -c "./bin/train --synthetic --steps 50 --batch 64 --hidden 256 --layers 2 --lr 0.05 && \
           ./bin/train --data $CIFAR_DIR --epochs 5 --batch 256 --hidden 1024 --layers 2 --lr 0.02"
cat logs/mddp-<jobid>.out
```
Expected: the thresholds from Step 1. Also record the CIFAR run's average `compute_s` per step. It's the baseline for the speedup plots in Plan 2.

- [ ] **Step 5: Commit**

```bash
git add src/train.c
git commit -m "Add single-GPU trainer with CIFAR-10 and synthetic data"
```

---

### Task 8: Data-parallel training and the equality test (T2)

**Files:**
- Create: `scripts/test_dp.sh`
- Modify: none. `src/train.c` already runs the staged all-reduce when ranks > 1; this task proves it correct.

**Interfaces:**
- Consumes: `bin/train` output format from Task 7.
- Produces: `scripts/test_dp.sh NP` returns exit 0 only if every all-reduce algorithm on `NP` ranks reproduces the 1-rank loss curve (max relative difference ≤ 1e-3 over 20 steps).

- [ ] **Step 1: Write the test `scripts/test_dp.sh`**

```bash
#!/bin/bash
# usage: scripts/test_dp.sh NP   -- N-rank data-parallel loss curve must match 1 rank
set -euo pipefail
NP=${1:-2}
ARGS="--synthetic --steps 20 --batch 64 --hidden 256 --layers 2 --lr 0.05"
launch() {
  local n=$1; shift
  if [ -n "${SLURM_JOB_ID:-}" ]; then srun --mpi="${SRUN_MPI:-pmix}" --ntasks="$n" "$@"
  else mpirun --oversubscribe -np "$n" "$@"; fi
}
ref=$(mktemp); out=$(mktemp)
trap 'rm -f "$ref" "$out"' EXIT
launch 1 ./bin/train $ARGS | awk '$1 == "step" { print $4 }' > "$ref"
status=0
for ar in naive tree ring mpi; do
  launch "$NP" ./bin/train $ARGS --allreduce "$ar" | awk '$1 == "step" { print $4 }' > "$out"
  paste "$ref" "$out" | awk -v ar="$ar" -v np="$NP" '
    { d = $1 - $2; if (d < 0) d = -d; r = d / ($1 > 1e-6 ? $1 : 1e-6); if (r > m) m = r; n++ }
    END {
      if (n != 20 || m > 1e-3) { printf "test_dp %s np=%s: FAIL (rows=%d maxrel=%g)\n", ar, np, n, m; exit 1 }
      printf "test_dp %s np=%s: PASS (maxrel=%g)\n", ar, np, m
    }' || status=1
done
exit $status
```

```bash
chmod +x scripts/test_dp.sh
```

- [ ] **Step 2: Run it on 2 GPUs (one node)**

```bash
sbatch --account=<class account> scripts/slurm/run.sh scripts/test_dp.sh 2
cat logs/mddp-<jobid>.out
```
Expected: four `PASS` lines (naive, tree, ring, mpi), each with `maxrel` well under 1e-3; values around 1e-6 are normal. A `FAIL` with `rows=0` means a launch error, so read the rest of the log. A `FAIL` with a large `maxrel` means the ranks see different data or apply different updates: check the sample-index formula and the `1.0f / p` scale.

- [ ] **Step 3: Run it across 2 nodes (4 GPUs) to cover InfiniBand**

```bash
sbatch --account=<class account> --nodes=2 --ntasks=4 --ntasks-per-node=2 scripts/slurm/run.sh scripts/test_dp.sh 4
cat logs/mddp-<jobid>.out
```
Expected: four `PASS` lines with `np=4`. Cost: about 2× the single-node job.

- [ ] **Step 4: Commit**

```bash
git add scripts/test_dp.sh
git commit -m "Add data-parallel loss-curve equality test"
```

---

### Task 9: Communication microbenchmark (T2)

**Files:**
- Create: `bench/bench_comm.c`

**Interfaces:**
- Consumes: `allreduce_sum`, `ar_name`, `ar_algo` (Task 2).
- Produces: `bin/bench_comm [max_bytes]` prints CSV on rank 0: header `kind,algo,np,bytes,seconds`, then `pingpong,-,<np>,<bytes>,<one-way seconds>` rows (rank 0 ↔ last rank) and `allreduce,<algo>,<np>,<bytes>,<seconds>` rows (max over ranks), with bytes = 4, 16, 64, … up to `max_bytes` (default 256 MB). Plan 2 fits the α–β model and draws experiment E3 from this output.

- [ ] **Step 1: Write the check first (already in the Makefile's `check-cpu`)**

With `max_bytes = 65536` there are 8 sizes (4^1 … 4^8). The output must have 1 header line + 8 ping-pong rows + 4 algorithms × 8 sizes = **41 lines**.

Run: `make bin/bench_comm`
Expected: `No rule to make target 'bench/bench_comm.c'`.

- [ ] **Step 2: Implement `bench/bench_comm.c`**

```c
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
```

- [ ] **Step 3: Run the full local check**

```bash
make check-cpu
```
Expected: `test_cifar: PASS`, four `test_allreduce np=N: PASS` lines, and the bench line-count check exits 0 (make prints no error).

- [ ] **Step 4: Commit**

```bash
git add bench/bench_comm.c
git commit -m "Add ping-pong and all-reduce microbenchmark with CSV output"
```

---

## Milestone 1 done when

- `make check-cpu` passes on a laptop.
- On CARC: `test_gemm`, `test_layers`, `test_mlp` print PASS; `scripts/test_dp.sh 2` and `scripts/test_dp.sh 4` (2 nodes) print four PASS lines each.
- 1-GPU CIFAR-10 reaches test accuracy ≥ 0.35 in 5 epochs, and the per-step `compute_s` baseline is recorded.

## Plans that follow (written after Milestone 1, using its measurements)

- **Plan 2: communication and scaling.**
  - Bucketed overlap through `grad_hook` with a comm thread
  - GPU-aware MPI path and NCCL baseline
  - Register-tiled GEMM
  - α–β fit from `bench_comm`
  - Strong and weak scaling sweeps (E3–E6)
- **Plan 3: advanced parallelism and reporting.**
  - WMMA tensor-core GEMM and FP16 mixed precision (E1, E2, E9)
  - GPipe pipeline and 2D parallelism (E7, E8)
  - Nsight timelines
  - README report and presentation
