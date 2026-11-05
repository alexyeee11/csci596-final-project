# Mini-DDP Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide gradient communication behind backprop with bucketed overlap, speed up our own GEMM with register tiling, explain all-reduce cost with a fitted α–β model, and produce the first 1 → 8 GPU strong/weak scaling plot.

**Architecture:** Builds on Milestone 1 (all on `main`): `gemm()` with naive/tiled/cuBLAS, the MLP with a flat `grads` buffer and a `grad_hook` called per layer during backward (last layer first), host all-reduce `allreduce_sum()`, and `bin/train`. This milestone adds a `regtile` GEMM, an overlap module (`src/comm/overlap.c`) that plugs into `grad_hook` and all-reduces fixed-size buckets of the flat gradient buffer on a background thread while backward continues, an optional NCCL path, and Slurm sweep + Python plot scripts whose outputs are committed under `results/`.

**Tech Stack:** C (gnu11), CUDA C++ (nvcc), POSIX threads, OpenMPI 5 (`MPI_THREAD_SERIALIZED`), cuBLAS (reference), NCCL (optional baseline), Slurm on CARC Discovery, Python 3 + matplotlib for plots (laptop only).

**Spec:** `docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md`
**Previous plan:** `docs/superpowers/plans/2026-10-28-mini-ddp-milestone1.md` (its Global Constraints still apply)

## Global Constraints

- Everything in Milestone 1's Global Constraints still holds: one MPI rank per GPU (`local_rank % device_count`), row-major matrices, `W` is `in x out`, `.cu` files contain no MPI, `.c` files contain no kernels, `extern "C"` guards on shared headers, tests print `<name>: PASS` or `FAIL ...` and exit non-zero on failure.
- Dependencies on CARC: MPI, CUDA runtime, cuBLAS (reference only), pthreads, NCCL (optional, behind `USE_NCCL`). Plot scripts use Python 3 + matplotlib and run on a laptop, never inside a Slurm job.
- `bin/train` stdout keeps its Milestone 1 fields in the same positions; this milestone only appends `step_s` at the end of each step line: `step <i> loss <L> compute_s <c> comm_s <m> step_s <s>`. `scripts/test_dp.sh` reads field 4 (the loss).
- Timing definitions: `compute_s` = forward + backward until the default stream is idle; `comm_s` = gradient communication still outstanding after that (exposed communication); `step_s` = forward start to SGD finished.
- Every parallel configuration must still match the 1-GPU loss curve: `scripts/test_dp.sh` max relative loss difference ≤ 1e-3 over 20 steps.
- Measurements that go in the report are committed as CSV/log + PNG under `results/<experiment>/`. Scaling numbers within one plot use one GPU type: P100 (`--gres=gpu:p100:N`) for this milestone.
- CARC: account `anakano_429`; there is no `/project` on Discovery, CIFAR-10 is at `~/data/cifar-10-batches-bin`. Multi-node jobs need `--partition=gpu` (`debug` allows one node). Max 2 GPUs per node. Costs per reserved minute: P100 = 4 SU per GPU, A40/A100 = 8 SU, plus 1 SU per CPU and per 4 GB.
- Team workflow: each task is a branch → pull request whose description lists what changed and the CARC test output → the other teammate looks it over → merge. Small doc-only changes may merge directly.
- Git commit messages and PR descriptions: no attribution lines (no `Co-Authored-By`, no "Generated with").
- Milestone target: Nov 17, 2026. Final deadline Dec 11, 2026.

## Ownership and order

| Task | Owner | Needs GPU? | Depends on |
|---|---|---|---|
| 1. Register-tiled GEMM | T1 Alex (kernels) | yes | — |
| 2. GEMM benchmark + plot (E1) | T1 Alex | yes | 1 |
| 3. α–β model from `bench_comm` (E3) | T2 Shaozhe (communication) | no (CPU nodes) | — |
| 4. Bucketed overlap of all-reduce with backward | T2 Shaozhe | yes | — |
| 5. NCCL baseline | T2 Shaozhe | yes | 4 (shares `train.c` edits) |
| 6. 1 → 8 GPU scaling sweep + plot (E4, E5) | T1 Alex (both review) | yes | 1, 4 |

Tasks 1, 3 and 4 can start in parallel on day one. Do Task 5 after Task 4 is merged, since both edit `src/train.c`. Task 6 runs last.

## File structure

```
src/kernels/gemm.{h,cu}        + GEMM_REGTILE (Task 1)
tests/test_gemm.cu             covers 4 implementations (Task 1)
bench/bench_gemm.cu            GEMM timing, CSV out (Task 2)
scripts/plot_gemm.py           GFLOP/s plot (Task 2)
scripts/slurm/bench_comm_sweep.sh  ping-pong + all-reduce, 2..16 ranks (Task 3)
scripts/fit_alpha_beta.py      α–β fit + measured-vs-model plot (Task 3)
src/comm/overlap.{h,c}         bucketed comm thread (Task 4)
src/train.c                    --overlap, --bucket-mb, step_s (Task 4); --allreduce nccl (Task 5)
scripts/test_dp.sh             extra train args + ALGOS override (Task 4)
scripts/env.sh                 module load nccl + NCCL_HOME (Task 5)
scripts/slurm/sweep_scaling.sh strong + weak scaling, 1..8 GPUs (Task 6)
scripts/plot_scaling.py        speedup / efficiency / time breakdown (Task 6)
results/{gemm,comm,scaling}/   committed CSVs, logs, PNGs
Makefile                       new rules per task
```

---

### Task 1: Register-tiled GEMM (T1)

**Files:**
- Modify: `src/kernels/gemm.h`, `src/kernels/gemm.cu`, `tests/test_gemm.cu`

**Interfaces:**
- Consumes: existing `ldA`, `ldB`, `gemm()` in `src/kernels/gemm.cu`.
- Produces: `typedef enum { GEMM_NAIVE, GEMM_TILED, GEMM_REGTILE, GEMM_CUBLAS } gemm_impl;` (REGTILE inserted before CUBLAS so `tests/test_mlp.cu`'s loop `impl <= GEMM_CUBLAS` covers it automatically); `gemm_parse("regtile") == GEMM_REGTILE`; `bin/train --gemm regtile`.

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull && git checkout -b alex/regtile-gemm
```

- [ ] **Step 2: Extend the test to four implementations (the failing test)**

In `tests/test_gemm.cu`, replace

```cuda
  const char* names[] = {"naive", "tiled", "cublas"};
  int fails = 0;
  for (int impl = 0; impl < 3; impl++)
```

with

```cuda
  const char* names[] = {"naive", "tiled", "regtile", "cublas"};
  int fails = 0;
  for (int impl = 0; impl < 4; impl++)
```

and add a shape that spans several 128×128 tiles with ragged edges: replace

```cuda
  const int shapes[][3] = {{1, 1, 1}, {33, 17, 65}, {128, 256, 64}, {257, 129, 300}};
```

with

```cuda
  const int shapes[][3] = {{1, 1, 1}, {33, 17, 65}, {128, 256, 64}, {257, 129, 300}, {300, 400, 129}};
```

and change both loops over shapes from `s < 4` to `s < 5`.

Then, as the first statements of `main`, add a check that fails until the new implementation is selectable by name (without it, an unhandled implementation would silently fall through to the tiled kernel and pass):

```cuda
  if (gemm_parse("regtile") != GEMM_REGTILE) { printf("FAIL gemm_parse(\"regtile\")\n"); return 1; }
```

- [ ] **Step 3: Update the header**

In `src/kernels/gemm.h`, replace

```c
typedef enum { GEMM_NAIVE, GEMM_TILED, GEMM_CUBLAS } gemm_impl;
```

with

```c
typedef enum { GEMM_NAIVE, GEMM_TILED, GEMM_REGTILE, GEMM_CUBLAS } gemm_impl;
```

- [ ] **Step 4: Run the test to verify it fails (CARC login node)**

```bash
source scripts/env.sh && make clean && make bin/test_gemm
sbatch --account=anakano_429 --ntasks=1 --gres=gpu:p100:1 scripts/slurm/run.sh ./bin/test_gemm
cat logs/mddp-<jobid>.out
```
Expected: `FAIL gemm_parse("regtile")`, because `gemm_parse` still only knows three names.

- [ ] **Step 5: Implement the kernel and dispatch in `src/kernels/gemm.cu`**

Add after `gemm_tiled_k`:

```cuda
/* Register tiling: a 256-thread block computes a 128x128 tile of C and each thread keeps an 8x8
   sub-tile in registers. op(A) and op(B) tiles are staged k-major in shared memory, so the inner
   loop reads each thread's 8 A values and 8 B values as two float4 loads apiece. */
#define RT_BM 128
#define RT_BN 128
#define RT_BK 8
#define RT_T 8

__global__ void __launch_bounds__(256) gemm_regtile_k(int ta, int tb, int M, int N, int K, float alpha,
                                                      const float* A, const float* B, float beta, float* C) {
  __shared__ __align__(16) float As[RT_BK][RT_BM];
  __shared__ __align__(16) float Bs[RT_BK][RT_BN];
  int tid = threadIdx.x, tr = tid / 16, tc = tid % 16;
  int row0 = blockIdx.y * RT_BM, col0 = blockIdx.x * RT_BN;
  float acc[RT_T][RT_T] = {};
  for (int k0 = 0; k0 < K; k0 += RT_BK) {
    /* index order follows each operand's memory layout so neighboring threads read neighboring addresses */
    /* ponytail: k-major stores bank-conflict when ta == 0 / tb == 1; pad or vectorize if Nsight shows it */
    for (int e = tid; e < RT_BK * RT_BM; e += 256) {
      int i = ta ? e % RT_BM : e / RT_BK, k = ta ? e / RT_BM : e % RT_BK;
      int gi = row0 + i, gk = k0 + k;
      As[k][i] = gi < M && gk < K ? ldA(A, ta, M, K, gi, gk) : 0.f;
    }
    for (int e = tid; e < RT_BK * RT_BN; e += 256) {
      int j = tb ? e / RT_BK : e % RT_BN, k = tb ? e % RT_BK : e / RT_BN;
      int gj = col0 + j, gk = k0 + k;
      Bs[k][j] = gj < N && gk < K ? ldB(B, tb, N, K, gk, gj) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < RT_BK; k++) {
      float4 a0 = *(const float4*)&As[k][tr * RT_T], a1 = *(const float4*)&As[k][tr * RT_T + 4];
      float4 b0 = *(const float4*)&Bs[k][tc * RT_T], b1 = *(const float4*)&Bs[k][tc * RT_T + 4];
      float a[RT_T] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      float b[RT_T] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < RT_T; i++)
#pragma unroll
        for (int j = 0; j < RT_T; j++) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
  for (int i = 0; i < RT_T; i++) {
    int gi = row0 + tr * RT_T + i;
    if (gi >= M) break;
    for (int j = 0; j < RT_T; j++) {
      int gj = col0 + tc * RT_T + j;
      if (gj >= N) break;
      size_t c = (size_t)gi * N + gj;
      C[c] = alpha * acc[i][j] + (beta == 0.f ? 0.f : beta * C[c]);
    }
  }
}
```

Replace `gemm_parse`:

```cuda
int gemm_parse(const char* s) {
  const char* names[] = {"naive", "tiled", "regtile", "cublas"};
  for (int i = 0; i < 4; i++) if (!strcmp(s, names[i])) return i;
  return -1;
}
```

In `gemm()`, insert before the line `dim3 blk(TILE, TILE), grd(...)`:

```cuda
  if (impl == GEMM_REGTILE) {
    dim3 grd((N + RT_BN - 1) / RT_BN, (M + RT_BM - 1) / RT_BM);
    gemm_regtile_k<<<grd, 256>>>(transA, transB, M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
```

Update the usage text in `src/train.c`: replace `[--gemm naive|tiled|cublas]` with `[--gemm naive|tiled|regtile|cublas]`.

- [ ] **Step 6: Run the kernel tests on a GPU**

```bash
make clean && make bin/test_gemm bin/test_mlp bin/train
sbatch --account=anakano_429 --ntasks=1 --gres=gpu:p100:1 scripts/slurm/run.sh \
  bash -c './bin/test_gemm && ./bin/test_mlp && ./bin/train --synthetic --steps 50 --batch 64 --hidden 256 --layers 2 --lr 0.05 --gemm regtile | tail -1'
cat logs/mddp-<jobid>.out
```
Expected: `test_gemm: PASS` (80 cases: 4 implementations × 5 shapes × 4 transpose combinations), `test_mlp: PASS` (its loop now also compares regtile gradients), and `epoch 0 test_acc` ≥ 0.90.

- [ ] **Step 7: Commit, push, open the PR**

```bash
git add src/kernels/gemm.h src/kernels/gemm.cu tests/test_gemm.cu src/train.c
git commit -m "Add register-tiled GEMM (128x128 block, 8x8 per thread)"
git push -u origin alex/regtile-gemm
gh pr create --title "Register-tiled GEMM" --body "<what changed + the three PASS lines from the CARC log>"
```

---

### Task 2: GEMM benchmark and plot — E1 (T1)

**Files:**
- Create: `bench/bench_gemm.cu`, `scripts/plot_gemm.py`, `results/gemm/p100.csv`, `results/gemm/p100.png`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `gemm()`, `gemm_impl` with REGTILE (Task 1).
- Produces: `bin/bench_gemm` → CSV `impl,M,N,K,transA,transB,seconds,gflops`; `scripts/plot_gemm.py <csv> <png>`.

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull && git checkout -b alex/bench-gemm
```

- [ ] **Step 2: Write `bench/bench_gemm.cu`**

```cuda
#include <stdio.h>
#include "cuda_check.h"
#include "kernels/gemm.h"

/* usage: bench_gemm > out.csv   (one GPU) */
static double seconds_per_gemm(int impl, int ta, int tb, int M, int N, int K,
                               const float* A, const float* B, float* C) {
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  gemm((gemm_impl)impl, ta, tb, M, N, K, 1.f, A, B, 0.f, C);  /* warm-up */
  int iters = 2.0 * M * N * K < 2e10 ? 20 : 3;
  CUDA_CHECK(cudaEventRecord(e0));
  for (int i = 0; i < iters; i++) gemm((gemm_impl)impl, ta, tb, M, N, K, 1.f, A, B, 0.f, C);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / 1e3 / iters;
}

int main(void) {
  const char* names[] = {"naive", "tiled", "regtile", "cublas"};
  const int shapes[][5] = {
    {256, 256, 256, 0, 0}, {512, 512, 512, 0, 0}, {1024, 1024, 1024, 0, 0},
    {2048, 2048, 2048, 0, 0}, {4096, 4096, 4096, 0, 0},
    /* the training model's GEMMs at batch 512, hidden 4096 */
    {512, 4096, 3072, 0, 0},  /* forward, first layer */
    {512, 4096, 4096, 0, 0},  /* forward, hidden layer */
    {4096, 4096, 512, 1, 0},  /* dW = A^T dZ */
    {512, 4096, 4096, 0, 1},  /* dA = dZ W^T */
  };
  const size_t maxn = (size_t)4096 * 4096;
  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, maxn * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&B, maxn * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&C, maxn * sizeof(float)));
  /* ponytail: zero inputs; FP32 FMA throughput does not depend on the values */
  CUDA_CHECK(cudaMemset(A, 0, maxn * sizeof(float)));
  CUDA_CHECK(cudaMemset(B, 0, maxn * sizeof(float)));
  printf("impl,M,N,K,transA,transB,seconds,gflops\n");
  for (int impl = 0; impl < 4; impl++)
    for (int s = 0; s < 9; s++) {
      int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2], ta = shapes[s][3], tb = shapes[s][4];
      double t = seconds_per_gemm(impl, ta, tb, M, N, K, A, B, C);
      printf("%s,%d,%d,%d,%d,%d,%.9f,%.1f\n", names[impl], M, N, K, ta, tb, t, 2.0 * M * N * K / t / 1e9);
      fflush(stdout);
    }
  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
```

- [ ] **Step 3: Add the Makefile rule**

Change the `gpu:` line to

```make
gpu: cpu bin/smoke bin/test_gemm bin/test_layers bin/test_mlp bin/train bin/bench_gemm
```

and add after the `bin/test_gemm` rule:

```make
bin/bench_gemm: bench/bench_gemm.cu build/gemm.o | bin
	$(NVCC) $(NVFLAGS) $^ -o $@ -lcublas
```
(recipe line starts with a real tab)

- [ ] **Step 4: Write `scripts/plot_gemm.py`**

```python
#!/usr/bin/env python3
"""usage: plot_gemm.py results/gemm/p100.csv results/gemm/p100.png"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1])))
impls = ["naive", "tiled", "regtile", "cublas"]
square = [r for r in rows if r["M"] == r["N"] == r["K"]]
model = [r for r in rows if not r["M"] == r["N"] == r["K"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for impl in impls:
    pts = sorted((int(r["M"]), float(r["gflops"])) for r in square if r["impl"] == impl)
    ax1.plot([n for n, _ in pts], [g for _, g in pts], marker="o", label=impl)
ax1.set(xscale="log", xlabel="matrix size N (N x N x N)", ylabel="GFLOP/s", title="Square GEMM")
ax1.set_xticks([256, 512, 1024, 2048, 4096], ["256", "512", "1024", "2048", "4096"])
ax1.legend()

def label(r):
    return f'{r["M"]}x{r["N"]}x{r["K"]}' + (" Aᵀ" if r["transA"] == "1" else "") + (" Bᵀ" if r["transB"] == "1" else "")


labels = list(dict.fromkeys(label(r) for r in model))
w = 0.2
for i, impl in enumerate(impls):
    g = {label(r): float(r["gflops"]) for r in model if r["impl"] == impl}
    ax2.bar([x + (i - 1.5) * w for x in range(len(labels))], [g[k] for k in labels], w, label=impl)
ax2.set_xticks(range(len(labels)), labels, rotation=20, fontsize=8)
ax2.set(ylabel="GFLOP/s", title="Training GEMMs (batch 512, hidden 4096)")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
print("wrote", sys.argv[2])
```

- [ ] **Step 5: Run on one P100 and pull the CSV back**

```bash
# CARC
make bin/bench_gemm && mkdir -p results/gemm
sbatch --account=anakano_429 --ntasks=1 --gres=gpu:p100:1 scripts/slurm/run.sh \
  bash -c './bin/bench_gemm > results/gemm/p100.csv'
cat results/gemm/p100.csv
# laptop (from the repo root; replace <netid>)
mkdir -p results/gemm
scp <netid>@discovery.usc.edu:csci596-final-project/results/gemm/p100.csv results/gemm/
python3 -m pip install --user matplotlib
python3 scripts/plot_gemm.py results/gemm/p100.csv results/gemm/p100.png
```
Expected: 37 lines (header + 4 × 9). At 4096³, `regtile` reaches at least 2× the `tiled` GFLOP/s; cuBLAS is the ceiling (roughly 8–9 TFLOP/s on a P100). Write the regtile-vs-tiled ratio and regtile's percentage of cuBLAS into the PR description. If regtile is under 2× tiled, profile it once with `ncu --set full ./bin/bench_gemm` on a single shape and record the finding instead of tuning blindly.

- [ ] **Step 6: Commit, push, open the PR**

```bash
git add bench/bench_gemm.cu scripts/plot_gemm.py Makefile results/gemm/p100.csv results/gemm/p100.png
git commit -m "Add GEMM benchmark and GFLOP/s plot (E1, P100)"
git push -u origin alex/bench-gemm
gh pr create --title "GEMM benchmark (E1)" --body "<GFLOP/s at 4096 per impl, regtile/tiled ratio, % of cuBLAS, plot>"
```

---

### Task 3: α–β model from `bench_comm` — E3 (T2)

**Files:**
- Create: `scripts/slurm/bench_comm_sweep.sh`, `scripts/fit_alpha_beta.py`, `results/comm/*.csv`, `results/comm/alpha_beta.txt`, `results/comm/allreduce_model.png`

**Interfaces:**
- Consumes: `bin/bench_comm` (Milestone 1 Task 9), CSV `kind,algo,np,bytes,seconds`.
- Produces: `results/comm/alpha_beta.txt` with lines `intra alpha_s=<a> beta_s_per_byte=<b>` and `inter alpha_s=<a> beta_s_per_byte=<b>`; the plot. Task 6's report text uses these numbers.

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull && git checkout -b t2-alpha-beta
```

- [ ] **Step 2: Write `scripts/slurm/bench_comm_sweep.sh`**

`bench_comm` is CPU-only, so this runs on the cheap `main` partition with no GPUs.

```bash
#!/bin/bash
#SBATCH --job-name=bench_comm
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs results/comm && sbatch --account=anakano_429 scripts/slurm/bench_comm_sweep.sh
# cost: 16 CPUs + 8 nodes x 8 GB  ~= 32 SU per minute
# ponytail: CPU nodes, not GPU nodes; if training comm_s disagrees with the model, re-run inter_np2 on the gpu partition
set -euo pipefail
source scripts/env.sh
R=results/comm
srun --mpi="$SRUN_MPI" --nodes=1 --ntasks=2 --ntasks-per-node=2 ./bin/bench_comm > $R/intra_np2.csv
srun --mpi="$SRUN_MPI" --nodes=2 --ntasks=2 --ntasks-per-node=1 ./bin/bench_comm > $R/inter_np2.csv
for n in 4 8 16; do
  srun --mpi="$SRUN_MPI" --nodes=$((n / 2)) --ntasks=$n --ntasks-per-node=2 ./bin/bench_comm > $R/allreduce_np$n.csv
done
```

```bash
chmod +x scripts/slurm/bench_comm_sweep.sh
```

- [ ] **Step 3: Run it on CARC**

```bash
source scripts/env.sh && make bin/bench_comm && mkdir -p logs results/comm
sbatch --account=anakano_429 scripts/slurm/bench_comm_sweep.sh
# when squeue --me is empty:
wc -l results/comm/*.csv
```
Expected: every CSV has 1 + 14 ping-pong + 4 × 14 all-reduce = 71 lines (14 sizes: 4 B … 256 MB in ×4 steps). If `srun` rejects `--mpi=pmix` on `main`, check `srun --mpi=list` and set `SRUN_MPI`.

- [ ] **Step 4: Write `scripts/fit_alpha_beta.py`**

```python
#!/usr/bin/env python3
"""Fit latency alpha and inverse bandwidth beta from ping-pong, then compare measured all-reduce
times with the alpha-beta model. usage: fit_alpha_beta.py results/comm"""
import csv
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pathlib.Path(sys.argv[1])


def rows(name):
    return list(csv.DictReader(open(d / name)))


def fit(rs):
    pp = [(int(r["bytes"]), float(r["seconds"])) for r in rs if r["kind"] == "pingpong"]
    small = sorted(t for b, t in pp if b <= 1024)
    alpha = small[len(small) // 2]  # median one-way time of tiny messages
    big = [(b, t - alpha) for b, t in pp if b >= 1 << 20]
    beta = sum(b * t for b, t in big) / sum(b * b for b, _ in big)  # least squares through the origin
    return alpha, beta


def model(algo, p, n, a, b):
    if algo == "naive":  # root receives p-1 messages, then sends p-1
        return 2 * (p - 1) * (a + n * b)
    if algo == "tree":  # recursive doubling, p a power of two
        return math.log2(p) * (a + n * b)
    return 2 * (p - 1) * (a + n / p * b)  # ring: reduce-scatter + all-gather


params = {"intra": fit(rows("intra_np2.csv")), "inter": fit(rows("inter_np2.csv"))}
with open(d / "alpha_beta.txt", "w") as f:
    for k, (a, b) in params.items():
        f.write(f"{k} alpha_s={a:.3e} beta_s_per_byte={b:.3e}  (~{1 / b / 1e9:.2f} GB/s)\n")
print(open(d / "alpha_beta.txt").read(), end="")

# ranks span nodes for p >= 4, so the slowest (inter-node) link sets the pace
a, b = params["inter"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, p in zip(axes, [4, 8, 16]):
    rs = [r for r in rows(f"allreduce_np{p}.csv") if r["kind"] == "allreduce"]
    for algo, color in [("naive", "C0"), ("tree", "C1"), ("ring", "C2"), ("mpi", "C3")]:
        pts = sorted((int(r["bytes"]), float(r["seconds"])) for r in rs if r["algo"] == algo)
        ax.plot([x for x, _ in pts], [t for _, t in pts], color=color, marker=".", label=f"{algo} measured")
        if algo != "mpi":
            ax.plot([x for x, _ in pts], [model(algo, p, x, a, b) for x, _ in pts], color=color, ls="--",
                    label=f"{algo} model")
    ax.set(xscale="log", yscale="log", xlabel="message bytes", title=f"{p} ranks ({p // 2} nodes)")
axes[0].set_ylabel("all-reduce seconds")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(d / "allreduce_model.png", dpi=150)
print("wrote", d / "allreduce_model.png")
```

- [ ] **Step 5: Fit and plot on the laptop**

```bash
mkdir -p results/comm
scp '<netid>@discovery.usc.edu:csci596-final-project/results/comm/*.csv' results/comm/
python3 scripts/fit_alpha_beta.py results/comm
```
Expected: two lines in `alpha_beta.txt` with inter-node α of a few microseconds and bandwidth in the GB/s range (intra-node faster), and `allreduce_model.png`. Ring's model line should track its measurement within about 2× for messages ≥ 1 MB; naive is slowest at 16 ranks. The model ignores reduction arithmetic, so it under-predicts at large sizes; say so in the PR description rather than adding a γ term.

- [ ] **Step 6: Commit, push, open the PR**

```bash
git add scripts/slurm/bench_comm_sweep.sh scripts/fit_alpha_beta.py results/comm
git commit -m "Add all-reduce sweep on 2-16 ranks and alpha-beta model fit (E3)"
git push -u origin t2-alpha-beta
gh pr create --title "α–β model (E3)" --body "<alpha_beta.txt contents + plot + where model and measurement disagree>"
```
---

### Task 4: Bucketed overlap of all-reduce with backward (T2)

**Files:**
- Create: `src/comm/overlap.h`, `src/comm/overlap.c`
- Modify: `src/train.c`, `scripts/test_dp.sh`, `Makefile`

**Interfaces:**
- Consumes: `grad_hook` / `mlp_forward_backward(m, X, y, B, hook, ctx)` (`src/model/mlp.h`): the hook runs on the main thread right after layer `l`'s `dW`/`db` kernels are *enqueued* on the default stream, last layer first, with `grad` = start of that layer's `[dW | db]` region inside `m->grads`. `allreduce_sum(float*, size_t, ar_algo, MPI_Comm)` on host buffers.
- Produces:
  - `Overlap* overlap_create(float* dgrads, size_t n, size_t bucket_floats, ar_algo algo, MPI_Comm comm, int device);`
  - `void overlap_begin(Overlap* o);` (once per step, before backward)
  - `void overlap_on_grad(int layer, float* grad, size_t n, void* ctx);` (a `grad_hook`)
  - `void overlap_wait(Overlap* o);` (all buckets reduced and copied back to `dgrads`)
  - `void overlap_free(Overlap* o);`
  - `bin/train --overlap [--bucket-mb X]` (default 25), and `step_s` appended to step lines.
  - `scripts/test_dp.sh NP [extra train args...]`, with `ALGOS` env override (default `naive tree ring mpi`).

**Design:** buckets are fixed-size slices of the flat gradient buffer counted from its end, because backward finishes the last layer first. Bucket `b` covers `[max(0, n - (b+1)·S), n - b·S)`. When the hook reports that layer `l` (starting at offset `off`) is done, every not-yet-queued bucket whose start is `≥ off` is complete: record a CUDA event for it on the default stream and queue it. A background thread, created per step, takes buckets in order 0, 1, 2…: wait on the bucket's event on a non-blocking copy stream, copy GPU → pinned host, `allreduce_sum`, copy host → GPU. Every rank queues the same buckets in the same order, so the MPI calls match up. Only the comm thread calls MPI between `overlap_begin` and `overlap_wait`, so `MPI_THREAD_SERIALIZED` is enough.

Two CUDA details that silently kill overlap if missed: the copy stream must be created with `cudaStreamNonBlocking` (otherwise the legacy default stream serializes with it), and the comm thread must call `cudaSetDevice` itself (the current device is per thread and defaults to 0).

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull && git checkout -b t2-overlap
```

- [ ] **Step 2: Make `scripts/test_dp.sh` accept extra args (the failing test)**

Replace

```bash
NP=${1:-2}
ARGS="--synthetic --steps 20 --batch 64 --hidden 256 --layers 2 --lr 0.05"
```

with

```bash
NP=${1:-2}
shift || true
ARGS="--synthetic --steps 20 --batch 64 --hidden 256 --layers 2 --lr 0.05 $*"
```

and replace `for ar in naive tree ring mpi; do` with `for ar in ${ALGOS:-naive tree ring mpi}; do`.

- [ ] **Step 3: Run it to verify it fails (CARC, 2 GPUs)**

```bash
make bin/train
sbatch --account=anakano_429 scripts/slurm/run.sh scripts/test_dp.sh 2 --overlap --bucket-mb 0.1
cat logs/mddp-<jobid>.out
```
Expected: `usage: train ...` and four `FAIL (rows=0 ...)` lines, because `--overlap` is not a flag yet.

- [ ] **Step 4: Write `src/comm/overlap.h`**

```c
#pragma once
#include <stddef.h>
#include "comm/allreduce.h"

/* Bucketed gradient all-reduce on a background thread, overlapped with backward.
   Gradients are staged GPU -> pinned host -> allreduce_sum -> GPU on a separate CUDA stream. */
typedef struct Overlap Overlap;

Overlap* overlap_create(float* dgrads, size_t n, size_t bucket_floats, ar_algo algo, MPI_Comm comm, int device);
void overlap_begin(Overlap* o);                                     /* once per step, before backward */
void overlap_on_grad(int layer, float* grad, size_t n, void* ctx);  /* grad_hook; ctx is the Overlap* */
void overlap_wait(Overlap* o);                                      /* all buckets reduced and back on the GPU */
void overlap_free(Overlap* o);
```

- [ ] **Step 5: Write `src/comm/overlap.c`**

```c
#include <pthread.h>
#include <stdlib.h>
#include "comm/overlap.h"
#include "cuda_check.h"

struct Overlap {
  float *dgrads, *hgrads;
  size_t n, bucket;
  int nb, queued, device;
  ar_algo algo;
  MPI_Comm comm;
  cudaStream_t stream;
  cudaEvent_t* ready;  /* one per bucket, recorded on the default stream when the bucket is final */
  pthread_t thread;
  pthread_mutex_t mu;
  pthread_cond_t cv;
};

/* Bucket b counts from the end of the buffer: backward finishes the last layer first. */
static void bucket_range(const Overlap* o, int b, size_t* start, size_t* len) {
  size_t end = o->n - (size_t)b * o->bucket;
  *start = end > o->bucket ? end - o->bucket : 0;
  *len = end - *start;
}

static void* comm_main(void* arg) {
  Overlap* o = arg;
  CUDA_CHECK(cudaSetDevice(o->device));  /* the current device is per thread */
  for (int b = 0; b < o->nb; b++) {
    pthread_mutex_lock(&o->mu);
    while (o->queued <= b) pthread_cond_wait(&o->cv, &o->mu);
    pthread_mutex_unlock(&o->mu);
    size_t s, len;
    bucket_range(o, b, &s, &len);
    CUDA_CHECK(cudaStreamWaitEvent(o->stream, o->ready[b], 0));
    CUDA_CHECK(cudaMemcpyAsync(o->hgrads + s, o->dgrads + s, len * sizeof(float), cudaMemcpyDeviceToHost, o->stream));
    CUDA_CHECK(cudaStreamSynchronize(o->stream));
    allreduce_sum(o->hgrads + s, len, o->algo, o->comm);
    CUDA_CHECK(cudaMemcpyAsync(o->dgrads + s, o->hgrads + s, len * sizeof(float), cudaMemcpyHostToDevice, o->stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(o->stream));
  return NULL;
}

Overlap* overlap_create(float* dgrads, size_t n, size_t bucket_floats, ar_algo algo, MPI_Comm comm, int device) {
  Overlap* o = calloc(1, sizeof *o);
  o->dgrads = dgrads;
  o->n = n;
  o->bucket = bucket_floats == 0 ? 1 : bucket_floats < n ? bucket_floats : n;
  o->nb = (int)((n + o->bucket - 1) / o->bucket);
  o->algo = algo;
  o->comm = comm;
  o->device = device;
  CUDA_CHECK(cudaMallocHost((void**)&o->hgrads, n * sizeof(float)));
  /* non-blocking: the legacy default stream would otherwise serialize with these copies */
  CUDA_CHECK(cudaStreamCreateWithFlags(&o->stream, cudaStreamNonBlocking));
  o->ready = malloc(o->nb * sizeof(cudaEvent_t));
  for (int b = 0; b < o->nb; b++) CUDA_CHECK(cudaEventCreateWithFlags(&o->ready[b], cudaEventDisableTiming));
  pthread_mutex_init(&o->mu, NULL);
  pthread_cond_init(&o->cv, NULL);
  return o;
}

/* ponytail: one thread per step (~tens of microseconds); keep a persistent thread if that shows up in step_s */
void overlap_begin(Overlap* o) {
  o->queued = 0;
  pthread_create(&o->thread, NULL, comm_main, o);
}

void overlap_on_grad(int layer, float* grad, size_t n, void* ctx) {
  (void)layer;
  (void)n;
  Overlap* o = ctx;
  size_t off = (size_t)(grad - o->dgrads), s, len;
  int q = o->queued;  /* only this thread writes queued */
  while (q < o->nb) {
    bucket_range(o, q, &s, &len);
    if (s < off) break;  /* this bucket still has gradients from earlier layers */
    CUDA_CHECK(cudaEventRecord(o->ready[q], 0));
    q++;
  }
  pthread_mutex_lock(&o->mu);
  o->queued = q;
  pthread_cond_signal(&o->cv);
  pthread_mutex_unlock(&o->mu);
}

void overlap_wait(Overlap* o) { pthread_join(o->thread, NULL); }

void overlap_free(Overlap* o) {
  for (int b = 0; b < o->nb; b++) cudaEventDestroy(o->ready[b]);
  free(o->ready);
  cudaStreamDestroy(o->stream);
  cudaFreeHost(o->hgrads);
  pthread_mutex_destroy(&o->mu);
  pthread_cond_destroy(&o->cv);
  free(o);
}
```

- [ ] **Step 6: Wire it into `src/train.c`**

Add the include after `#include "comm/allreduce.h"`:

```c
#include "comm/overlap.h"
```

Extend `Opts` and its defaults. Replace

```c
  float lr;
  unsigned seed;
} Opts;
```

with

```c
  float lr;
  unsigned seed;
  int overlap;
  float bucket_mb;
} Opts;
```

and in `parse()` replace the defaults line with

```c
  Opts o = {NULL, 0, 1, 0, 256, 4096, 3, GEMM_TILED, AR_RING, 0.01f, 42, 0, 25.0f};
```

In `parse()`, after the `--synthetic` line add

```c
    if (!strcmp(a, "--overlap")) { o.overlap = 1; continue; }
```

and before the final `else usage();` add

```c
    else if (!strcmp(a, "--bucket-mb")) o.bucket_mb = (float)atof(v);
```

Change the validation line to also reject a non-positive bucket:

```c
  if ((!o.data && !o.synthetic) || o.layers < 1 || o.layers > 14 || o.batch < 1 || o.bucket_mb <= 0) usage();
```

In `usage()`, replace `"             [--allreduce naive|tree|ring|mpi]\n"` with
`"             [--allreduce naive|tree|ring|mpi] [--overlap] [--bucket-mb X]\n"`.

In `main()`, replace `MPI_Init(&argc, &argv);` and the declaration line after it with

```c
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  int rank, p, lrank, ndev;
```

and right after `Opts o = parse(argc, argv);` add

```c
  if (o.overlap && provided < MPI_THREAD_SERIALIZED) {
    if (!rank) fprintf(stderr, "--overlap needs MPI_THREAD_SERIALIZED\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
```

Replace the staging-buffer allocation

```c
  float* hgrad = NULL;  /* pinned staging buffer for the host all-reduce */
  if (p > 1) CUDA_CHECK(cudaMallocHost((void**)&hgrad, m->nparams * sizeof(float)));
```

with

```c
  Overlap* ov = NULL;
  if (p > 1 && o.overlap)
    ov = overlap_create(m->grads, m->nparams, (size_t)(o.bucket_mb * (1 << 20) / sizeof(float)),
                        (ar_algo)o.allreduce, MPI_COMM_WORLD, lrank % ndev);
  float* hgrad = NULL;  /* pinned staging buffer for the blocking host all-reduce */
  if (p > 1 && !ov) CUDA_CHECK(cudaMallocHost((void**)&hgrad, m->nparams * sizeof(float)));
```

Replace the timed section of the step, from `double t0 = MPI_Wtime();` through the `printf("step ...` line, with

```c
      double t0 = MPI_Wtime();
      if (ov) overlap_begin(ov);
      float loss = mlp_forward_backward(m, dX, dy, b, ov ? overlap_on_grad : NULL, ov);
      CUDA_CHECK(cudaStreamSynchronize(0));  /* default stream only; overlap copies run on their own stream */
      double t1 = MPI_Wtime();
      if (ov) {
        overlap_wait(ov);
      } else if (p > 1) {
        CUDA_CHECK(cudaMemcpy(hgrad, m->grads, m->nparams * sizeof(float), cudaMemcpyDeviceToHost));
        allreduce_sum(hgrad, m->nparams, (ar_algo)o.allreduce, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(m->grads, hgrad, m->nparams * sizeof(float), cudaMemcpyHostToDevice));
      }
      double t2 = MPI_Wtime();
      mlp_sgd(m, o.lr, 1.0f / p);  /* sum of per-rank means / p = global-batch mean */
      CUDA_CHECK(cudaStreamSynchronize(0));
      double t3 = MPI_Wtime();

      float gl;
      MPI_Allreduce(&loss, &gl, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      if (!rank)
        printf("step %d loss %.6f compute_s %.6f comm_s %.6f step_s %.6f\n", step, gl / p, t1 - t0, t2 - t1, t3 - t0);
```

In the cleanup block, after `if (hgrad) cudaFreeHost(hgrad);` add

```c
  if (ov) overlap_free(ov);
```

- [ ] **Step 7: Link it**

In the `Makefile`, replace the `bin/train` rule with

```make
bin/train: src/train.c src/comm/allreduce.c src/comm/overlap.c src/data/cifar.c $(KERNELS) | bin
	$(MPICC) $(CFLAGS) $(GPU_INC) $^ -o $@ $(GPU_LIBS) -lpthread
```

- [ ] **Step 8: Run the equality tests on CARC**

```bash
source scripts/env.sh && make clean && make bin/train
sbatch --account=anakano_429 scripts/slurm/run.sh bash -c \
  'scripts/test_dp.sh 2 && scripts/test_dp.sh 2 --overlap --bucket-mb 0.1 && scripts/test_dp.sh 2 --overlap --bucket-mb 1'
sbatch --account=anakano_429 --partition=gpu --nodes=2 --ntasks=4 --ntasks-per-node=2 \
  scripts/slurm/run.sh scripts/test_dp.sh 4 --overlap --bucket-mb 0.1
cat logs/mddp-<jobid>.out
```
Expected: 12 PASS lines from the first job and 4 from the second. Bucket 0.1 MB splits the 3.3 MB test model into 33 buckets that cross layer boundaries; 1 MB gives 4. Results may differ from the 1-rank run in the last printed digit (bucket boundaries change the ring's summation order), which the 1e-3 tolerance allows.

- [ ] **Step 9: Check that overlap actually overlaps (4 GPUs, training-size model)**

```bash
sbatch --account=anakano_429 --partition=gpu --nodes=2 --ntasks=4 --ntasks-per-node=2 scripts/slurm/run.sh bash -c \
  'for f in "" "--overlap"; do srun --mpi=$SRUN_MPI ./bin/train --synthetic --steps 30 --batch 512 --gemm regtile $f | awk "\$1==\"step\" && \$2>=5 {c+=\$6; m+=\$8; s+=\$10; n++} END {printf \"%s compute %.4f comm %.4f step %.4f\n\", \"$f\", c/n, m/n, s/n}"; done'
cat logs/mddp-<jobid>.out
```
Expected: with `--overlap`, mean `comm_s` (exposed) drops and `step_s` is lower than without. If `step_s` does not improve, record the numbers in the PR anyway; Task 6 and Plan 3's Nsight timeline (E6) explain why. If Task 1 is not merged yet, drop `--gemm regtile`.

- [ ] **Step 10: Commit, push, open the PR**

```bash
git add src/comm/overlap.h src/comm/overlap.c src/train.c scripts/test_dp.sh Makefile
git commit -m "Overlap bucketed gradient all-reduce with backward on a comm thread"
git push -u origin t2-overlap
gh pr create --title "Bucketed overlap of all-reduce with backward" --body "<test_dp PASS lines + Step 9 numbers>"
```
Alex reviews this one: it is the first change that calls into `mlp_forward_backward`'s hook.

---

### Task 5: NCCL baseline (T2)

**Files:**
- Modify: `scripts/env.sh`, `Makefile`, `src/train.c`

**Interfaces:**
- Consumes: Task 4's `train.c` (`Opts` with `overlap`, `bucket_mb`; step timing block).
- Produces: `bin/train --allreduce nccl` when built with NCCL (device-buffer all-reduce, no host staging); `ALGOS=nccl scripts/test_dp.sh NP` passes.

- [ ] **Step 1: Find NCCL on CARC (decides whether this task runs)**

```bash
module spider nccl
```
If it lists no module, stop here: write "NCCL not available as a module on Discovery (checked <date>)" in the README's Expected Results note in a direct doc commit, and skip the rest of this task. Otherwise run `module show nccl/<version>` (the version compatible with the loaded CUDA 12.6) and note the install prefix it prints (the directory containing `include/nccl.h` and `lib/libnccl.so`; if the library is under `lib64/`, use `lib64` in the Makefile line below).

- [ ] **Step 2: Branch and load NCCL in `scripts/env.sh`**

```bash
git checkout main && git pull && git checkout -b t2-nccl
```

Append to `scripts/env.sh` (with the version and prefix from Step 1):

```bash
module load nccl/<version>
export NCCL_HOME=<install prefix from module show>
```

- [ ] **Step 3: Make the build pick NCCL up when present**

In the `Makefile`, after the `KERNELS` line add

```make
ifneq ($(NCCL_HOME),)
NCCL_INC  = -DUSE_NCCL -I$(NCCL_HOME)/include
NCCL_LIBS = -L$(NCCL_HOME)/lib -lnccl
endif
```

and change the `bin/train` recipe to

```make
	$(MPICC) $(CFLAGS) $(GPU_INC) $(NCCL_INC) $^ -o $@ $(NCCL_LIBS) $(GPU_LIBS) -lpthread
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
source scripts/env.sh && make clean && make bin/train
sbatch --account=anakano_429 scripts/slurm/run.sh bash -c 'ALGOS=nccl scripts/test_dp.sh 2'
```
Expected: `usage: train ...` and `test_dp nccl np=2: FAIL (rows=0 ...)`.

- [ ] **Step 5: Add the NCCL path to `src/train.c`**

After the other includes add

```c
#ifdef USE_NCCL
#include <nccl.h>
#endif
```

Add a field: replace `  float bucket_mb;\n} Opts;` with

```c
  float bucket_mb;
  int nccl;
} Opts;
```

and the defaults line with

```c
  Opts o = {NULL, 0, 1, 0, 256, 4096, 3, GEMM_TILED, AR_RING, 0.01f, 42, 0, 25.0f, 0};
```

Replace the `--allreduce` parse line with

```c
    else if (!strcmp(a, "--allreduce")) {
      if (!strcmp(v, "nccl")) o.nccl = 1;
      else if ((o.allreduce = ar_parse(v)) < 0) usage();
    }
```

After the validation line add

```c
#ifndef USE_NCCL
  if (o.nccl) { fprintf(stderr, "built without NCCL: source scripts/env.sh and rebuild\n"); exit(2); }
#endif
  if (o.nccl && o.overlap) { fprintf(stderr, "--overlap stages through the host; use it without nccl\n"); exit(2); }
```

In `usage()`, replace `[--allreduce naive|tree|ring|mpi]` with `[--allreduce naive|tree|ring|mpi|nccl]`.

After the `Overlap* ov` / `hgrad` setup, add

```c
#ifdef USE_NCCL
  ncclComm_t nccl = NULL;
  if (o.nccl && p > 1) {
    ncclUniqueId id;
    if (!rank) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (ncclCommInitRank(&nccl, p, id, rank) != ncclSuccess) MPI_Abort(MPI_COMM_WORLD, 1);
  }
#endif
```

and change the staging allocation condition from `if (p > 1 && !ov)` to `if (p > 1 && !ov && !o.nccl)`.

In the step, replace

```c
      if (ov) {
        overlap_wait(ov);
      } else if (p > 1) {
```

with

```c
      if (ov) {
        overlap_wait(ov);
#ifdef USE_NCCL
      } else if (nccl) {
        ncclAllReduce(m->grads, m->grads, m->nparams, ncclFloat, ncclSum, nccl, 0);
        CUDA_CHECK(cudaStreamSynchronize(0));
#endif
      } else if (p > 1) {
```

In cleanup add

```c
#ifdef USE_NCCL
  if (nccl) ncclCommDestroy(nccl);
#endif
```

- [ ] **Step 6: Run the equality test and a timing comparison**

```bash
make clean && make bin/train
sbatch --account=anakano_429 scripts/slurm/run.sh bash -c 'ALGOS=nccl scripts/test_dp.sh 2 && scripts/test_dp.sh 2'
sbatch --account=anakano_429 --partition=gpu --nodes=2 --ntasks=4 --ntasks-per-node=2 scripts/slurm/run.sh bash -c \
  'ALGOS=nccl scripts/test_dp.sh 4 && for a in ring mpi nccl; do srun --mpi=$SRUN_MPI ./bin/train --synthetic --steps 30 --batch 512 --allreduce $a | awk -v a=$a "\$1==\"step\" && \$2>=5 {m+=\$8; n++} END {printf \"%s comm_s %.4f\n\", a, m/n}"; done'
```
Expected: `test_dp nccl np=2: PASS`, the usual four PASS lines, `test_dp nccl np=4: PASS`, then three `comm_s` lines. NCCL skips the host copies, so it should have the lowest `comm_s`; put the three numbers in the PR description.

- [ ] **Step 7: Commit, push, open the PR**

```bash
git add scripts/env.sh Makefile src/train.c
git commit -m "Add NCCL all-reduce baseline on device buffers"
git push -u origin t2-nccl
gh pr create --title "NCCL baseline" --body "<PASS lines + comm_s for ring / mpi / nccl at 4 GPUs>"
```

---

### Task 6: 1 → 8 GPU scaling sweep and plot — E4, E5 (T1, both review)

**Files:**
- Create: `scripts/slurm/sweep_scaling.sh`, `scripts/plot_scaling.py`, `results/scaling/*.log`, `results/scaling/summary.csv`, `results/scaling/scaling.png`
- Modify: `README.md` (Expected Results → first measured results)

**Interfaces:**
- Consumes: `bin/train` with `--overlap`, `--bucket-mb`, `--gemm regtile`, and the `step_s` field (Tasks 1, 4).
- Produces: `results/scaling/{strong,weak}_n{1,2,4,8}_{off,on}.log`, `summary.csv` with columns `kind,overlap,np,compute_s,comm_s,step_s,speedup,efficiency`, and `scaling.png`.

**Configuration:** the spec's model (3072 → 4096 → 4096 → 4096 → 10, the trainer's defaults `--hidden 4096 --layers 3`), synthetic data (timing does not depend on the data), ring all-reduce, our `regtile` GEMM, 30 steps per run with the first 5 dropped as warm-up, medians per run. Strong scaling: global batch 512 at every N. Weak scaling: 128 per GPU (global 128·N). Overlap off and on (25 MB buckets).

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull && git checkout -b alex/scaling-sweep
```

- [ ] **Step 2: Write `scripts/slurm/sweep_scaling.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs results/scaling && sbatch --account=anakano_429 scripts/slurm/sweep_scaling.sh
# cost: 8 P100 (32) + 16 CPUs (16) + 4 x 16 GB (16) ~= 64 SU per minute for the whole allocation
# ponytail: one allocation for all N, so small-N runs leave GPUs idle; split into per-N jobs if SU gets tight
set -euo pipefail
source scripts/env.sh
OUT=results/scaling
ALGO=${ALGO:-ring}
COMMON="--synthetic --steps 30 --hidden 4096 --layers 3 --lr 0.01 --gemm regtile --allreduce $ALGO"
for n in 1 2 4 8; do
  nodes=$(( (n + 1) / 2 )); per=$(( n < 2 ? 1 : 2 ))
  run="srun --mpi=$SRUN_MPI --nodes=$nodes --ntasks=$n --ntasks-per-node=$per ./bin/train $COMMON"
  for ov in off on; do
    flags=""; [ "$ov" = on ] && flags="--overlap --bucket-mb 25"
    $run --batch 512 $flags > $OUT/strong_n${n}_${ov}.log
    $run --batch $((128 * n)) $flags > $OUT/weak_n${n}_${ov}.log
  done
done
echo "sweep done"
```

```bash
chmod +x scripts/slurm/sweep_scaling.sh
```

- [ ] **Step 3: Dry-run on 2 GPUs first (cheap check that the script and logs are right)**

```bash
mkdir -p logs results/scaling
sbatch --account=anakano_429 --partition=debug --nodes=1 --time=00:15:00 scripts/slurm/sweep_scaling.sh
```
This fails at `n=4` (only one node), which is expected; check that `results/scaling/strong_n{1,2}_{off,on}.log` and `weak_n{1,2}_{off,on}.log` each contain 30 `step` lines with a `step_s` field, then delete them: `rm results/scaling/*.log`.

- [ ] **Step 4: Write `scripts/plot_scaling.py`**

```python
#!/usr/bin/env python3
"""usage: plot_scaling.py results/scaling  -> summary.csv and scaling.png in the same directory"""
import pathlib
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = re.compile(r"^step (\d+) loss \S+ compute_s (\S+) comm_s (\S+) step_s (\S+)")
d = pathlib.Path(sys.argv[1])
NS = [1, 2, 4, 8]


def medians(path):
    rows = [tuple(map(float, m.groups()[1:])) for m in map(STEP.match, open(path)) if m and int(m.group(1)) >= 5]
    return [statistics.median(col) for col in zip(*rows)]  # compute_s, comm_s, step_s


res = {(k, ov, n): medians(d / f"{k}_n{n}_{ov}.log") for k in ("strong", "weak") for ov in ("off", "on") for n in NS}
with open(d / "summary.csv", "w") as f:
    f.write("kind,overlap,np,compute_s,comm_s,step_s,speedup,efficiency\n")
    for (k, ov, n), (c, m, s) in res.items():
        base = res[(k, ov, 1)][2]
        speedup = base / s if k == "strong" else n * base / s  # weak: throughput ratio
        f.write(f"{k},{ov},{n},{c:.6f},{m:.6f},{s:.6f},{speedup:.3f},{speedup / n:.3f}\n")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for ov, style in (("off", "o-"), ("on", "s--")):
    ax[0].plot(NS, [res[("strong", ov, 1)][2] / res[("strong", ov, n)][2] for n in NS], style, label=f"overlap {ov}")
    ax[1].plot(NS, [res[("weak", ov, 1)][2] / res[("weak", ov, n)][2] for n in NS], style, label=f"overlap {ov}")
ax[0].plot(NS, NS, ":", color="gray", label="ideal")
ax[0].set(xlabel="GPUs", ylabel="speedup", title="Strong scaling (global batch 512)", xticks=NS)
ax[1].axhline(1, ls=":", color="gray", label="ideal")
ax[1].set(xlabel="GPUs", ylabel="efficiency", title="Weak scaling (128 per GPU)", xticks=NS, ylim=(0, 1.1))
w = 0.38
for i, ov in enumerate(("off", "on")):
    xs = [j + (i - 0.5) * w for j in range(len(NS))]
    comp = [res[("strong", ov, n)][0] * 1e3 for n in NS]
    comm = [res[("strong", ov, n)][1] * 1e3 for n in NS]
    ax[2].bar(xs, comp, w, color="C0", alpha=1 if ov == "off" else 0.6, label=f"compute (overlap {ov})")
    ax[2].bar(xs, comm, w, bottom=comp, color="C3", alpha=1 if ov == "off" else 0.6, label=f"exposed comm (overlap {ov})")
ax[2].set_xticks(range(len(NS)), [str(n) for n in NS])
ax[2].set(xlabel="GPUs", ylabel="ms per step", title="Strong scaling: where the time goes")
for a in ax:
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(d / "scaling.png", dpi=150)
print(open(d / "summary.csv").read(), end="")
```

- [ ] **Step 5: Run the full sweep on 4 nodes × 2 P100**

```bash
sbatch --account=anakano_429 scripts/slurm/sweep_scaling.sh
# when squeue --me is empty:
tail -1 logs/scaling-<jobid>.out          # "sweep done"
grep -c '^step' results/scaling/*.log    # 16 files, 30 each
```
Expected: about 15–25 minutes of runtime, roughly 1,000–1,600 SU. If the job sits in the queue for hours, swap `p100` for `v100` in the script's `--gres` line and note the GPU type in the PR.

- [ ] **Step 6: Plot on the laptop**

```bash
mkdir -p results/scaling
scp '<netid>@discovery.usc.edu:csci596-final-project/results/scaling/*.log' results/scaling/
python3 scripts/plot_scaling.py results/scaling
```
Expected: `summary.csv` (16 rows) and `scaling.png` with three panels. What to look for, and to write in the PR description:
- Strong scaling speedup flattens as N grows and the red (exposed communication) share grows; 1 → 2 GPUs (same node) should scale better than 2 → 4 (first InfiniBand hop).
- Overlap on has a smaller red bar than overlap off at N ≥ 2. If it doesn't, say so: the D2H/H2D copies and the host reduction may be competing with backward for PCIe and CPU time, and Plan 3's Nsight timeline (E6) will show it.
- Weak scaling efficiency stays closer to 1 than strong scaling at the same N.

- [ ] **Step 7: Put the first numbers in the README (doc change, same PR)**

In `README.md`, under `## Expected Results`, add after the table:

```markdown
**First measurements (Milestone 2, P100):**
- GEMM: ![GEMM GFLOP/s](results/gemm/p100.png)
- All-reduce vs. α–β model: ![all-reduce](results/comm/allreduce_model.png)
- Scaling 1 → 8 GPUs: ![scaling](results/scaling/scaling.png)
```

Mark the Nov 17 milestone done in the Milestones list when this PR merges.

- [ ] **Step 8: Commit, push, open the PR**

```bash
git add scripts/slurm/sweep_scaling.sh scripts/plot_scaling.py results/scaling README.md
git commit -m "Add 1-8 GPU strong and weak scaling sweep and plot (E4, E5)"
git push -u origin alex/scaling-sweep
gh pr create --title "Scaling 1 → 8 GPUs (E4, E5)" --body "<summary.csv highlights + scaling.png + the three observations above>"
```
Shaozhe reviews this one: the comm numbers are his to explain.

---

## Milestone 2 done when

- `test_gemm` (80 cases) and `test_mlp` print PASS with `regtile` included; `regtile` is at least 2× `tiled` at 4096³ on a P100, or the PR records an Nsight finding explaining why not.
- `results/comm/alpha_beta.txt` and `allreduce_model.png` are committed.
- `scripts/test_dp.sh 2 --overlap --bucket-mb 0.1` and `scripts/test_dp.sh 4 --overlap --bucket-mb 0.1` print four PASS lines each, and the plain `scripts/test_dp.sh 2` still does.
- NCCL either passes `ALGOS=nccl scripts/test_dp.sh 4` or is documented as unavailable.
- `results/scaling/summary.csv` and `scaling.png` are committed and linked from the README.

## Plans that follow

- **Plan 3: advanced parallelism and reporting.** WMMA tensor-core GEMM and FP16 mixed precision (E1 on V100/A100, E2 roofline, E9); GPipe pipeline and 2D parallelism (E7, E8); bucket-size sweep with Nsight Systems timelines (E6); CUDA-aware MPI if `ompi_info --parsable --all | grep mpi_built_with_cuda_support:value` reports true; 16-GPU runs; README report and the one-minute presentation.
