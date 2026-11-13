# Mini-DDP Milestone 3a (Alex): Pipeline Stage API, Tensor-Core GEMM, Mixed Precision

> **For agentic workers:** REQUIRED SUB-SKILLS: superpowers:subagent-driven-development to run the tasks, and superpowers:test-driven-development inside each task. This plan says *what* to build and *how to know it works*; the code is written test-first during execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the model a per-stage, per-micro-batch API that Shaozhe's GPipe pipeline builds on, add a WMMA tensor-core GEMM (FP16 in, FP32 accumulate), train with mixed precision (`--fp16`), and measure FP32 vs FP16 (E1 on V100, E9).

**Architecture:** Builds on Milestone 2 (all on `main`). `src/model/mlp.{h,cu}` gains a layer-range constructor and stage forward/backward that work on one micro-batch's rows; `mlp_forward_backward` is re-expressed through them. `src/kernels/gemm_h.{h,cu}` adds an FP16-input GEMM with a WMMA kernel and a cuBLAS reference. `--fp16` keeps FP32 master weights and gradients and runs every GEMM on FP16 copies with a static ×1024 loss scale.

**Tech Stack:** CUDA C++ (`nvcuda::wmma`, `cuda_fp16.h`), cuBLAS (`cublasGemmEx`, reference only), C (gnu11), OpenMPI 5, Slurm on CARC Discovery, Python 3 + matplotlib (laptop only).

**Spec:** `docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md` (§4 pipeline, §5 kernels and mixed precision, §8 E1/E9)
**Sibling plan:** `docs/superpowers/plans/2026-11-12-mini-ddp-milestone3-pipeline.md` (Shaozhe). Its Task 1 consumes this plan's Task 1.

## Global Constraints

- Milestone 1 and 2 constraints still hold: one MPI rank per GPU (`local_rank % device_count`), row-major matrices, `W` is `in x out`, `.cu` files contain no MPI, `.c` files contain no kernels, `extern "C"` guards on headers shared with C, tests are plain executables printing `<name>: PASS` or `FAIL ...` with a non-zero exit on failure.
- `bin/train` keeps its step line unchanged: `step <i> loss <L> compute_s <c> comm_s <m> step_s <s>`. `scripts/test_dp.sh` reads field 4.
- Correctness: every FP32 parallel mode matches the 1-GPU loss curve over 20 steps with max relative difference ≤ 1e-3. FP16 runs compare against an FP16 reference with the same tolerance; FP16 vs FP32 uses the looser checks stated in Task 3.
- Hardware: P100 has no tensor cores. WMMA and `--fp16` run on V100 (`--gres=gpu:v100:N`). The build still targets sm_60 too, so WMMA code is compiled only for `__CUDA_ARCH__ >= 700`, and the host refuses with a clear message on an older GPU. Plots use one GPU type each.
- Flag combinations that no test covers are rejected with a clear message, not silently run: `--fp16` with `--pp` (until a pipeline FP16 test exists; whichever of the two PRs lands second adds this check), `--gemm wmma` without `--fp16`, `--fp16` with `--gemm naive|tiled|regtile`.
- Ownership: Alex owns `src/model/` and `src/kernels/`. Shaozhe's plan edits `src/train.c` and `src/comm/`; this plan's `train.c` edits stay small (one flag) to keep rebases easy.
- Measurements are committed under `results/<experiment>/` (CSV/logs + PNG). Plot scripts run on a laptop, never in a Slurm job.
- CARC: account `anakano_429`; CIFAR-10 at `~/data/cifar-10-batches-bin`; V100 and P100 cost 4 SU per GPU-minute plus 1 SU per CPU and per 4 GB; run `make clean && make gpu` after header changes (no header dependency tracking); submit multi-line commands as a script file.
- Workflow: one branch per task (`alex/...`) → PR with what changed + CARC output → Shaozhe looks it over → merge. Doc-only changes merge directly. No attribution lines in commits or PR descriptions.
- Dates: Tasks 1–3 by the Nov 24 milestone; Task 1 first and fast, because Shaozhe's pipeline waits on it. Task 4 by Dec 1. Final deadline Dec 11.

## Order

| Task | GPU | Depends on | Unblocks |
|---|---|---|---|
| 1. Stage API in the MLP | any (P100) | — | Shaozhe's Task 1 |
| 2. FP16 GEMM: WMMA + cuBLAS reference | V100 | — | 3, 4 |
| 3. Mixed-precision training `--fp16` | V100 | 1, 2 | 4 |
| 4. E1 on V100 + E9 FP32 vs FP16 | V100 | 2, 3 | — |

## Files

```
src/model/mlp.{h,cu}           stage API (Task 1); FP16 mode (Task 3)
tests/test_mlp.cu              stage-API and FP16 checks (Tasks 1, 3)
src/kernels/gemm_h.{h,cu}      FP16-input GEMM: WMMA + cublasGemmEx (Task 2)
src/kernels/gemm.{h,cu}        GEMM_WMMA enum value + parse (Task 2)
src/kernels/layers.{h,cu}      FP32 -> FP16 conversion kernel, loss-scale hook (Task 3)
tests/test_gemm.cu             FP16 cases, skipped below sm_70 (Task 2)
bench/bench_gemm.cu            FP16 rows on sm_70+ (Task 4)
src/train.c                    --fp16 (Task 3)
scripts/slurm/fp16_checks.sh   one V100 job for Tasks 2-4 (Tasks 2-4)
results/gemm/v100.{csv,png}, results/fp16/   (Task 4)
```

---

### Task 1: Stage API in the MLP (Alex; unblocks Shaozhe)

**Interfaces (produced; Shaozhe's pipeline consumes these exact names):**
- `MLP* mlp_create_range(const int* sizes, int nsizes, int first, int last, int max_batch, unsigned seed, gemm_impl impl);` builds only layers `[first, last)` of the full network described by `sizes`. Weights are bit-identical to the same layers of `mlp_create(sizes, ...)`: the initializer draws random numbers for every layer in order and keeps only its range. `mlp_create` becomes `mlp_create_range(..., 0, nsizes - 1, ...)`.
- New `MLP` fields: `int first;` (global index of its first layer) and `int head;` (1 if it holds the network's last layer, which has no ReLU and produces logits).
- `float* mlp_stage_forward(MLP* m, const float* X, int b, int k);` runs micro-batch `k` (`b` rows) through the stage. It writes rows `[k*b, (k+1)*b)` of each layer's `Y`, so all micro-batches' activations stay alive for GPipe's backward. Returns the device pointer to the stage's output rows (`b x out` of its last layer).
- `void mlp_stage_backward(MLP* m, const float* X, const float* dOut, float* dX, int b, int k);` backward for micro-batch `k`. `X` is that micro-batch's stage input, `dOut` the gradient of the stage output, `dX` (may be NULL on the first stage) receives the gradient of the stage input. Gradients **overwrite** `grads` when `k == 0` and **accumulate** when `k > 0` (GEMM `beta` 0 or 1; the bias-gradient kernel needs the same option).
- Contract the caller relies on: `max_batch` is the full batch (M micro-batches × b rows); for the head, the caller computes `softmax_xent` on the returned logits and passes `dlogits` as `dOut`. Summing M micro-batches gives M × the batch-mean gradient, so the caller scales by `1/M` in `mlp_sgd`.

**Design:**
- Re-express `mlp_forward_backward` as one stage covering all layers with `k = 0, b = B`, so `test_mlp`, `test_dp.sh` and the overlap path keep covering the shared code. The stage backward has no hook parameter, so both call one internal backward that takes an optional `grad_hook`.
- `relu_backward` needs `Y` rows of the same micro-batch: offset by `k*b*out`.
- `bias_grad` gets an accumulate flag (add to `db` instead of overwrite). Extend its test in `test_layers`.

**Tests (write first, in `test_mlp`):**
- Range init: for an 8→16→12→4 network, `mlp_create_range(.., 1, 3, ..)` params equal the matching slice of `mlp_create` params exactly.
- Split equivalence: two stages `[0,1)` and `[1,3)`, batch 8 as M = 2 micro-batches of 4, driven by hand (forward stage 0, forward stage 1, softmax_xent per micro-batch, backward stage 1, backward stage 0, in GPipe order). After scaling by 1/M, both stages' gradients match the full model's `mlp_forward_backward` gradients for the whole batch (relative error ≤ 1e-4 against max |g|), and the mean micro-batch loss matches the full-batch loss.
- The new `bias_grad` accumulate case in `test_layers`. Both new tests fail to compile before the API exists; that is the expected RED.

**Acceptance:** `test_layers`, `test_mlp` (all 4 FP32 impls) PASS on one P100; `scripts/test_dp.sh 2` still prints 4 PASS lines.

- [ ] Branch `alex/stage-api`; write the failing tests; implement; run on 1 P100 (`--gres=gpu:p100:1`) plus `test_dp.sh 2`.
- [ ] PR "MLP stage API for pipeline parallelism"; tell Shaozhe as soon as it merges.

### Task 2: FP16 GEMM — WMMA + cuBLAS reference (Alex)

**Interfaces:**
- `src/kernels/gemm_h.h` (included from `.cu` files only; uses `cuda_fp16.h`): `typedef enum { GEMMH_WMMA, GEMMH_CUBLAS } gemmh_impl;` and `void gemm_h(gemmh_impl impl, int transA, int transB, int M, int N, int K, float alpha, const __half* A, const __half* B, float beta, float* C);` — same layout and transpose contract as `gemm()`, FP16 inputs, FP32 accumulate and FP32 output, `beta == 0` means C is not read.
- `int gemm_h_supported(void);` 1 when the current device is sm_70+.
- `gemm_impl` gains `GEMM_WMMA` **after** `GEMM_CUBLAS` (so `impl <= GEMM_CUBLAS` loops are unchanged); `gemm_parse("wmma")` returns it. FP32 `gemm()` called with `GEMM_WMMA` exits with "wmma needs --fp16".

**Design:**
- Map transposes to WMMA fragment layouts instead of separate kernels: `transA` (A stored K×M) is a col_major M×K view, likewise for B.
- Block tile 64×64 with 4 warps, each owning a 32×32 output (2×2 `16×16×16` fragments); stage A and B tiles through shared memory with bounds checks and zero fill, so ragged M, N, K and unaligned leading dimensions work (WMMA loads need 16-byte-aligned rows with `ldm` a multiple of 8 halves; padded shared memory guarantees it).
- Epilogue: store accumulators to shared memory, then write `alpha*acc + beta*C` with bounds checks (fragments can't be stored to ragged global edges).
- Kernel body under `#if __CUDA_ARCH__ >= 700`; the host checks `gemm_h_supported()` first.
- cuBLAS path: `cublasGemmEx` with `CUDA_R_16F` inputs, `CUDA_R_32F` output and `CUBLAS_COMPUTE_32F`, using the same row-major trick as the FP32 cuBLAS call.

**Tests (write first, in `test_gemm`):**
- On sm_70+: both FP16 impls × the 5 existing shapes × 4 transpose combos = 40 cases, against a CPU reference computed from the **FP16-rounded** inputs in double. Tolerance: |err| ≤ 1e-3 · Σ|a||b| per element (FP32 accumulate order only).
- `gemm_parse("wmma") == GEMM_WMMA`.
- On sm_60, the FP16 cases print `test_gemm: fp16 cases skipped (needs sm_70+)` and the FP32 cases still decide PASS/FAIL.

**Acceptance:** `test_gemm` PASS on one V100 (120 cases) and on one P100 (80 cases + skip line); `-Xptxas -v` for the WMMA kernel shows 0 spill bytes.

- [ ] Branch `alex/wmma-gemm`; write failing tests; implement; add `scripts/slurm/fp16_checks.sh` (one V100, debug or gpu partition — check which has V100s with `sinfo -o "%P %G"`) running the spill report and `test_gemm`; run on V100 and on P100.
- [ ] PR "Tensor-core FP16 GEMM (WMMA)" with both test outputs.

### Task 3: Mixed-precision training `--fp16` (Alex)

**Interfaces:**
- `void mlp_enable_fp16(MLP* m);` switches an existing model to FP16 GEMMs: allocates FP16 copies of the weights, of each layer's activations and of the backward `dZ` buffers, and sets `m->loss_scale = 1024`. GEMMs use `gemm_h` with `GEMMH_CUBLAS` if `m->impl == GEMM_CUBLAS`, else `GEMMH_WMMA`.
- `mlp_sgd` divides by `m->loss_scale` internally, so callers keep passing `1/p` (or `1/(M·dp)`) unchanged. Master weights and `grads` stay FP32, so all-reduce and overlap code is untouched.
- `void f32_to_f16(__half* dst, const float* src, size_t n);` in `layers`.
- `bin/train --fp16 [--gemm wmma|cublas]` (default `wmma` when `--fp16`). Combinations listed in Global Constraints are rejected.

**Design:**
- Forward: convert the input batch to FP16 once; after each `bias_act_forward`, convert `Y` to its FP16 copy for the next GEMM.
- Backward: scale `dlogits` by the loss scale; after `relu_backward`, convert `dZ` to FP16 for the next two GEMMs. GEMM outputs (`Y`, `dW`, `dA`) are FP32.
- After `mlp_sgd`, refresh the FP16 weight copy from the FP32 master weights (one conversion over the whole params buffer).
- Static scale ×1024 per the spec; dynamic scaling is a stretch goal. Leave a `ponytail:` note naming the overflow risk and the upgrade.
- Stage API (Task 1) with FP16 is not required here; `train.c` rejects `--fp16 --pp`.

**Tests (write first):**
- `test_mlp` FP16 check (sm_70+ only, skip line otherwise): on a small network, FP16 gradients vs FP32 gradients have ‖g16 − g32‖ / ‖g32‖ ≤ 1e-2 per layer, and loss differs by ≤ 1e-2 relative.
- `test_dp.sh 2 --fp16` on 2 V100s: FP16 N-rank vs FP16 1-rank, 4 PASS lines (only the FP32 summation order differs).

**Acceptance (V100):**
- Synthetic sanity: `./bin/train --synthetic --steps 50 --batch 64 --hidden 256 --layers 2 --lr 0.05 --fp16` ends with `test_acc` ≥ 0.90.
- CIFAR-10, 5 epochs, `--batch 256 --hidden 1024 --layers 2 --lr 0.02`: FP16 test accuracy within 0.02 of the FP32 run in the same job.
- `test_dp.sh 2 --fp16` prints 4 PASS lines.

- [ ] Branch `alex/fp16-training`; write failing tests; implement; extend `fp16_checks.sh` with the three acceptance runs; run on 2 V100s.
- [ ] PR "Mixed-precision training (--fp16)" with the outputs.

### Task 4: E1 on V100 and E9 FP32 vs FP16 (Alex; Shaozhe reviews)

**What to measure:**
- `bench_gemm` adds rows `wmma` and `cublas_fp16` on sm_70+ (same shapes; inputs converted once outside the timed loop). On V100 the CSV has 1 + 6 × 9 = 55 lines; on P100 it stays at 37. → `results/gemm/v100.{csv,png}` (check `plot_gemm.py` handles the extra impls).
- Training step time on 1 V100, full model, batch 512, synthetic, 30 steps, median of steps 5–29: FP32 `regtile`, FP32 `cublas`, FP16 `wmma`, FP16 `cublas`. → `results/fp16/{fp32_regtile,fp32_cublas,fp16_wmma,fp16_cublas}.log` + `summary.txt`.

**Acceptance:**
- At 4096³ on V100, `wmma` ≥ 2× `regtile` (FP32). Report `wmma` as a percentage of `cublas_fp16`. If it falls short of 2×, profile one shape with `ncu --set full` and record the finding (this doubles as a first E2 roofline data point).
- PR discusses: GEMM speedup vs end-to-end step speedup (the non-GEMM kernels and conversions don't speed up), and what FP16 would buy for the 1→8 GPU scaling result, where communication dominates — estimated with Shaozhe's α–β model at half the bytes (an FP16 all-reduce itself is not in this plan).

- [ ] Branch `alex/fp16-bench`; extend `bench_gemm` and `fp16_checks.sh`; run on 1 V100; plot on the laptop.
- [ ] PR "FP16 vs FP32 (E1 on V100, E9)" with the table and plot.

---

## Milestone 3a done when

- Shaozhe has the stage API on `main` (Task 1) and his pipeline equality test passes on it.
- `test_gemm` (120 cases on V100) and `test_mlp` PASS; WMMA ≥ 2× regtile at 4096³ on V100 or an Nsight finding explains why not.
- `--fp16` trains CIFAR-10 to within 0.02 of FP32 accuracy and passes `test_dp.sh 2 --fp16`.
- `results/gemm/v100.png` and `results/fp16/summary.txt` are committed and linked from the README.

## Not in this plan

- FP16 gradient all-reduce (halves the bytes on the wire): joint with Shaozhe; worth doing after both milestone 3 plans because communication dominates the scaling result.
- Dynamic loss scaling, `--fp16` with `--pp`, Nsight Compute roofline across all kernels (E2), CNN stretch.
