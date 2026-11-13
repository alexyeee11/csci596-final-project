# Mini-DDP Milestone 3b (Shaozhe): GPipe Pipeline and 2D Parallelism

> **For agentic workers:** REQUIRED SUB-SKILLS: superpowers:subagent-driven-development to run the tasks, and superpowers:test-driven-development inside each task. This plan says *what* to build and *how to know it works*; the code is written test-first during execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the MLP's layers across GPUs with a GPipe schedule (`--pp P --microbatches M`), measure the pipeline bubble against (P−1)/(M+P−1) (E7), then combine pipeline and data parallelism on a `dp × pp` grid and compare layouts at 8 GPUs (E8).

**Architecture:** Builds on Milestone 2 and on Alex's stage API (Milestone 3a, Task 1): `mlp_create_range`, `mlp_stage_forward`, `mlp_stage_backward`. A new `src/comm/pipeline.{h,c}` runs one GPipe step per call: all micro-batch forwards with activations sent to the next stage, then all backwards with activation gradients sent back, staged through pinned host memory like the all-reduce. `train.c` builds one stage per rank and, for 2D, splits `MPI_COMM_WORLD` into a pipeline communicator and a data communicator; the existing `allreduce_sum` runs on the data communicator.

**Tech Stack:** C (gnu11), OpenMPI 5 (`MPI_Isend`/`MPI_Irecv`, `MPI_Comm_split`), CUDA runtime, Slurm on CARC Discovery, Python 3 + matplotlib (laptop only).

**Spec:** `docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md` (§4 pipeline and 2D, §8 E7/E8)
**Sibling plan:** `docs/superpowers/plans/2026-11-12-mini-ddp-milestone3-fp16.md` (Alex). Its Task 1 produces the stage API this plan consumes.

## Global Constraints

- Milestone 1 and 2 constraints still hold: one MPI rank per GPU (`local_rank % device_count`), row-major matrices, `.cu` files contain no MPI, `.c` files contain no kernels, `extern "C"` guards on headers shared with C, tests are plain executables or scripts printing `<name>: PASS` or `FAIL ...` with a non-zero exit on failure.
- `bin/train` keeps its step line unchanged: `step <i> loss <L> compute_s <c> comm_s <m> step_s <s>`, printed by world rank 0. For pipeline runs, `compute_s` is the mean over stages of each stage's busy time (its forward/backward kernels, synchronized), and `comm_s = step_s − compute_s` (transfers plus bubble).
- Correctness: every FP32 parallel mode matches the 1-GPU loss curve over 20 steps with max relative difference ≤ 1e-3 (spec §4: "pipeline with P stages must match 1 GPU likewise").
- Flag combinations that no test covers are rejected with a clear message: `--pp` with `--overlap`, `--allreduce nccl` or `--fp16` (the `--fp16` check is added by whichever of this PR and Alex's `--fp16` PR lands second); `--pp` larger than the layer count; a world size that isn't a multiple of `--pp`; a per-replica batch not divisible by `--microbatches`.
- Multi-node jobs: one Slurm job per layout, and every `srun` spans its whole allocation. A step on a subset of a multi-node allocation fails in MPI start-up on Discovery (Milestone 2, Task 6: `pml_ucx.c:188 Failed to receive UCX worker address`, then a segfault in `MPI_Comm_split_type`). Follow `scripts/slurm/sweep_scaling_submit.sh`. `--partition=gpu` for more than one node; max 2 GPUs per node.
- Hardware: P100 for all runs in this plan (`--gres=gpu:p100:N`); plots use one GPU type each.
- Ownership: Shaozhe owns `src/comm/` and the `--pp` wiring in `src/train.c`. `src/model/` is Alex's: if the stage API is missing something, ask for it in Alex's PR rather than editing `mlp.cu` directly.
- Measurements are committed under `results/<experiment>/` (logs/CSV + PNG). Plot scripts run on a laptop, never in a Slurm job.
- CARC: account `anakano_429`; CIFAR-10 at `~/data/cifar-10-batches-bin`; P100 costs 4 SU per GPU-minute plus 1 SU per CPU and per 4 GB; `make clean && make gpu` after header changes.
- Workflow: one branch per task (`t2-...`) → PR with what changed + CARC output → Alex looks it over → merge. Doc-only changes may merge directly. No attribution lines in commits or PR descriptions.
- Dates: Tasks 1–2 by the Nov 24 milestone; Tasks 3–4 by Dec 1 (before the Nov 30 / Dec 2 presentation if possible). Final deadline Dec 11.

## Order

| Task | GPUs | Depends on |
|---|---|---|
| 1. GPipe pipeline `--pp`, `--microbatches` + equality test | 2–4 P100 | Alex's Task 1 (stage API) |
| 2. E7: bubble vs micro-batches | 2 and 4 P100 | 1 |
| 3. 2D parallelism (`dp × pp`) + equality test | 4 P100 | 1 |
| 4. E8: layouts at 8 GPUs | 8 P100 | 3 |

While Alex's stage API is in review, Task 1's schedule, transfers and test script can be written against the interface in this plan and compiled with the declarations from Alex's branch.

## Files

```
src/comm/pipeline.{h,c}           GPipe step, stage split, pipeline eval (Task 1)
src/train.c                       --pp, --microbatches (Task 1); dp x pp communicators (Task 3)
scripts/test_pp.sh                pipeline / 2D loss curve vs 1 rank (Tasks 1, 3)
scripts/slurm/test_pp.sh          equality job (Tasks 1, 3)
scripts/slurm/sweep_pipeline*.sh  E7 jobs (Task 2)
scripts/plot_pipeline.py          E7 plot (Task 2)
scripts/slurm/sweep_2d*.sh        E8 jobs (Task 4)
scripts/plot_2d.py                E8 plot (Task 4)
results/pipeline/, results/2d/    committed measurements
```

---

### Task 1: GPipe pipeline (Shaozhe; Alex reviews)

**Interfaces:**
- Consumes (Alex, Milestone 3a Task 1): `mlp_create_range(sizes, nsizes, first, last, max_batch, seed, impl)`, `mlp_stage_forward(m, X, b, k)` → stage output rows, `mlp_stage_backward(m, X, dOut, dX, b, k)` (overwrites grads at `k == 0`, accumulates after), fields `m->first` and `m->head`; `softmax_xent` from `kernels/layers.h`. Summing M micro-batches gives M × the batch-mean gradient, so SGD is called with `grad_scale = 1/M` (times `1/dp` in Task 3).
- `void pp_split(int nlayers, int P, int stage, int* first, int* last);` contiguous layer ranges, as equal in layer count as possible, earlier stages taking the extra layer.
- `float pp_step(MLP* stage, const float* X, const int* y, int b_total, int M, MPI_Comm pipe, double* busy_s);` one GPipe step on this replica's `b_total` rows (`X` used by the first stage, `y` by the last). Returns the mean loss on every rank of `pipe`; `*busy_s` is this rank's kernel time.
- `int pp_correct(MLP* stage, const float* X, const int* y, int b, MPI_Comm pipe);` forward-only pass for evaluation; returns the number of correct predictions on every rank.
- `bin/train --pp P --microbatches M` (default `--pp 1 --microbatches 1` = today's behaviour). For this task, the world size must equal P.

**Design:**
- Schedule (spec §4): forwards k = 0..M−1 (receive input rows k from the previous stage, `mlp_stage_forward`, send output rows k to the next stage), then backwards k = M−1..0 (receive `dOut` k from the next stage — the head computes it with `softmax_xent` instead — `mlp_stage_backward`, send `dX` k to the previous stage).
- Keep every micro-batch's received input in a stage-input buffer of `b_total` rows (the backward needs it), next to the activations `Y` that the stage API already keeps per micro-batch.
- Transfers: copy device → pinned host → `MPI_Isend`, and `MPI_Irecv` → pinned host → device, as the all-reduce does. Post sends non-blocking so a stage can start micro-batch k+1 while k's output is in flight; complete them before reusing a host buffer. Use distinct tags for forward and backward and per micro-batch.
- Every rank loads the dataset as today; only the first stage uses `X` and only the last uses labels. Loss: the head sums micro-batch losses; broadcast the mean over `pipe`.
- `evaluate()` with `--pp` goes through `pp_correct`, so `epoch <e> test_acc <a>` keeps working.
- Model and stage balance: the default model (3072→4096→4096→4096→10) has 4 layers. With P = 4 the last stage holds only 4096→10, so stages are unbalanced; that is real and belongs in Task 2's discussion, not a reason to change the split.

**Tests (write first):**
- `scripts/test_pp.sh P [extra args]`: like `test_dp.sh`, runs the 1-rank reference (`--synthetic --steps 20 --batch 64 --hidden 256 --layers 3 --lr 0.05`) and P ranks with `--pp P` for `--microbatches 1`, `2` and `4`, comparing loss field 4. Prints one `test_pp: PASS (P=…, M=…, maxrel …)` per M.
- Before implementation, `--pp` is an unknown flag, so every line FAILs: the expected RED.
- `scripts/slurm/test_pp.sh`: 2 GPUs on one node, and a separate 2-node job for P = 4.

**Acceptance:**
- `test_pp.sh 2` and `test_pp.sh 4` (2 nodes) each print 3 PASS lines; `test_dp.sh 2` still prints 4 PASS lines.
- Synthetic run with `--pp 2 --microbatches 4` ends with the same `test_acc` as the 1-rank run.

- [ ] Branch `t2-pipeline`; write `test_pp.sh` and confirm it fails; implement `pipeline.{h,c}` and the `train.c` wiring; run both equality jobs.
- [ ] PR "GPipe pipeline (--pp, --microbatches)" with the PASS lines.

### Task 2: E7 — pipeline bubble vs micro-batches (Shaozhe)

**What to measure:**
- Full model, synthetic, global batch 512, `--gemm regtile`, 30 steps, median of steps 5–29.
- P = 2 (one node) and P = 4 (two nodes, one job per layout); M ∈ {1, 2, 4, 8, 16}; plus the 1-GPU run as the baseline.
- Measured bubble fraction = 1 − `compute_s` / `step_s`, against the model (P−1)/(M+P−1).
- Output: `results/pipeline/p{2,4}_m{1,2,4,8,16}.log`, `summary.csv` (`P,M,compute_s,comm_s,step_s,bubble_measured,bubble_model`) and `bubble.png` (measured vs model, per P), from `scripts/plot_pipeline.py results/pipeline`.

**Acceptance:**
- 10 logs with 30 step lines each, plus the baseline.
- PR discusses: does the measured bubble track (P−1)/(M+P−1) as M grows; where it departs (unbalanced stages — the 4096→10 stage at P = 4 — and per-micro-batch transfer and launch overhead at large M with small micro-batches); and pipeline step time vs data parallelism on the same GPU count, from `results/scaling/summary.csv` (pipeline sends only activations — 1 MB-scale per micro-batch — instead of all-reducing ~152 MB of gradients).

- [ ] Branch `t2-pipeline-sweep`; write the per-layout jobs and submitter (one job per P); dry-run P = 2 on `debug`; full sweep.
- [ ] Copy the logs to the laptop; write and run the plot script.
- [ ] PR "Pipeline bubble (E7)" with the summary, plot and discussion.

### Task 3: 2D parallelism — dp × pp (Shaozhe; Alex reviews)

**Interfaces:**
- `bin/train --pp P` with world size `p = dp × P`; `dp = p / P`.
- Rank layout: `stage = rank / dp`, `replica = rank % dp`. With 2 GPUs per node and dp ≥ 2, the two GPUs of a node are data-parallel peers of the same stage, so the large gradient all-reduce stays inside a node where possible. Record this choice in the PR; Task 4 shows its effect.
- Communicators from `MPI_Comm_split`: `pipe` (same replica, ordered by stage) and `data` (same stage, ordered by replica).
- Each replica takes its shard of the global batch exactly like data parallelism today; `pp_step` runs on `pipe`; then `allreduce_sum` over the stage's own `grads` on `data` (host-staged, `--allreduce` algorithm), then `mlp_sgd(stage, lr, 1.0f / (M * dp))`.
- The loss printed by rank 0 is the mean over replicas (`MPI_Allreduce` on `data` or the world), so `test_pp.sh` can compare it with the 1-rank curve.

**Tests (write first):**
- Extend `test_pp.sh` to accept the world size separately from P, e.g. `test_pp.sh NP P`: with 4 ranks, P = 2 (dp = 2) for M = 1, 2, 4 against the 1-rank curve. RED before the change: a world size larger than P is rejected.
- Keep `test_pp.sh 2 2`-style pure pipeline cases and `test_dp.sh` passing (P = 1 is plain data parallelism).

**Acceptance:** On 4 GPUs (2 nodes), `test_pp.sh 4 2` prints 3 PASS lines; `test_pp.sh 4 4` and `test_dp.sh 4` still pass.

- [ ] Branch `t2-2d`; write the failing test; implement the communicator split and the per-stage all-reduce; run the 2-node equality job.
- [ ] PR "2D parallelism (dp × pp)" with the PASS lines.

### Task 4: E8 — layouts at 8 GPUs (Shaozhe; Alex reviews)

**What to measure:**
- 8 P100s (4 nodes × 2), one job per layout: (dp, pp) = (8, 1), (4, 2), (2, 4). Full model, synthetic, global batch 512, `--gemm regtile`, ring, M = 8 for pp > 1, 30 steps, median of steps 5–29.
- Output: `results/2d/dp{8,4,2}_pp{1,2,4}.log`, `summary.csv` (`dp,pp,compute_s,comm_s,step_s`) and `layouts.png` (stacked compute vs comm+bubble per layout), via `scripts/plot_2d.py`.

**Acceptance:**
- 3 logs with 30 step lines each.
- PR discusses which layout trains fastest and why, in terms of the two costs this project measured: the gradient all-reduce shrinks with fewer replicas and smaller stages (α–β model, E3) while the bubble grows with more stages (E7). README: link `results/pipeline/bubble.png` and `results/2d/layouts.png` under the first measurements.

- [ ] Branch `t2-2d-sweep`; write the per-layout jobs and submitter; run; plot on the laptop.
- [ ] Update the README; PR "Data × pipeline layouts at 8 GPUs (E8)".

---

## Milestone 3b done when

- `test_pp.sh` passes for pure pipeline (P = 2, 4) and 2D (4 ranks, P = 2), each for M = 1, 2, 4, and the existing `test_dp.sh` still passes.
- `results/pipeline/bubble.png` and `results/2d/layouts.png` are committed and linked from the README.

## Not in this plan

- 1F1B schedule (spec stretch), `--overlap` or NCCL inside 2D, `--fp16` with `--pp`, CUDA-aware MPI for activation transfers, 16-GPU runs, the joint FP16 gradient all-reduce.
