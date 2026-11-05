# Mini-DDP Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILLS: superpowers:subagent-driven-development to run the tasks, and superpowers:test-driven-development inside each task. This plan says *what* to build and *how to know it works*; the code is written test-first during execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide gradient communication behind backprop with bucketed overlap, speed up our own GEMM with register tiling, explain all-reduce cost with a fitted α–β model, and produce the first 1 → 8 GPU strong/weak scaling plot.

**Architecture:** Builds on Milestone 1 (all on `main`): `gemm()` with naive/tiled/cuBLAS, the MLP with a flat `grads` buffer and a `grad_hook` called per layer during backward (last layer first), host all-reduce `allreduce_sum()`, and `bin/train`. This milestone adds a `regtile` GEMM, an overlap module that plugs into `grad_hook` and all-reduces buckets of the flat gradient buffer on a background thread while backward continues, an optional NCCL path, and Slurm sweep + plot scripts whose outputs are committed under `results/`.

**Tech Stack:** C (gnu11), CUDA C++ (nvcc), POSIX threads, OpenMPI 5, cuBLAS (reference), NCCL (optional), Slurm on CARC Discovery, Python 3 + matplotlib for plots (laptop only).

**Spec:** `docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md`
**Previous plan:** `docs/superpowers/plans/2026-10-28-mini-ddp-milestone1.md` (its Global Constraints still apply)

## Global Constraints

- Milestone 1's constraints still hold: one MPI rank per GPU (`local_rank % device_count`), row-major matrices, `W` is `in x out`, `.cu` files contain no MPI, `.c` files contain no kernels, `extern "C"` guards on shared headers, tests are plain executables printing `<name>: PASS` or `FAIL ...` with a non-zero exit on failure.
- Dependencies on CARC: MPI, CUDA runtime, cuBLAS (reference only), pthreads, NCCL (optional, compiled in only when available). Plot scripts use Python 3 + matplotlib and run on a laptop, never inside a Slurm job.
- `bin/train` keeps every Milestone 1 output field in place and only appends `step_s`: `step <i> loss <L> compute_s <c> comm_s <m> step_s <s>`. `scripts/test_dp.sh` reads field 4.
- Timing definitions: `compute_s` = forward + backward until the default stream is idle; `comm_s` = gradient communication still outstanding after that (exposed communication); `step_s` = forward start to SGD finished.
- Every parallel configuration must match the 1-GPU loss curve: `scripts/test_dp.sh` max relative loss difference ≤ 1e-3 over 20 steps.
- Measurements for the report are committed as CSV/logs + PNG under `results/<experiment>/`. All points in one plot use one GPU type; this milestone uses P100.
- CARC: account `anakano_429`; no `/project` on Discovery (CIFAR-10 lives at `~/data/cifar-10-batches-bin`); multi-node jobs need `--partition=gpu`; max 2 GPUs per node. Cost per reserved minute: P100 4 SU/GPU, A40/A100 8 SU/GPU, plus 1 SU per CPU and per 4 GB. Submit multi-line commands as a script file, not a pasted one-liner.
- Workflow: one branch per task → PR whose description has what changed + the CARC test output → the other teammate looks it over → merge. Doc-only changes may merge directly.
- No attribution lines in commits or PR descriptions.
- Milestone target: Nov 17, 2026. Final deadline Dec 11, 2026.

## Ownership and order

| Task | Owner | GPU? | Depends on |
|---|---|---|---|
| 1. Register-tiled GEMM | Alex (T1) | yes | — |
| 2. GEMM benchmark + plot (E1) | Alex | yes | 1 |
| 3. α–β model from `bench_comm` (E3) | Shaozhe (T2) | no (CPU nodes) | — |
| 4. Bucketed overlap of all-reduce with backward | Shaozhe | yes | — |
| 5. NCCL baseline | Shaozhe | yes | 4 (both edit `train.c`) |
| 6. 1 → 8 GPU scaling sweep + plot (E4, E5) | Alex (Shaozhe reviews) | yes | 1, 4 |

Tasks 1, 3 and 4 start in parallel. Task 5 waits for Task 4's merge. Task 6 runs last.

## Files

```
src/kernels/gemm.{h,cu}            + regtile (Task 1)
tests/test_gemm.cu                 + regtile cases (Task 1)
bench/bench_gemm.cu                GEMM timing → CSV (Task 2)
scripts/plot_gemm.py               (Task 2)
scripts/slurm/bench_comm_sweep.sh  bench_comm on 2..16 CPU ranks (Task 3)
scripts/fit_alpha_beta.py          (Task 3)
src/comm/overlap.{h,c}             bucketed comm thread (Task 4)
src/train.c                        --overlap, --bucket-mb, step_s (Task 4); --allreduce nccl (Task 5)
scripts/test_dp.sh                 extra train args, ALGOS override (Task 4)
scripts/env.sh, Makefile           NCCL detection (Task 5); new targets (all tasks)
scripts/slurm/sweep_scaling.sh     (Task 6)
scripts/plot_scaling.py            (Task 6)
results/{gemm,comm,scaling}/       committed measurements
```

---

### Task 1: Register-tiled GEMM (Alex)

**Interfaces:**
- `gemm_impl` becomes `{ GEMM_NAIVE, GEMM_TILED, GEMM_REGTILE, GEMM_CUBLAS }`. REGTILE goes before CUBLAS so `test_mlp`'s existing loop `impl <= GEMM_CUBLAS` covers it with no edit.
- `gemm_parse("regtile")` returns `GEMM_REGTILE`; `bin/train --gemm regtile` works and the usage text lists it.
- Same contract as the other implementations: row-major, transpose flags, `beta == 0` means C is not read.

**Design:**
- Block tile 128×128, K-step 8, 256 threads; each thread accumulates an 8×8 sub-tile in registers (spec §5).
- Stage op(A) and op(B) tiles k-major in shared memory, so a thread reads its 8 A values and 8 B values as two `float4` loads each.
- Global loads go through the existing `ldA`/`ldB` helpers, so transposes and ragged edges work. Order the load loop by each operand's memory layout, so neighbouring threads read neighbouring addresses.
- Known trade-off, to leave as a `ponytail:` note: k-major stores cause bank conflicts for some transpose cases. Fix only if profiling shows it matters.

**Tests (write first):**
- `test_gemm` covers 4 implementations. Add a shape that spans several 128×128 tiles with ragged edges, e.g. 300×400×129.
- Add a check that `gemm_parse("regtile") == GEMM_REGTILE`. This is the check that fails before the implementation exists. Without it, an unhandled value would quietly fall through to the tiled kernel and pass.
- `test_mlp` must still pass. It now also compares regtile gradients against the other implementations.

- [ ] Branch `alex/regtile-gemm`; write the failing tests; confirm on CARC that the `gemm_parse` check fails.
- [ ] Implement until `test_gemm` (80 cases: 4 impls × 5 shapes × 4 transpose combos) and `test_mlp` print PASS on one P100.
- [ ] Sanity run: `./bin/train --synthetic --steps 50 --batch 64 --hidden 256 --layers 2 --lr 0.05 --gemm regtile` ends with `epoch 0 test_acc` ≥ 0.90.
- [ ] PR "Register-tiled GEMM" with the three results.

### Task 2: GEMM benchmark and plot — E1 (Alex)

**Interfaces:**
- `bin/bench_gemm` (one GPU) prints CSV `impl,M,N,K,transA,transB,seconds,gflops` to stdout. Add it to `make gpu`.
- `scripts/plot_gemm.py <csv> <png>`.

**What to measure:**
- All 4 implementations, timed with CUDA events after one warm-up call. GFLOP/s = 2MNK / t.
- Square sizes 256, 512, 1024, 2048, 4096.
- The training model's four GEMMs at batch 512, hidden 4096: forward 512×4096×3072 and 512×4096×4096; `dW = Aᵀ·dZ` as 4096×4096×512 with transA; `dA = dZ·Wᵀ` as 512×4096×4096 with transB.
- Plot: GFLOP/s vs size for the square sizes, and grouped bars for the training shapes.

**Acceptance:**
- `results/gemm/p100.csv` has 37 lines (header + 4 × 9).
- At 4096³, regtile is at least 2× tiled. Report regtile as a percentage of cuBLAS (cuBLAS is roughly 8–9 TFLOP/s on a P100).
- If regtile falls short of 2×, profile one shape with `ncu --set full` and record the finding instead of tuning blindly.

- [ ] Branch `alex/bench-gemm`; write the benchmark and run it on one P100 (`--gres=gpu:p100:1`) into `results/gemm/p100.csv`.
- [ ] Copy the CSV to the laptop; write and run the plot script → `results/gemm/p100.png`.
- [ ] PR "GEMM benchmark (E1)" with the 4096 numbers, the regtile/tiled ratio, the % of cuBLAS, and the plot.

### Task 3: α–β model from `bench_comm` — E3 (Shaozhe)

**Interfaces:**
- Consumes `bin/bench_comm` CSV (`kind,algo,np,bytes,seconds`; ping-pong between rank 0 and the last rank, then all-reduce for each algorithm).
- Produces `results/comm/{intra_np2,inter_np2,allreduce_np4,allreduce_np8,allreduce_np16}.csv`.
- Produces `results/comm/alpha_beta.txt`, with lines `intra alpha_s=… beta_s_per_byte=…` and `inter …`.
- Produces `results/comm/allreduce_model.png`.

**Design:**
- `bench_comm` is CPU-only, so run the sweep as one Slurm job on the `main` partition, with no GPUs and 8 nodes × 2 tasks:
  - intra-node: 2 ranks on 1 node
  - inter-node: 2 ranks on 2 nodes
  - all-reduce: 4, 8 and 16 ranks at 2 per node, matching the GPU layout
- Fit α as the median ping-pong time for messages ≤ 1 KB. Fit β as the least-squares slope of (t − α) against bytes for messages ≥ 1 MB.
- Model costs, using inter-node α and β for 4 or more ranks, since the slowest link sets the pace:
  - naive: 2(p−1)(α + nβ)
  - tree: log₂p · (α + nβ)
  - ring: 2(p−1)(α + nβ/p)
- Plot measured times (solid) against the model (dashed) for each algorithm, one panel per rank count. Include `mpi` measured only.
- Known limits, to state in the PR rather than fix:
  - The model ignores the reduction arithmetic.
  - CPU nodes may have a different network path than GPU nodes.

**Acceptance:**
- Every CSV has 71 lines: header, plus 14 message sizes (4 B … 256 MB) × (1 ping-pong + 4 all-reduce algorithms).
- The fit gives an inter-node α of a few microseconds and bandwidth in the GB/s range, with intra-node faster.
- Ring's model tracks its measurement within about 2× for messages ≥ 1 MB.

- [ ] Branch `t2-alpha-beta`; write the sweep script and run it on CARC (≈ 32 SU/min; a few minutes).
- [ ] Copy the CSVs to the laptop; write and run the fit script.
- [ ] PR "α–β model (E3)" with `alpha_beta.txt`, the plot, and where the model and measurement disagree.

### Task 4: Bucketed overlap of all-reduce with backward (Shaozhe; Alex reviews)

**Interfaces:**
```c
Overlap* overlap_create(float* dgrads, size_t n, size_t bucket_floats, ar_algo algo, MPI_Comm comm, int device);
void overlap_begin(Overlap* o);                                     /* once per step, before backward */
void overlap_on_grad(int layer, float* grad, size_t n, void* ctx);  /* a grad_hook; ctx = Overlap* */
void overlap_wait(Overlap* o);                                      /* all buckets reduced and back on the GPU */
void overlap_free(Overlap* o);
```
- `bin/train --overlap [--bucket-mb X]`: default 25; reject X ≤ 0.
- `scripts/test_dp.sh NP [extra train args...]`, with an `ALGOS` environment override (default `naive tree ring mpi`).
- When the hook fires: it runs on the main thread right after layer `l`'s `dW`/`db` kernels are *enqueued* (not finished), last layer first. `grad` points at that layer's `[dW | db]` region inside `m->grads`.

**Design:**
- **Buckets:** fixed-size slices of the flat gradient buffer, counted from its end, because backward finishes the last layer first.
- **When a bucket is ready:** when the hook reports the layer starting at offset `off`, every not-yet-queued bucket that starts at or after `off` is complete. Record a CUDA event for it on the default stream, then queue it.
- **The comm thread:** one per step. It takes buckets in order. For each one, it makes its copy stream wait on that bucket's event, copies GPU → pinned host, calls `allreduce_sum`, and copies host → GPU.
- **Why MPI is safe from a second thread:** every rank queues the same buckets in the same order, so the MPI calls match across ranks. Only the comm thread calls MPI between begin and wait. Initialize with `MPI_Init_thread(MPI_THREAD_SERIALIZED)`, and abort if the library provides less while `--overlap` is set.
- **Pitfall:** the copy stream must be created `cudaStreamNonBlocking`. Otherwise the legacy default stream serializes with it and nothing overlaps.
- **Pitfall:** the comm thread must call `cudaSetDevice` itself, because the current device is per thread and defaults to 0.
- **Timing:** synchronize on the default stream only (`cudaStreamSynchronize(0)`), not `cudaDeviceSynchronize`, which would wait for the copies too. The timing definitions are in Global Constraints. Append `step_s`.

**Tests (write first):**
- `scripts/test_dp.sh 2 --overlap --bucket-mb 0.1` must print four PASS lines. That bucket size splits the test model (~3.3 MB) into about 33 buckets that cross layer boundaries.
- Also run `--bucket-mb 1` (4 buckets) and the plain `scripts/test_dp.sh 2` (no regression).
- On 2 nodes, `scripts/test_dp.sh 4 --overlap --bucket-mb 0.1` must also print four PASS lines.
- Bucket boundaries change the ring's summation order, so the last printed digit may differ. The 1e-3 tolerance covers that.

**Acceptance (beyond the tests):**
- On 4 GPUs over 2 nodes, train the full model: synthetic data, 30 steps, batch 512, regtile if merged.
- With `--overlap`, mean exposed `comm_s` and `step_s` should be lower than without (ignore the first 5 steps).
- If `step_s` doesn't improve, record the numbers in the PR anyway. Task 6 and Plan 3's Nsight timeline will explain why.

- [ ] Branch `t2-overlap`; make `test_dp.sh` accept extra args; confirm the overlap test fails (`--overlap` is an unknown flag).
- [ ] Implement `overlap.{h,c}` and the `train.c` wiring until all the equality tests pass.
- [ ] Run the overlap on/off timing comparison.
- [ ] PR "Bucketed overlap" with the PASS lines and the timing numbers.

### Task 5: NCCL baseline (Shaozhe)

**Interfaces:**
- `bin/train --allreduce nccl`: an all-reduce on the device gradient buffer with no host staging. It's compiled in only when `NCCL_HOME` is set at build time. Without NCCL, the flag exits with a clear message.
- `--overlap` together with `nccl` is rejected, because the overlap path stages through the host.
- `ALGOS=nccl scripts/test_dp.sh NP` passes.

**Design:**
- Rank 0 creates the NCCL unique id and broadcasts it with MPI. Every rank then calls `ncclCommInitRank`.
- The all-reduce runs on the default stream, then that stream is synchronized, so `comm_s` stays comparable.

**First step decides whether this task runs:**
- Run `module spider nccl` on CARC.
- **If there's no module:** add a line to the README saying "NCCL not available on Discovery (checked <date>)" as a direct doc commit, and stop.
- **If there is one:** load the CUDA 12.6–compatible version in `scripts/env.sh`. Set `NCCL_HOME` to the prefix that `module show` prints, and use `lib64` if that's where `libnccl.so` lives.

**Acceptance:**
- `ALGOS=nccl scripts/test_dp.sh 2` and `… 4` both PASS, and the existing algorithms still pass.
- On 4 GPUs, record mean `comm_s` for `ring`, `mpi` and `nccl` on the full model. NCCL is expected to be lowest.

- [ ] Run the NCCL check; branch `t2-nccl`; write the failing test (`ALGOS=nccl` → unknown algorithm).
- [ ] Implement until it passes; run the 4-GPU comparison.
- [ ] PR "NCCL baseline" with the PASS lines and the three `comm_s` numbers.

### Task 6: 1 → 8 GPU scaling sweep and plot — E4, E5 (Alex; Shaozhe reviews)

**Interfaces:**
- `scripts/slurm/sweep_scaling.sh`: one job on 4 nodes × 2 P100s in `--partition=gpu`. It writes `results/scaling/{strong,weak}_n{1,2,4,8}_{off,on}.log`, which is 16 runs. `ALGO` is an environment override, default `ring`.
- `scripts/plot_scaling.py results/scaling` writes `summary.csv` (`kind,overlap,np,compute_s,comm_s,step_s,speedup,efficiency`) and `scaling.png`.

**Configuration:**
- **Model:** the spec's 3072 → 4096 → 4096 → 4096 → 10, which is the trainer's default `--hidden 4096 --layers 3`.
- **Data and settings:** synthetic data (timing doesn't depend on the data), ring all-reduce, regtile GEMM, 30 steps per run.
- **Measurement:** use the median of steps 5–29 for each run.
- **Strong scaling:** global batch 512 at every N.
- **Weak scaling:** 128 per GPU.
- **Overlap:** each configuration runs off, and on with 25 MB buckets.
- **Placement:** 2 ranks per node; N = 1 uses a single rank.

**Plots:**
1. Strong-scaling speedup vs N, overlap off/on, plus the ideal line.
2. Weak-scaling efficiency, t(1)/t(N).
3. Strong-scaling ms per step split into compute and exposed communication, overlap off/on side by side.

**Acceptance:**
- **Dry run:** run the sweep on one `debug` node first. It fails at N = 4 by design. Check that the N = 1 and N = 2 logs have 30 step lines with `step_s`, then delete them.
- **Full run:** 16 logs with 30 step lines each. It takes about 15–25 minutes and roughly 1,000–1,600 SU. If P100s queue for hours, switch to V100 and say so in the PR.
- **What the PR should discuss:**
  - Whether speedup flattens and the exposed-comm share grows with N.
  - Whether going from 1 to 2 GPUs (same node) scales better than 2 to 4 (the first InfiniBand hop).
  - Whether overlap shrinks exposed communication.
  - Whether weak scaling stays closer to ideal than strong scaling.
- **README (same PR):** under Expected Results, link `results/gemm/p100.png`, `results/comm/allreduce_model.png` and `results/scaling/scaling.png` as the first measurements. Check off the Nov 17 milestone when the PR merges.

- [ ] Branch `alex/scaling-sweep`; write the sweep script; dry-run on 2 GPUs.
- [ ] Full sweep; copy the logs to the laptop; write and run the plot script.
- [ ] Update the README; PR "Scaling 1 → 8 GPUs (E4, E5)" with the summary and the observations.

---

## Milestone 2 done when

- `test_gemm` (80 cases) and `test_mlp` PASS with regtile. Regtile is ≥ 2× tiled at 4096³ on a P100, or the PR records an Nsight finding explaining why not.
- `results/comm/alpha_beta.txt` and `allreduce_model.png` are committed.
- `scripts/test_dp.sh 2 --overlap --bucket-mb 0.1` and the 4-GPU, 2-node version each print four PASS lines, and plain `test_dp.sh 2` still does.
- NCCL either passes `ALGOS=nccl scripts/test_dp.sh 4` or is documented as unavailable.
- `results/scaling/summary.csv` and `scaling.png` are committed and linked from the README.

## Plans that follow

- **Plan 3: advanced parallelism and reporting.**
  - WMMA tensor-core GEMM and FP16 mixed precision (E1 on V100/A100, E2 roofline, E9)
  - GPipe pipeline and 2D parallelism (E7, E8)
  - Bucket-size sweep with Nsight Systems timelines (E6)
  - CUDA-aware MPI, if `ompi_info --parsable --all | grep mpi_built_with_cuda_support:value` reports true
  - 16-GPU runs
  - README report and the one-minute presentation
