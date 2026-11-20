# Mini-DDP Milestone 4a (Alex): Roofline (E2) and a Faster Tensor-Core GEMM

> **For agentic workers:** REQUIRED SUB-SKILLS: superpowers:subagent-driven-development to run the tasks, and superpowers:test-driven-development inside each task. This plan says *what* to build and *how to know it works*; the code is written test-first during execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put every GEMM variant on a V100 roofline measured with Nsight Compute (E2). Use it to drive two WMMA optimizations (16-byte loads, then register double buffering), each added to the plot as its own point. Finish with the final V100 numbers and the kernel half of the README report.

**Architecture:** All on `main` after Milestone 3a.
- `bench/bench_gemm.cu` gains real input data and a single-case mode for profiling.
- A new `scripts/slurm/roofline.sh` runs `ncu` per case and appends labeled rows to `results/roofline/v100.csv`.
- `scripts/plot_roofline.py` draws the roofs and points.
- The WMMA kernel in `src/kernels/gemm.cu` changes only inside its tile-staging path and K loop.
- Its interface (`gemm_h`) and its callers do not change.

**Tech Stack:** CUDA C++ (`nvcuda::wmma`, `cuda_fp16.h`), cuBLAS, Nsight Compute CLI (`ncu`, ships with the CUDA 12.6 module), Slurm on CARC Discovery, Python 3 + matplotlib (laptop only).

**Spec:** `docs/superpowers/specs/2026-11-20-mini-ddp-milestone4-design.md` (§2). Parent: `docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md` (§5, §8 E1/E2).
**Sibling plan:** `docs/superpowers/plans/2026-11-20-mini-ddp-milestone4b-experiments.md` (Shaozhe). No code dependency in either direction. Only the README's last task is shared.

## Global Constraints

- **Earlier constraints still hold:**
  - One MPI rank per GPU.
  - Row-major matrices; `W` is `in x out`.
  - `.cu` files contain no MPI; `.c` files contain no kernels.
  - `extern "C"` guards on headers shared with C.
  - Tests are plain executables printing `<name>: PASS` or `FAIL ...`, with a non-zero exit on failure.
  - The `bin/train` step line is unchanged.
- **`gemm_h` contract unchanged:** same signature, transpose semantics, ragged shapes, unaligned pointers and `beta == 0` behavior. Every existing `test_gemm` case keeps passing unmodified (120 cases on V100, 80 plus a skip line on P100).
- **Hardware:**
  - WMMA code is compiled only for `__CUDA_ARCH__ >= 700`.
  - The build still targets sm_60.
  - The WMMA kernel must show 0 spill bytes in `-Xptxas -v`. Each WMMA task's PR states its register count.
- **Roofline coordinates** (spec §2.1):
  - FLOPs = 2·M·N·K.
  - Time comes from `bench_gemm`'s normal timing, not ncu.
  - Bytes = ncu `dram__bytes_read.sum + dram__bytes_write.sum`.
- **Roofs** (spec §2.2):
  - V100-SXM2: 15.7 TFLOP/s FP32, 125 TFLOP/s tensor.
  - V100-PCIE: 14 FP32, 112 tensor.
  - 900 GB/s HBM2 for both.
  - The variant is taken from `nvidia-smi --query-gpu=name`.
- **`bench_gemm` CSV format** (`impl,M,N,K,ta,tb,seconds,gflops`) and its 55-line V100 / 37-line P100 output are unchanged. `plot_gemm.py` must keep working.
- **Measurements:**
  - Committed under `results/<experiment>/` (CSV/logs + PNG).
  - Plot scripts run on a laptop, never in a Slurm job.
  - `.ncu-rep` files are not committed.
- **CARC:**
  - Account `anakano_429`. V100s are in partition `gpu` (`--gres=gpu:v100:1`).
  - V100 costs 4 SU per GPU-minute plus 1 SU per CPU and per 4 GB.
  - Run `make clean && make gpu` after header changes.
  - Commands are given to the user one line at a time; multi-line work goes in a script.
  - Jobs that build in the same checkout must not overlap: submit the next only after the previous one shows `R` and has finished building.
- **Workflow:**
  - One branch per task (`alex/...`) → PR with what changed plus the CARC output → Shaozhe looks it over → the user merges.
  - Doc-only changes merge directly.
  - No attribution lines in commits or PR descriptions.
- **Dates:** Tasks 1–2 by Nov 26, Tasks 3–4 by Dec 1, Task 5 by Dec 5, Task 6 by Dec 10. Final deadline Dec 11.

## Order

| Task | GPU | Depends on | Unblocks |
|---|---|---|---|
| 1. Benchmark fixes + single-case mode | V100 (P100 for the sm_60 check) | — | 2 |
| 2. Roofline baseline (E2) | V100 | 1 | 3, 4 |
| 3. WMMA 16-byte loads | V100 | 2 | 4 |
| 4. WMMA register double buffering | V100 | 3 | 5 |
| 5. Final V100 numbers + README results | V100 | 4 | 6 |
| 6. README report, kernel half | — | 5 | — |

## Files

```
bench/bench_gemm.cu             deterministic input fill; single-case mode (Task 1)
src/kernels/gemm.cu             CUBLAS_CHECK (Task 1); WMMA staging + K loop (Tasks 3, 4)
tests/test_gemm.cu              fast-path and forced-slow-path FP16 cases (Task 3)
scripts/slurm/roofline.sh       ncu counter check + per-case profiling, labeled rows (Task 2; reused in 3-5)
scripts/plot_roofline.py        roofline plot (Task 2)
scripts/slurm/bench_fp16.sh     unchanged; rerun in Task 5
results/roofline/v100.{csv,png} (Tasks 2-5)
results/gemm/v100.{csv,png}, results/fp16/   refreshed (Task 5)
README.md                       (Tasks 5, 6)
```

---

### Task 1: Benchmark fixes and single-case mode (Alex)

**Interfaces:**
- `bench_gemm` with no arguments behaves as now, including the CSV layout and row order.
- `bench_gemm <impl> M N K ta tb` (impl is one of the 6 CSV names):
  - Runs exactly one warm-up call, then one call between `cudaProfilerStart()` and `cudaProfilerStop()`.
  - Then times the case as the sweep does and prints one CSV row with no header.
  - Exits 2 with a usage line on a bad impl name, or on an FP16 impl below sm_70.
- `CUBLAS_CHECK(call)` in `gemm.cu`:
  - On a non-success status, prints file, line and status code, then exits.
  - Wraps both `cublasSgemm` and `cublasGemmEx`.

**Design:**
- Fill A and B once with a cheap on-GPU deterministic hash of the element index, mapped to [-1, 1). The FP16 copies are converted from these as now.
- The fill runs before any timed loop.
- Leave a short comment saying why the inputs are not zeros: zero operands lower switching power and inflate sustained clocks.

**Tests:**
- No new unit test; the benchmark is the check.
- Acceptance on 1 V100:
  - `bench_gemm` prints 55 lines.
  - `bench_gemm wmma 4096 4096 4096 0 0` prints one row.
  - `bench_gemm bogus 1 1 1 0 0` exits 2.
  - `test_gemm` passes.
- On 1 P100: `bench_gemm` prints 37 lines.
- Note in the PR how much the 4096³ rows changed against the committed zero-input `v100.csv`.

- [ ] Branch `alex/bench-hygiene`; implement; add the checks to a small job script or reuse `bench_fp16.sh` with `wc -l`; run on V100 and P100.
- [ ] PR "bench_gemm: real inputs, single-case mode; check cuBLAS status".

### Task 2: Roofline baseline (E2) (Alex)

**What to build:**
- `scripts/slurm/roofline.sh [LABEL] [IMPLS...]`, a 1-V100 job:
  - LABEL defaults to the impl name. IMPLS defaults to all 6.
  - Step 0, counter check: print the GPU name. Run `ncu --metrics gpu__time_duration.sum` on one tiny single-case run. If the output contains `ERR_NVGPUCTRPERM`, print `roofline: counters blocked (ERR_NVGPUCTRPERM)`, write the analytical rows (below) and exit 0.
  - For each impl and each shape (4096×4096×4096 and the training forward 512×4096×4096):
    - Run `ncu --profile-from-start off --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum --csv` on the single-case mode.
    - Sum the metrics over every kernel profiled: cuBLAS may launch more than one.
    - Run the single case again without ncu for the time.
    - Append `label,impl,gpu,M,N,K,flops,dram_bytes,ncu_time_s,bench_time_s,source` to `results/roofline/v100.csv`, with `source` = `ncu`.
    - Write the header only if the file is new, so later tasks append.
  - Analytical fallback row: `dram_bytes` = min bytes (A and B read once at their element size, C written once as FP32) and `source` = `analytic`.
- `scripts/plot_roofline.py <csv> <png>`:
  - Log-log axes. x = flops/dram_bytes, y = flops/bench_time_s.
  - Roofs: the memory slope, the FP32 roof and the tensor roof, chosen from the `gpu` column per the Global Constraints.
  - One marker per impl and one color per shape. The latest row wins for a repeated `(label, M, N, K)`.
  - Labels are annotated next to points.
  - The title says "analytical" if any plotted row has `source` = `analytic`.

**Acceptance:**
- The job either prints the counter-blocked line or produces 12 `ncu` rows (6 impls × 2 shapes).
- The plot is made on the laptop and committed with the CSV.
- The PR answers, with the numbers, where each kernel sits relative to the roofs:
  - Which are memory-bound and which are compute-bound.
  - How far WMMA is below the tensor roof compared to cuBLAS FP16.
  - Whether WMMA's DRAM bytes at 4096³ are higher than cuBLAS's, which would indicate the 64×64 tile's lower reuse.
- Include `ncu --set full` output for WMMA at 4096³ (as a file under `results/roofline/`, text only) if counters work. Its "Memory Workload" and "Warp State" sections are what Tasks 3 and 4 act on.

- [ ] Branch `alex/roofline`; write `roofline.sh` and `plot_roofline.py`; run on 1 V100; plot on the laptop.
- [ ] PR "Roofline from Nsight Compute (E2)" with the plot and the findings.

### Task 3: WMMA 16-byte loads (Alex)

**Interfaces:** none change.

**Design:**
- In the tile-staging helper, take the 16-byte path (one `uint4` load and one `uint4` shared-memory store per 8 halves) only when all of the following hold:
  - The tile lies fully inside the matrix.
  - The source pointer is 16-byte aligned.
  - The leading dimension is a multiple of 8.
- Otherwise keep the current scalar bounds-checked path.
- Decide per tile with a uniform branch, not per thread.
- Shared-memory rows already have 16-byte-aligned strides (40 and 72 halves). Keep that invariant and note it next to `WPAD`.
- Replace the existing `ponytail:` note on scalar loads with one that names what is left (double buffering, then larger tiles).

**Tests (write first, in `test_gemm`):**
- FP16 case where every tile takes the fast path: 128×128×128, all 4 transpose combos.
- FP16 case that forces the slow path on an otherwise aligned shape: A passed at an offset of 1 half from an aligned allocation, so the pointer is misaligned.
- Both compare against the FP16-rounded CPU reference with the existing tolerance.
- These cases pass before the change too, because they test the contract and not the path. Their job is to keep both paths covered afterwards.

**Acceptance:**
- `test_gemm` passes on V100 and P100.
- 0 spill bytes.
- `roofline.sh wmma_vec wmma` appends 2 rows.
- `bench_gemm wmma 4096 4096 4096 0 0` is faster than the Task 2 `wmma` row. The PR states the factor, and the DRAM bytes and achieved FLOP/s against the Task 2 point.

- [ ] Branch `alex/wmma-vec`; write the tests; implement; run on V100 (tests, spills, roofline rows) and P100 (`test_gemm`).
- [ ] PR "WMMA: 16-byte tile loads" with the before/after numbers and the updated plot.

### Task 4: WMMA register double buffering (Alex)

**Interfaces:** none change.

**Design:**
- In the K loop, load tile `k+1` from global memory into registers, using the 16-byte path where Task 3 allows it, before running the MMAs on tile `k` from shared memory.
- After the MMAs and a barrier, store the registers into shared memory.
- Shared memory stays single-buffered; V100 has no `cp.async`.
- The first tile is loaded before the loop. The edge-tile scalar path may stay synchronous if that keeps the code simple; say so in a `ponytail:` note.
- Watch registers: the kernel is at 160–168 with `__launch_bounds__(128)`. Prefetching 32 halves per thread per operand adds about 16 registers. 0 spills is required.

**Tests:** the Task 3 cases and all existing FP16 cases. No new cases unless the K-loop change introduces a new edge (K < 32 and K not a multiple of 32 are already covered by `33x17x65` and `1x1x1`).

**Acceptance:**
- `test_gemm` passes on V100 and P100.
- 0 spill bytes.
- `roofline.sh wmma_db wmma` appends 2 rows.
- At 4096³, `wmma_db` ≥ 1.5× the Task 2 `wmma` row (≥ 33 TFLOP/s if the baseline stays at about 22). Training shapes are not slower than the Task 2 rows.
- If 1.5× is missed, the PR records the ncu `--set full` finding for `wmma_db` (stall reasons, achieved occupancy, shared-memory bank conflicts) and what the next step would be.

- [ ] Branch `alex/wmma-db`; implement; run on V100 and P100.
- [ ] PR "WMMA: register double buffering" with the numbers and the updated plot.

### Task 5: Final V100 numbers and README results (Alex)

**What to run:**
- `bench_fp16.sh` again (1 V100) with the final kernel. This refreshes `results/gemm/v100.csv`, `results/fp16/*.log` and `summary.txt`.
- Re-plot `results/gemm/v100.png` and `results/roofline/v100.png` on the laptop.

**README (results section, "FP16 on a V100" and a new "Roofline" entry):**
- Link the roofline plot.
- Update the 4096³ numbers and the FP32 vs FP16 step times.
- Give the before/after chain as it happened: zero-input baseline → real inputs → 16-byte loads → double buffering.
- Say where the remaining gap to cuBLAS comes from, per the ncu data. Larger tiles are future work.

**Acceptance:** CSV is 55 lines; the plots are regenerated and linked; the numbers in the README match the committed CSVs.

- [ ] Branch `alex/m4-results`; run; plot; update the README.
- [ ] PR "Final V100 GEMM and FP16 numbers; roofline in README".

### Task 6: README report, kernel half (Alex; Shaozhe writes the communication half in 4b)

**What to write** (doc-only, so it goes directly to `main`; tell Shaozhe before pushing so the two halves don't conflict):
- **Report structure:** problem → approach → findings → so what → who did what. Put it at the top of the README, above the existing build and run sections.
- **Kernel findings:**
  - The GEMM ladder: naive → tiled → regtile on P100, then WMMA on V100, as a fraction of cuBLAS at each step.
  - What the roofline shows.
  - The FP16 training speedup, and why the full step speeds up less than the GEMM.
  - FP16 accuracy parity.
  - The α–β estimate for an FP16 all-reduce.
- **Who did what:** Alex's half. Shaozhe fills his own.
- **Housekeeping:**
  - Update the status line.
  - Tick the Nov 30 / Dec 2 and Dec 11 milestones once they happen.

**Acceptance:** every plot in `results/*/*.png` that belongs to T1 is linked; Shaozhe has read the kernel half.

- [ ] Write the sections; push to `main` after Shaozhe's read-through.

---

## Milestone 4a done when

- `results/roofline/v100.png` is committed and linked from the README. It is either ncu-measured, or analytical and labeled as such.
- WMMA is ≥ 1.5× its Milestone 3a speed at 4096³, or an ncu finding in the PR explains why not.
- `test_gemm` passes on V100 and P100 with 0 spills.
- The README's kernel half of the report is written and reviewed by Shaozhe.

## Not in this plan

- 128×128 WMMA tiles or 8-warp blocks (the next step after this milestone).
- `cp.async` or other sm_80-only paths.
- Dynamic loss scaling.
- FP16 all-reduce.
- `--fp16 --pp`.
- Profiling the FP32 kernels on P100. The roofline is one GPU type: the V100 runs all six variants.
