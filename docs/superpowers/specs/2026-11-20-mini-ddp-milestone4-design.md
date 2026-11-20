# Mini-DDP Milestone 4 Design: Roofline, Faster Tensor-Core GEMM, Final Experiments

**Date:** 2026-11-20. **Parent spec:** `2026-10-28-mini-ddp-cuda-mpi-design.md`. This document only covers what changes or gets decided for the last milestone; everything else in the parent spec still holds.

**Plans:**
- `docs/superpowers/plans/2026-11-20-mini-ddp-milestone4a-roofline.md` (Alex, T1)
- `docs/superpowers/plans/2026-11-20-mini-ddp-milestone4b-experiments.md` (Shaozhe, T2)

## 1. Where the project stands

Done and on `main`:
- Milestones 1 and 2.
- Milestone 3a: the stage API, the WMMA GEMM, `--fp16`, and E1 on V100 plus E9.
- Milestone 3b Tasks 1–3: GPipe, E7 bubble data, and 2D parallelism.

Still open, against the parent spec's experiment table (§8):

| # | Experiment | Status | Owner | Plan |
|---|---|---|---|---|
| E2 | Roofline from Nsight Compute | not started | Alex | 4a |
| E6 | Bucket-size sweep + Nsight Systems timeline | only overlap on/off measured | Shaozhe | 4b |
| E7 | Pipeline bubble | data and plot committed, not in README | Shaozhe | 4b |
| E8 | 2D layouts at 8 GPUs | planned in 3b Task 4, not run | Shaozhe | 3b Task 4 (4b points to it) |
| E4/E5 | Scaling to 16 GPUs | 1 → 8 done | both | 4b |
| — | Accuracy parity across configurations | no table | both | 4b |
| — | README report (problem → finding → so what, who did what) | partial | both | 4a and 4b, last tasks |

Success criterion 5 of the parent spec ("every plot above is in the README") is the bar for Dec 11.

## 2. Alex (4a): roofline and a faster WMMA kernel

**Why:** The WMMA kernel reaches 22.1 TFLOP/s at 4096³ on a V100, which is 25% of cuBLAS FP16. E2 should show *why*, and the fix should be driven by that measurement.

**Decisions:**

1. **Roofline coordinates.**
   - x = FLOPs per DRAM byte, using the DRAM bytes measured by Nsight Compute (`dram__bytes_read.sum + dram__bytes_write.sum`). FLOPs = 2·M·N·K.
   - y = FLOP/s from `bench_gemm`'s own timing, not ncu's `gpu__time_duration`. ncu locks clocks to base and flushes caches, so its times are slower than a normal run. The ncu duration is still recorded for reference.
   - ncu's FLOP counters miss tensor-core work, which is why FLOPs come from the formula.
2. **Roofs.**
   - The V100 variant (SXM2 or PCIe) is read from `nvidia-smi` in the job and decides the roofs. SXM2: FP32 15.7 TFLOP/s, tensor 125 TFLOP/s. PCIe: 14 and 112. HBM2 is 900 GB/s for both.
   - The plot script takes the variant from the CSV.
3. **Counter access.**
   - The first step of the roofline job is a single-kernel ncu run.
   - If it fails with `ERR_NVGPUCTRPERM`, E2 falls back to an analytical roofline, labeled as such in the plot title and the README.
   - The analytical roofline uses minimum bytes: each input read once, the output written once.
4. **Kernel optimizations, in this order, each measured separately as a new labeled row on the same roofline:**
   - (a) 16-byte global loads when a tile lies fully inside the matrix and the source is 16-byte aligned with a leading dimension that is a multiple of 8 halves. Every other tile keeps the scalar path.
   - (b) Register double buffering of the next K tile. V100 has no `cp.async`, so shared memory stays single-buffered.
   - Larger 128×128 tiles are out of scope and are named as future work.
5. **Target.**
   - At least 1.5× the original WMMA at 4096³ (≥ 33 TFLOP/s).
   - Training shapes must not regress.
   - If the target is missed, an ncu finding explaining why is recorded instead.
6. **Benchmark fixes.**
   - `bench_gemm` inputs get a deterministic on-GPU fill instead of zeros.
   - Both cuBLAS calls check their status.
   - `bench_gemm <impl> M N K ta tb` runs one case for profiling; with no arguments it runs the existing 55-line sweep.

## 3. Shaozhe (4b): the remaining T2 experiments

1. **E8** stays exactly as specified in 3b Task 4. 4b only schedules it first and adds the README links it requires.
2. **E6 bucket sweep:**
   - 8 P100s, strong scaling (batch 512), ring, `--overlap`.
   - Bucket sizes 1, 5, 25, 100 MB, per the parent spec's 1–100 MB range.
   - Median of steps 5–29.
   - Output: `results/overlap/bucket_sweep.csv` and `.png`.
3. **E6 timeline:**
   - One Nsight Systems capture on 2 GPUs, with overlap off and overlap on (25 MB).
   - The capture uses `nsys profile -t cuda,mpi,osrt` around a 10-step run.
   - Output: two committed PNG screenshots that show the comm thread's all-reduce next to backward kernels.
   - The `.nsys-rep` files are large and stay off git.
4. **Final scaling:**
   - Rerun E4/E5 at 1, 2, 4, 8 GPUs.
   - Add 16 GPUs only if the class allocation allows it. Check `myaccount` first; a 16-P100 job costs about 160 SU per minute.
   - Same P100 FP32 configuration as Milestone 2, so the curves are comparable.
   - The WMMA work in 4a does not affect these runs: they are FP32 on P100, so the two plans have no ordering dependency.
5. **Accuracy parity:**
   - One CIFAR-10 run per configuration, all using the same model and epochs:
     - 1 GPU
     - dp = 8
     - pp = 4
     - (dp, pp) = (4, 2)
     - `--fp16` on 1 V100
   - Final test accuracy goes to `results/accuracy/summary.csv`.
   - Pass: every configuration is within 0.01 of the 1-GPU accuracy, except FP16, which gets 0.02 as in Milestone 3a.

## 4. Shared rules

- Every parent-spec and Milestone 3 global constraint still holds.
- One branch and PR per task, with CARC output in the PR. The other person looks it over.
- Doc-only changes go directly to `main`. No attribution lines in commits or PRs.
- CARC commands are given one line at a time. Multi-line work goes in a script file.
- Plots run on a laptop, never in a Slurm job.
- **README ownership:**
  - Alex writes the structure plus the kernel, FP16 and roofline sections.
  - Shaozhe writes the communication, pipeline, 2D, overlap and scaling sections.
  - Both review the whole report before Dec 11.

## 5. Schedule

| Dates | Alex (4a) | Shaozhe (4b) | Shared |
|---|---|---|---|
| Nov 20 – 26 | bench fixes; roofline baseline | E8; README links for E7/E8 | — |
| Nov 27 – Dec 1 | 16-byte loads; double buffering | E6 sweep and timeline | 1-minute presentation (Nov 30 / Dec 2) |
| Dec 2 – 5 | final V100 numbers | final scaling; accuracy table | — |
| Dec 6 – 10 | README: kernels, FP16, roofline | README: comm, pipeline, scaling | full read-through |
| Dec 11 | — | — | submit repo URL to Brightspace |

## 6. Out of scope

- 128×128 WMMA tiles.
- Dynamic loss scaling.
- FP16 gradient all-reduce (only estimated via α–β, in E9).
- `--fp16` with `--pp`.
- 1F1B.
- CNN.
- FP16 scaling on multiple V100 nodes.
