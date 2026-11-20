# Mini-DDP Milestone 4b (Shaozhe): E8, Bucket Sweep and Timeline (E6), Final Scaling, Accuracy Parity

> **For agentic workers:** REQUIRED SUB-SKILLS: superpowers:subagent-driven-development to run the tasks, and superpowers:test-driven-development inside each task where code changes. This plan says *what* to build and *how to know it works*. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the communication-side experiments so every plot the parent spec promises is committed and linked from the README:
- E8 (2D layouts)
- E6 (bucket-size sweep plus an Nsight Systems timeline)
- E4/E5 final scaling, to 16 GPUs if the allocation allows
- An accuracy-parity table across parallel configurations

Then write the communication half of the README report.

**Architecture:** No new training code is expected; every experiment uses existing flags:
- `--allreduce`, `--overlap`, `--bucket-mb`, `--pp`, `--microbatches`, `--fp16`.
- Work is Slurm jobs modeled on `sweep_scaling.sh` and `sweep_pipeline.sh` (one layout per job, submitted by a `*_submit.sh`), plus laptop plot scripts.

**Tech Stack:** Bash + Slurm on CARC Discovery, OpenMPI 5, Nsight Systems CLI (`nsys`), Python 3 + matplotlib (laptop only).

**Spec:** `docs/superpowers/specs/2026-11-20-mini-ddp-milestone4-design.md` (§3). Parent: `docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md` (§6, §8 E4–E8).
**Sibling plans:**
- `docs/superpowers/plans/2026-11-20-mini-ddp-milestone4a-roofline.md` (Alex). No ordering dependency: 4a changes only the FP16 WMMA kernel on V100, and this plan runs FP32 on P100 except the one FP16 accuracy run.
- `docs/superpowers/plans/2026-11-12-mini-ddp-milestone3-pipeline.md`. Its Task 4 is this plan's Task 1.

## Global Constraints

- All earlier constraints hold, in particular:
  - The step line: `step <i> loss <L> compute_s <c> comm_s <m> step_s <s>`.
  - Medians are over steps 5–29.
  - Every `srun` in a multi-node job spans the whole allocation; a step on a subset segfaults in PMIx/UCX wire-up on Discovery.
  - Two ranks (= 2 GPUs) per node.
- Same model and settings as the Milestone 2/3 sweeps unless a task says otherwise: synthetic, full model (default `--hidden 4096 --layers 3`), `--gemm regtile`, `--allreduce ring`, global batch 512 (strong) or 128 per GPU (weak), P100s.
- **Budget:**
  - Run `myaccount` on Discovery before Task 3 and record the remaining SU in the PR.
  - P100 costs 4 SU per GPU-minute plus 1 SU per CPU and per 4 GB. An 8-GPU job is about 80 SU per minute; a 16-GPU job is about 160.
  - Every new job script states its cost in its header comment, as the existing ones do.
  - Dry-run each new sweep on 2 GPUs in `debug` first.
- **Measurements:**
  - Committed under `results/<experiment>/` (logs + CSV + PNG).
  - Plot scripts run on a laptop.
  - Nsight `.nsys-rep` files are not committed; screenshots are.
- **Workflow:**
  - One branch per task (`t2-...`) → PR with the CARC output → Alex looks it over → merge.
  - Doc-only changes merge directly.
  - No attribution lines in commits or PR descriptions.
- **Dates:** Task 1 by Nov 27, Task 2 by Dec 1, Tasks 3–4 by Dec 5, Task 5 by Dec 10. Final deadline Dec 11.

## Order

| Task | GPUs | Depends on | Unblocks |
|---|---|---|---|
| 1. E8 layouts at 8 GPUs (3b Task 4) + README links for E7/E8 | 8 P100 | — | 5 |
| 2. E6 bucket sweep + Nsight Systems timeline | 8 P100; 2 P100 | — | 5 |
| 3. Final scaling E4/E5 (1 → 8, 16 if budget allows) | up to 16 P100 | — | 5 |
| 4. Accuracy parity table | 1, 8 P100; 1 V100 | — | 5 |
| 5. README report, communication half | — | 1–4 | — |

## Files

```
scripts/slurm/sweep_2d.sh, sweep_2d_submit.sh      E8 (Task 1, per 3b Task 4)
scripts/plot_2d.py                                 E8 plot (Task 1)
scripts/slurm/sweep_bucket.sh                      E6 sweep (Task 2)
scripts/slurm/nsys_overlap.sh                      E6 timeline (Task 2)
scripts/plot_bucket.py                             E6 plot (Task 2)
scripts/slurm/sweep_scaling*.sh, plot_scaling.py   N=16 support if needed (Task 3)
scripts/slurm/accuracy.sh                          accuracy runs (Task 4)
results/2d/, results/overlap/, results/scaling/, results/accuracy/
README.md                                          (Tasks 1, 5)
```

---

### Task 1: E8, layouts at 8 GPUs (Shaozhe; Alex reviews)

Exactly as specified in `2026-11-12-mini-ddp-milestone3-pipeline.md`, Task 4:
- (dp, pp) = (8, 1), (4, 2), (2, 4).
- M = 8 for pp > 1.
- Output: `results/2d/dp{8,4,2}_pp{1,2,4}.log`, `summary.csv` (`dp,pp,compute_s,comm_s,step_s`) and `layouts.png`.

Additions here:
- The README links both `results/pipeline/bubble.png` (E7, already committed) and `results/2d/layouts.png` under the measurements, each with one line of finding.
- The PR explains which layout wins using the two measured costs: the per-stage all-reduce shrinks with fewer replicas and smaller stages (α–β, E3), while the bubble grows with more stages (E7).

- [ ] Branch `t2-2d-sweep`; write the per-layout jobs and submitter; dry-run on 2 GPUs in `debug`; run; plot on the laptop.
- [ ] Update the README; PR "Data × pipeline layouts at 8 GPUs (E8)".

### Task 2: E6, bucket-size sweep and Nsight Systems timeline (Shaozhe)

**What to measure:**
- **Sweep:**
  - 8 P100s (4 nodes × 2), strong scaling (batch 512), ring, `--overlap`.
  - `--bucket-mb` 1, 5, 25 and 100, plus one overlap-off run in the same job as the reference.
  - One job, all runs sequential in it.
  - Output: `results/overlap/bucket_{off,1,5,25,100}.log`, and `bucket_sweep.csv` (`bucket_mb,compute_s,comm_s,step_s`, with `off` for the reference).
  - Plot: `bucket_sweep.png`, step time and exposed comm against bucket size, with the off reference as a horizontal line.
- **Timeline:**
  - 2 P100s on one node, 10 steps.
  - `nsys profile -t cuda,mpi,osrt -o <name>` once with overlap off and once with `--overlap --bucket-mb 25`.
  - First check that `nsys` is on the path after `source scripts/env.sh`. If it isn't, find the module (`module spider nsight`) and record it in `env.sh`.
  - Commit two PNG screenshots, `results/overlap/timeline_{off,on}.png`, taken in the Nsight Systems GUI on the laptop. Each shows one step: backward kernels on the GPU row and the comm thread's all-reduce on the CPU rows.

**Acceptance:**
- 5 logs with 30 step lines each.
- The CSV and plot are committed. The PR names the best bucket size and explains the trend at both ends: small buckets pay α per bucket, and large buckets start too late to overlap.
- The two timeline PNGs show the overlap visibly. If they don't, the PR says why.

- [ ] Branch `t2-bucket-sweep`; write the jobs; dry-run on 2 GPUs in `debug`; run; plot and screenshot on the laptop.
- [ ] PR "Bucket-size sweep and overlap timeline (E6)".

### Task 3: Final scaling, E4/E5 (Shaozhe; both)

**What to run:**
- Rerun `sweep_scaling_submit.sh` for 1, 2, 4, 8 GPUs on the current `main`. This shows whether anything since Milestone 2 changed the curves; nothing on the P100 FP32 path should have.
- 16 GPUs (8 nodes × 2) only if `myaccount` shows enough SU. A 16-GPU job is about 160 SU per minute, and each scaling job runs a few minutes.
- Extend `NS` in `plot_scaling.py` to include 16 when its logs exist. The plot must still work without them.
- If the budget is short, ask the instructor for more SU (parent spec §7) or report 1 → 8 only and say so in the README.

**Acceptance:**
- `results/scaling/summary.csv` and `scaling.png` are regenerated.
- Every strong-scaling run ends at the 1-GPU loss.
- The PR compares the new 1 → 8 numbers against the Milestone 2 commit, and explains any difference above about 5%.

- [ ] Branch `t2-final-scaling`; check `myaccount`; run; plot on the laptop.
- [ ] PR "Final scaling sweep (E4/E5)".

### Task 4: Accuracy parity (Shaozhe; Alex runs the FP16 row)

**What to run:**
- CIFAR-10 at `~/data/cifar-10-batches-bin`, the same model and schedule for every row:
  - `--hidden 1024 --layers 3 --batch 256 --lr 0.02 --epochs 5`.
  - `--layers 3` gives 4 weight layers, so pp = 4 has one per stage.
- Configurations: 1 GPU; dp = 8 (ring); pp = 4 (M = 4); (dp, pp) = (4, 2) (M = 4). These are FP32 on P100.
- The `--fp16` row runs on 1 V100 against its own same-job 1-V100 FP32 run, as in Milestone 3a.
- `scripts/slurm/accuracy.sh` runs the P100 rows. The FP16 row can reuse `fp16_checks.sh`'s CIFAR step with this row's settings.
- Output: `results/accuracy/*.log` and `summary.csv` (`config,gpus,test_acc,ref_acc,diff`).

**Acceptance:**
- FP32 rows: within 0.01 of the 1-GPU accuracy.
- FP16 row: within 0.02 of its V100 FP32 run.
- Any miss is investigated before it is reported. Data parallel is exact to 1e-3 in loss, so a miss points to a bug or a seed or data-order difference, not noise.

- [ ] Branch `t2-accuracy`; write the job; run (P100 rows; Alex runs the V100 row); fill in the CSV.
- [ ] PR "Accuracy parity across parallel configurations".

### Task 5: README report, communication half (Shaozhe; Alex writes the kernel half in 4a)

**What to write** (doc-only; directly to `main`; tell Alex before pushing):
- **Communication findings:**
  - All-reduce algorithms against the α–β model (E3).
  - Why host-staged all-reduce dominates the 1 → 8 scaling (E4/E5).
  - What overlap and bucket size buy (E6).
  - The bubble against (P−1)/(M+P−1) (E7).
  - The best 2D layout and why (E8).
  - The accuracy table.
- **Who did what:** Shaozhe's half.
- Every T2 plot in `results/*/*.png` is linked.

**Acceptance:** Alex has read the communication half; every parent-spec experiment E3–E8 has a linked plot or an explicit "not run, because ..." line.

- [ ] Write the sections; push to `main` after Alex's read-through.

---

## Milestone 4b done when

- `results/2d/layouts.png`, `results/overlap/bucket_sweep.png`, both timeline PNGs, the regenerated `results/scaling/scaling.png` and `results/accuracy/summary.csv` are committed and linked from the README.
- Every configuration in the accuracy table meets its tolerance.
- The communication half of the README report is written and reviewed by Alex.

## Not in this plan

- FP16 gradient all-reduce. It is only estimated, in E9.
- CUDA-aware MPI for activations.
- 1F1B.
- `--overlap` or NCCL inside 2D.
- FP16 multi-node scaling on V100.
- Repeated runs for error bars. Add them only if budget remains after Task 3.
