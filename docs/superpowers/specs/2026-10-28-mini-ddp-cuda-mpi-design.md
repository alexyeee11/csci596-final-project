# Mini-DDP: Data- and Pipeline-Parallel Neural Network Training from Scratch in CUDA + MPI

CSCI 596 final project (team of 2), Fall 2026. Due Dec 11, 2026.

## 1. Goal

Training neural networks is the dominant compute cost in modern AI, and every large model is trained
across many GPUs with frameworks (PyTorch DDP, Megatron, DeepSpeed) whose internals most users never see.
We build a minimal distributed trainer from scratch in C/CUDA/MPI, writing both the GPU compute kernels and
the inter-GPU communication ourselves, and measure where time goes as training scales from 1 to 16 GPUs on
USC CARC Discovery.

Prior CSCI596 repos (116 surveyed) include one MPI data-parallel MNIST timing study in mpi4py and several
k-means projects; none implement GPU kernels plus hand-written collectives, overlap, or pipeline parallelism.

## 2. Team split

| Teammate | Owns |
|---|---|
| T1 — kernels | GEMM variants, layer forward/backward kernels, mixed precision, kernel verification, Nsight Compute / roofline |
| T2 — communication | All-reduce variants, GPU data path, α–β model, bucketing + overlap, GPipe pipeline, 2D parallelism, Slurm sweeps |
| Shared | Interface, CIFAR-10 loader, CARC environment, plots, README report, presentation |

## 3. Architecture

One C/CUDA codebase built with `nvcc` + `mpicc`. One MPI rank drives one GPU.

```
src/
  kernels/   gemm.cu  elementwise.cu  loss.cu     (T1)
  model/     mlp.cu   layers, forward/backward, SGD step
  comm/      allreduce.c  overlap.c  pipeline.c   (T2)
  data/      cifar.c
  train.c    main: parse flags -> build model -> train loop
scripts/     slurm/*.sh, plot.py
```

**Interface between halves:** each layer owns a contiguous device gradient buffer. The model calls
`on_grad_ready(layer)` when a layer's backward finishes; the comm layer decides what to do (nothing,
blocking all-reduce, or bucketed non-blocking all-reduce). T2 can develop against random buffers before
kernels exist.

**Run modes are flags on one binary:**
`--allreduce=naive|tree|ring|mpi|nccl`, `--overlap`, `--bucket-mb=<n>`, `--pp=<stages>`,
`--microbatches=<m>`, `--gemm=naive|tiled|regtile|wmma|cublas`, `--fp16`.

## 4. Training and data flow

**Model:** MLP for CIFAR-10, 3072 → 4096 → 4096 → 4096 → 10 (~38M parameters, ~150 MB FP32 gradients).
Stretch: CNN via im2col + GEMM, reusing T1's GEMM.

**Data-parallel step (per rank):**
1. Load shard of global batch (rank r takes samples r, r+N, r+2N, ...).
2. Forward on the compute stream.
3. Backward, last layer first; record a `cudaEvent` after each layer's gradient kernel.
4. Gradient sync:
   - no overlap: synchronize, then all-reduce all gradients;
   - overlap: a dedicated comm thread waits on each bucket's event and all-reduces it while the GPU
     continues backward on earlier layers. Only the comm thread calls MPI during backward.
5. Scale by 1/N (fused into SGD step), update, next iteration.

**Pipeline (GPipe):** layers split into P contiguous stages, one per rank. Batch split into M
micro-batches; all forwards (activations sent via `MPI_Isend`/`Irecv`), then all backwards (activation
gradients sent back). Predicted bubble fraction (P−1)/(M+P−1). Stretch: 1F1B schedule.

**2D:** ranks form a `dp × pp` grid; `MPI_Comm_split` yields a pipeline communicator (activations
between stages) and a data communicator (gradient all-reduce among same-stage replicas).

**Correctness:** all ranks initialize from the same seed. Data parallel on N GPUs with global batch B must
match 1 GPU with batch B to floating-point tolerance; pipeline with P stages must match 1 GPU likewise.
These curve-match tests are the primary integration tests.

## 5. Kernels (T1)

| Variant | Idea |
|---|---|
| `naive` | One thread per output, global memory |
| `tiled` | 32×32 shared-memory tiles |
| `regtile` | 8×8 register sub-tile per thread, `float4` loads |
| `wmma` | FP16 inputs, FP32 accumulate, tensor cores (`nvcuda::wmma`) |
| `cublas` | Reference ceiling |

GEMM takes transpose flags (cuBLAS-style) so forward `Y = XW`, backward `dX = dY·Wᵀ` and
`dW = Xᵀ·dY` use one kernel.

Other kernels: fused bias+ReLU forward; ReLU backward mask; bias gradient (column reduction);
fused numerically stable softmax + cross-entropy producing loss and `dlogits = (p − onehot)/B`;
SGD step fused with 1/N gradient scaling.

**Mixed precision:** FP32 master weights, FP16 copies for GEMM, static loss scale ×1024 (stretch:
dynamic). FP16 gradient all-reduce halves communication volume (joint T1/T2 result).

**Hardware:** P100 has no tensor cores. Develop FP32 kernels on P100 (`debug` partition); WMMA on V100+.

**Verification:** each kernel vs CPU reference or cuBLAS with relative-error tolerance (looser for FP16);
finite-difference gradient check on a tiny MLP (8→16→4).

## 6. Communication (T2)

| All-reduce | Algorithm | Cost (N ranks, message S) |
|---|---|---|
| `naive` | Gather to rank 0, sum, broadcast (hand-written Send/Recv) | Root moves ~2(N−1)·S |
| `tree` | Recursive doubling | log₂N steps of S |
| `ring` | Reduce-scatter + all-gather | 2(N−1) steps of S/N |
| `mpi`, `nccl` | Library baselines | — |

**GPU data path:** check CUDA-aware OpenMPI on CARC (`ompi_info | grep -i cuda`); if available pass
device pointers, else stage through pinned host memory with `cudaMemcpyAsync`. Measure both if possible.

**α–β model:** ping-pong microbenchmark fits latency α and inverse bandwidth β, separately intra-node and
inter-node (InfiniBand). Plot predicted vs measured all-reduce time.

**Bucketing:** default 25 MB; sweep 1–100 MB.

## 7. Platform and budget

CARC Discovery, class allocation: 600,000 SU shared by ~100 students (~12,000 SU for the team).
GPU charges per reserved minute: A100/A40/L40S = 8 SU, V100/P100 = 4 SU, plus 1 SU per CPU and per 4 GB.
Max 2 GPUs per node (3 for L40S), so runs above 2 GPUs are multi-node. Max 36 concurrent GPUs per user.

Plan: develop on 1–2 P100/V100 in `debug`; dry-run the full sweep on 2 GPUs; run E4–E8 once as a scripted
batch. Estimated 5–8k SU total. Request additional SU from the instructor for 16-GPU runs and repeats.

## 8. Experiments

| # | Experiment | Owner |
|---|---|---|
| E1 | GEMM GFLOP/s vs size, all variants | T1 |
| E2 | Roofline from Nsight Compute | T1 |
| E3 | All-reduce bandwidth vs message size, N = 2…16, intra vs inter-node, with α–β model | T2 |
| E4 | Strong scaling, fixed global batch, 1→16 GPUs: time/epoch, efficiency, compute vs comm breakdown | both |
| E5 | Weak scaling, fixed per-GPU batch | both |
| E6 | Overlap on/off, bucket-size sweep, Nsight Systems timeline | T2 |
| E7 | Pipeline bubble vs M, measured vs (P−1)/(M+P−1) | T2 |
| E8 | 2D layouts at 8 GPUs: (dp, pp) = (8,1), (4,2), (2,4) | T2 |
| E9 | FP32 vs FP16, compute and all-reduce | both |
| — | Accuracy parity: every configuration reaches the 1-GPU test accuracy (MLP on CIFAR-10 tops out ~55%) | both |

**Success criteria:** (1) all parallel configurations match the 1-GPU loss curve; (2) ring all-reduce is
within a reasonable factor of `MPI_Allreduce`/NCCL and beats naive at large N; (3) overlap measurably
reduces step time; (4) measured pipeline bubble tracks the analytical model; (5) every plot above is in
the README.

## 9. Scope tiers

- **Minimum (~40 h/person):** tiled GEMM, single-GPU training, ring all-reduce, overlap, 1→8 GPU scaling.
- **Planned (~70–80 h/person):** everything in sections 5–8.
- **Stretch:** CNN via im2col, 1F1B schedule, dynamic loss scaling, FlashAttention-style tiled attention.

## 10. Schedule

| Week | T1 | T2 | Milestone |
|---|---|---|---|
| Oct 28 – Nov 3 | Naive + tiled GEMM | Naive/tree/ring on host buffers; ping-pong | Shared: CARC env, skeleton, interface, CIFAR loader |
| Nov 4 – 10 | Full fwd/bwd; 1-GPU training converges | GPU buffers, CUDA-aware check, NCCL; sync DP | DP curve matches 1-GPU |
| Nov 11 – 17 | Register-tiled GEMM; gradient check | Overlap + bucketing | First 1→8 GPU scaling plot |
| Nov 20 | — | — | Topic agreement deadline (aim to meet instructor week of Nov 3) |
| Nov 18 – 24 | WMMA + FP16 | GPipe pipeline | — |
| Nov 25 – Dec 1 | Nsight / roofline | 2D parallelism; full sweep | 1-minute presentation Nov 30 / Dec 2 |
| Dec 2 – 11 | Re-runs, plots | Re-runs, plots | README report (problem → finding → so what, who did what); submit Dec 11 |

## 11. Open items

- Confirm remaining class SU with `myaccount` on Discovery.
- Confirm CUDA-aware MPI availability and module versions on CARC.
- Confirm team member names for the README contributions section.
