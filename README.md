# Mini-DDP: Multi-GPU Neural Network Training from Scratch in CUDA + MPI

Final project for CSCI 596 (Scientific Computing and Visualization), Fall 2026.

> **Status:** Milestone 3 complete; Milestone 4 in progress (roofline and the faster WMMA kernel are done). On top of Milestones 1 and 2 (kernels, MLP, data-parallel training that matches the 1-GPU loss curve, register-tiled GEMM, α–β all-reduce model, overlap, NCCL baseline, 1 → 8 GPU scaling), we added a tensor-core WMMA GEMM (34 TFLOP/s on a V100, 4.4× our register-tiled kernel), a roofline, mixed-precision training with FP32 master weights, and GPipe pipeline parallelism with micro-batches. See the [design spec](docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md) and the plans for [Milestone 1](docs/superpowers/plans/2026-10-28-mini-ddp-milestone1.md), [Milestone 2](docs/superpowers/plans/2026-11-05-mini-ddp-milestone2.md), [Milestone 3 FP16](docs/superpowers/plans/2026-11-12-mini-ddp-milestone3-fp16.md), [Milestone 3 pipeline](docs/superpowers/plans/2026-11-12-mini-ddp-milestone3-pipeline.md), and Milestone 4 ([design](docs/superpowers/specs/2026-11-20-mini-ddp-milestone4-design.md), [kernels](docs/superpowers/plans/2026-11-20-mini-ddp-milestone4a-roofline.md), [experiments](docs/superpowers/plans/2026-11-20-mini-ddp-milestone4b-experiments.md)).

## Problem

Adding GPUs doesn't make neural network training proportionally faster.

Large models are trained on many GPUs at once with **data parallelism**: each GPU computes gradients on its own slice of the batch, then all GPUs average their gradients before updating the weights. As GPUs are added, each one computes less, but each still has to exchange the full gradient every step. Communication, not arithmetic, comes to dominate training time.

Frameworks like PyTorch DDP and NCCL hide this behind black boxes, so it is hard to see where the time goes or to change it. On USC CARC Discovery, where a node has at most 2 GPUs, any run beyond 2 GPUs also has to cross nodes over InfiniBand.

**Goal:** build the training stack ourselves, both the GPU kernels and the gradient exchange, then measure and reduce the communication cost from 1 to 16 GPUs.

## Methods

One MPI rank drives one GPU. The model is an MLP for CIFAR-10 (3072 → 4096 → 4096 → 4096 → 10, about 38M parameters).

**GPU kernels (CUDA)**
- GEMM, built up step by step: naive → shared-memory tiled → register-tiled → tensor-core (WMMA, FP16 in / FP32 accumulate), compared against cuBLAS
- Fused bias + ReLU, a numerically stable fused softmax + cross-entropy, and SGD
- Mixed precision: FP32 master weights with FP16 compute and loss scaling

**Communication (MPI)**
- Hand-written all-reduce: naive (gather to rank 0), tree (recursive doubling) and ring (reduce-scatter + all-gather), compared against `MPI_Allreduce` and NCCL
- Gradient bucketing, with each bucket's all-reduce overlapped with the rest of backpropagation (CUDA streams + a communication thread)
- An α–β (latency–bandwidth) cost model fitted from ping-pong measurements, within a node and across nodes

**Parallelism**
- Data parallelism across GPUs
- GPipe-style pipeline parallelism: layers split across GPUs, with micro-batches
- 2D (data × pipeline) parallelism using `MPI_Comm_split`

**Correctness:** every parallel configuration must reproduce the 1-GPU loss curve to floating-point tolerance. Kernels are checked against CPU references, and backpropagation is checked with finite differences.

## Expected Results

| Experiment | What it shows |
|---|---|
| GEMM GFLOP/s vs. matrix size, plus a roofline plot | How far each kernel optimization gets toward cuBLAS and hardware peak |
| All-reduce bandwidth vs. message size, 2–16 GPUs | Ring vs. tree vs. naive, within a node vs. across nodes, measured vs. α–β model |
| Strong and weak scaling, 1 → 16 GPUs | Speedup, efficiency, and the split between compute and communication time |
| Overlap on/off and bucket-size sweep | How much communication can be hidden behind backprop (with Nsight timelines) |
| Pipeline bubble vs. micro-batch count | Measured idle time vs. the theoretical (P−1)/(M+P−1) |
| 2D layouts at 8 GPUs | Which data × pipeline split trains fastest |
| FP32 vs. FP16 | Speedup from tensor cores, and an α–β estimate of what FP16 all-reduce would save |

All configurations must reach the same test accuracy as the 1-GPU run. An MLP on CIFAR-10 tops out around 55%, so the point is matching accuracy, not high accuracy.

**First measurements (P100):**
- [GEMM GFLOP/s](results/gemm/p100.png): register tiling is 3.1× the shared-memory tiled kernel at 4096³ and reaches 63% of cuBLAS.
- [All-reduce cost vs. the α–β model](results/comm/allreduce_model.png), with the fitted parameters in [alpha_beta.txt](results/comm/alpha_beta.txt).
- [Strong and weak scaling, 1 → 8 GPUs](results/scaling/scaling.png) ([summary.csv](results/scaling/summary.csv)): with ~185 MB of gradients (46.2M parameters) and only ~17 ms of compute per step, the host-staged all-reduce dominates, so 8 GPUs are ~6× slower per step than 1. Overlap hides 15–18% of the step. Every strong-scaling run ends at the same loss as the 1-GPU run.
- [Pipeline bubble vs. micro-batches](results/pipeline/bubble.png) ([summary.csv](results/pipeline/summary.csv)): the measured bubble follows (P−1)/(M+P−1) but stays above it, because the stages are unequal (the last one holds only the 4096→10 layer). Pipelining is 3–5× faster than data parallelism on the same GPUs, since only activations cross between GPUs. At batch 512 the best is M = 2: more micro-batches make each GEMM smaller than its 128-row tile, so compute time grows faster than the bubble shrinks.
- [Data × pipeline layouts at 8 GPUs](results/2d/layouts.png) ([summary.csv](results/2d/summary.csv)): dp=2 × pp=4 (74 ms per step at M = 2) is 2.7× faster than pure data parallelism (202 ms). Each stage all-reduces only its own layers' gradients, between 2 replicas on the same node, instead of all ~185 MB of gradients over 8 GPUs on 4 nodes. All layouts end at the same loss.
- [Overlap bucket size at 8 GPUs](results/overlap/bucket_sweep.png) ([csv](results/overlap/bucket_sweep.csv)): 5 MB buckets are best, at 158 ms per step against 206 ms with overlap off (−23%). With 1 MB buckets each of the 185 buckets pays about 0.25 ms of fixed cost (14 ring messages, a stream sync), so overlap barely helps. With 100 MB, the first bucket is ready only near the end of backward, so little is left to overlap. Nsight Systems timelines of one step on 2 GPUs, [overlap off](results/overlap/timeline_off.png) and [on](results/overlap/timeline_on.png), show the comm thread starting the first bucket's all-reduce while backward is still running.

**FP16 on a V100 (PCIe, 14 TFLOP/s FP32 / 112 TFLOP/s tensor peak):**
- [GEMM GFLOP/s](results/gemm/v100.png) ([csv](results/gemm/v100.csv)): our tensor-core WMMA kernel reaches 34.1 TFLOP/s at 4096³. That is 4.4× our register-tiled FP32 kernel and 2.6× FP32 cuBLAS, and 39% of cuBLAS FP16 (88.3 TFLOP/s).
- [Roofline](results/roofline/v100.png) ([csv](results/roofline/v100.csv)): CARC blocks GPU performance counters (`ERR_NVGPUCTRPERM`), so the DRAM bytes are the analytical minimum, not measured by Nsight Compute. At these sizes every GEMM sits far right of the ridge, so each kernel's distance below its roof is its own inefficiency, not a bandwidth limit.
- The WMMA kernel step by step, at 4096³ (1 V100, real inputs):

  | Version | TFLOP/s | % of tensor peak |
  |---|---|---|
  | scalar 2-byte tile loads (Milestone 3a) | 22.1 | 20% |
  | + 16-byte loads for aligned, fully inside tiles | 29–32 | 27% |
  | + register double buffering of the next K tile | 34.1 | 30% |
  | cuBLAS FP16 (reference) | 88.3 | 79% |

  Most of the remaining gap to cuBLAS is the 64×64 block tile: a 128×128 tile reuses each loaded byte twice as often. That is future work.
- [Training step time, FP32 vs FP16](results/fp16/summary.txt) (1 GPU, batch 512, hidden 4096): with our own kernels a step goes from 22.5 ms (FP32 regtile) to 5.9 ms (FP16 WMMA), 3.8× faster and 1.9× faster than FP32 cuBLAS (11.3 ms). With cuBLAS, FP16 is 3.4 ms (3.3×). FP16 test accuracy matches FP32 within 0.001 after 5 CIFAR-10 epochs.
- Run-to-run variance between CARC jobs is about ±5–10%, depending on the node; numbers above are single jobs.

## Build and Run

> Everything below works today. Multi-node jobs need `--partition=gpu`, because the `debug` partition only allows one node.

**Dependencies:** C compiler, OpenMPI, CUDA toolkit (with cuBLAS), Make. Runs on CARC Discovery (P100, V100, A100, A40 or L40S).

```bash
# Laptop (communication code and data loader only, no GPU needed)
brew install open-mpi
make check-cpu

# CARC Discovery
source scripts/env.sh
make gpu
mkdir -p logs
sbatch --account=<class account> scripts/slurm/run.sh ./bin/test_mlp

# Train on 1 GPU (reaches ~0.46 test accuracy, ~2.6 ms/step on an A40)
./bin/train --data <path>/cifar-10-batches-bin --epochs 5 --batch 256 --hidden 1024 --layers 2 --lr 0.02

# Check that N-GPU training matches 1 GPU, for every all-reduce algorithm (prints 4 PASS lines)
sbatch --account=<class account> scripts/slurm/run.sh scripts/test_dp.sh 2
sbatch --account=<class account> --partition=gpu --nodes=2 --ntasks=4 --ntasks-per-node=2 \
  scripts/slurm/run.sh scripts/test_dp.sh 4

# Data-parallel on 4 GPUs across 2 nodes, with ring all-reduce (--batch must divide by the GPU count)
sbatch --account=<class account> --partition=gpu --nodes=2 --ntasks=4 --ntasks-per-node=2 \
  scripts/slurm/run.sh srun --mpi=pmix ./bin/train --data <path>/cifar-10-batches-bin --allreduce ring
```

## Team

| Member | Role |
|---|---|
| Alex Ye | Kernels (T1) |
| Shaozhe Zhang | Communication (T2) |

- **Kernels:** GEMM variants, layer kernels, mixed precision, kernel profiling
- **Communication:** all-reduce algorithms, overlap, pipeline and 2D parallelism, scaling runs

## Milestones

- [x] Oct 28 — topic chosen, design spec and Milestone 1 plan written
- [x] Nov 10 — data-parallel training matches the 1-GPU loss curve (done Nov 5: 2 and 4 GPUs, all four all-reduce algorithms)
- [x] Nov 17 — overlap + bucketing; first 1 → 8 GPU scaling plot
- [x] Nov 20 — agree on topic with the instructor (discussed and agreed)
- [x] Nov 24 — tensor-core FP16 GEMM; GPipe pipeline (done Nov 19)
- [ ] Nov 30 / Dec 2 — one-minute in-class presentation (from this repo)
- [ ] Dec 11 — final repo due; submit URL to Brightspace

## References

- [Assignment handout](Final.pdf)
- [Design spec](docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md)
- [Milestone 1 implementation plan](docs/superpowers/plans/2026-10-28-mini-ddp-milestone1.md)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CARC GPU programming guide](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming)
- Li et al., [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/abs/2006.15704), VLDB 2020
- Huang et al., [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965), NeurIPS 2019
- Patarasuk & Yuan, Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations, JPDC 2009
