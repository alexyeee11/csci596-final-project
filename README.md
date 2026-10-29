# Mini-DDP: Multi-GPU Neural Network Training from Scratch in CUDA + MPI

Final project for CSCI 596 (Scientific Computing and Visualization), Fall 2026.

> **Status:** design complete, implementation starting. See the [design spec](docs/superpowers/specs/2026-10-28-mini-ddp-cuda-mpi-design.md) and the [Milestone 1 plan](docs/superpowers/plans/2026-10-28-mini-ddp-milestone1.md).

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
| FP32 vs. FP16 | Speedup from tensor cores and from halving all-reduce volume |

All configurations must reach the same test accuracy as the 1-GPU run. An MLP on CIFAR-10 tops out around 55%, so the point is matching accuracy, not high accuracy.

## Build and Run

> Planned interface from the [Milestone 1 plan](docs/superpowers/plans/2026-10-28-mini-ddp-milestone1.md). The code is not in the repository yet.

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

# Train on 1 GPU
./bin/train --data <path>/cifar-10-batches-bin --epochs 5 --batch 256

# Data-parallel on 4 GPUs across 2 nodes, with ring all-reduce
srun --mpi=pmix --ntasks=4 ./bin/train --data <path>/cifar-10-batches-bin --allreduce ring
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
- [ ] Nov 10 — data-parallel training matches the 1-GPU loss curve
- [ ] Nov 17 — overlap + bucketing; first 1 → 8 GPU scaling plot
- [ ] Nov 20 — agree on topic with the instructor
- [ ] Nov 24 — tensor-core FP16 GEMM; GPipe pipeline
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
