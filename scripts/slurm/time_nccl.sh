#!/bin/bash
#SBATCH --job-name=time-nccl
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/time_nccl.sh
# Task 5 acceptance on 4 GPUs over 2 nodes: NCCL equality test, then the full model's comm_s for ring, mpi and nccl
# (no overlap); logs go to results/nccl/. cost: ~40 SU per minute while running (4 P100 + 8 CPU + 2 x 32 GB)
set -euo pipefail
source scripts/env.sh
echo "== ALGOS=nccl scripts/test_dp.sh 4"
ALGOS=nccl scripts/test_dp.sh 4 || echo "(equality test failed; timing below is not meaningful)"
# synthetic data is 4096 samples = 8 steps per epoch at batch 512, so 4 epochs to reach 30 steps
ARGS="--synthetic --epochs 4 --steps 30 --batch 512 --gemm regtile"  # default --hidden 4096 --layers 3
mkdir -p results/nccl
for algo in ${ALGOS:-ring mpi nccl}; do
  srun --mpi="$SRUN_MPI" ./bin/train $ARGS --allreduce "$algo" > "results/nccl/$algo.log"
done
# mean of steps 5..29 (steps 0-4 warm up); fields: step i loss L compute_s c comm_s m step_s s
for algo in ${ALGOS:-ring mpi nccl}; do
  awk -v a="$algo" '$1 == "step" && $2 >= 5 { c += $6; m += $8; s += $10; n++ }
    END { printf "%-5s steps=%d  compute_s %.4f  comm_s %.4f  step_s %.4f\n", a, n, c / n, m / n, s / n }' \
    "results/nccl/$algo.log"
done | tee results/nccl/summary.txt
