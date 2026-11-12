#!/bin/bash
#SBATCH --job-name=sweep-scaling
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/sweep_scaling.sh
# debug dry run (fails at N=4 by design, N=1 and N=2 logs are kept for checking):
#   sbatch --account=<acct> --partition=debug --nodes=1 --gres=gpu:p100:2 --time=00:15:00 scripts/slurm/sweep_scaling.sh
# DRY_RUN=1 bash scripts/slurm/sweep_scaling.sh only echoes the 16 srun commands.
# Task 6 (E4, E5): 1/2/4/8 GPUs, strong (batch 512) and weak (128 per GPU), overlap off/on; logs go to results/scaling/.
# cost: ~4 P100 SU per GPU-minute + 1 SU per CPU + 1 SU per 4 GB; ~80 SU per minute for the full 4-node job (8 GPUs)
set -euo pipefail
[ -n "${DRY_RUN:-}" ] || source scripts/env.sh
ALGO=${ALGO:-ring}
BUCKET_MB=${BUCKET_MB:-25}
OUT=results/scaling
mkdir -p "$OUT"
# synthetic data is 4096 samples, so weak N=8 (batch 1024) has 4 steps per epoch: 100 epochs, --steps stops the run at 30
BASE="--synthetic --epochs 100 --steps 30 --gemm regtile --allreduce $ALGO"  # default --hidden 4096 --layers 3
for n in 1 2 4 8; do
  # pack 2 ranks per node: N=1 one rank, N=2 one node, N=4 two nodes, N=8 four nodes
  per=$(( n < 2 ? n : 2 )); nodes=$(( n / per ))
  for kind in strong weak; do
    batch=512; [ "$kind" = weak ] && batch=$(( 128 * n ))
    for ov in off on; do
      extra=""; [ "$ov" = on ] && extra="--overlap --bucket-mb $BUCKET_MB"
      cmd="srun --mpi=${SRUN_MPI:-pmix} --nodes=$nodes --ntasks=$n --ntasks-per-node=$per ./bin/train $BASE --batch $batch $extra"
      log="$OUT/${kind}_n${n}_${ov}.log"
      if [ -n "${DRY_RUN:-}" ]; then echo "$cmd > $log"; else $cmd > "$log"; fi
    done
  done
done
