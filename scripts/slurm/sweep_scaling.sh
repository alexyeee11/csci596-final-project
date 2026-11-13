#!/bin/bash
#SBATCH --job-name=sweep-scaling
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# one GPU count N per job (N=1: 1 node x 1 rank, 2: 1x2, 4: 2x2, 8: 4x2); submitted by scripts/slurm/sweep_scaling_submit.sh,
# which sets --nodes/--ntasks-per-node/--gres. srun always spans the whole allocation: a step on a subset of a
# multi-node job segfaults in PMIx/UCX wire-up on Discovery.
# usage: mkdir -p logs && sbatch --account=<acct> --nodes=1 --ntasks-per-node=2 --gres=gpu:p100:2 scripts/slurm/sweep_scaling.sh 2
# debug dry run: sbatch --account=<acct> --partition=debug --nodes=1 --ntasks-per-node=2 --gres=gpu:p100:2 scripts/slurm/sweep_scaling.sh 2
# Task 6 (E4, E5): strong (batch 512) and weak (128 per GPU), overlap off/on; logs go to results/scaling/.
# cost: ~4 P100 SU per GPU-minute + 1 SU per CPU + 1 SU per 4 GB; ~80 SU per minute for the 8-GPU job, each job a few minutes
set -euo pipefail
n=$1
source scripts/env.sh
ALGO=${ALGO:-ring}
BUCKET_MB=${BUCKET_MB:-25}
OUT=results/scaling
mkdir -p "$OUT"
echo "== N=$n: $SLURM_NTASKS ranks on $SLURM_JOB_NUM_NODES node(s): $(srun --mpi="$SRUN_MPI" hostname -s | sort | uniq -c | xargs)"
# synthetic data is 4096 samples, so weak N=8 (batch 1024) has 4 steps per epoch: 100 epochs, --steps stops the run at 30
BASE="--synthetic --epochs 100 --steps 30 --gemm regtile --allreduce $ALGO"  # default --hidden 4096 --layers 3
for kind in strong weak; do
  batch=512; [ "$kind" = weak ] && batch=$(( 128 * n ))
  for ov in off on; do
    extra=""; [ "$ov" = on ] && extra="--overlap --bucket-mb $BUCKET_MB"
    srun --mpi="$SRUN_MPI" ./bin/train $BASE --batch "$batch" $extra > "$OUT/${kind}_n${n}_${ov}.log"
  done
done
