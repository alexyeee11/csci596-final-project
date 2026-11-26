#!/bin/bash
#SBATCH --job-name=sweep-bucket
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# E6 (Milestone 4b Task 2): overlap bucket-size sweep, strong scaling (batch 512) with ring all-reduce. One job runs the
# overlap-off reference and --overlap --bucket-mb B for each B in BUCKETS (default 1 5 25 100), all on every rank.
# usage: mkdir -p logs && sbatch --account=<acct> --nodes=4 --ntasks-per-node=2 --gres=gpu:p100:2 scripts/slurm/sweep_bucket.sh
# debug dry run (2 GPUs, logs kept out of results/):
#   OUT=logs/bucket-dry sbatch --account=<acct> --partition=debug --nodes=1 --ntasks-per-node=2 --gres=gpu:p100:2 \
#     scripts/slurm/sweep_bucket.sh
# Logs: $OUT/bucket_off.log and $OUT/bucket_<B>.log. Plot on the laptop: python3 scripts/plot_bucket.py results/overlap
# cost: ~4 P100 SU per GPU-minute + 1 SU per CPU + 1 SU per 4 GB; ~80 SU per minute for 8 GPUs, about 4 minutes for the
#   5 runs (~320 SU)
set -euo pipefail
source scripts/env.sh
OUT=${OUT:-results/overlap}
mkdir -p "$OUT"
echo "== $SLURM_NTASKS ranks on $SLURM_JOB_NUM_NODES node(s): $(srun --mpi="$SRUN_MPI" hostname -s | sort | uniq -c | xargs)"
# synthetic data is 4096 samples = 8 steps per epoch at batch 512, so 4 epochs to reach 30 steps
BASE="--synthetic --epochs 4 --steps 30 --batch 512 --gemm regtile --allreduce ring"  # default --hidden 4096 --layers 3
echo "== overlap off"
srun --mpi="$SRUN_MPI" ./bin/train $BASE > "$OUT/bucket_off.log"
for B in ${BUCKETS:-1 5 25 100}; do
  echo "== overlap, bucket $B MB"
  srun --mpi="$SRUN_MPI" ./bin/train $BASE --overlap --bucket-mb "$B" > "$OUT/bucket_$B.log"
done
