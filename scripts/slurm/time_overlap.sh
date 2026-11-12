#!/bin/bash
#SBATCH --job-name=time-overlap
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/time_overlap.sh
# Task 4 acceptance: full model on 4 GPUs over 2 nodes, overlap off vs on; logs go to results/overlap/.
# cost: ~40 SU per minute while running (4 P100 + 8 CPU + 2 x 32 GB)
set -euo pipefail
source scripts/env.sh
ALGO=${ALGO:-ring}
BUCKET_MB=${BUCKET_MB:-25}
ARGS="--synthetic --steps 30 --batch 512 --gemm regtile --allreduce $ALGO"  # default --hidden 4096 --layers 3
mkdir -p results/overlap
srun --mpi="$SRUN_MPI" ./bin/train $ARGS > results/overlap/off.log
srun --mpi="$SRUN_MPI" ./bin/train $ARGS --overlap --bucket-mb "$BUCKET_MB" > results/overlap/on.log
# mean of steps 5..29 (step 0-4 warm up); fields: step i loss L compute_s c comm_s m step_s s
for f in off on; do
  awk -v f="$f" '$1 == "step" && $2 >= 5 { c += $6; m += $8; s += $10; n++ }
    END { printf "overlap %-3s steps=%d  compute_s %.4f  comm_s %.4f  step_s %.4f\n", f, n, c / n, m / n, s / n }' \
    "results/overlap/$f.log"
done | tee results/overlap/summary.txt
