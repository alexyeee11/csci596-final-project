#!/bin/bash
#SBATCH --job-name=sweep-2d
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# E8 (Milestone 4b Task 1): data x pipeline layouts on the same GPUs. Usage: sweep_2d.sh [PP...] (default 1 2 4).
#   Runs every layout dp = SLURM_NTASKS / pp in this one allocation. All layouts use every rank, so each srun still
#   spans the whole allocation (a step on a subset segfaults in PMIx/UCX wire-up on Discovery), and the 8-GPU
#   allocation is queued for once instead of three times. Submitted by scripts/slurm/sweep_2d_submit.sh.
# debug dry run (2 GPUs, logs kept out of results/):
#   OUT=logs/2d-dry sbatch --account=<acct> --partition=debug --nodes=1 --ntasks-per-node=2 --gres=gpu:p100:2 \
#     scripts/slurm/sweep_2d.sh 1 2
# Logs: $OUT/dp<dp>_pp<pp>.log (M = 8 for pp > 1, the plan's runs) and, for each other M in MBS, dp<dp>_pp<pp>_m<M>.log.
# cost: ~4 P100 SU per GPU-minute + 1 SU per CPU + 1 SU per 4 GB; ~80 SU per minute for 8 GPUs, about 5 minutes for the
#   default 5 runs (~400 SU)
set -euo pipefail
source scripts/env.sh
OUT=${OUT:-results/2d}
mkdir -p "$OUT"
np=$SLURM_NTASKS
echo "== $np ranks on $SLURM_JOB_NUM_NODES node(s): $(srun --mpi="$SRUN_MPI" hostname -s | sort | uniq -c | xargs)"
# synthetic data is 4096 samples = 8 steps per epoch at batch 512, so 4 epochs to reach 30 steps
BASE="--synthetic --epochs 4 --steps 30 --batch 512 --gemm regtile --allreduce ring"  # default --hidden 4096 --layers 3
for pp in ${@:-1 2 4}; do
  if (( np % pp )); then echo "skip pp=$pp: $np ranks are not a multiple"; continue; fi
  dp=$(( np / pp ))
  if [ "$pp" = 1 ]; then
    echo "== dp=$dp pp=1"
    srun --mpi="$SRUN_MPI" ./bin/train $BASE > "$OUT/dp${dp}_pp1.log"
    continue
  fi
  # M = 8 is the plan's setting; M = 2 is the best M at batch 512 in E7, where 8 micro-batches of 512/dp/8 rows are
  # far below the 128-row GEMM tile
  for m in ${MBS:-8 2}; do
    name=dp${dp}_pp${pp}; [ "$m" = 8 ] || name=${name}_m${m}
    echo "== dp=$dp pp=$pp M=$m"
    srun --mpi="$SRUN_MPI" ./bin/train $BASE --pp "$pp" --microbatches "$m" > "$OUT/$name.log"
  done
done
