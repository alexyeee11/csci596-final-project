#!/bin/bash
#SBATCH --job-name=sweep-pipeline
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# one pipeline depth P per job (P=2: 1 node x 2, P=4: 2 nodes x 2); submitted by scripts/slurm/sweep_pipeline_submit.sh,
# which sets --nodes/--ntasks-per-node/--gres. Every srun of a multi-node job spans the whole allocation (a step on a
# subset segfaults in PMIx/UCX wire-up on Discovery); the 1-GPU baseline runs inside the single-node P=2 job.
# debug dry run: sbatch --account=<acct> --partition=debug --nodes=1 --ntasks-per-node=2 --gres=gpu:p100:2 \
#   scripts/slurm/sweep_pipeline.sh 2
# Task 2 (E7): bubble vs micro-batches M; logs go to results/pipeline/. cost: ~40 SU per minute for the P=4 job
set -euo pipefail
P=$1
source scripts/env.sh
OUT=results/pipeline
mkdir -p "$OUT"
echo "== P=$P: $SLURM_NTASKS ranks on $SLURM_JOB_NUM_NODES node(s): $(srun --mpi="$SRUN_MPI" hostname -s | sort | uniq -c | xargs)"
# synthetic data is 4096 samples = 8 steps per epoch at batch 512, so 4 epochs to reach 30 steps
BASE="--synthetic --epochs 4 --steps 30 --batch 512 --gemm regtile"  # default --hidden 4096 --layers 3
if [ "$P" = 2 ] && [ "${BASELINE:-1}" = 1 ]; then
  srun --mpi="$SRUN_MPI" --ntasks=1 ./bin/train $BASE > "$OUT/p1_m1.log"
fi
for m in ${MBS:-1 2 4 8 16}; do
  srun --mpi="$SRUN_MPI" ./bin/train $BASE --pp "$P" --microbatches "$m" > "$OUT/p${P}_m${m}.log"
done
