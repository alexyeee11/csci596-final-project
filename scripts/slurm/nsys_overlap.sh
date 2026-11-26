#!/bin/bash
#SBATCH --job-name=nsys-overlap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# E6 (Milestone 4b Task 2): Nsight Systems timelines of 10 data-parallel steps on 2 P100s, overlap off and on
# (--overlap --bucket-mb $BUCKET_MB, default 25). One report per rank and run:
#   logs/nsys/timeline_{off,on}.<rank>.nsys-rep (not committed; copy them to the laptop and screenshot one step in the
#   Nsight Systems GUI as results/overlap/timeline_{off,on}.png).
# usage: mkdir -p logs && sbatch --account=<acct> scripts/slurm/nsys_overlap.sh
# cost: ~4 P100 SU per GPU-minute + 1 SU per CPU + 1 SU per 4 GB; ~20 SU per minute, a few minutes in all
set -euo pipefail
source scripts/env.sh
# nsys ships with the CUDA toolkit; it is not always on PATH
NSYS=$(command -v nsys || ls "$CUDA_HOME"/bin/nsys "$CUDA_HOME"/nsight-systems-*/bin/nsys 2>/dev/null | head -1 || true)
if [ -z "$NSYS" ]; then echo "nsys not found: try 'module spider nsight' and add the module to scripts/env.sh"; exit 1; fi
echo "== nsys: $NSYS ($("$NSYS" --version))"
OUT=logs/nsys
mkdir -p "$OUT"
# 10 steps = 2 epochs of 8 steps at batch 512 (--steps stops the run)
BASE="--synthetic --epochs 2 --steps 10 --batch 512 --gemm regtile --allreduce ring"  # default --hidden 4096 --layers 3
for run in off on; do
  extra=""; [ "$run" = on ] && extra="--overlap --bucket-mb ${BUCKET_MB:-25}"
  echo "== overlap $run"
  # %q{SLURM_PROCID} gives each rank its own report
  srun --mpi="$SRUN_MPI" "$NSYS" profile -t cuda,mpi,osrt --mpi-impl=openmpi --force-overwrite true \
    -o "$OUT/timeline_$run.%q{SLURM_PROCID}" ./bin/train $BASE $extra
done
ls -l "$OUT"
