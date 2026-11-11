#!/bin/bash
#SBATCH --job-name=test-overlap
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/test_overlap.sh                   (2 GPUs)
#        sbatch --account=<class account> --partition=gpu --nodes=2 --ntasks-per-node=2 --ntasks=4 \
#          scripts/slurm/test_overlap.sh                                                                   (4 GPUs, 2 nodes)
# Task 4 equality tests: overlap with ~33 buckets (crossing layer boundaries), with 4 buckets, and without overlap.
set -uo pipefail
source scripts/env.sh
np=$SLURM_NTASKS
status=0
echo "== overlap, 0.1 MB buckets"; scripts/test_dp.sh "$np" --overlap --bucket-mb 0.1 || status=1
echo "== overlap, 1 MB buckets";   scripts/test_dp.sh "$np" --overlap --bucket-mb 1 || status=1
echo "== no overlap";              scripts/test_dp.sh "$np" || status=1
exit $status
