#!/bin/bash
#SBATCH --job-name=test-pp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/test_pp.sh                       (P = 2)
#        sbatch --account=<class account> --partition=gpu --nodes=2 --ntasks-per-node=2 --ntasks=4 \
#          scripts/slurm/test_pp.sh                                                                       (P = 4, 2 nodes)
# Milestone 3b Task 1 acceptance: GPipe with P = $SLURM_NTASKS stages for M = 1, 2, 4 against 1 rank; then
# data parallelism on the same ranks still passes.
set -uo pipefail
source scripts/env.sh
make gpu || exit 1
np=$SLURM_NTASKS
status=0
echo "== pipeline, P=$np"; scripts/test_pp.sh "$np" || status=1
echo "== data parallel"; scripts/test_dp.sh "$np" || status=1
exit $status
