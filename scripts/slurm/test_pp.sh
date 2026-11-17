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
# Milestone 3b Tasks 1 and 3 acceptance: on NP = $SLURM_NTASKS ranks, GPipe with P = NP stages and, for NP >= 4,
# 2D with P = NP/2 (dp = 2), each for M = 1, 2, 4 against 1 rank; then data parallelism on the same ranks.
# PPS="..." overrides the pipeline depths.
set -uo pipefail
source scripts/env.sh
make gpu || exit 1
np=$SLURM_NTASKS
status=0
pps=${PPS:-$np}
[ -z "${PPS:-}" ] && [ "$np" -ge 4 ] && pps="$np $(( np / 2 ))"
for P in $pps; do
  echo "== pipeline, NP=$np P=$P (dp=$(( np / P )))"; scripts/test_pp.sh "$np" "$P" || status=1
done
echo "== data parallel"; scripts/test_dp.sh "$np" || status=1
exit $status
