#!/bin/bash
#SBATCH --job-name=test-stage
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/test_stage.sh
# Milestone 3a Task 1 acceptance: layer + MLP tests (incl. stage API) on one GPU, then 2-rank data parallel.
set -uo pipefail
source scripts/env.sh
make clean && make gpu || exit 1
status=0
./bin/test_layers || status=1
./bin/test_mlp || status=1
scripts/test_dp.sh 2 || status=1
exit $status
