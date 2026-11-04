#!/bin/bash
#SBATCH --job-name=mddp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:p100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/run.sh <command...>
# cost: ~16 SU per minute while running (2 P100 + 4 CPU + 16 GB)
set -euo pipefail
source scripts/env.sh
"$@"
