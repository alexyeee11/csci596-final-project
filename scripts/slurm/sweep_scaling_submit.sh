#!/bin/bash
# usage (login node): source scripts/env.sh && make && mkdir -p logs && scripts/slurm/sweep_scaling_submit.sh <class account> [N...]
# submits scripts/slurm/sweep_scaling.sh once per GPU count (default 1 2 4 8), one job per layout so each srun spans its
# whole allocation. --dependency=singleton (same job name) runs them one at a time. Pass e.g. `4 8` to redo only those.
# DRY_RUN=1 echoes the sbatch commands.
set -euo pipefail
account=$1; shift
for n in ${@:-1 2 4 8}; do
  per=$(( n < 2 ? n : 2 ))  # 2 ranks (= 2 P100) per node
  cmd="sbatch --account=$account --dependency=singleton --nodes=$(( n / per )) --ntasks-per-node=$per --gres=gpu:p100:$per scripts/slurm/sweep_scaling.sh $n"
  if [ -n "${DRY_RUN:-}" ]; then echo "$cmd"; else $cmd; fi
done
