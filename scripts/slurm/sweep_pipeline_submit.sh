#!/bin/bash
# usage (login node): source scripts/env.sh && make gpu && mkdir -p logs && scripts/slurm/sweep_pipeline_submit.sh <class account> [P...]
# submits scripts/slurm/sweep_pipeline.sh once per pipeline depth (default 2 4), one job per layout so each srun spans
# its whole allocation; --dependency=singleton runs them one at a time. DRY_RUN=1 echoes the sbatch commands.
set -euo pipefail
account=$1; shift
for P in ${@:-2 4}; do
  cmd="sbatch --account=$account --dependency=singleton --nodes=$(( P / 2 )) --ntasks-per-node=2 --gres=gpu:p100:2 scripts/slurm/sweep_pipeline.sh $P"
  if [ -n "${DRY_RUN:-}" ]; then echo "$cmd"; else $cmd; fi
done
