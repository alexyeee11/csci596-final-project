#!/bin/bash
# usage (login node): source scripts/env.sh && make gpu && mkdir -p logs && scripts/slurm/sweep_2d_submit.sh <class account> [PP...]
# submits scripts/slurm/sweep_2d.sh as one 8-GPU job (4 nodes x 2 P100) that runs every layout (default pp = 1 2 4).
# Pass e.g. `4` to redo only that layout. DRY_RUN=1 echoes the sbatch command.
set -euo pipefail
account=$1; shift
cmd="sbatch --account=$account --nodes=4 --ntasks-per-node=2 --gres=gpu:p100:2 scripts/slurm/sweep_2d.sh ${*:-1 2 4}"
if [ -n "${DRY_RUN:-}" ]; then echo "$cmd"; else $cmd; fi
