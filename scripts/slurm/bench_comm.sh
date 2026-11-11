#!/bin/bash
#SBATCH --job-name=bench-comm
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
# one bench_comm layout per job; submitted by scripts/slurm/bench_comm_sweep.sh, which sets --nodes/--ntasks-per-node.
# srun always spans the whole allocation: a step on a subset of a multi-node job hangs in PMIx/UCX wire-up on Discovery.
set -euo pipefail
name=$1
source scripts/env.sh
mkdir -p results/comm
echo "== $name: $SLURM_NTASKS ranks on $SLURM_JOB_NUM_NODES node(s): $(srun --mpi="$SRUN_MPI" hostname -s | sort | uniq -c | xargs)"
srun --mpi="$SRUN_MPI" ./bin/bench_comm > "results/comm/$name.csv"
echo "$(wc -l < "results/comm/$name.csv") lines (expect 71)"
