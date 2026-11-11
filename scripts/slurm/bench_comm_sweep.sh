#!/bin/bash
#SBATCH --job-name=bench-comm
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/bench_comm_sweep.sh
# runs bench_comm (CPU only, no GPUs) at the GPU jobs' layouts, one CSV per layout, for Task 3 (E3)
# cost: ~32 SU per minute while running (16 CPU + 8 x 8 GB)
set -euo pipefail
source scripts/env.sh
make bin/bench_comm
mkdir -p results/comm

# run NAME NODES RANKS: bench_comm on RANKS ranks spread evenly over NODES nodes
run() {
  local name=$1 nodes=$2 ranks=$3
  local opts=(--mpi="$SRUN_MPI" --nodes="$nodes" --ntasks="$ranks" --ntasks-per-node=$((ranks / nodes)))
  echo "== $name: $ranks ranks on $nodes node(s): $(srun "${opts[@]}" hostname -s | sort | uniq -c | xargs)"
  srun "${opts[@]}" ./bin/bench_comm > "results/comm/$name.csv"
  echo "$(wc -l < "results/comm/$name.csv") lines (expect 71)"
}

run intra_np2     1 2
run inter_np2     2 2
run allreduce_np4 2 4
run allreduce_np8 4 8
run allreduce_np16 8 16
