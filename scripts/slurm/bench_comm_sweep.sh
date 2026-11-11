#!/bin/bash
# usage (login node): source scripts/env.sh && make cpu && mkdir -p logs && scripts/slurm/bench_comm_sweep.sh <class account>
# submits bench_comm (CPU only, no GPUs) at the GPU jobs' layouts for Task 3 (E3): one job per layout, so each srun
# spans its whole allocation. --dependency=singleton runs them one at a time so they don't share the network.
# cost: 1 SU per CPU + 2 per node per minute, ~32 SU/min for the 16-rank job; each job takes a few minutes.
# main mixes node generations (epyc + HDR100/HDR200 InfiniBand, xeon-4116 + FDR56), and one fitted model only
# describes one of them, so every job is pinned to one node feature. The default, fdr56, is the network of the
# P100 (and V100) GPU nodes the training runs use. Override with CONSTRAINT=<feature>, e.g. CONSTRAINT='epyc-7513&hdr200'.
set -euo pipefail
account=$1
constraint=${CONSTRAINT:-fdr56}
test -x bin/bench_comm || { echo "build first: make cpu"; exit 1; }

# sub NAME NODES RANKS_PER_NODE -> results/comm/NAME.csv
sub() {
  sbatch --account="$account" --dependency=singleton --constraint="$constraint" \
    --nodes="$2" --ntasks-per-node="$3" scripts/slurm/bench_comm.sh "$1"
}

sub intra_np2      1 2
sub inter_np2      2 1
sub allreduce_np4  2 2
sub allreduce_np8  4 2
sub allreduce_np16 8 2
