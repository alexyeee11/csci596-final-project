#!/bin/bash
#SBATCH --job-name=bench-gemm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/bench_gemm.sh
# runs the Task 1 + Task 2 checks (spills, tests, regtile training, GEMM benchmark) in one job
set -euo pipefail
source scripts/env.sh
make clean && make gpu
echo "== regtile register/spill report (sm_60) =="
nvcc -O2 -gencode arch=compute_60,code=sm_60 -Isrc -Xptxas -v -c src/kernels/gemm.cu -o /dev/null
echo "== tests =="
./bin/test_gemm
./bin/test_mlp
echo "== train sanity (regtile, 1 rank) =="
srun --mpi="${SRUN_MPI:-pmix}" --ntasks=1 ./bin/train --synthetic --steps 50 --batch 64 --hidden 256 --layers 2 --lr 0.05 --gemm regtile
echo "== bench_gemm =="
mkdir -p results/gemm
./bin/bench_gemm > results/gemm/p100.csv
wc -l results/gemm/p100.csv
