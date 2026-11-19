#!/bin/bash
#SBATCH --job-name=bench-fp16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/bench_fp16.sh
# Task 4 (E1 on V100, E9): bench_gemm CSV, then 1-V100 step time, full model (default hidden 4096, 3 layers),
# batch 512: FP32 regtile/cublas vs FP16 wmma/cublas. Median of steps 5-29 goes to results/fp16/summary.txt.
set -euo pipefail
source scripts/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
make clean && make gpu
mkdir -p results/gemm results/fp16
./bin/bench_gemm > results/gemm/v100.csv
wc -l results/gemm/v100.csv   # want 55
# same style as sweep_scaling.sh: synthetic data is 4096 samples (8 steps/epoch), --steps stops the run at 30
BASE="--synthetic --epochs 100 --steps 30 --batch 512"
run() { srun --mpi="$SRUN_MPI" --ntasks=1 ./bin/train $BASE "${@:2}" > "results/fp16/$1.log"; }
run fp32_regtile --gemm regtile
run fp32_cublas  --gemm cublas
run fp16_wmma    --fp16 --gemm wmma
run fp16_cublas  --fp16 --gemm cublas
median() { sort -n | awk '{ v[NR] = $1 } END { print NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2 }'; }
{
  echo "median over steps 5-29, 1 V100, batch 512, seconds"
  printf "%-14s %10s %10s\n" run compute_s step_s
  for r in fp32_regtile fp32_cublas fp16_wmma fp16_cublas; do
    c=$(awk '$1 == "step" && $2 >= 5 && $2 <= 29 { print $6 }' "results/fp16/$r.log" | median)
    s=$(awk '$1 == "step" && $2 >= 5 && $2 <= 29 { print $10 }' "results/fp16/$r.log" | median)
    printf "%-14s %10s %10s\n" "$r" "$c" "$s"
  done
} | tee results/fp16/summary.txt
