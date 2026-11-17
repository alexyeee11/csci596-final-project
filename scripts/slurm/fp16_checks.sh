#!/bin/bash
#SBATCH --job-name=fp16-checks
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=anakano_429 scripts/slurm/fp16_checks.sh
#   P100 run (FP32 cases + skip line): sbatch --account=anakano_429 --gres=gpu:p100:1 scripts/slurm/fp16_checks.sh
# V100s are assumed to be in the gpu partition: check with `sinfo -o "%P %G"` (add --partition=... if not)
set -euo pipefail
source scripts/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
make clean && make gpu
echo "== wmma register/spill report (sm_70; want 0 bytes spill for gemm_wmma_k) =="
nvcc -O2 -gencode arch=compute_70,code=sm_70 -Isrc -Xptxas -v -c src/kernels/gemm.cu -o /dev/null
echo "== tests =="
./bin/test_gemm
