#!/bin/bash
#SBATCH --job-name=fp16-checks
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=anakano_429 scripts/slurm/fp16_checks.sh
#   P100 run (FP32 cases + skip lines; the --fp16 runs then refuse with a message):
#     sbatch --account=anakano_429 --gres=gpu:p100:2 scripts/slurm/fp16_checks.sh
# Tests run as plain ./bin/...; bin/train runs on one rank via srun (as in bench_gemm.sh); test_dp.sh uses both GPUs.
# V100s are assumed to be in the gpu partition: check with `sinfo -o "%P %G"` (add --partition=... if not)
set -uo pipefail
source scripts/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
make clean && make gpu || exit 1
echo "== wmma register/spill report (sm_70; want 0 bytes spill for gemm_wmma_k) =="
nvcc -O2 -gencode arch=compute_70,code=sm_70 -Isrc -Xptxas -v -c src/kernels/gemm.cu -o /dev/null
echo "== tests =="
status=0
./bin/test_gemm || status=1
./bin/test_layers || status=1
./bin/test_mlp || status=1
train1() { srun --mpi="$SRUN_MPI" --ntasks=1 ./bin/train "$@"; }
echo "== rejected combinations (want 3 refusals) =="
train1 --synthetic --steps 1 --gemm wmma 2>&1 | tail -1
train1 --synthetic --steps 1 --fp16 --gemm tiled 2>&1 | tail -1
train1 --synthetic --steps 1 --fp16 --gemm regtile 2>&1 | tail -1
echo "== acceptance 1: synthetic --fp16 (want test_acc >= 0.90) =="
train1 --synthetic --steps 50 --batch 64 --hidden 256 --layers 2 --lr 0.05 --fp16 | grep test_acc |
  awk '{ print } $3 >= 0.90 { ok = 1 } END { print ok ? "synthetic fp16: PASS" : "synthetic fp16: FAIL"; exit !ok }' ||
  status=1
echo "== acceptance 2: CIFAR-10 5 epochs, FP32 vs FP16 (want |diff| <= 0.02) =="
CIFAR=${CIFAR:-$HOME/data/cifar-10-batches-bin}
CARGS="--data $CIFAR --epochs 5 --batch 256 --hidden 1024 --layers 2 --lr 0.02"
echo "-- fp32"; f32=$(train1 $CARGS | grep test_acc | tee /dev/stderr | tail -1 | awk '{ print $3 }')
echo "-- fp16"; f16=$(train1 $CARGS --fp16 | grep test_acc | tee /dev/stderr | tail -1 | awk '{ print $3 }')
awk -v a="$f32" -v b="$f16" 'BEGIN {
  d = a - b; if (d < 0) d = -d
  ok = a != "" && b != "" && d <= 0.02
  printf "cifar fp32 test_acc %s fp16 test_acc %s diff %.4f: %s\n", a, b, d, ok ? "PASS" : "FAIL"
  exit !ok }' || status=1
echo "== acceptance 3: 2-rank data parallel, FP16 (want 4 PASS) =="
scripts/test_dp.sh 2 --fp16 || status=1
exit $status
