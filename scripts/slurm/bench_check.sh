#!/bin/bash
#SBATCH --job-name=bench-check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=anakano_429 scripts/slurm/bench_check.sh
#   P100 run (no FP16 rows: 37 lines, wmma must refuse with exit 2):
#     sbatch --account=anakano_429 --gres=gpu:p100:1 scripts/slurm/bench_check.sh
# Checks bench_gemm (real inputs, single-case mode) and test_gemm. Does not touch results/gemm/*.csv.
set -uo pipefail
source scripts/env.sh
nvidia-smi --query-gpu=name --format=csv,noheader
make clean && make gpu || exit 1
status=0
echo "== test_gemm =="
./bin/test_gemm || status=1
sm70=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | awk -F. '{ print ($1 >= 7) ? 1 : 0 }')
want=$((sm70 ? 55 : 37))
echo "== full sweep (want $want lines) =="
sweep=$(./bin/bench_gemm) || status=1
echo "$sweep"
got=$(echo "$sweep" | wc -l)
echo "lines: $got (want $want)"
[ "$got" -eq "$want" ] || { echo "line count: FAIL"; status=1; }
echo "== 4096^3 rows (compare with the zero-input rows in results/gemm/*.csv) =="
echo "$sweep" | grep -E '^[a-z0-9_]+,4096,4096,4096,0,0,' || status=1
echo "== single case: wmma 4096 4096 4096 0 0 =="
rc=0; out=$(./bin/bench_gemm wmma 4096 4096 4096 0 0) || rc=$?
echo "$out"
if [ "$sm70" = 1 ]; then
  [ "$rc" -eq 0 ] && [ "$(echo "$out" | wc -l)" -eq 1 ] && echo "single-case: PASS" || { echo "single-case: FAIL (exit $rc)"; status=1; }
else
  [ "$rc" -eq 2 ] && echo "wmma refusal exit 2: PASS" || { echo "wmma refusal: FAIL (exit $rc, want 2)"; status=1; }
fi
echo "== bad impl (want exit 2) =="
rc=0; ./bin/bench_gemm bogus 1 1 1 0 0 || rc=$?
[ "$rc" -eq 2 ] && echo "bogus exit 2: PASS" || { echo "bogus: FAIL (exit $rc)"; status=1; }
exit $status
