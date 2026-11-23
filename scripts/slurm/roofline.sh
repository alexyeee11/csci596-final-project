#!/bin/bash
#SBATCH --job-name=roofline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# usage: mkdir -p logs && sbatch --account=<class account> scripts/slurm/roofline.sh [LABEL] [IMPLS...]
# E2 roofline: per impl and shape, DRAM bytes from ncu, time from bench_gemm, appended to results/roofline/v100.csv.
# LABEL tags every row (default: the impl). IMPLS default: all 6. ~6 min of V100 (24 SU) plus ncu replays.
# Counters blocked (ERR_NVGPUCTRPERM): analytic rows (min bytes), no ncu. Keep the usage text above in sync.
set -euo pipefail
source scripts/env.sh
LABEL=${1:-}
[ $# -gt 0 ] && shift
IMPLS=${*:-naive tiled regtile cublas wmma cublas_fp16}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$GPU"
make clean && make gpu
mkdir -p results/roofline
OUT=results/roofline/v100.csv
[ -f $OUT ] || echo label,impl,gpu,M,N,K,flops,dram_bytes,ncu_time_s,bench_time_s,source > $OUT
TMP=$(mktemp)
trap 'rm -f $TMP $TMP.out' EXIT
METRICS=dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum

# counter check on a tiny case
rc=0
ncu --metrics gpu__time_duration.sum --csv --log-file $TMP ./bin/bench_gemm cublas 256 256 256 0 0 > $TMP.out 2>&1 || rc=$?
BLOCKED=0
if grep -q ERR_NVGPUCTRPERM $TMP $TMP.out; then
  BLOCKED=1
  echo "roofline: counters blocked (ERR_NVGPUCTRPERM)"
elif [ $rc -ne 0 ]; then
  cat $TMP.out; exit $rc
fi
rm -f $TMP.out

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
RUN=
for i in $IMPLS; do
  case $i in wmma|cublas_fp16) [ "${CC%%.*}" -ge 7 ] || { echo "skip $i: needs sm_70+ (GPU is $CC)"; continue; } ;; esac
  RUN="$RUN $i"
done

for i in $RUN; do
  case $i in wmma|cublas_fp16) es=2 ;; *) es=4 ;; esac
  for s in "4096 4096 4096" "512 4096 4096"; do
    set -- $s
    M=$1 N=$2 K=$3
    bench=$(./bin/bench_gemm $i $M $N $K 0 0 | cut -d, -f7)
    if [ $BLOCKED = 1 ]; then
      bytes=$(( (M * K + K * N) * es + M * N * 4 )) nt= src=analytic
    else
      : > $TMP
      ncu --profile-from-start off --metrics $METRICS --csv --print-units base --log-file $TMP ./bin/bench_gemm $i $M $N $K 0 0 > /dev/null || { cat $TMP; exit 1; }
      read -r bytes nt < <(python3 -I scripts/ncu_sum.py $TMP)
      src=ncu
    fi
    echo "${LABEL:-$i},$i,$GPU,$M,$N,$K,$((2 * M * N * K)),$bytes,$nt,$bench,$src" | tee -a $OUT
  done
done

# Memory Workload / Warp State sections drive Tasks 3 and 4
case " $RUN " in *" wmma "*) [ $BLOCKED = 1 ] || ncu --set full --profile-from-start off ./bin/bench_gemm wmma 4096 4096 4096 0 0 > results/roofline/ncu_full_${LABEL:-wmma}.txt ;; esac
