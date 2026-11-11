#!/bin/bash
# usage: [ALGOS="ring mpi"] scripts/test_dp.sh NP [extra train args...]
#   N-rank data-parallel loss curve must match 1 rank, for each all-reduce algorithm in ALGOS.
#   Extra args go to every run, except --overlap/--bucket-mb, so the reference is plain 1-rank training.
set -euo pipefail
NP=${1:-2}
shift || true
ARGS="--synthetic --steps 20 --batch 64 --hidden 256 --layers 2 --lr 0.05"
extra=("$@") ref_extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --overlap) ;;
    --bucket-mb) shift ;;
    *) ref_extra+=("$1") ;;
  esac
  shift || true
done
launch() {
  local n=$1; shift
  if [ -n "${SLURM_JOB_ID:-}" ]; then srun --mpi="${SRUN_MPI:-pmix}" --ntasks="$n" "$@"
  else mpirun --oversubscribe -np "$n" "$@"; fi
}
ref=$(mktemp); out=$(mktemp)
trap 'rm -f "$ref" "$out"' EXIT
# ${a[@]+"${a[@]}"} expands an empty array safely under set -u in older bash
launch 1 ./bin/train $ARGS ${ref_extra[@]+"${ref_extra[@]}"} | awk '$1 == "step" { print $4 }' > "$ref"
status=0
for ar in ${ALGOS:-naive tree ring mpi}; do
  # a crashed run leaves $out short, which the check below reports as FAIL
  launch "$NP" ./bin/train $ARGS ${extra[@]+"${extra[@]}"} --allreduce "$ar" | awk '$1 == "step" { print $4 }' > "$out" || true
  paste "$ref" "$out" | awk -v ar="$ar" -v np="$NP" '
    NF < 2 { next }
    { d = $1 - $2; if (d < 0) d = -d; r = d / ($1 > 1e-6 ? $1 : 1e-6); if (r > m) m = r; n++ }
    END {
      if (n != 20 || m > 1e-3) { printf "test_dp %s np=%s: FAIL (rows=%d maxrel=%g)\n", ar, np, n, m; exit 1 }
      printf "test_dp %s np=%s: PASS (maxrel=%g)\n", ar, np, m
    }' || status=1
done
exit $status
