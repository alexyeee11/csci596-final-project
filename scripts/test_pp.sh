#!/bin/bash
# usage: [MBS="1 2 4"] scripts/test_pp.sh P [extra train args...]
#   GPipe with P stages on P ranks must match the 1-rank loss curve, for each micro-batch count in MBS.
set -euo pipefail
P=${1:-2}
shift || true
ARGS="--synthetic --steps 20 --batch 64 --hidden 256 --layers 3 --lr 0.05"
launch() {
  local n=$1; shift
  if [ -n "${SLURM_JOB_ID:-}" ]; then srun --mpi="${SRUN_MPI:-pmix}" --ntasks="$n" "$@"
  else mpirun --oversubscribe -np "$n" "$@"; fi
}
ref=$(mktemp); out=$(mktemp)
trap 'rm -f "$ref" "$out"' EXIT
launch 1 ./bin/train $ARGS "$@" | awk '$1 == "step" { print $4 }' > "$ref"
status=0
for m in ${MBS:-1 2 4}; do
  # a crashed or rejected run leaves $out short, which the check below reports as FAIL
  launch "$P" ./bin/train $ARGS "$@" --pp "$P" --microbatches "$m" | awk '$1 == "step" { print $4 }' > "$out" || true
  paste "$ref" "$out" | awk -v p="$P" -v mb="$m" '
    NF < 2 { next }
    { d = $1 - $2; if (d < 0) d = -d; r = d / ($1 > 1e-6 ? $1 : 1e-6); if (r > x) x = r; n++ }
    END {
      if (n != 20 || x > 1e-3) { printf "test_pp: FAIL (P=%s, M=%s, rows=%d maxrel %g)\n", p, mb, n, x; exit 1 }
      printf "test_pp: PASS (P=%s, M=%s, maxrel %g)\n", p, mb, x
    }' || status=1
done
exit $status
