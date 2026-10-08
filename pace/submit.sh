#!/bin/bash
# Submit the pipeline as a dependency chain, from the project directory on ICE:
#
#   bash pace/submit.sh                    # prepare -> e0 -> smoke
#   bash pace/submit.sh --with-baselines   # ... -> full baselines (several GPU-hours)
#
# Each job starts only if the previous one succeeded (afterok); if E0 fails, nothing
# after it runs. GPU=h200 (or another ICE type: a100, l40s, ...) overrides the default h100;
# ACCOUNT=<charge account> adds -A if your allocation needs one.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

GPU="${GPU:-h100}"
extra=()
[ -n "${ACCOUNT:-}" ] && extra+=(-A "$ACCOUNT")
gpu=(--gres="gpu:${GPU}:1")

j1=$(sbatch --parsable "${extra[@]}" pace/prepare.sbatch)
j2=$(sbatch --parsable "${extra[@]}" "${gpu[@]}" --dependency=afterok:"$j1" pace/e0.sbatch)
j3=$(sbatch --parsable "${extra[@]}" "${gpu[@]}" --dependency=afterok:"$j2" pace/smoke.sbatch)
echo "prepare $j1 -> e0 $j2 -> smoke $j3"
if [ "${1:-}" = "--with-baselines" ]; then
    j4=$(sbatch --parsable "${extra[@]}" "${gpu[@]}" --dependency=afterok:"$j3" pace/baselines.sbatch)
    echo "-> baselines $j4"
fi
echo "watch: squeue -u \$USER ; logs in logs/, results in results/"
