#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-peak-holdout-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
raw=/data/users/akolchina/gfp_project/artificial_peak_holdout_seed42_46_runs
for _ in $(seq 1 180); do
  count=$(find "$raw/fits" -name metrics.json -type f | wc -l)
  if [[ "$count" -eq 120 ]]; then
    exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.peak_holdout.finalize \
      --directory "$raw" --output /nethome/akolchina/gfp_project/results/artificial_peak_holdout_seed42_46
  fi
  sleep 120
done
echo "Timed out waiting for 120 fits" >&2
exit 1
