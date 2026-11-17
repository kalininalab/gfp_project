#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-peak-rotation-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
raw=/data/users/akolchina/gfp_project/artificial_peak_rotation_seed42_46_runs
count=$(find "$raw/fits" -name metrics.json -type f | wc -l)
if [[ "$count" -ne 160 ]]; then
  echo "Expected 160 fits, found $count" >&2
  exit 1
fi
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.peak_rotation.finalize \
  --directory "$raw" --output /nethome/akolchina/gfp_project/results/artificial_peak_holdout_seed42_46
