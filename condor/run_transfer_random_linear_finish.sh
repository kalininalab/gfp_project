#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-random-linear-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 172800))
while (( SECONDS < deadline )); do
  count=$(find -H results/transfer_random_linear_cgreGFP_seed42_46/fits -name complete.json -type f | wc -l)
  if [[ "$count" -eq 40 ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.finalize_sampling --model aubin_linear --random results/transfer_random_linear_cgreGFP_seed42_46
    exit 0
  fi
  sleep 120
done
exit 1
