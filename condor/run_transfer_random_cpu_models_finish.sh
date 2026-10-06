#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-random-cpu-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 172800))
while (( SECONDS < deadline )); do
  aubin=$(find -H results/transfer_random_aubin_cgreGFP_seed42_46/fits -name complete.json -type f | wc -l)
  mlp=$(find -H results/transfer_random_mlp_cgreGFP_seed42_46/fits -name complete.json -type f | wc -l)
  if [[ "$aubin" -eq 40 && "$mlp" -eq 40 ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.finalize_sampling_cpu_models
    exit 0
  fi
  sleep 120
done
exit 1
