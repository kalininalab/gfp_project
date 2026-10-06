#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-ensemble-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 345600))
while (( SECONDS < deadline )); do
  ensemble=$(find -H results/transfer_al_ensemble_cpu_seed42_46/fits -name complete.json -type f | wc -l)
  aubin_random=$(find -H results/transfer_random_aubin_cgreGFP_seed42_46/fits -name complete.json -type f | wc -l)
  mlp_random=$(find -H results/transfer_random_mlp_cgreGFP_seed42_46/fits -name complete.json -type f | wc -l)
  if [[ "$ensemble" -eq 120 && "$aubin_random" -eq 40 && "$mlp_random" -eq 40 ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.finalize_sampling_cpu_models \
      --fancy results/transfer_al_ensemble_cpu_seed42_46 \
      --output results/transfer_ensemble_sampling_cpu_seed42_46
    exit 0
  fi
  sleep 120
done
exit 1
