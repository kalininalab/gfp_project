#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-al-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 172800))
while (( SECONDS < deadline )); do
  count=$(find -H results/transfer_al_cgreGFP_seed42_46/fits -name complete.json -type f | wc -l)
  if [[ "$count" -eq 160 ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.plot
    exit 0
  fi
  echo "Waiting for transfer-AL trajectories: ${count}/160"
  sleep 120
done
echo "Timed out waiting for transfer-AL trajectories" >&2
exit 1
