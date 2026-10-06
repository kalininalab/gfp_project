#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 172800))
while (( SECONDS < deadline )); do
  count=$(find -H results/transfer_cgreGFP_seed42_46_runs/fits -name metrics.json -type f | wc -l)
  if [[ "$count" -eq 160 ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_benchmark.plot --directory results/transfer_cgreGFP_seed42_46_runs --output results/transfer_cgreGFP_seed42_46
    exit 0
  fi
  echo "Waiting for transfer fits: ${count}/160"
  sleep 120
done
echo "Timed out waiting for 160 transfer fits" >&2
exit 1
