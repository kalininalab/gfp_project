#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-target-adapt-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 172800))
while (( SECONDS < deadline )); do
  count=$(find -H results/target_adaptation_seed42_46/fits -name complete.json -type f | wc -l)
  if [[ "$count" -eq 90 ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.target_adaptation.finalize
    exit 0
  fi
  echo "Waiting for target-adaptation jobs: ${count}/90"
  sleep 120
done
exit 1
