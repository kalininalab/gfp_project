#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-transfer-random-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 172800))
while (( SECONDS < deadline )); do
  count=$(find -H results/transfer_random_cgreGFP_seed42_46/fits -name complete.json -type f | wc -l)
  if [[ "$count" -eq 40 ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.transfer_active_learning.plot_followups
    exit 0
  fi
  echo "Waiting for random transfer trajectories: ${count}/40"
  sleep 120
done
echo "Timed out waiting for random transfer trajectories" >&2
exit 1
