#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
kind=$1
if [[ "$kind" == "proteins" ]]; then
  directory=results/target_adaptation_cnn_seed42_46
  expected=30
else
  directory=results/target_adaptation_peaks_cnn_seed42_46
  expected=10
fi
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-target-cnn-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 604800))
while (( SECONDS < deadline )); do
  count=$(find -H "$directory/fits" -name complete.json -type f | wc -l)
  if [[ "$count" -eq "$expected" ]]; then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.target_adaptation.finalize --directory "$directory"
    exit 0
  fi
  sleep 120
done
exit 1
