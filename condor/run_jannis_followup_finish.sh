#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export MPLCONFIGDIR="$(mktemp -d /tmp/gfp-followup-final.XXXXXX)"
trap 'rm -rf "$MPLCONFIGDIR"' EXIT
deadline=$((SECONDS + 43200))
while (( SECONDS < deadline )); do
  missing=0
  for seed in 42 43 44 45 46; do
    for model in aubin_1_10_1 mlp_small mlp_deep CNN_Jannis_OHE CNN_Jannis_ESM; do
      [[ -f results/paper_reproduction_cgre_80_20/seed_${seed}/${model}/complete.json ]] || missing=$((missing + 1))
    done
    [[ -f results/experiment_2_training_size/seed_${seed}/complete.json ]] || missing=$((missing + 1))
    [[ -f results/experiment_3_sampling/seed_${seed}/fancy/complete.json ]] || missing=$((missing + 1))
  done
  if (( missing == 0 )); then
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.paper_reproduction.finalize
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.paper_reproduction.plot
    /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.jannis_experiments.finalize
    echo "All follow-up experiments finalized."
    exit 0
  fi
  echo "Waiting for ${missing} completion markers..."
  sleep 120
done
echo "Timed out waiting for follow-up experiments" >&2
exit 1
