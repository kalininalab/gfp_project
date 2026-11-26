#!/bin/bash
set -euo pipefail
MODEL="$1"
SEED="$2"
STRATEGY="$3"
DEVICE="$4"
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.amac_active_learning.run \
  --model "$MODEL" --seed "$SEED" --strategy "$STRATEGY" --device "$DEVICE"
