#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
/nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.pool_mixing.finalize \
  --raw "$1" --output "$3" --models aubin_1_10_1 mlp_small CNN_Jannis_OHE
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.pool_mixing_acquisition.finalize \
  --reference-raw "$1" --acquisition-raw "$2" --output "$3"
