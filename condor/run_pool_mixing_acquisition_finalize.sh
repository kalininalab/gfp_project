#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.pool_mixing_acquisition.finalize \
  --reference-raw "$1" --acquisition-raw "$2" --output "$3"
