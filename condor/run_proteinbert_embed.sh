#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
runtime="${TMPDIR:-/tmp}/proteinbert-runtime"
mkdir -p "$runtime"
/nethome/akolchina/miniconda3/envs/combi/bin/python -m pip install --quiet --no-deps --target "$runtime" \
  /data/users/akolchina/gfp_project_proteinbert/wheels/*.whl
export PYTHONPATH="$runtime:/data/users/akolchina/gfp_project_proteinbert/python_packages"
export TF_CPP_MIN_LOG_LEVEL=1
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.hurdle_predictor.proteinbert_embed
