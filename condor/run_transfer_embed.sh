#!/bin/bash
set -euo pipefail
cd /nethome/akolchina/gfp_project
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.esm_benchmark.embed --gene "$1" --output "esm_embeddings/${1}_t30" --revision a695f6045e2e32885fa60af20c13cb35398ce30c
