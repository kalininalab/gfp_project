#!/bin/bash
set -euo pipefail
exec /nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.protein_saturation.run --directory "$1" --gene "$2" --seed "$3" --model "$4" --device "$5"
