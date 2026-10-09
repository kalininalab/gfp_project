#!/bin/bash
set -euo pipefail

ROOT=/data/users/akolchina/gfp_project_kermut
MPNN="$ROOT/software/ProteinMPNN"
mkdir -p "$ROOT/features/proteinmpnn_alphafold"
exec /nethome/akolchina/miniconda3/envs/combi/bin/python "$MPNN/protein_mpnn_run.py" \
  --pdb_path "$ROOT/inputs/AF-D7PM05-F1-model_v6.pdb" \
  --save_score 1 \
  --conditional_probs_only 1 \
  --num_seq_per_target 10 \
  --batch_size 1 \
  --out_folder "$ROOT/features/proteinmpnn_alphafold" \
  --seed 37
