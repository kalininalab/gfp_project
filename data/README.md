# GFP datasets

## Which folder should I use?

- [`raw/`](raw/) contains the versioned experimental input tables and reference
  FASTA sequences. Share this folder when someone needs the original records or
  wants to audit preprocessing.
- `processed/baseline_v1/` is the canonical model-ready dataset used by current
  experiments. It is generated locally and ignored by Git because
  `sequences.csv` is approximately 88 MB.
- Other committed files under `processed/` are older intermediate tables kept
  for provenance and legacy analyses. They are not the canonical input for new
  experiments.

## Build the model-ready dataset

From the repository root:

```bash
conda env create -f environment.yml
conda activate gfp-benchmark
python scripts/prepare_data.py
```

This creates:

- `processed/baseline_v1/sequences.csv`: sequences, canonical genotypes,
  `target_log10`, frozen splits, identifiers and source provenance;
- `processed/baseline_v1/exclusions.csv`: rejected rows and reasons;
- `processed/baseline_v1/manifest.json`: hashes, versions, counts and overlap
  checks.

Complete column definitions, scaling rules, landscape counts and validation
checks are in [Data and baselines](../docs/DATA_AND_BASELINES.md).
