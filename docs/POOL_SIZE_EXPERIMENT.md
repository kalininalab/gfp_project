# Cross-protein labelled-pool size experiment

This workflow trains a model from scratch on a fixed cgreGFP source set plus a
cumulative, randomly ordered labelled set from ppluGFP or amacGFP. It evaluates
every fit on unchanged held-out tests for cgreGFP and the target protein. It is
a supervised pool-size experiment and does not use an active-learning
acquisition function.

The matched acquisition extension keeps this entire protocol fixed and changes
only how target-training records are chosen. Its publication figures overlay
the random curves with acquisition curves on the same axes.

## Code map

- [`scripts/pool_mixing/prepare.py`](../scripts/pool_mixing/prepare.py) freezes
  the source subset, target order, validation schedule, test policy and protein
  alignment, then creates an HTCondor DAG.
- [`scripts/pool_mixing/run.py`](../scripts/pool_mixing/run.py) trains and
  evaluates one `(target, cumulative target size, seed, model)` fit.
- [`scripts/pool_mixing/finalize.py`](../scripts/pool_mixing/finalize.py)
  validates completed fits and creates metric tables, curves, distributions and
  the report.
- [`scripts/pool_mixing_acquisition/prepare.py`](../scripts/pool_mixing_acquisition/prepare.py)
  creates dependent round-by-round DAGs for the matched acquisition arm.
- [`scripts/pool_mixing_acquisition/run_round.py`](../scripts/pool_mixing_acquisition/run_round.py)
  retrains one primary model, evaluates the frozen tests, estimates uncertainty
  and selects the next target plate.
- [`scripts/pool_mixing_acquisition/finalize.py`](../scripts/pool_mixing_acquisition/finalize.py)
  overlays random and acquisition results in the existing publication figures.
- [`scripts/transfer_benchmark/models.py`](../scripts/transfer_benchmark/models.py)
  defines the aligned positional-OHE wrappers.
- [`scripts/aubin_model.py`](../scripts/aubin_model.py) defines Aubin 1–10–1;
  [`scripts/baseline_models.py`](../scripts/baseline_models.py) defines the MLP.
- [`scripts/train_aubin.py`](../scripts/train_aubin.py) contains their shared
  training and early-stopping loop.

## Set up once

Run commands from the repository root:

```bash
conda env create -f environment.yml
conda activate gfp-benchmark
python scripts/prepare_data.py
```

The last command creates the ignored processed table
`data/processed/baseline_v1/sequences.csv` from the versioned raw inputs.

## Recreate the 7,350-cgreGFP protocol

Choose a new output directory. `prepare.py` deliberately refuses to overwrite
an existing frozen protocol.

```bash
python -m scripts.pool_mixing.prepare \
  --raw /data/users/USERNAME/gfp_project/pool_size_cgre7350_runs \
  --base-train-n 7350 \
  --models mlp_small aubin_1_10_1 \
  --dag condor/pool_size_cgre7350_student.dag
```

This freezes five seeds (42–46), both targets, plate sizes 9, 48, 96, 192 and
384, and all cumulative target sizes needed to end near 1,920 records. The
resulting `protocol.json`, `base_pools/`, `target_pools/` and `alignment.json`
fully specify the experiment.

## Run one fit

This example trains Aubin 1–10–1 on 7,350 cgreGFP records plus the first 960
ppluGFP records in the frozen seed-42 order:

```bash
python -m scripts.pool_mixing.run \
  --target ppluGFP \
  --n-target 960 \
  --seed 42 \
  --model aubin_1_10_1 \
  --device cpu \
  --raw /data/users/USERNAME/gfp_project/pool_size_cgre7350_runs
```

Use `--model mlp_small` for the MLP. Each command creates a fresh model; model
weights are not continued from a previous target-pool size. Outputs include the
checkpoint, learning history, predictions on both frozen tests, metrics and a
completion manifest.

## Run the complete grid on the LSV cluster

The generated DAG uses the repository's CPU submit file and shared cluster
wrapper:

```bash
condor_submit_dag condor/pool_size_cgre7350_student.dag
```

The current Condor submit and wrapper files point to the shared checkout at
`/nethome/akolchina/gfp_project`. The Python commands above are portable; a
clone in another location needs its Condor `initialdir` and wrapper path
changed accordingly.

## Build the report after the full grid finishes

```bash
python -m scripts.pool_mixing.finalize \
  --raw /data/users/USERNAME/gfp_project/pool_size_cgre7350_runs \
  --output results/pool_size_cgre7350_student \
  --models mlp_small aubin_1_10_1
```

Published reference results are in
[`results/cross_protein_pool_size_cgre7350_seed42_46/`](../results/cross_protein_pool_size_cgre7350_seed42_46/).

## Matched acquisition experiment

The acquisition extension is run for both source sizes (14,709 and 7,350
cgreGFP training records), both target proteins, plate sizes 9, 48, 96, 192 and
384, seeds 42–46, and Aubin, MLP and CNN on OHE. At every round the primary
model is newly initialized and fitted on the complete cumulative labelled set,
exactly as in the random experiment.

The next plate is the highest-scoring set from the remaining frozen target
training pool:

```text
score = 0.62 × scaled hidden-space distance
      + 0.38 × scaled predictive variance
```

Aubin and MLP predictive variance comes from five fitted ensemble members.
CNN predictive variance comes from 25 MC-dropout predictions. Target
validation grows by the same frozen rule as in the random arm. Both
protein-specific tests are identical between arms and are never available to
training, early stopping or acquisition.

One acquisition round depends on the selected records from the preceding
round, so the generated DAG uses a separate dependency chain for each
`(source size, target, plate size, seed, model)` trajectory. Raw outputs and
Condor logs belong under `/data/users/USERNAME/`, outside the Git checkout.
