# Cross-protein labelled-pool size experiment

This workflow trains a model from scratch on a fixed cgreGFP source set plus a
cumulative, randomly ordered labelled set from ppluGFP or amacGFP. It evaluates
every fit on unchanged held-out tests for cgreGFP and the target protein. It is
a supervised pool-size experiment and does not use an active-learning
acquisition function.

## Code map

- [`scripts/pool_mixing/prepare.py`](../scripts/pool_mixing/prepare.py) freezes
  the source subset, target order, validation schedule, test policy and protein
  alignment, then creates an HTCondor DAG.
- [`scripts/pool_mixing/run.py`](../scripts/pool_mixing/run.py) trains and
  evaluates one `(target, cumulative target size, seed, model)` fit.
- [`scripts/pool_mixing/finalize.py`](../scripts/pool_mixing/finalize.py)
  validates completed fits and creates metric tables, curves, distributions and
  the report.
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
