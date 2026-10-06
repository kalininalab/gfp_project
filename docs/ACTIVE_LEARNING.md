# Active learning on cgreGFP

This experiment asks: **how does prediction of measured cgre fluorescence change
when the same active-learning procedure uses a different base model?** It is a
retrospective simulation: all fluorescence measurements already exist, but a
pool measurement becomes available for training only when that sequence is
queried. Acquisition functions never receive pool targets.

The primary result is **Pearson correlation on one fixed random 20% test**, using
deterministic evaluation as in the supervised manuscript comparison. The test is
selected first and is unavailable to training, validation and acquisition. All
models share its record IDs. The student's Pearson on the remaining acquisition
pool, using 100 MC-dropout passes, is retained as a diagnostic; that population
diverges between models and cannot support a paired model comparison.

## Files to read or reuse

| File | Purpose |
|---|---|
| [acquisition.py](../scripts/active_learning/acquisition.py) | Scores, cluster allocation and sequence diversity; no target labels or experiment paths |
| [run.py](../scripts/active_learning/run.py) | Data checks, model adapters, training, querying, evaluation and restart |
| [plot.py](../scripts/active_learning/plot.py) | Copyable `plot_pearson(rows, partition, ax)`; only CSV, NumPy and Matplotlib |
| [test_active_learning.py](../tests/test_active_learning.py) | Disjoint splits, query budgets, reference diversity parity, hidden features and uncertainty |

No imports from the external student repository are needed to run this pipeline.
The optional reference-parity test reads its functions if that checkout exists.

## Required data

Run commands from the repository root in the environment from
[environment.yml](../environment.yml). Prepare the common sequence table once
with `python scripts/prepare_data.py` if it does not already exist.

1. `data/processed/baseline_v1/sequences.csv`: one row per variant. This runner uses
   `record_id` (unique stable identifier), `gene` (`cgreGFP`), `sequence` (235 native
   amino-acid residues, canonical 20-letter alphabet, no stop token), and
   `target_log10` (already log10-transformed fluorescence). Do **not** log-transform
   this column again. Duplicate sequences/IDs and nonfinite targets are rejected.
   See [data provenance and exclusions](DATA_AND_BASELINES.md).
2. Jannis ESM additionally needs `esm_embeddings/cgreGFP_t30/` containing
   `manifest.json`, `records.csv`, and `residue_embeddings.npy`. These are frozen
   ESM-2 t30 150M features, shape `[24516, 235, 640]`, float16 on disk. They are
   full per-residue embeddings, not pooled means. Model training uses FP32.
   Generation commands are in [Running models](RUNNING_MODELS.md). The runner
   checks the dataset hash, sequence/record order, feature shape and feature hash.

The original student's `main.py` refers to different local paths that are not
present here. We use the existing project data preparation and its audited target
definition instead of recreating those missing inputs.

## What the student's code actually does

Audited reference files: `master_thesis_jaca00001/main.py`, `src/configs.py`,
`src/data_loading.py`, `src/model.py`, and `src/utls.py`. The local checkout is an
external reference, excluded from this repository.

1. In `main.py`, training is commented out and `use_al=False`; executing that
   script alone does not launch AL. Its explicit split settings are 1% initial
   training, approximately 10% validation and 10% unlabeled pool. About 79% of
   records are discarded. `test_size` means acquisition pool here, not an
   independent held-out test set. `configs.py` has different dataclass defaults;
   this runner uses the explicit `main.py` experiment sizes.
2. For AL, train the first model for at most `11 × 9 = 99` epochs, then warm-start
   for at most 11 epochs per round. Restore the best validation checkpoint each
   time. Optimizer state is restarted each round. Targets are standardized using
   the current training set's mean and sample standard deviation.
3. Compute deterministic last-hidden-layer features for labeled and pool records.
   Find each pool point's nearest Euclidean distance to a labeled point. Estimate
   uncertainty with the **variance**, not standard deviation, of 25 MC-dropout
   predictions. Independently min-max scale the two arrays with epsilon `1e-8`.
   The submitted code computes `0.38 × variance + 0.62 × distance`, but the
   manuscript specifies `0.38 × distance + 0.62 × variance`. Experiment 2 follows
   the manuscript equation. A normalized predicted mean is computed in the
   reference but is not used in the score.
4. Spectrally cluster pool features into two clusters with dense RBF affinity,
   gamma 1 and random state 0. Allocate one query per cluster, then distribute
   the remainder proportionally to cluster size, using largest remainders.
5. Per cluster, shortlist the highest-scoring `5 × cluster_budget` candidates.
   The intended diversity rule starts with the highest-scoring sequence, then
   greedily adds the farthest sequence using 3-mer overlap. Its denominator is
   `2 × (sequence_length − 2) − intersection_size`; because repeated 3-mers
   are collapsed into sets, this is **not exact Jaccard distance**. However, the
   actual caller passes `base_dataset[i][0]`, an OHE tensor, instead of a sequence
   string. Tensor slices in Python sets are hashed by object identity, so distinct
   candidates' intersections are empty and all distances tie. On the normal
   cgre input this reduces selection to the first `cluster_budget` shortlisted
   indices, disabling the intended diversity. **Our pilot corrects this caller
   bug by passing amino-acid strings**, while retaining the distance formula.
6. Acquire 96 measurements per round, 960 in total over 10 acquisitions. Randomly
   put `max(1, floor(0.05 × 96)) = 4` into validation and 92 into training.
   Acquired validation examples consume measurement budget too. They never
   enter the training subset or the nearest-labeled-distance reference set.
7. Remove all queried examples from the pool, then evaluate the current model
   on the remainder. Those new labels are used in the **next** fit. Thus round 0
   already excludes the first 96 queried points, although its model has only
   seen the initial training set. Round 10 fits the last acquired training points
   and performs no further acquisition. There are 11 fits and 10 queries.

Training uses AdamW (`lr=0.00063`, `weight_decay=0.000023`), cosine annealing,
effective batch size 1024, gradient norm clipping at 1 and patience 20. Training
Huber delta is 1; validation Huber delta is 1.35. These differ in the source and
remain different here. Despite the `new_query_weight=7` setting, the reference
overwrites all weights with ones. We preserve **uniform weights** rather than
silently activating a different loss.

## What changes, and what stays fixed

Only the base model/representation varies within the comparison. Split, budget,
acquisition rule, training loss and optimizer recipe are common.

| CLI model | Input | Last hidden features | MC uncertainty |
|---|---|---|---|
| `aubin_1_10_1` | Positional OHE | 10 sigmoid activations | Zero: no dropout |
| `mlp_small` | Positional OHE | 64 ReLU activations | Zero: no dropout |
| `mlp_deep` | Positional OHE | 32 ReLU activations | Zero: no dropout |
| `CNN_Jannis_OHE` | Native per-residue OHE | 128 features | MC-dropout variance |
| `CNN_Jannis_ESM` | Frozen per-residue ESM | 128 features | MC-dropout variance |

These are the **existing benchmark architectures**. In particular, both Jannis
models use the repository's `CNN_Jannis` (4 convolutions and its existing head),
not the student's separate AL architecture defaults (5 convolutions, different
head and dropout). Aubin/MLP do not acquire extra dropout layers. Their selection
therefore uses learned distance, clustering and sequence diversity, with a zero
uncertainty component. This is a material limitation of interpreting the model
comparison as a comparison of uncertainty estimates.

Explicit implementation differences from the reference:

- Seed 42 and saved record-level assignments replace unseeded splits.
- The manuscript's random 20% test is frozen first. The student's 1% initial
  train, 10% validation and 10% acquisition pool are sampled only from the other
  80%; the remaining 59% is unused. This preserves a tractable spectral pool.
- The common prepared cgre dataset replaces absent student-specific input files.
- Training uses FP32 and memory microbatches (default 64, accumulated to 1024)
  instead of CUDA FP16 and physical batches of 1024. This bounds memory and works
  on CPU; dropout random-number consumption differs from the original batching.
- Best weights are copied with `.clone()` to avoid CPU tensor aliasing.
- The diversity function receives sequence strings, fixing the reference caller
  that supplied OHE tensors and therefore did not actually compare 3-mers. This
  can materially change queried records; this pilot is not a byte-for-byte
  reproduction of the buggy source trajectory.
- Cluster allocations cannot exceed cluster capacity; leftover slots are filled
  so exactly the requested number of unique examples is acquired.
- Dense models without dropout skip redundant repeated forward passes and return
  exactly zero MC variance.
- The **primary** fixed test comprises 20% of all records and uses the same split
  as Experiment 1. Its predictions use deterministic evaluation, are never used
  for tuning or selection, and are stored as `fixed_test`. `remaining_pool`
  retains MC100 solely as a reference-code diagnostic.

The additional holdout comes from a dataset already used in earlier project
experiments: this is exploratory validation, not a new external validation set.
There is no random-acquisition control in this first model-only comparison; it
cannot establish that AL outperforms random sampling.

## Run and restart

```bash
# Scientific checks
python -m unittest discover -s tests -p 'test_active_learning.py' -v

# One full trajectory, original budget and epoch schedule
python -m scripts.active_learning.run --model aubin_1_10_1 --seed 42
python -m scripts.active_learning.run --model mlp_small --seed 42
python -m scripts.active_learning.run --model mlp_deep --seed 42
python -m scripts.active_learning.run --model CNN_Jannis_OHE --seed 42 --device cuda
python -m scripts.active_learning.run --model CNN_Jannis_ESM --seed 42 --device cuda

# Regenerate tables and all three vector/raster plot formats
python -m scripts.active_learning.plot
```

Use `--seed 43` and `--seed 44` for additional matched initial splits. One seed is
a pilot, not an uncertainty estimate. Repeating the same command resumes at the
last completed round and restores model weights and RNG states. An existing run
with different source hashes or settings is rejected: choose a new
`--directory`. Only load `resume.pt` files generated by your own trusted runs;
they contain Python/NumPy RNG state and require pickle deserialization.

On our HTCondor cluster, the submit files launch the five seed-42 trajectories
and one lightweight result-finalization job:

```bash
mkdir -p results/active_learning_cgre_fixed_test/condor
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/active_learning_cpu.sub'
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/active_learning_gpu.sub'
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/active_learning_finish.sub'
```

The wrappers contain our cluster paths and interpreter; adapt them elsewhere.
Check the queue before submitting to avoid duplicates. The corrected fixed-test
pilot completed as CPU cluster **63668** and GPU cluster **63669** on 2026-10-05;
finish cluster **63670** monitored the run.
The lightweight finish job refreshes figures as new rounds arrive, validates
completion and exits. It times out with an error after 24 hours if a model job
fails or remains unfinished; inspect the model logs in that case.

For a short **software smoke test**, use a separate directory:

```bash
python -m scripts.active_learning.run --model aubin_1_10_1 \
  --rounds 1 --budget 8 --epochs 1 --directory /tmp/gfp-al-smoke
```

Epoch overrides, smaller budgets and incomplete trajectories must not be mixed
into scientific result figures. Spectral clustering has quadratic memory cost;
do not silently expand the pool to the whole dataset. Full CNN trajectories
require GPU time, especially MC100 evaluation. ESM data remain memory-mapped;
only batches move to the GPU.

## Outputs and plotting in a notebook

Each `results/active_learning_cgre_fixed_test/seed_<seed>/<model>/` contains:

- `protocol.json`: input/code hashes, versions, settings, initial counts.
- `splits.csv`: all record IDs and initial partitions, including unused records.
- `queries.csv`: record ID, selection round, first subsequent fit round,
  train/validation destination, acquisition score and MC variance.
- `history.csv`: training/validation losses per epoch and round.
- `predictions_round_XX.csv`: individual predictions, targets and population.
- `metrics.csv`: Pearson, Spearman, R², RMSE and counts at every round. Blank
  Pearson means undefined correlation for a constant array, not zero.
- `resume.pt`: latest fitted model, target scaling, next-round memberships, RNGs.
- `complete.json`: written only after the trajectory completes.

The plotting command merges available metrics into `summary.csv`. It deliberately
also supports monitoring partial runs; check that all five models have rounds
0–10 and completion markers before interpreting a final comparison.
It also writes `RESULTS.md`, checks protocol/split compatibility and labels
incomplete figures as preliminary. Use `--require-complete` for final figures;
this rejects missing models/rounds and epoch-overridden smoke tests.

```python
import csv
from scripts.active_learning.plot import plot_pearson

with open('results/active_learning_cgre_fixed_test/summary.csv') as handle:
    rows = list(csv.DictReader(handle))
fig, ax = plot_pearson(rows)  # primary: shared fixed 20% test
ax.set_title('cgreGFP active learning')
fig.savefig('cgre_active_learning.pdf', bbox_inches='tight')
```

The x-axis counts **all acquired labels available before the fit**, including
those allocated to validation. Initial training labels and the initial ~10%
validation set are not included in this axis; actual counts are in `metrics.csv`.
With multiple seeds, shading is sample SD, not a confidence interval.
