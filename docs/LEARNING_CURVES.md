# Training-size experiments

We vary the number of training sequences for **Aubin 1–10–1, small MLP,
deep MLP, and Jannis OHE**. Colors remain blue, orange, purple, and rose.
All inputs use native one-hot coordinates: these cgre-derived sequences are
235 residues long. No ESM embeddings or ortholog alignment are involved.

## What is compared?

- Natural cgre → natural cgre, using the original ten CV folds.
- Artificial → artificial, artificial → natural, natural → artificial, and
  natural → natural, using paired source/target folds.

“Natural” means the original cgreGFP landscape, including mutants. “Artificial”
means the four approved peaks pooled: cgre132, cgre1338, cgre4111, cgre9708.
There are 24,516 natural and 26,902 artificial unique sequences, with no exact
sequence duplicates between these domains. The excluded cgre12minis file remains
excluded. Fluorescence targets retain the previous calibration and log10 scale.

Each source model is tested on **both** target domains. Natural → natural is
therefore the same result in both plot families, not another training run.

## Sampling and evaluation

**Training sizes:** 500, 1,000, 2,500, 5,000, 10,000, 15,000, 17,500.
The maximum fits every natural training fold (which has 17,651–17,652 available
training records). It is slightly smaller than the full-data main benchmark.

1. Reuse the original natural-cgre outer folds and inner validation assignment
   (seed 42). Preparation verifies existing split files when present; the same
   deterministic algorithm can regenerate them in a fresh checkout.
2. Make ten artificial outer folds, stratified by peak with seed 42. Reserve 20%
   of each outer training pool for source-validation, also stratified by peak.
   These are new CV partitions, not the earlier artificial 60/20/20 holdout.
3. Keep **4,413 source-validation examples per fold** at every training size and
   for every model. Natural uses the original validation set; artificial
   subsamples its larger validation pool proportionally. Extra records stay unused.
4. Within each source training pool, create nested samples without replacement.
   Artificial samples preserve peak proportions using deterministic quotas and
   fixed within-peak random orders. Every model gets the same sample at each size.
5. Fit from scratch at every size with seed 42; never warm-start from a smaller
   model. Restore the source-validation checkpoint, then predict both fixed
   target-test folds. No target-domain labels select a transfer checkpoint.
6. Plot the mean ± sample SD of **ten fold metrics**, not ten seed replicates or
   a confidence interval. Each target sequence is tested once per model and size.

**The x-axis counts training labels only.** Every fit additionally uses 4,413
validation labels. Thus these curves measure training-set size with a fixed
validation bank; they are not learning curves for the total labeling budget.
Training keeps the original recipes: at most 30 epochs for Aubin/MLPs, 60 for the
CNN, patience 10. Smaller datasets have fewer optimizer updates per epoch; we do
not tune a separate optimization schedule for each size.

Test size is **2,451–2,452 natural** or **2,690–2,691 artificial** records per fold
(10% of each target domain). Because training size changes, the proportions of
actually used records also change:

| Training n | Extra validation n | Used train/validation/test %, natural target | Used train/validation/test %, artificial target |
|---:|---:|---|---|
| 500 | 4,413 | 6.8/59.9/33.3 | 6.6/58.0/35.4 |
| 1,000 | 4,413 | 12.7/56.1/31.2 | 12.3/54.5/33.2 |
| 2,500 | 4,413 | 26.7/47.1/26.2 | 26.0/45.9/28.0 |
| 5,000 | 4,413 | 42.1/37.2/20.7 | 41.3/36.5/22.2 |
| 10,000 | 4,413 | 59.3/26.2/14.5 | 58.5/25.8/15.7 |
| 15,000 | 4,413 | 68.6/20.2/11.2 | 67.9/20.0/12.2 |
| 17,500 | 4,413 | 71.8/18.1/10.1 | 71.1/17.9/10.9 |

Percentages above use the larger test fold and are rounded. Exact per-fold counts
and used fractions are saved in `per_fold_scores.csv`. The initial source pools
are approximately 72/18/10; unused data are not reassigned to another partition.

Artificial → artificial tests held-out variants **within represented peaks**,
not an unseen peak. We save per-peak and equal-peak macro scores alongside pooled
scores. Cross-domain CV pairs source fold i with target fold i; error bars reflect
both source sampling and which target sequences are held out. Prior experiments
already used these landscapes, so this is exploratory evaluation, not a new blind test.

## Reproduce

From the repository root, activate the pinned environment and prepare data once
as described in [Running models](RUNNING_MODELS.md). Then:

```bash
python -m scripts.learning_curves.prepare
```

This freezes splits, nested subsets, source/data hashes and job lists in
`results/learning_curves_cgre_cv10/`. It refuses to overwrite an existing protocol.
Keep scientific training files unchanged during jobs. Use a new directory for a
changed experiment; CLI `--directory` selects that directory for training/plotting.

Run one model and fold locally (all seven sizes, independent fits):

```bash
python -m scripts.learning_curves.run --source natural --fold 0 --model mlp_small
python -m scripts.learning_curves.run --source artificial --fold 0 --model CNN_Jannis_OHE --device cuda
```

Add `--size 1000` to run one frozen size. Sources are `natural` or `artificial`;
models are `aubin_1_10_1`, `mlp_small`, `mlp_deep`, `CNN_Jannis_OHE`; folds are 0–9.
Completed fits are reused only after provenance and artifact checks. Each fit
saves predictions, weights, history, metrics, hardware and runtime.

For the complete experiment on our cluster:

```bash
mkdir -p results/learning_curves_cgre_cv10/condor
cp condor/learning_curves.dag results/learning_curves_cgre_cv10/workflow.dag
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit_dag results/learning_curves_cgre_cv10/workflow.dag'
```

The DAG launches **60 CPU jobs + 20 GPU jobs**, each with seven sizes: **560 fits**.
After every training job succeeds, it validates all predictions and draws the
figures automatically. Wrappers contain cluster-specific paths; adapt them on
another installation. Do not submit duplicate jobs. If interrupted, preserve
completed fits and use the Condor rescue workflow or rerun only pending jobs.

To regenerate figures after all fits finish:

```bash
MPLCONFIGDIR=/tmp/gfp-matplotlib python -m scripts.learning_curves.plot
```

## Outputs

All outputs are under `results/learning_curves_cgre_cv10/`:

| File | Contents |
|---|---|
| `cgre_learning_curve_spearman.png` | Four-model natural → natural learning curves |
| `peak_learning_curves_spearman.png` | Four panels for natural/artificial directions |
| Corresponding `_pearson`, `_r2`, `_rmse` files | Companion metrics; every plot also has PDF and SVG |
| `RESULTS.md`, `summary.csv` | Fold-mean results and sample SD |
| `per_fold_scores.csv` | Individual fold metrics, sample counts and exact used ratios |
| `pooled_oof_scores.csv` | Scores over all out-of-fold predictions, distinct from mean fold scores |
| `per_peak_summary.csv`, `artificial_macro_summary.csv` | Per-peak and equally weighted peak results |
| `runtimes.csv`, `runtime_summary.csv` | Whole-fit and inference times; CPU/GPU hardware varies |
| `sample_counts.csv` | Exact peak contributions at each size/fold |
| `protocol.json`, `finalization.json` | Frozen design and successful final validation marker |

R² is squared-error reduction against the target-test mean; it can be negative
and is not Pearson squared. Undefined correlations remain undefined. Fits include
training and validation time, but exclude scheduler waiting and shared context
loading; these are not per-sample runtimes.
