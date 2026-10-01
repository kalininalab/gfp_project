# Training mixtures and artificial-peak transfer

## Completed results

All **96 fits** completed successfully; the finalizer verified artifact hashes,
test record identities and targets, and recomputed prediction metrics.
[Full results](../results/transfer_cgreGFP_seed42_44/RESULTS.md) include Pearson,
R², RMSE and runtime, alongside the two requested figures:
[ortholog mixtures](../results/transfer_cgreGFP_seed42_44/ortholog_mixtures_spearman.png)
and [artificial-peak transfer](../results/transfer_cgreGFP_seed42_44/artificial_peak_transfer_spearman.png).

[amacGFP + ppluGFP → cgreGFP measured-vs-predicted panel](../results/transfer_cgreGFP_seed42_44/amac_pplu_to_cgre_true_vs_predicted/amac_pplu_to_cgre_all_seeds.png)
shows all four selected models and all three independent fits. It uses the same
4,904 held-out cgreGFP sequences in every subplot. The plotted predictions are
saved outputs; the plotting step does not refit any model.

Natural-only training gives the highest mean natural-test Spearman for every
model. Deep MLP leads this comparison (0.893 ± 0.001; RMSE 0.248), followed by
small MLP (0.889), Aubin (0.887), and CNN Jannis (0.884). Mean fitting times for
these controls are respectively 4.25, 2.81, 1.70 CPU minutes and 35.63 GPU minutes;
hardware varies and embedding extraction is additional. Aubin remains an
efficient baseline; CNN Jannis does not improve this natural-data control.

Replacing natural training examples with ortholog examples reduces performance
at this fixed total budget. This does not test adding ortholog data while keeping
all natural examples. Training without any natural cgre examples gives weak or
unstable ortholog transfer; individual seed points show this explicitly.

Artificial → natural ranking is strongest for Aubin (Spearman 0.843 ± 0.003),
then CNN Jannis (0.824 ± 0.009). However, their log10 RMSE values are 1.274 and
1.323: good rank transfer does not establish accurate absolute fluorescence
calibration. Natural → artificial ranking is highest for small MLP (0.748),
closely followed by deep MLP (0.747); CNN Jannis has the lowest RMSE (1.092).
Artificial → artificial scores are pooled over represented peaks, not unseen
peaks; per-peak and macro scores are saved separately. Three seeds measure run
variability, not uncertainty over new data splits. The natural test was used in
earlier comparisons, so this is exploratory evaluation rather than a fresh blind test.

This experiment uses **Aubin 1–10–1, small MLP, deep MLP, and CNN_Jannis**.
The first three use aligned positional one-hot inputs. CNN_Jannis uses full,
unpooled ESM-2 t30 embeddings. We keep the previous training recipes; this is a
comparison of training sources, with a fixed training budget.

## What is held out?

We preserve the prepared per-landscape **60% train / 20% validation / 20% test**
partitions. No source record changes partition. Each fit uses **14,709 training
records and 4,903 validation records**. The held-out target records are identical
for every model, source mixture, and seed:

| Target | Test sequences | Fraction of that landscape |
|---|---:|---:|
| Natural cgreGFP | 4,904 | 20% |
| cgre132 | 854 | approximately 20% |
| cgre1338 | 2,049 | approximately 20% |
| cgre4111 | 1,643 | approximately 20% |
| cgre9708 | 836 | 20% |
| All four artificial peaks pooled | 5,382 | approximately 20% |

**Natural cgreGFP** means the original measured cgreGFP landscape, including its
mutants. This is what “WT cgreGFP” means in the reference figure; it does not mean
training on a single wild-type sequence.

With the fixed training and validation budgets, the assembled train/validation/
test counts for a natural-target evaluation are **14,709 / 4,903 / 4,904**, or
approximately **60/20/20**. For an artificial-target evaluation, they are
**14,709 / 4,903 / 5,382**, or **58.8/19.6/21.5**. The latter test count still
represents 20% of the original artificial landscapes; the denominator changes
because we match the training budget to the natural-data experiment.

Subsampling leaves some source records unused. `sample_counts.csv` records
**exact counts, available counts, and fractions of the original landscape** for
each source, partition, mixture, and seed. Test counts and ratios are printed
on both main figures.

## First figure: training mixtures → natural cgreGFP

The seven columns use these training sources:

| Sources | Train counts, in listed order | Validation counts | Source ratio |
|---|---|---|---|
| cgreGFP | 14,709 | 4,903 | 100% |
| amacGFP | 14,709 | 4,903 | 100% |
| ppluGFP | 14,709 | 4,903 | 100% |
| cgreGFP + amacGFP | 7,355 + 7,354 | 2,452 + 2,451 | approximately 50:50 |
| cgreGFP + ppluGFP | 7,355 + 7,354 | 2,452 + 2,451 | approximately 50:50 |
| amacGFP + ppluGFP | 7,355 + 7,354 | 2,452 + 2,451 | approximately 50:50 |
| cgreGFP + amacGFP + ppluGFP | 4,903 each | 1,635 + 1,634 + 1,634 | approximately 1:1:1 |

All are evaluated on the **same 4,904 natural cgreGFP test variants**. This asks
what happens when we replace some training examples with another protein at a
fixed total budget. It does not measure the benefit of adding unlimited extra
ortholog data while retaining all cgreGFP training records.

Validation has the same source composition as training. For example, the
amacGFP-only model selects its checkpoint using **amacGFP validation labels only**;
it cannot select a checkpoint using cgreGFP validation performance. Thus the
source-only bars represent transfer without labeled target data for selection.

## Second figure: natural cgreGFP ↔ artificial peaks

The four directions are:

1. Artificial → artificial.
2. Artificial → natural.
3. Natural → artificial.
4. Natural → natural.

Only two sets of models are needed: models trained on natural cgreGFP and models
trained on the artificial pool. Each model is evaluated on both held-out targets.
The natural-only fits are **reused from the first figure**, so their scores and
model weights are exactly the same in both figures.

The artificial pool contains **cgre132, cgre1338, cgre4111, and cgre9708**. Training
and validation examples are sampled proportionally to the peak's available
partition size, keeping the total budget at 14,709/4,903. The excluded
`cgre12minis` file remains excluded. Brightness calibration and log10 targets
are exactly those in the prepared dataset; there is no target-domain rescaling.

| Artificial training source | Training records | Validation records |
|---|---:|---:|
| cgre132 | 2,333 | 777 |
| cgre1338 | 5,599 | 1,867 |
| cgre4111 | 4,491 | 1,497 |
| cgre9708 | 2,286 | 762 |

Artificial → artificial holds out new variants **within the four represented
peaks**. It is not leave-one-peak-out generalization. The primary plot reports
pooled Spearman correlation. Pooled scores can be influenced by differences
between peak brightness distributions, so we also save **per-peak scores and
an unweighted macro-average across peaks**.

## How are different protein sequences handled?

The orthologs have different lengths; simple cropping or treating the same raw
position number as homologous would be inappropriate. For the one-hot models:

- Align the three **wild-type sequences only**, anchored to natural cgreGFP.
- Use Biopython 1.86 global pairwise alignment, BLOSUM62, gap opening −10, gap
  extension −0.5, and the first optimum for deterministic tie handling.
- Merge insertion slots into 256 shared columns, left-aligning insertions in
  the same slot. Every native residue appears exactly once; none is discarded.
- Every mutant inherits its own WT's mapping. Mutants are not independently
  realigned. This preserves validated mutation coordinates.
- Encode alignment gaps as all-zero amino-acid features. Artificial peaks have
  the same native length/coordinates as cgreGFP and inherit its coordinate map.

The alignment is a modeling assumption, not a structure-validated correspondence.
Inspect `aligned_wildtypes.fasta`, `alignment_coordinates.csv`, and
`alignment.json` (including pairwise alignments, scores, and optimal-alignment
counts). See [Biopython's alignment documentation](https://biopython.org/docs/1.86/Tutorial/chapter_pairwise.html).

All one-hot fits, including cgreGFP-only controls, use the same aligned feature
space. Consequently, their initialization/input width differs from the earlier
unaligned within-cgreGFP benchmark; these are fresh matched controls.

**CNN_Jannis retains every ESM residue at its native sequence position.** For a
mixed minibatch, we forward each protein's same-length subgroup separately,
restore the original row order, and take one optimizer step on the combined
sample-weighted loss. This requires no cropping, padding, or modification of
Jannis's architecture. GroupNorm does not mix batch statistics between samples.

## Repeats, metrics, and limitations

- Three seeds: **42, 43, 44**. They control initialization, minibatch order, and
  subsampling where a source pool is larger than its assigned budget.
- The test and validation partitions stay fixed. Validation subsets are drawn
  only from their original source validation pools.
- Bars are the **mean across the three fits**; error bars are **sample SD**.
  Small white markers show the individual runs, so unstable or sign-changing
  transfer is visible rather than hidden by its mean.
  They are not confidence intervals, CV uncertainty, or uncertainty over new
  protein backgrounds. Individual run metrics are saved.
- Report Spearman, Pearson, RMSE, and R² on log10 fluorescence. The two main
  figures use Spearman; matching Pearson and R² figures are also generated.
- Constant predictions have undefined correlation. We retain that as undefined,
  rather than plotting it as zero or dropping a failed ranking silently.
- The natural cgreGFP test partition appeared in earlier model comparisons.
  These are exploratory comparisons on a consistent test set, not a new blind
  test of a model chosen without any previous results.
- Source sequence IDs, validation IDs, and test IDs are checked for overlap.
  The seven included landscapes have no exact cross-landscape sequence duplicates.
- Fit, prediction, and total elapsed times are recorded. Neural training settings
  match the earlier comparison: 30 epochs maximum for Aubin/MLPs, 60 for the CNN,
  patience 10, and restored minimum-validation-MSE checkpoints. No target-test
  score is used for early stopping or hyperparameter selection.

## Run the experiment

From the project root, activate the environment from `environment.yml` (now also
including Biopython 1.86). On this cluster the wrappers use the existing `combi`
environment with those versions.

```bash
cd /nethome/akolchina/gfp_project
conda activate gfp-benchmark
python -m unittest discover -s tests -p 'test_transfer_benchmark.py' -v
```

**Prepare once** after `python scripts/prepare_data.py`. Published summary files
are included in Git; the protocol and per-seed subsets are generated locally.
Skip this command only if the complete local preparation already exists:

```bash
python -m scripts.transfer_benchmark.prepare
```

The experiment directory is `results/transfer_cgreGFP_seed42_44`. Preparation
freezes the source-code/data hashes, WT mapping, test records, per-seed subsets,
and model settings. Do not edit scientific training files while jobs are running.
A changed experiment requires a new protocol/output directory.

The new ortholog/artificial embeddings use the same pinned ESM checkpoint as
before. The natural cgreGFP cache is reused. Submit their generation:

```bash
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/transfer_embeddings.sub'
```

Skip that submission if all seven `esm_embeddings/GENE_t30/manifest.json` files
already exist. A `.npy` file alone does not prove that generation finished.

The one-hot models do not need embeddings and can start immediately:

```bash
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/transfer_onehot.sub'
```

**After all embeddings finish**, verify their checksums and sequence identities
once, then submit the CNN fits:

```bash
python -m scripts.transfer_benchmark.audit_features
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/transfer_cnn.sub'
```

The audit reads the large feature files; use the CPU Condor audit job supplied
in `condor/transfer_feature_audit.sub` on this cluster if preferred. Each CNN fit
loads only the selected rows. It checks the frozen file size/timestamp and
manifest against that checksum audit, avoiding repeated hashing of every full
cache in every GPU job. Any changed cache requires a fresh audit.

Alternatively, after embeddings **and all 72 one-hot fits** finish, run the
remaining stages automatically as a Condor DAG instead of the separate audit/CNN
commands above:

```bash
cp condor/transfer_finish.dag results/transfer_cgreGFP_seed42_44/workflow.dag
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit_dag results/transfer_cgreGFP_seed42_44/workflow.dag'
```

The DAG runs the feature audit, then all 24 CNN fits, then final validation and
plotting. Do not submit the standalone CNN jobs as well as this DAG.

There are **72 CPU fits + 24 GPU fits = 96 fits**. Queue files explicitly list
mixture, seed, and model. Monitor with `ssh lsv-submit 'condor_q -submitter akolchina'`.
Logs are under the experiment's `condor/` directory. Do not submit duplicates
while jobs are still running; per-fit locks also prevent concurrent writers.

Run one particular model without Condor, for example:

```bash
python -m scripts.transfer_benchmark.run --mix cgre_amac --seed 42 --model mlp_small
python -m scripts.transfer_benchmark.run --mix artificial --seed 42 --model CNN_Jannis --device cuda
```

Complete matching fits are reused. Incomplete fits restart deterministically.
Each fit saves `model.pt`, `history.csv`, predictions with record IDs, and a
completion-marker `metrics.json`.

**After all fits finish**, validate predictions and create the final plots:

```bash
MPLCONFIGDIR=/tmp/gfp-matplotlib python -m scripts.transfer_benchmark.plot
```

The finalizer requires all 96 fits, verifies checkpoint/prediction hashes,
recomputes metrics against original targets, and checks exact target membership.
It produces PNG, PDF, and SVG versions of:

- `ortholog_mixtures_spearman`: the seven mixtures with training-source dots.
- `artificial_peak_transfer_spearman`: the four transfer directions.
- Corresponding `*_pearson` and `*_r2` figures.

The palette is blue, orange, purple, and rose; no green. Ratios and sample counts
are printed directly on the figures. `summary.csv`, `per_run_scores.csv`,
`per_peak_scores.csv`, `artificial_macro_scores.csv`, `runtimes.csv`, and
`RESULTS.md` hold the numerical results. `finalization.json` records final checks.

## True versus predicted fluorescence on cgreGFP

[Four-model test scatter panel (all seeds)](../results/transfer_cgreGFP_seed42_44/cgre_true_vs_predicted/cgre_test_all_seeds.png)
uses saved predictions from the **cgre-only transfer controls**: aligned one-hot
Aubin/MLPs and native full ESM embeddings for CNN Jannis. Every panel contains
the same 4,904 held-out cgre sequences. Training/validation/test is 60/20/20.
Rows show seeds 42, 43 and 44 separately; predictions are not averaged into an
ensemble. Separate figures for each seed are in the same folder. PNG, PDF, SVG
and a metrics CSV are generated with:

```bash
MPLCONFIGDIR=/tmp/gfp-matplotlib python -m scripts.transfer_benchmark.plot_cgre_predictions
MPLCONFIGDIR=/tmp/gfp-matplotlib python -m scripts.transfer_benchmark.plot_ortholog_predictions
```

This requires the saved per-fit predictions and prepared sequence table; it does
not retrain models. The script verifies prediction hashes, test IDs and targets.

R² is included in the result table and scatter annotations. Companion R² plots:
[ortholog mixtures](../results/transfer_cgreGFP_seed42_44/ortholog_mixtures_r2.png) ·
[artificial peaks](../results/transfer_cgreGFP_seed42_44/artificial_peak_transfer_r2.png).
Negative R² means worse squared error than predicting the test-target mean;
high rank correlation alone does not imply accurate fluorescence calibration.
