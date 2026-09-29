# GFP fluorescence prediction

This detailed guide preserves the data audit and original one-hot benchmark.
For the current environment, ESM/CNN commands, and complete comparison, start
with the [main README](../README.md). All ten original CV folds are now complete.

This project compares fluorescence predictors within GFP fitness landscapes and,
subsequently, across protein backgrounds. The first reproducible stage reconstructs
protein sequences, prepares log10 fluorescence targets, and freezes independent
training, validation, and test partitions. Reusable Aubin linear and nonlinear
models have now been trained and evaluated on cgreGFP. The benchmark also includes
ordinary least squares, ridge regression, and conventional shallow/deep MLPs.

## Reference baseline

We use the [Orthologous GFP Fitness Peaks paper](https://elifesciences.org/articles/75842)
and its local notebooks in `Orthologous_GFP_Fitness_Peaks/analysis/ML` as references:

- `01_Load_data.ipynb`, cells 12–14: target transformation, exclusion of genotypes
  containing `*`, and positive log-fluorescence filtering.
- `01_preprocessing.ipynb`, cell 11: two `train_test_split` calls, first holding out
  40%, then dividing that remainder equally, both with `random_state=42`.
- `02_fit_various_models.ipynb`: ordinary linear regression, a one-node linear
  neural network, and several nonlinear models.

We call the paper's one-hot, single-linear-output baseline the **Aubin model**.
The paper describes MSE training for up to 30 epochs with validation early stopping
(patience 10). The notebook also provides an ordinary least-squares implementation;
these implementations will be named explicitly when implemented. We have not
established that the linear model is the best-performing model.

## Reusable Aubin models and cgreGFP reproduction

`scripts/aubin_model.py` implements two explicitly named variants in PyTorch:

- **Aubin linear:** flattened positional one-hot encoding → one linear output.
- **Aubin 1–10–1:** the same encoding → one linear node → ten sigmoid nodes →
  one linear output. This adds a nonlinear mapping from an additive fitness
  potential to fluorescence; it is not a purely linear predictor.

Both use a fixed 20-amino-acid alphabet, Glorot-uniform weights, zero biases,
Adam (`lr=0.001`, betas `0.9/0.999`, epsilon `1e-7`), batch size 32, MSE, a
maximum of 30 epochs, and validation-loss early stopping with patience 10.
The checkpoint with minimum validation MSE is restored. No rare-mutation filter,
weight decay, dropout, target normalization, or test-driven tuning is applied.
An all-zero terminal position preserves the notebook's 4,720-dimensional cgreGFP
input: the notebook retained the FASTA stop position but removed its `*` features.

The implementation follows the notebook architecture and training settings, but
uses PyTorch instead of TensorFlow/Keras. Random-number streams, shuffling, and
Adam epsilon placement differ across these frameworks. It is an approximate
reproduction, not a bit-for-bit recreation. CPU operation, one thread,
deterministic algorithms, and seed 42 are the defaults; environment versions and
source/data hashes are recorded in each run.

### First cgreGFP result (seed 42)

| Model | Validation Spearman | Validation R² | Test Spearman | Test R² | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Aubin linear | 0.8386 | 0.6008 | 0.8315 | 0.5851 | 0.4968 |
| Aubin 1–10–1 | 0.8919 | 0.8870 | 0.8892 | 0.8870 | 0.2592 |

RMSE is in log10 fluorescence units. The split contains 14,709 training,
4,903 validation, and 4,904 test sequences. The linear run stopped after epoch
21 and restored epoch 11. The nonlinear run restored epoch 30, the prescribed
budget limit. These are single-seed results, not uncertainty estimates.

The archived `cgreGFP_lm_with_NN.csv` gives validation Spearman **0.8285** and
R² **0.5964**; `cgreGFP_output_subnetwork.csv` gives validation Spearman **0.8727**
and R² **0.8613**. These were recalculated from the saved predictions rather than
inferred from figure labels. We did not establish a reference Spearman of 0.858.
The nonlinear variant outperformed the linear variant in this run.

The archived cgreGFP preprocessing retained 23,665 sequences versus our 24,516.
A comparison by mutation description finds 23,487 shared genotypes, 178 only in
the archived set and 1,029 only in ours; 10,272 shared brightness values differ
at relative tolerance `1e-10`. Consequently, membership and fluorescence changes,
as well as framework/initialization differences, preclude a strict superiority
claim over the archived model. Our validation metrics are the appropriate
comparison to the archived validation scores; our test metrics are reported
separately.

### Train and reuse

```bash
python -m pip install -r requirements-aubin.txt
python -m unittest discover -s tests -v
python scripts/train_aubin.py --gene cgreGFP --output results/aubin_cgreGFP_seed42
python scripts/audit_aubin_reference.py --gene cgreGFP \
  --output results/aubin_cgreGFP_seed42/reference_metrics.json
MPLCONFIGDIR=/tmp/gfp-matplotlib python scripts/plot_aubin.py results/aubin_cgreGFP_seed42
```

Run directories must be new/empty. Each architecture saves `model.pt`,
`metrics.json`, `history.csv`, and `predictions.csv`; the run saves `run.json`
with provenance. The plotting command saves `evaluation.png` and `evaluation.pdf`.
Results are local artifacts ignored by Git; the numerical summary above is
retained in this README. The current results are under
`results/aubin_cgreGFP_seed42/`.

To train the same architecture separately on another prepared protein:

```bash
python scripts/train_aubin.py --gene amacGFP --architectures 1_10_1 \
  --output results/aubin_amacGFP_seed42
```

To load a trained model from Python, from the repository root:

```python
from scripts.aubin_model import load_model

model, metadata = load_model('results/aubin_cgreGFP_seed42/1_10_1/model.pt')
# sequences: list of full amino-acid strings, without terminal '*'
predicted_log10_fluorescence = model.predict(sequences)
```

The sequence length must match training. Retraining on another protein is
supported; cross-protein transfer still needs the alignment and leakage controls
described below. Checkpoint reload predictions were verified against the trained
models on all three partitions. Seven tests cover preprocessing, one-hot/additive
semantics, checkpoint round trips, and restoration of the best validation state.

## Additional baseline models

`scripts/train_baselines.py` adds the following models, using the same fixed
positional one-hot encoding, targets, and partitions as the Aubin runs:

| Name | Model | Selection rule |
|---|---|---|
| `mean` | Predict the training-set mean | None; reference floor |
| `linear_regression` | scikit-learn `LinearRegression`, intercept included | Ordinary least squares, no regularization |
| `ridge` | scikit-learn `Ridge`, intercept included | Lowest validation MSE among alpha = 0.1, 1, 10, 100 |
| `mlp_small` | 4,720 → 64 ReLU → 1 linear | Best validation checkpoint |
| `mlp_deep` | 4,720 → 128 ReLU → 64 ReLU → 32 ReLU → 1 linear | Best validation checkpoint |
| `knn` | scikit-learn nearest-neighbor regression on sequence Hamming distance | Lowest validation MSE over k and weighting |

Input width 4,720 is specific to cgreGFP (235 amino acids plus the all-zero
terminal position, each with 20 features). Other proteins use their own lengths.
The MLPs have no one-dimensional bottleneck before their nonlinear layers. They
can learn multiple sequence features and non-additive effects. Sizes were chosen
before running this benchmark; no hidden-width search was performed.

MLPs reuse the Aubin training loop: Adam, learning rate 0.001, MSE, batch size 32,
maximum 30 epochs, and early-stopping patience 10. Initial weights are Glorot
uniform with zero biases. No dropout, batch normalization, weight decay, feature
scaling, or target scaling is applied. Training budget and random seed match the
first Aubin experiment. More parameters do not necessarily improve generalization.

The sklearn models use sparse positional one-hot matrices and float64 arithmetic.
`LinearRegression` uses sparse LSQR with tolerance `1e-8`, while ridge uses LSQR
with tolerance `1e-8` and an iteration cap of 10,000. This avoids allocating a
dense one-hot design for a rank-deficient feature system. It does not add a
penalty to ordinary least squares. Ridge penalizes large coefficients, which can
help with correlated or infrequently observed features; see the
[scikit-learn linear-model documentation](https://scikit-learn.org/1.8/modules/linear_model.html).
Ridge candidates fit training data only; the selected candidate is not refit on
validation data. Test data never enter model fitting or hyperparameter selection.

Train all additional models and include existing Aubin metrics in a comparison:

```bash
python -m pip install -r requirements-aubin.txt
python scripts/train_baselines.py --gene cgreGFP \
  --output results/baselines_cgreGFP_seed42 \
  --aubin-run results/aubin_cgreGFP_seed42
MPLCONFIGDIR=/tmp/gfp-matplotlib python scripts/plot_baselines.py results/baselines_cgreGFP_seed42
```

Use `--models linear_regression ridge mlp_small mlp_deep` to select a subset,
or `--gene amacGFP --output results/baselines_amacGFP_seed42` to retrain on another
protein. When supplied, `--aubin-run` must match the gene and dataset checksum.
Dependency versions are the same as for the Aubin workflow.

Each model saves its predictions, metrics, and reloadable weights/estimator.
MLPs also save learning histories; ridge saves the complete alpha-search table.
`run.json` records configurations, software versions, and script/data hashes;
`comparison.csv` includes separate training, validation, and test metrics.
The plotting command saves `comparison.png` and `comparison.pdf` with the
validation and test scores shown separately for each model.
Constant mean predictions have undefined Spearman correlation, represented by
JSON `null` and an empty CSV field. Outputs are ignored by Git as with Aubin.

```python
from scripts.baseline_models import load_baseline

model, metadata = load_baseline('results/baselines_cgreGFP_seed42/ridge/model.pkl')
predicted_log10_fluorescence = model.predict(sequences)
# For an MLP, load mlp_small/model.pt or mlp_deep/model.pt in the same way.
```

Only load sklearn pickle artifacts from trusted sources. Models expect the same
sequence length and positional interpretation as training. These are separately
trained within-protein baselines; transfer evaluation is a separate experiment.

### cgreGFP baseline comparison (seed 42)

All rows use the same 14,709 / 4,903 / 4,904 training/validation/test partition.

| Model | Validation Spearman | Validation R² | Test Spearman | Test R² | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Training mean | — | -0.0002 | — | -0.0011 | 0.7716 |
| Sklearn linear regression | 0.8463 | 0.5825 | 0.8366 | 0.5694 | 0.5061 |
| Ridge (alpha 10) | 0.8565 | 0.6066 | 0.8503 | 0.5960 | 0.4902 |
| Aubin linear | 0.8386 | 0.6008 | 0.8315 | 0.5851 | 0.4968 |
| Aubin 1–10–1 | 0.8919 | 0.8870 | 0.8892 | 0.8870 | 0.2592 |
| Small MLP (64) | 0.8955 | 0.8713 | 0.8896 | 0.8741 | 0.2737 |
| Deep MLP (128–64–32) | 0.8946 | 0.8929 | 0.8904 | 0.8957 | 0.2491 |

The deeper MLP has the lowest validation and test MSE in this first run. The
three nonlinear models have very similar test Spearman correlations; a difference
around 0.001 is not evidence of a reliable ranking from a single seed. Spearman
measures ordering, while R²/RMSE measure numerical prediction accuracy, so they
need not favor the same model.

The small MLP has 302,209 parameters and selected epoch 30. The deeper MLP has
614,657 parameters and selected epoch 13, stopping at epoch 23. By comparison,
Aubin 1–10–1 has only 4,752 parameters. Its accuracy remains competitive despite
being much smaller. Training histories are retained; both MLPs showed fluctuating
validation losses, and the saved checkpoints restore the minimum validation MSE.
No test-based adjustment or second architecture search followed these results.

All 12 tests passed. Full-partition predictions from reloaded artifacts matched
their original models exactly. The ordinary least-squares solution also passed
an independent numerical check of its normal equations (maximum absolute feature
gradient divided by sample count: approximately `2.2e-9`). These checks establish
implementation consistency, not performance uncertainty or transferability.

### kNN, Pearson correlation, and true-versus-predicted plots

kNN uses the fraction of amino-acid positions that differ (Hamming distance),
so integer amino-acid codes are compared for equality rather than treated as
ordered numerical values. This gives the same neighbor ordering as Euclidean
distance on full one-hot sequences, but inverse-distance weights differ between
these metrics. The search tests k = 1, 3, 5, 11, 21, 51 with `uniform` and
`distance` weights, selecting the minimum validation MSE. The predictor averages
neighbors' **log10 targets**. Only training sequences are stored as neighbors.
Distances are computed by brute force with one worker.

Each k is queried separately, sharing the resulting neighbors between weighting
choices. Exact matches receive all the weight under distance weighting. Ties at
the neighbor boundary follow scikit-learn's behavior and can depend on training
row order; the frozen source ordering and pinned environment are therefore part
of reproducibility. See [KNeighborsRegressor documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.neighbors.KNeighborsRegressor.html).
Training-set kNN predictions include the query itself as a neighbor and are
optimistic; validation/test scores are the relevant generalization estimates.

Run just the new model, preserving earlier model artifacts:

```bash
python scripts/train_baselines.py --gene cgreGFP --models knn \
  --output results/knn_cgreGFP_seed42
MPLCONFIGDIR=/tmp/gfp-matplotlib python scripts/evaluate_models.py \
  --runs results/aubin_cgreGFP_seed42 results/baselines_cgreGFP_seed42 results/knn_cgreGFP_seed42 \
  --output results/evaluation_cgreGFP_seed42
```

The evaluation command reads saved predictions without retraining. It verifies
matching protein, dataset checksum, record IDs, split assignments, and targets
before combining runs. `comparison.csv` and `run.json` contain **Pearson r**,
Spearman, R², MSE, and RMSE for every model and partition. Pearson and Spearman
are undefined (`null`/blank) for the constant mean predictor. Pearson is not R²:
a shifted or rescaled prediction can have high correlation but poor R².
Metrics are evaluated in log10 fluorescence space, not raw fluorescence space.
Existing archived run files remain unchanged; the new evaluation is the complete
metric report. Minor floating-point differences from older reported MSE/R² can
occur because evaluation reads full-precision targets from saved CSVs.

`true_vs_pred_train`, `true_vs_pred_validation`, and `true_vs_pred_test` are saved
as both PNG and PDF. Each figure contains all models, with true values on x,
predictions on y, the equality diagonal, shared full-range axes, equal aspect
ratios, and a common logarithmic count color scale. No points are clipped or
subsampled. Per-panel labels include Pearson, Spearman, R², and sample count.
`comparison.png`/`.pdf` show validation/test Spearman, Pearson, and R² together.
Input prediction hashes, source run hashes, evaluation code hashes, and library
versions are recorded in the evaluation manifest.

The selected kNN uses **51 neighbors with inverse-Hamming-distance weighting**.
Validation MSE is 0.5892; held-out test MSE is 0.6088 (R² = -0.0235), worse than
the training-mean baseline's test MSE of 0.5954. Thus simple sequence proximity
is not an adequate fluorescence predictor here, despite moderate correlation.
The perfect kNN training score is self-neighbor lookup, not evidence of learning
or generalization. The full validation grid is in
`results/knn_cgreGFP_seed42/knn/neighbor_search.csv`.

| Model | Test Pearson | Test Spearman | Test R² |
|---|---:|---:|---:|
| Training mean | — | — | -0.0011 |
| Sklearn linear regression | 0.7610 | 0.8366 | 0.5694 |
| Ridge | 0.7721 | 0.8503 | 0.5960 |
| Aubin linear | 0.7668 | 0.8315 | 0.5851 |
| Aubin 1–10–1 | 0.9442 | 0.8892 | 0.8870 |
| Small MLP (64) | 0.9355 | 0.8896 | 0.8741 |
| Deep MLP (128–64–32) | 0.9469 | 0.8904 | 0.8957 |
| kNN (51, distance) | 0.6532 | 0.7135 | -0.0235 |

All 15 tests passed, including correlation-versus-R² distinctions, constant
predictions, kNN weighting with exact matches and tied neighbors, and saved kNN
sequence encoding. Reloaded kNN predictions matched the original estimator on
all partitions. Other models were not retrained for this evaluation.

## Ten-fold cross-validation on cgreGFP

`scripts/cross_validate.py` evaluates all eight models on all 24,516 prepared
cgreGFP sequences. Outer folds use `KFold(n_splits=10, shuffle=True,
random_state=42)`. Each sequence appears in exactly one outer test fold.
Within each remaining 90%, `train_test_split(test_size=0.2,
random_state=42 + fold_index)` reserves validation data. Thus approximately
72% of the full dataset fits model weights, 18% selects settings/checkpoints,
and 10% evaluates that fold. Training/validation/test membership is identical
across models within each fold and is written to `splits/fold_XX.csv`.

The architectures, 30-epoch neural-network budget, Adam settings, patience 10,
ridge alpha grid, and kNN neighbor/weighting grid remain unchanged. Each fold
initializes a fresh model with seed 42; fold variation primarily measures
partition sensitivity, not a separate initialization sweep. Ridge and kNN
settings are selected anew using only that fold's validation labels. Neural
checkpoints use that fold's minimum validation MSE. Validation samples are not
merged into the fitting data afterward, including for linear regression and
the mean baseline, so all models have the same fitting examples. The original
60/20/20 benchmark is preserved as a separate experiment.

Outputs include all 80 fitted model artifacts, fold-specific search/learning
histories and validation/test predictions, per-fold metrics, and pooled
out-of-fold (OOF) predictions. Summaries report both **mean ± sample SD across
10 test folds** and metrics computed on pooled OOF predictions. These are
different estimands, especially for correlation and R². Fold SD is not a
confidence interval: the training sets overlap. Pooled mean-baseline predictions
can vary across folds even though the predictor is constant within each fold;
within-fold Pearson/Spearman remain undefined for that baseline.

This is random sequence-level cross-validation, not a protein-held-out or
sequence-similarity-controlled evaluation. Related mutants can occur in different
folds. The model family choices were already explored in the original benchmark;
this CV quantifies their performance across partitions without test-fold tuning.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/cross_validate.py --gene cgreGFP --workers 4 \
  --output results/cv10_cgreGFP_seed42
MPLCONFIGDIR=/tmp/gfp-matplotlib python scripts/plot_cross_validation.py results/cv10_cgreGFP_seed42
```

Workers are ordinary local training processes, each limited to one CPU thread.
Use `--resume` to skip fully completed folds after interruption; source code,
configuration, and dataset hashes must match the saved run. An incomplete fold
is rerun in full. `summary.csv`/`.json`, `fold_metrics.csv`, and one OOF CSV per
model are the main numerical outputs. `fold_comparison.png`/`.pdf` display fold
variation; `true_vs_pred_oof.png`/`.pdf` show pooled held-out predictions.

kNN uses an exact, label-free Hamming-distance cache to avoid repeating sequence
comparisons across folds. One-hot vectors are translated by a common reference
sequence, preserving distances while making calculation sparse. Integer mismatch
counts are stored in `hamming_counts.npy` (about 601 MB for cgreGFP). Each query
still selects only its fold's training neighbors. Tie handling follows the
pinned scikit-learn brute-Hamming implementation; tests and a per-fold prediction
comparison against the ordinary saved sklearn estimator verify agreement.

### HTCondor execution

The first four folds completed locally. At the user's request, the six unfinished
folds were submitted to the LSV pool through the configured SSH alias `lsv-submit`
as **cluster 63263**. Each job requests one CPU, 4 GB memory, and no GPU. The
submit file is `condor/cv10_cgreGFP.sub`, and the executable wrapper is
`condor/run_cv_fold.sh`. Shared storage and the existing `combi` Python environment
are used; input files are not uploaded elsewhere.

The local worker process group was stopped before submission to prevent duplicate
training. `scripts/run_cv_fold.py` checks the frozen code/data hashes and reuses
models whose checkpoint, predictions, and metrics were already completed. Only
an interrupted model is retrained. Fold-specific file locks prevent duplicate
batch jobs from writing simultaneously. This scheduler change does not alter
fold membership, model settings, or validation rules. The handoff's completed
models, cluster ID, job-to-fold mapping, resource requests, and wrapper hashes
are recorded in `scheduler_handoff.json` alongside the original run config.

```bash
# Submit from the LSV submit host; do not resubmit an already-running cluster.
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/cv10_cgreGFP.sub'
ssh lsv-submit 'condor_q 63263'
# After all folds complete:
MPLCONFIGDIR=/tmp/gfp-matplotlib python scripts/finalize_cv.py results/cv10_cgreGFP_seed42
```

The submit file currently lists zero-based folds 4–9, which were unfinished at
handoff. For a fresh experiment, prepare its configuration and splits first and
update the fold list/run paths. Condor stdout, stderr, and event logs are under
`results/cv10_cgreGFP_seed42/condor/`. Finalization refuses incomplete folds,
verifies OOF coverage, and writes aggregate metrics and figures.

## Audit of the supplied DataSail file

`data/processed/fluorescence.csv` contains **avGFP**, not cgreGFP. Exact sequence
matching against our prepared data found:

| Check | Result |
|---|---:|
| External rows / unique sequences | 54,025 / 54,025 |
| Exact cgreGFP matches | **0 / 24,516** |
| Exact avGFP matches | **51,715 / 51,715** |
| Duplicate or cross-split sequence groups | 0 |
| Additional external sequences outside our filtered set | 2,310 |

The external labels contain 21,446 train, 5,362 valid, and 27,217 test records.
Among sequences retained by our avGFP preprocessing, these counts become 20,963,
5,235, and 25,517 respectively. Two sequences also match amacGFP; these are the
previously documented exact identities shared between amacGFP and avGFP, not
evidence that this is a mixed cgreGFP dataset.

The remaining 2,310 external sequences match the excluded stop-mutant avGFP
genotypes if all `*` characters are removed after applying the mutations. Their
lengths are 237 (2,239 rows) or 236 (71 rows), rather than the 238-residue reference.
Removing a stop marker does not represent translation termination; these records
remain excluded under our established policy.

We have **not adopted these split labels for cgreGFP**, nor trained additional
avGFP models. The audit records the supplied split alongside our own target for
every exact match in `results/datasail_audit_cgreGFP/sequence_matches.csv`, making
future avGFP adoption possible without substituting the friend's target values.
The CSV does not establish how the split was generated; DataSail settings and
provenance were not provided.

Among exact avGFP matches, train and validation contain only 0–3 mutations,
while test contains 4–15 mutations. This matches the mutation-distance split
described in the [TAPE paper](https://papers.nips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf).
We infer that this is TAPE-style membership, but do not claim a verified DataSail
generation procedure. `split_profile.json` records the observed mutation-count
histograms and input hashes. The audit therefore establishes both a different
protein and a different evaluation question from random cgreGFP CV.

```bash
python scripts/audit_external_split.py --output results/datasail_audit_cgreGFP
```

`audit.json` records match counts, duplication checks, and input/code hashes.
This audit does not require invoking DataSail or installing another package.

## Data sources and target definitions

| Landscape | Source | Target before any model fitting |
|---|---|---|
| amacGFP, ppluGFP | `data/processed/amac_pplu__aadata_rescaled_by_WTctrls.csv` | `log10(scaled_WTctrl_fit)` |
| cgreGFP | `data/raw/amacGFP_cgreGFP_ppluGFP2__final_aminoacid_genotypes_to_brightness.csv` | `log10(replicates_mean_brightness)` |
| avGFP | `data/raw/avGFP__rf_aminoacid_genotypes_to_brightness.csv` | `log_brightness`, already logged |
| Artificial peaks: cgre132, cgre1338, cgre4111, cgre9708 | Corresponding `data/raw/240228__ntdata_<gene>_c075-genotypes__err1__cgreWTgates.csv` | `log10(brightness)` after protein-genotype aggregation |

The rescaled amac/pplu source is used as supplied, following the biologist's
recommendation. Its genotype sets match the raw data, and the supplied
`scaled_WTctrl_fit__log10` agrees with the computed log10 values. The calibration
script and its provenance are not yet available; this workflow does not refit it.
The similarly named file in `data/processed/additional_proteins` differs at the
byte level, but has matching genotype sets and calibrated values (checked at
relative tolerance `1e-12`). It is not substituted for the requested file.

The biologist confirmed that artificial-peak brightness can be used as supplied.
We apply no additional calibration. Repeated amino-acid genotypes are collapsed
using the `pseudocell_count`-weighted arithmetic mean of brightness, **before**
log10 transformation. This extends the existing aggregation approach in
`scripts/epistasis_selection.py`, without subtracting wildtype brightness.
Aggregation is a documented project choice, not a claim of exact reproduction
of the paper's four natural landscapes. Replicate counts and cell counts are not
interchangeable, and neither is used as a model training weight at this stage.

`cgre12minis` is explicitly excluded. Nucleotide files for the natural landscapes
and existing epistasis-selected datasets are not additional training observations.

## Sequence validation and exclusions

References come from `data/raw/fasta_sequences/protein_seqs.fa`. Use
`aa_genotype_native` for natural landscapes and `aa_genotype` for artificial peaks.
Positions are zero-based, including the initial methionine. Each ordinary
substitution must match its original residue in the appropriate background FASTA.
The four artificial peaks use their own reference sequences, not cgreGFP's sequence.
Mismatches, repeated positions, and no-op substitutions fail the run rather than
silently changing a sequence. `wt` reconstructs the unmutated reference.

Exclusion reasons are recorded per source row:

- `wt_ctrl_*` calibration controls;
- unsupported mutation notation, including `.` insertion/deletion descriptions;
- mutations introducing or replacing `*`, following Aubin's stop-codon filter;
- noncanonical amino acids, nonfinite targets, nonpositive linear fluorescence,
  nonpositive log10 targets, or invalid aggregation weights.

Filters are applied in this order by the implementation; each rejected row gets
one reason. The unchanged terminal FASTA `*` is removed from model-ready sequences.
No sequences are padded, truncated, or aligned at this stage. Sequence hashes
identify exact protein identity independently of mutation order.

## Splitting and evaluation

Each landscape receives a separate random 60/20/20 train/validation/test split
with seed 42, using the same two calls as Aubin. Source order is preserved after
filtering; aggregated records retain their first occurrence. Exact sequences are
unique within each prepared landscape, so synonymous nucleotide records cannot
cross partitions. Fractions have the rounding behavior of scikit-learn.

These reproduce the **splitting procedure**, not the historical membership:
the input versions, calibration, and explicit handling of unsupported genotypes
can differ. There is no additional standardization, wildtype subtraction, or
min–max scaling. Any future fitted feature/target transform must fit on training
data only. Validation selects hyperparameters and early stopping; test data stay
untouched until final evaluation.

Rare-mutation filtering is not applied to this initial linear benchmark. The
paper discusses it for optimized nonlinear models. The local notebook's later
filter combines train and validation data and its loop can stop before counts
converge; we will not inherit that behavior. If added, frequency thresholds must
be derived from training data only, with filtered evaluation clearly distinguished
from the full evaluation set.

The current partitions evaluate **within-landscape prediction**, not transfer.
Transfer experiments must hold out whole protein backgrounds and exclude exact
sequence overlaps between source and destination. The manifest reports any such
overlaps. Artificial peaks are closely related cgre backgrounds and should be
reported separately from transfer among natural orthologs. Positional one-hot
transfer will also require an explicit alignment; equal numeric indices are not
necessarily homologous positions.

## Reproduce preprocessing

Use Python 3.11 or newer with a dedicated environment:

```bash
python -m pip install -r requirements-preprocessing.txt
python -m unittest discover -s tests -v
python scripts/prepare_data.py
```

The default output is `data/processed/baseline_v1`. The script refuses to overwrite
a nonempty directory. To compare an independent regeneration:

```bash
python scripts/prepare_data.py --output /tmp/gfp-baseline-check --seed 42
```

Outputs:

- `sequences.csv`: full sequences, canonical mutations, log10 targets, split
  membership, sequence/record IDs, source filenames and CSV line numbers,
  aggregation counts and weights, and source-scale target values.
- `exclusions.csv`: rejected source rows and reasons.
- `manifest.json`: input and output SHA-256 hashes, preprocessing script hash,
  software versions, seed, per-landscape counts, and cross-landscape overlaps.

For grouped artificial records, `source_value` is the weighted linear brightness;
for avGFP it remains the supplied logged brightness. `target_log10` is always the
model target. Source line numbers include the header as line 1. Generated outputs
are not committed by default; preserve the manifest alongside experiment results.

### Initial prepared dataset

| Landscape | Retained sequences | Train | Validation | Test |
|---|---:|---:|---:|---:|
| amacGFP | 33,511 | 20,106 | 6,702 | 6,703 |
| avGFP | 51,715 | 31,029 | 10,343 | 10,343 |
| cgreGFP | 24,516 | 14,709 | 4,903 | 4,904 |
| ppluGFP | 31,402 | 18,841 | 6,280 | 6,281 |
| cgre132 | 4,267 | 2,560 | 853 | 854 |
| cgre1338 | 10,241 | 6,144 | 2,048 | 2,049 |
| cgre4111 | 8,214 | 4,928 | 1,643 | 1,643 |
| cgre9708 | 4,180 | 2,508 | 836 | 836 |

The initial run used Python 3.11.14, NumPy 2.4.2, and scikit-learn 1.8.0 from
the existing `combi` environment. The preprocessing dependencies are also pinned
in the current `environment.yml`, alongside ESM and CNN training dependencies.

Validation passed four unit tests covering residue indexing, exclusion policies,
invalid mutations, and split reproducibility. An independent full regeneration
produced byte-identical CSVs and manifest. All artificial-peak weighted targets
were independently recalculated from their source rows; source-row accounting
and within-landscape sequence uniqueness also passed. There are 168,046 prepared
records and two exact sequence identities shared across backgrounds, recorded
in the manifest for future transfer filtering.

## Legacy code

`CNN/model_legacy.py` and `scripts/epistasis_selection.py` are historical references,
not the new data pipeline. The CNN contains an independently sampled sequence/
label branch, uses test data for early stopping, and relies on external embedding
files. Existing processed epistasis datasets also use a different target definition.
Do not use their scores as validated baselines without repairing and auditing them.
