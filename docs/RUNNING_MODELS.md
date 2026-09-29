# Running the models

Detailed reproduction instructions, model definitions, and legacy CNN fixes.
See the [project overview](../README.md) for results and the shortest setup path.

## 1. Open the project and activate the environment

Run these commands in a terminal:

```bash
cd /path/to/gfp_project
conda env create -f environment.yml
conda activate gfp-benchmark
python -m unittest discover -s tests -v
```

Create the environment once. On later visits, only run `cd ...` and
`conda activate gfp-benchmark`. [environment.yml](../environment.yml) pins the
scientific packages used by this benchmark. A CUDA-capable NVIDIA GPU is needed
for the full embedding/CNN workloads. The mean-embedding regressors run on CPU.
On this cluster, the already-installed `combi` environment has these package
versions and is the interpreter used by our Condor wrappers.

## 2. Use the prepared data and frozen splits

The prepared table is `data/processed/baseline_v1/sequences.csv`.
Each row contains the **record ID, full sequence, target_log10, and split**.
Never shuffle the sequence column independently of the target column.

If this file already exists, **keep it**. To rebuild it from the raw files in a
fresh checkout, run:

```bash
python scripts/prepare_data.py
```

The script refuses to overwrite prepared data. For an independent check use
`python scripts/prepare_data.py --output /tmp/gfp-preprocessing-check`.

There are two evaluations:

| Evaluation | Training | Validation (model selection) | Test (final scoring) |
|---|---:|---:|---:|
| Holdout | 14,709 | 4,903 | 4,904 |
| 10-fold CV | about 72% | about 18% | about 10% in each fold |

The ten split files are
`results/cv10_cgreGFP_seed42/splits/fold_00.csv` through `fold_09.csv`.
Every sequence is a test sequence exactly once across those ten folds.
Validation is sampled **inside each outer training partition**. All models reuse
these files; no new split is made for ESM. Seed 42 is fixed.

The friend's `data/processed/fluorescence.csv` matches **avGFP, not cgreGFP**:
0 cgreGFP matches; all 51,715 retained avGFP sequences match. We have audited it
but have not applied that split to cgreGFP or trained an avGFP benchmark.

## 3. Generate ESM-2 embeddings once

We use [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D),
pinned to commit `a695f6045e2e32885fa60af20c13cb35398ce30c`.
First download the model; this requires internet access:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/esm2_t30_150M_UR50D', revision='a695f6045e2e32885fa60af20c13cb35398ce30c', allow_patterns=['config.json','model.safetensors','tokenizer_config.json','special_tokens_map.json','vocab.txt'])"
```

Then, **on a GPU machine**, run:

```bash
python -m scripts.esm_benchmark.embed \
  --revision a695f6045e2e32885fa60af20c13cb35398ce30c
```

On our cluster, submit the GPU job instead of running it on the login machine:

```bash
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/esm_embed.sub'
```

**Wait until `esm_embeddings/cgreGFP_t30/manifest.json` exists.** This is the
completion marker. Large `.npy` files can exist before computation is finished.
If embeddings are already complete, skip this step.

| Output under `esm_embeddings/cgreGFP_t30/` | Meaning |
|---|---|
| `residue_embeddings.npy` | Full last-layer vectors: `[24516, 235, 640]`, float16 storage, about 7.4 GB. CNN input. |
| `mean_embeddings.npy` | Arithmetic mean across the 235 residues: `[24516, 640]`, float32. Classical/MLP input. |
| `records.csv` | Ordered record IDs and sequences; protects against label mismatches. |
| `manifest.json` | Model revision, input/output hashes, tensor shape, hardware, and timing. |

The model runs in evaluation mode, without gradients, in float32. BOS, EOS, and
padding tokens are excluded. The saved residue vectors are compacted to float16;
the means are calculated in float32 from those same saved vectors. “Full” means
**all residue positions from the last layer**, not a mean vector or all 30 layers.
No fluorescence labels are used when creating embeddings, so the frozen features
can safely be cached once and reused across folds. Fine-tuning ESM would require
a separate training procedure inside each fold and is not done here.

## 4. Train the models

### A single holdout run

Run all eight existing regressors on the mean embeddings (CPU):

```bash
python -m scripts.esm_benchmark.run --model all_mean
```

Run one CNN on the full embeddings (GPU):

```bash
python CNN/model.py --model CNN_old --device cuda
python CNN/model.py --model CNN_new --device cuda
python CNN/model.py --model CNN_Jannis --device cuda
```

These commands write to `results/esm_cgreGFP/holdout/MODEL_NAME/`.
For one particular classical model, replace `all_mean` with `ridge`, `knn`,
`linear_regression`, `mean`, `aubin_linear`, `aubin_1_10_1`, `mlp_small`, or
`mlp_deep`. A completed run is reused only if its inputs, code, and settings match.
For a changed experiment, supply a fresh `--output results/MY_NEW_EXPERIMENT`.

### The complete holdout + 10-fold benchmark on Condor

After embeddings are complete and the split files exist, submit:

```bash
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/esm_mean.sub'
ssh lsv-submit 'cd /nethome/akolchina/gfp_project && condor_submit condor/esm_cnn.sub'
```

This launches 11 CPU jobs (eight mean-feature models per split) and 33 GPU jobs
(three CNNs × 11 splits). Each job is independent. CNNs request one modern GPU
and 14 GB host memory. The full embedding tensor is read once into host memory;
only a batch is moved to the GPU. Logs are in `results/esm_cgreGFP/condor/`.

Check progress with `ssh lsv-submit 'condor_q -submitter akolchina'`.
If authentication has expired, run `kinit` in your terminal and retry.
Do not submit a second copy of a job that is already running.

To run a particular fold without Condor:

```bash
python -m scripts.esm_benchmark.run --model CNN_new --device cuda \
  --split results/cv10_cgreGFP_seed42/splits/fold_00.csv --split-name 00
```

Repeat with `00` through `09` for all folds. The shell wrappers in `condor/`
contain this cluster's absolute project/interpreter paths; edit those paths if
moving the project to another machine. The Python commands use relative paths
and run from the project root.

### Original one-hot models and splits, from scratch

These commands reproduce the non-ESM reference runs. Use new output folders if
existing results are present. They can take several minutes; prefer Condor for
CV. The detailed guide explains the per-fold Condor wrapper.

```bash
python scripts/train_aubin.py --gene cgreGFP --output results/aubin_cgreGFP_seed42
python scripts/train_baselines.py --gene cgreGFP --models mean linear_regression ridge mlp_small mlp_deep --output results/baselines_cgreGFP_seed42
python scripts/train_baselines.py --gene cgreGFP --models knn --output results/knn_cgreGFP_seed42
python scripts/evaluate_models.py --runs results/aubin_cgreGFP_seed42 results/baselines_cgreGFP_seed42 results/knn_cgreGFP_seed42 --output results/evaluation_cgreGFP_seed42
python scripts/cross_validate.py --gene cgreGFP --workers 1 --output results/cv10_cgreGFP_seed42
python scripts/finalize_cv.py results/cv10_cgreGFP_seed42
```

The original holdout kNN run lives separately in `results/knn_cgreGFP_seed42`;
see the exact commands in the detailed guide. To measure reference fit/prediction
times consistently after those original runs exist, submit
`condor/reference_timing.sub`. The timer verifies that retrained predictions
match the saved reference predictions.

## What are the three CNNs?

All take `[batch, amino-acid position, 640 ESM features]` and output one number.
A convolution learns local patterns along the protein sequence. Pooling combines
those position-wise patterns into a vector used to predict fluorescence.

| Model | Architecture | Trainable parameters |
|---|---|---:|
| `CNN_old` | Your 64-channel kernel-5 convolution → 32-channel dilated kernel-3 convolution → max pooling → 64-unit head. | 213,217 |
| `CNN_new` | 128-channel projection → three residual blocks, dilations 1/2/4 → mean + max pooling → 64-unit head. Group normalization and dropout. | 297,985 |
| `CNN_Jannis` | 512 channels, four convolutions, residual connections, group normalization, GELU, dropout → mean + max pooling → 512/256/128-unit head. | 6,920,961 |

`CNN_old` preserves your original layer sizes and padding. Its second convolution
shortens the intermediate sequence by two positions; that is legal, so it is
preserved rather than called a training bug. `CNN_new` keeps position lengths
unchanged and uses residual connections to help optimization. Its wider receptive
field combines nearby patterns without Jannis's much larger parameter count.
Whether that improves prediction is an empirical question, not a guarantee.

`CNN_Jannis` ports `ProtCNN` from `master_thesis_jaca00001/src/model.py`, using
`NON_AL_MODEL_DEFAULTS` from his `src/configs.py`: four convolutional and four
fully connected layers; head dropout 0.1324962. His convolutional dropout remains
0.15. His active-learning defaults use five convolutions and a different head;
we use the non-active-learning settings for this supervised benchmark. A unit
test loads identical weights into both implementations and checks their outputs.

### Bugs fixed in your original training script

The original source is preserved verbatim in `CNN/model_legacy.py`.
`CNN/model.py` now invokes the corrected shared trainer.

| Original problem | Why it matters | Correction |
|---|---|---|
| cgreGFP embeddings and labels sampled with two independent random draws in the mixed-protein branch | A sequence can receive another sequence's fluorescence. | Join/check record IDs and use the same indices for features and targets. |
| Test loader used for early stopping | Test labels influence model selection, making test performance optimistic. | Dedicated validation split; test labels used only for final scoring. |
| `best_state = model.state_dict()` | Stored tensors can change as training continues; the “best” model is lost. | Detached, cloned checkpoint tensors; restore the minimum-validation-loss epoch. |
| Train/test overlap checked only when the two gene lists were exactly identical, including order | Partially overlapping or reordered gene lists can leak examples. | Explicit disjoint sequence-ID splits and overlap checks. Sequence overlap is checked explicitly in the transfer experiment too. |
| Predictions created with gradients, then converted with `.numpy()` | Conversion can raise an error; inference also wastes memory. | Evaluation mode plus `torch.inference_mode()`. |
| Validation batch losses averaged equally | A small last batch gets too much weight. | MSE over all validation samples. |
| Crop every protein to the shortest train/test sequence | Discards residues and is not a biological alignment. | Preserve every cgreGFP residue; do not silently crop or claim cross-protein alignment. |
| All training/test tensors put on the GPU; fragile external paths and whole-model pickles | Memory blowups and hard-to-reuse experiments. | Host-side features, GPU minibatches, explicit paths, state dictionaries, hashes, and reusable loaders. |

We reuse **Jannis's architecture**, not his active-learning training pipeline.
His pipeline uses a different Huber delta for training versus validation,
overwrites acquisition weights with ones, and predicts with repeated dropout
passes. Those choices are not copied into this benchmark. All CNNs instead use
the same deterministic single-pass evaluation and training recipe below.

## Training rules and interpreting scores

- **Mean embeddings:** each feature is standardized using training-set statistics
  only. OLS is ordinary sklearn regression. Ridge selects alpha from
  `0.1, 1, 10, 100` using validation MSE. kNN uses Euclidean distance in standardized
  embedding space, choosing `k = 1,3,5,11,21,51` and uniform/distance weighting on
  validation MSE. Original sequence kNN instead uses Hamming distance.
- **Aubin/MLP on mean embeddings:** same layer widths as their one-hot versions;
  input width becomes 640. Adam, learning rate 0.001, batch 32, at most 30 epochs,
  patience 10, MSE. Aubin 1–10–1 has a sigmoid hidden layer; the conventional MLPs
  have ReLU layers of width 64 or 128/64/32. ESM features are nonlinear functions
  of sequence, so a linear head on ESM is not an additive mutation model.
- **All three CNNs:** AdamW, learning rate 0.0003, weight decay 0.0001, batch 64,
  at most 60 epochs, patience 10, MSE, gradient-norm clipping at 1. Targets are
  centered/scaled from training values only and predictions are converted back
  to log10 units. Full ESM features are not standardized. Tensors are float32;
  cuDNN's TF32 convolution setting is recorded (enabled by default on supporting
  GPUs), while TF32 matrix multiplication is disabled. Bitwise equality across
  different GPU generations is not assumed.
- The best validation checkpoint is restored. No fit includes validation or test
  labels. Hyperparameters were fixed before inspecting these new test results.
- **Spearman** measures ranking; **Pearson** measures linear agreement;
  **RMSE** measures error in actual log10 fluorescence units (lower is better).
  **R²** compares squared error with a constant predictor. Pearson squared is
  not generally R². A constant predictor has undefined correlations.
- CV tables show the **mean and sample SD across folds**, not a confidence
  interval. Pooled out-of-fold plots are also saved, but pooled correlations can
  differ from the mean of fold correlations. The holdout and CV evaluations reuse
  the same dataset and are not independent replications.
- These random within-protein splits test interpolation in cgreGFP. They do not
  establish transfer to another GFP or to a deliberately distant mutation split.

## Results, plots, runtimes, and saved models

Every model directory contains `metrics.json`, `predictions.csv`, and either
`model.pt` or `model.pkl`. Neural models also have `history.csv`; mean-feature
models have `feature_scaler.pkl`. Checkpoints and predictions retain original
record IDs. All scores are on the original log10 target scale.

After **all** jobs and the reference timer finish, create the final tables/plots:

```bash
MPLCONFIGDIR=/tmp/gfp-matplotlib python -m scripts.finalize_esm
```

The finalizer checks every prediction's record ID, target, and split, recomputes
metrics, and requires exactly one held-out prediction per sequence/model in CV.
It creates:

- `results/esm_cgreGFP/RESULTS.md`: the human-readable full comparison table.
- `comparison.csv`: holdout and CV Spearman/Pearson/R²/RMSE, fold SDs, and runtimes.
- `holdout_mean_true_vs_predicted.*` and `holdout_cnn_true_vs_predicted.*`:
  true-versus-predicted plots, PNG/PDF. Corresponding `cv_*` plots use out-of-fold
  predictions. The diagonal means a perfect prediction; hexagon color shows density.
- `accuracy_vs_runtime.png`: measured fitting cost versus CV performance.
- `MODEL_oof_predictions.csv`: all 24,516 held-out predictions per ESM model.

Fit time includes training, validation, hyperparameter selection, and checkpoint
handling. Prediction time covers validation plus test, separately. Feature
loading is outside fit time; the additional “including load” column reports the
ESM run's total elapsed time, including reading/checking the feature files. This
can be substantial for the 7.4 GB full tensor on shared storage. That total was
not measured for the older one-hot runs and is left blank. CNN times use a GPU; classical models use one CPU
thread. Exact hosts/GPU types are recorded: timings across different hardware
are descriptive, not a controlled hardware comparison. ESM extraction is a
**shared additional cost**, recorded separately; download, queue waiting, and
plotting are not model fit time. Processing new sequences requires ESM again.

Reuse a trained model, for example:

```bash
python -m scripts.esm_benchmark.predict \
  --model-dir results/esm_cgreGFP/holdout/CNN_new \
  --features esm_embeddings/cgreGFP_t30/residue_embeddings.npy \
  --output /tmp/cgreGFP_predictions_log10.npy --device cuda
```

For a mean-feature regressor, use its model directory and `mean_embeddings.npy`.
Outputs have the same row order as the supplied feature file. For new sequences,
use the same pinned ESM model/layer and token-exclusion procedure. Checkpoints
are fitted to cgreGFP; applying them to another protein is a new transfer experiment.

## Data rules that must not change silently

Mutation indices are validated against each file's own wild-type FASTA and
original residue. Natural-data mutation coordinates start with methionine at 0.
The preparer audits source files separately rather than assuming every file uses
the same convention. `replicates_mean_brightness` is the natural cgreGFP target,
then log10 transformed. amac/pplu use the supplied WT-control-rescaled file;
avGFP already supplies log brightness. Artificial-peak brightness is used on its
provided calibration, synonymous rows are aggregated before log10, and those
landscapes stay separate. The excluded `cgre12minis` file is still not used.
See the detailed guide and preparation manifest for exclusions and source hashes.
