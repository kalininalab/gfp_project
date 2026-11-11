# Cross-protein labelled-pool size experiment

## Question

When a model is trained on a fixed 7,350-record cgreGFP source pool, how does gradually adding a labelled pool from ppluGFP or amacGFP change prediction on the new protein and on the original cgreGFP protein? This is a supervised pool-size experiment, not active learning: target records are added by a frozen random order, without an acquisition score.

## Design

- Fixed cgreGFP training/validation: 7,350 / 4,903 records.
- Equal target training/validation pools: 14,709 / 4,903 records.
- Plate sizes: 9, 48, 96, 192 and 384 new target training records per round.
- Schedules end near the same cumulative budget: 213 rounds × 9 = 1,917; 40 × 48 = 1,920; 20 × 96 = 1,920; 10 × 192 = 1,920; 5 × 384 = 1,920.
- The cumulative target training size is `plate size × round`; all prefixes are nested within each seed.
- Target validation grows proportionally in the original 60/20 train/validation ratio and is used only for early stopping.
- Models included in this completed result: MLP, Aubin 1–10–1; seeds 42–46.
- Every round is evaluated on the same frozen test: 4,904 cgreGFP records and either 6,281 ppluGFP or 6,703 amacGFP records. Test records never enter training, validation, early stopping or sampling.
- R² is displayed on an ordinary linear scale limited to [−1, 1] for readability. More negative source-only target scores are clipped only in the figure; their complete values remain in the CSV tables.
- KL is D_KL(true || predicted) in bits, using 50 fixed bins over each protein's frozen-test range and a 0.5 pseudocount per bin; lower is better.

| Protein | Available training pool | Available validation pool | Frozen test |
|---|---:|---:|---:|
| cgreGFP | 7,350 used | 4,903 | 4,904 |
| ppluGFP | 14,709 | 4,903 | 6,281 |
| amacGFP | 14,709 | 4,903 | 6,703 |

The fixed 7,350-record cgreGFP subset and complete cgreGFP validation set are
used throughout. For the target protein, each round uses only the cumulative
prefix specified by its plate schedule; the remaining target-pool records are
unused, while the complete frozen test remains unchanged.

## Figures

### cgreGFP + ppluGFP

Each figure contains five panels, one for each plate size; the x-axis is round and ends when the cumulative target set reaches approximately 1,920 sequences.

![R²](figures/cgre_plus_pplu_r2.png)

![Spearman](figures/cgre_plus_pplu_spearman.png)

![KL divergence](figures/cgre_plus_pplu_kl_divergence.png)

True-versus-predicted frozen-test distributions before target addition and after 1,920 cumulative target training sequences; predictions are pooled across the five seeds.

#### MLP

![MLP true and predicted distributions](figures/cgre_plus_pplu_mlp_true_vs_pred_distributions.png)

#### Aubin 1–10–1

![Aubin true and predicted distributions](figures/cgre_plus_pplu_aubin_true_vs_pred_distributions.png)

### cgreGFP + amacGFP

Each figure contains five panels, one for each plate size; the x-axis is round and ends when the cumulative target set reaches approximately 1,920 sequences.

![R²](figures/cgre_plus_amac_r2.png)

![Spearman](figures/cgre_plus_amac_spearman.png)

![KL divergence](figures/cgre_plus_amac_kl_divergence.png)

True-versus-predicted frozen-test distributions before target addition and after 1,920 cumulative target training sequences; predictions are pooled across the five seeds.

#### MLP

![MLP true and predicted distributions](figures/cgre_plus_amac_mlp_true_vs_pred_distributions.png)

#### Aubin 1–10–1

![Aubin true and predicted distributions](figures/cgre_plus_amac_aubin_true_vs_pred_distributions.png)

Machine-readable results: [summary](summary.csv) and [all seeded metrics](all_metrics.csv).

## Code and reuse

The implementation is shared with the 14,709-cgreGFP experiment. The 7,350
source size is set with `--base-train-n 7350`; it is not a separate model
implementation. See the [student-facing code and reproduction guide](../../docs/POOL_SIZE_EXPERIMENT.md)
for the source-file map, a single-fit example and the complete HTCondor workflow.
