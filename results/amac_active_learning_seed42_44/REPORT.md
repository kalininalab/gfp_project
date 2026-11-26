# Preliminary active-learning benchmark on amacGFP

This benchmark compares CNN-on-OHE MC-dropout and a ten-member MLP ensemble across eight acquisition strategies. It uses three seeds, a fixed 96-sequence plate, ten acquisition rounds, matched splits within each seed, and a frozen held-out test.

## Main findings

After ten 96-sequence acquisitions, **Random is the strongest overall CNN strategy**: mean Spearman is 0.632, R² is 0.621 and RMSE is 0.256. Distance sampling gives similar rank correlation (0.625) but lower R² (0.541). Low-fitness sampling reaches R² 0.580. Strategies containing MC-dropout uncertainty perform poorly: Uncertainty alone reaches Spearman 0.222, while Uncertainty + distance and Uncertainty + low fitness are approximately uncorrelated with the held-out measurements (Spearman 0.022 and 0.017).

For the **ten-member MLP ensemble**, Distance is the best final strategy by Spearman (0.555) and R² (0.532), but its advantage over Random is small (Spearman 0.551, R² 0.495). Uncertainty + distance reaches Spearman 0.541 and R² 0.514. Random gives the lowest KL divergence (0.645 bits) and Wasserstein distance (0.124), so targeted acquisition does not clearly improve the complete predicted fitness distribution.

| Model | Strategy | Spearman | Pearson | R² | RMSE | KL, bits | Wasserstein |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN on OHE | Random | 0.632 | 0.830 | 0.621 | 0.256 | 4.551 | 0.137 |
| CNN on OHE | Distance | 0.625 | 0.838 | 0.541 | 0.282 | 4.528 | 0.177 |
| CNN on OHE | Uncertainty | 0.222 | 0.283 | 0.162 | 0.377 | 7.654 | 0.288 |
| CNN on OHE | Uncertainty + distance | 0.022 | 0.026 | -0.042 | 0.425 | 8.569 | 0.358 |
| MLP ensemble | Random | 0.551 | 0.708 | 0.495 | 0.296 | **0.645** | **0.124** |
| MLP ensemble | Distance | **0.555** | **0.730** | **0.532** | **0.285** | 0.666 | 0.125 |
| MLP ensemble | Uncertainty | 0.542 | 0.698 | 0.484 | 0.299 | 0.698 | 0.134 |
| MLP ensemble | Uncertainty + distance | 0.541 | 0.719 | 0.514 | 0.290 | 0.708 | 0.131 |

Values are means across seeds 42–44 at round 10. The complete eight-strategy table is in `all_seeded_metrics.csv`.

## Why the strategies differ

Uncertainty is only weakly associated with actual absolute prediction error. Across rounds, the mean Spearman uncertainty–error correlation is 0.159 for CNN MC dropout and 0.102 for the MLP ensemble. Distance is more informative for MLP (mean distance–error Spearman 0.228), which is consistent with its small advantage over Random.

The acquisition rules also change the fitness composition of the measured batches. For MLP, predicted-high-fitness sampling produces an operationally bright fraction of 0.919, predicted-low-fitness sampling 0.380, Distance 0.581 and Random 0.754. Distance also increases batch diversity: mean pairwise Hamming distance is 8.09 residues versus 5.80 for Random. High-fitness sampling therefore spends most of its budget on an already common bright region, while low-fitness and distance sampling cover more dark or sequence-distant variants.

The practical conclusion is that the current MC-dropout variance should **not** be used as the CNN acquisition signal without recalibration. For MLP, hidden-space distance is the most promising tested component, but the improvement over Random is preliminary and small.

### Reproducibility limitation

CNN strategy arms used identical record splits and nominal seeds but were trained as independent GPU jobs. Their round-0 metrics differ slightly, whereas all MLP round-0 metrics are identical. This indicates hardware-level numerical variation between CNN jobs. It does not explain the large late-round collapse of the uncertainty arms, but a confirmatory CNN comparison should reuse one shared round-0 checkpoint per seed before the acquisition paths diverge.

## Main outputs

- [Learning curves](figures/learning_curves/) for Spearman, Pearson, R², RMSE, MAE, KL divergence and Wasserstein distance.
- [True versus predicted plots](figures/true_vs_predicted/) and [fitness distributions](figures/distributions/).
- [Error versus uncertainty](figures/error_vs_uncertainty/) and [error versus distance](figures/error_vs_distance/).
- [Correlation heatmaps](figures/correlation_heatmaps/) and [acquired-sequence distributions](figures/acquired_batches/).
- [Performance by true fitness and mutation count](figures/stratified_performance/).

The raw per-seed pool scores, predictions, selected sequences, configurations and resumable checkpoints are stored outside Git at `/data/users/akolchina/gfp_project/amac_active_learning_seed42_44`. True pool fitness appears there only as a retrospective diagnostic; acquisition scores are computed exclusively from model predictions, uncertainty and/or hidden-space distance.

[Frozen protocol](protocol.json) · [all seeded metrics](all_seeded_metrics.csv) · [acquisition correlations](all_acquisition_correlations.csv) · [batch summary](acquired_batch_summary.csv)
