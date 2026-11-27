# Preliminary active-learning benchmark on cgreGFP

This benchmark compares CNN-on-OHE MC-dropout and a ten-member MLP ensemble across eight acquisition strategies. It uses three seeds, a fixed 96-sequence plate, ten acquisition rounds, matched splits within each seed, and a frozen held-out test.

## Main outputs

- [Learning curves](figures/learning_curves/) for Spearman, Pearson, R², RMSE, MAE, KL divergence and Wasserstein distance.
- [True versus predicted plots](figures/true_vs_predicted/) and [fitness distributions](figures/distributions/).
- [Error versus uncertainty](figures/error_vs_uncertainty/) and [error versus distance](figures/error_vs_distance/).
- [Correlation heatmaps](figures/correlation_heatmaps/) and [acquired-sequence distributions](figures/acquired_batches/).
- [Performance by true fitness and mutation count](figures/stratified_performance/).

The raw per-seed pool scores, predictions, selected sequences, configurations and resumable checkpoints are stored outside Git at `/data/users/akolchina/gfp_project/cgre_active_learning_seed42_44`. True pool fitness appears there only as a retrospective diagnostic; acquisition scores are computed exclusively from model predictions, uncertainty and/or hidden-space distance.

[Frozen protocol](protocol.json) · [all seeded metrics](all_seeded_metrics.csv) · [acquisition correlations](all_acquisition_correlations.csv) · [batch summary](acquired_batch_summary.csv)
