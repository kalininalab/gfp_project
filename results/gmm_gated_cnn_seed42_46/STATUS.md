# Two-GMM-gated CNN experiment

This five-seed experiment tests whether a low/high fluorescence division inferred from a two-component Gaussian mixture is more useful than separating only exact measurement-floor observations.

- The GMM boundary is fitted on the training labels only for each landscape and seed.
- A CNN on positional one-hot sequence predicts the probability of the high group and separate continuous predictions from low and high regression experts.
- Direct CNN, floor-hurdle CNN and two-GMM-gated CNN are evaluated on the same frozen test records.
- R², Pearson, Spearman, RMSE, Wasserstein distance and Jensen–Shannon divergence are reported.

The final report will appear at [REPORT.md](REPORT.md) after all 25 trajectories finish and the finalizer validates them.

## Live workflow

- Submitted 22 October 2026 as DAG **66803**.
- CNN trajectories: clusters **66804–66828**.
- Initial queue check: all 25 jobs entered the queue with no holds.
