# cgreGFP distribution and predictor study

This is the live status page for the requested two-part report.

1. **Distribution structure:** the completed Gaussian-mixture analysis explains what one- through four-component fits, AIC and BIC say about natural cgreGFP and four artificial peak landscapes.
2. **Predictor comparison:** ProteinBERT, ESM-2 and positional one-hot encodings are compared through matched MLP, XGBoost and CNN predictors. Direct regression and soft classification–regression use the same frozen 60/20/20 splits and seeds 42–46.

The final report will appear at [REPORT.md](REPORT.md) after every trajectory passes validation. It reports R², Pearson, Spearman, RMSE, Wasserstein distance and Jensen–Shannon divergence, with publication-ready comparison, distribution and observed-versus-predicted plots.

ProteinBERT uses the official pretrained checkpoint and its 512-dimensional final global hidden state. Embeddings and the checkpoint are stored under `/data/users/akolchina`; only compact results and figures are written to the repository.

## Live workflow

- Submitted 21 October 2026 as DAG **66532**.
- ProteinBERT embedding job: **66533** (1 GPU).
- ESM-2/OHE XGBoost jobs: **66534–66583** (50 CPU-only jobs).
- After embeddings finish, the DAG releases 25 ProteinBERT-XGBoost CPU jobs and 25 ProteinBERT-MLP GPU jobs, then builds and validates the final report automatically.
- Initial queue check: no held jobs.

The completed current-model reference is already available in the [neural direct-versus-hurdle report](../hurdle_neural_seed42_46/REPORT.md). CNN on OHE remains strongest there (natural cgreGFP direct R² 0.910 ± 0.004); the new workflow tests whether ProteinBERT or XGBoost improves on that reference.
