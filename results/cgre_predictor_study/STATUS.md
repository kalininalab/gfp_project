# cgreGFP distribution and predictor study

This is the live status page for the requested two-part report.

1. **Distribution structure:** the completed Gaussian-mixture analysis explains what one- through four-component fits, AIC and BIC say about natural cgreGFP and four artificial peak landscapes.
2. **Predictor comparison:** ProteinBERT, ESM-2 and positional one-hot encodings are compared through matched MLP, XGBoost and CNN predictors. Direct regression and soft classification–regression use the same frozen 60/20/20 splits and seeds 42–46.

The final report will appear at [REPORT.md](REPORT.md) after every trajectory passes validation. It reports R², Pearson, Spearman, RMSE, Wasserstein distance and Jensen–Shannon divergence, with publication-ready comparison, distribution and observed-versus-predicted plots.

ProteinBERT uses the official pretrained checkpoint and its 512-dimensional final global hidden state. Embeddings and the checkpoint are stored under `/data/users/akolchina`; only compact results and figures are written to the repository.
