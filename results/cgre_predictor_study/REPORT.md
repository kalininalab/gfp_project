# cgreGFP landscape distributions and predictor comparison

## Executive summary

This study first asks whether the fluorescence values form two clean Gaussian groups. It then tests whether pretrained protein encodings improve prediction when the data split and downstream models are held constant.

## Part 1 — What does the fluorescence distribution look like?

The [distribution analysis](../cgre_peak_mixture_analysis/REPORT.md) fits one to four Gaussian components. AIC and BIC reward fit but penalize unnecessary components; lower values are better. Both prefer four components over two for every tested landscape. This means that two Gaussians are too simple as a statistical description. It does not prove four biological states, because the assay floor creates a spike that is not Gaussian.

That result motivates the **soft hurdle** predictor: estimate the probability that a variant is above the measurement floor, predict fluorescence conditional on being above it, and combine the two predictions. Every soft-hurdle result below is compared with a direct regressor using the same encoding, split and seed.

## Part 2 — Do ProteinBERT encodings improve prediction?

### Design

- Five landscapes: natural cgreGFP and four artificial peak datasets.
- Existing frozen 60% train / 20% validation / 20% test splits; the test records are never used for fitting or early stopping.
- Five seeds (42–46). Error bars are standard deviations across seeds.
- Encodings: 512-dimensional final ProteinBERT global state, 640-dimensional mean ESM-2 state, and positional one-hot sequence.
- Predictors: MLP, XGBoost and the current CNN-on-OHE reference. Each is evaluated as direct regression and soft hurdle.
- Main accuracy metric: R² (higher is better). Distribution metric: Wasserstein distance in log₁₀-fluorescence units (lower is better). Pearson, Spearman, RMSE and Jensen–Shannon divergence are retained in the CSV.

![Model comparison](model_comparison.png)

### Best mean R² in each landscape

| Landscape | Encoding | Predictor | Form | R² | Pearson | Wasserstein |
|---|---|---|---|---:|---:|---:|
| cgreGFP | OHE | CNN | direct | 0.910 ± 0.004 | 0.954 ± 0.002 | 0.044 ± 0.010 |
| cgre132 | OHE | CNN | direct | 0.676 ± 0.014 | 0.826 ± 0.007 | 0.075 ± 0.015 |
| cgre1338 | OHE | CNN | hurdle | 0.449 ± 0.005 | 0.675 ± 0.006 | 0.159 ± 0.005 |
| cgre4111 | OHE | CNN | hurdle | 0.741 ± 0.005 | 0.863 ± 0.004 | 0.093 ± 0.011 |
| cgre9708 | OHE | XGBoost | hurdle | 0.672 ± 0.004 | 0.823 ± 0.002 | 0.082 ± 0.003 |

![Best-model distributions and dot plots](best_model_diagnostics.png)

### How to read the metrics

R² measures explained squared error: 1 is perfect, 0 matches predicting the test mean, and negative values are worse. Pearson measures linear agreement and Spearman measures rank agreement. RMSE reports typical error in log₁₀ fluorescence. Wasserstein distance measures the average horizontal movement needed to align the predicted and observed distributions, in the same units as the target. Jensen–Shannon divergence is a bounded, symmetric histogram comparison; 0 means identical distributions.

Machine-readable results: [all seeded metrics](all_metrics.csv) and [five-seed summary](summary.csv). ProteinBERT embeddings and the 192 MB checkpoint live under `/data/users/akolchina`, outside Git.
