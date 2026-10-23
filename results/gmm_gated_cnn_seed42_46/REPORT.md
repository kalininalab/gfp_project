# Two-GMM-gated CNN predictor

## Question

Can the low/high boundary suggested by a two-component Gaussian mixture provide a better classification–regression predictor than separating only the exact measurement-floor values?

## Model

For each seed and landscape, a two-component Gaussian mixture is fitted to the **training fluorescence labels only**. The intersection of its weighted component densities defines a low/high threshold. A shared CNN on positional one-hot sequence has three outputs: `p_high`, a low-fluorescence regression expert, and a high-fluorescence regression expert. Its final prediction is `prediction = (1 - p_high) × low_prediction + p_high × high_prediction`.

This is a two-expert mixture, not a classical floor hurdle. The validation split selects the neural checkpoint, and the frozen test split is used only once for final evaluation. The fact that four Gaussians fit the marginal histogram better by AIC/BIC does not invalidate this supervised test: here the question is whether a simpler low/high boundary helps sequence-to-fluorescence prediction.

![Comparison](comparison.png)

## Results

| Landscape | Direct CNN R² | Floor-hurdle R² | 2-GMM-gated R² | 2-GMM-gated Pearson | 2-GMM-gated Wasserstein |
|---|---:|---:|---:|---:|---:|
| cgreGFP | 0.910 ± 0.004 | 0.907 ± 0.007 | 0.909 ± 0.004 | 0.954 ± 0.002 | 0.042 ± 0.009 |
| cgre132 | 0.676 ± 0.014 | 0.665 ± 0.017 | 0.573 ± 0.015 | 0.760 ± 0.009 | 0.116 ± 0.019 |
| cgre1338 | 0.444 ± 0.013 | 0.449 ± 0.005 | 0.454 ± 0.009 | 0.679 ± 0.004 | 0.164 ± 0.009 |
| cgre4111 | 0.728 ± 0.013 | 0.741 ± 0.005 | 0.434 ± 0.396 | 0.650 ± 0.280 | 0.340 ± 0.345 |
| cgre9708 | 0.625 ± 0.021 | 0.589 ± 0.040 | 0.592 ± 0.029 | 0.774 ± 0.021 | 0.084 ± 0.022 |

The broader low-fluorescence class does not improve prediction consistently. It approximately matches the direct CNN on cgreGFP and improves R² slightly on cgre1338, but performs worse on cgre132 and cgre9708 and is highly unstable on cgre4111. The direct CNN therefore remains the recommended default. The GMM-gated model is retained as a documented negative result showing that a plausible split of the marginal label distribution need not define a useful boundary in sequence space.

Machine-readable outputs: [summary](summary.csv), [all seeded metrics](all_metrics.csv), [train-only thresholds](thresholds.csv), and [gate-classifier metrics](classifier_metrics.csv).
