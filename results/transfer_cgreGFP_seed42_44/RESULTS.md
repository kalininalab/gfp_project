# GFP transfer experiment results

Three seeds (42, 43, 44), with fixed held-out targets. Values are mean ± run SD, not confidence intervals.
Each fit uses 14,709 training and 4,903 source-validation records.
Natural target: 4,904 test sequences; artificial target: 5,382. Each is 20% of its target landscape.
Natural mixtures have equal source contributions; artificial training is proportional across the four peak datasets.

| Experiment | Training / direction | Model | Test n | Spearman | Pearson | R² | RMSE | Mean fit (min) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ortholog_mixtures | cgre | Aubin 1–10–1 | 4904 | 0.887 ± 0.001 | 0.945 ± 0.005 | 0.892 ± 0.009 | 0.254 ± 0.011 | 1.70 |
| peak_transfer | cgre_to_artificial | Aubin 1–10–1 | 5382 | 0.718 ± 0.036 | 0.721 ± 0.022 | -3.325 ± 0.435 | 1.410 ± 0.070 | 1.70 |
| peak_transfer | cgre_to_natural | Aubin 1–10–1 | 4904 | 0.887 ± 0.001 | 0.945 ± 0.005 | 0.892 ± 0.009 | 0.254 ± 0.011 | 1.70 |
| ortholog_mixtures | cgre | Small MLP | 4904 | 0.889 ± 0.000 | 0.935 ± 0.001 | 0.873 ± 0.002 | 0.275 ± 0.002 | 2.81 |
| peak_transfer | cgre_to_artificial | Small MLP | 5382 | 0.748 ± 0.001 | 0.783 ± 0.004 | -2.029 ± 0.187 | 1.181 ± 0.036 | 2.81 |
| peak_transfer | cgre_to_natural | Small MLP | 4904 | 0.889 ± 0.000 | 0.935 ± 0.001 | 0.873 ± 0.002 | 0.275 ± 0.002 | 2.81 |
| ortholog_mixtures | cgre | Deep MLP | 4904 | 0.893 ± 0.001 | 0.948 ± 0.000 | 0.897 ± 0.002 | 0.248 ± 0.002 | 4.25 |
| peak_transfer | cgre_to_artificial | Deep MLP | 5382 | 0.747 ± 0.001 | 0.664 ± 0.018 | -3.443 ± 0.226 | 1.430 ± 0.037 | 4.25 |
| peak_transfer | cgre_to_natural | Deep MLP | 4904 | 0.893 ± 0.001 | 0.948 ± 0.000 | 0.897 ± 0.002 | 0.248 ± 0.002 | 4.25 |
| ortholog_mixtures | cgre | CNN Jannis | 4904 | 0.884 ± 0.004 | 0.944 ± 0.003 | 0.891 ± 0.006 | 0.255 ± 0.007 | 35.63 |
| peak_transfer | cgre_to_artificial | CNN Jannis | 5382 | 0.723 ± 0.022 | 0.772 ± 0.029 | -1.591 ± 0.217 | 1.092 ± 0.046 | 35.63 |
| peak_transfer | cgre_to_natural | CNN Jannis | 4904 | 0.884 ± 0.004 | 0.944 ± 0.003 | 0.891 ± 0.006 | 0.255 ± 0.007 | 35.63 |
| ortholog_mixtures | amac | Aubin 1–10–1 | 4904 | -0.162 ± 0.518 | -0.171 ± 0.543 | -2.517 ± 2.994 | 1.364 ± 0.590 | 1.79 |
| ortholog_mixtures | amac | Small MLP | 4904 | 0.323 ± 0.029 | 0.326 ± 0.040 | -13.471 ± 2.366 | 2.927 ± 0.241 | 2.83 |
| ortholog_mixtures | amac | Deep MLP | 4904 | -0.275 ± 0.265 | -0.296 ± 0.288 | -9.714 ± 2.438 | 2.513 ± 0.299 | 4.25 |
| ortholog_mixtures | amac | CNN Jannis | 4904 | 0.033 ± 0.068 | -0.010 ± 0.037 | -1.076 ± 0.064 | 1.111 ± 0.017 | 35.86 |
| ortholog_mixtures | pplu | Aubin 1–10–1 | 4904 | -0.057 ± 0.208 | -0.029 ± 0.294 | -0.773 ± 0.521 | 1.020 ± 0.146 | 1.44 |
| ortholog_mixtures | pplu | Small MLP | 4904 | 0.109 ± 0.032 | 0.139 ± 0.059 | -23.721 ± 3.827 | 3.827 ± 0.297 | 3.13 |
| ortholog_mixtures | pplu | Deep MLP | 4904 | -0.119 ± 0.054 | -0.167 ± 0.079 | -15.237 ± 5.175 | 3.082 ± 0.490 | 4.29 |
| ortholog_mixtures | pplu | CNN Jannis | 4904 | 0.121 ± 0.065 | 0.100 ± 0.069 | -1.150 ± 0.015 | 1.131 ± 0.004 | 30.93 |
| ortholog_mixtures | cgre_amac | Aubin 1–10–1 | 4904 | 0.838 ± 0.005 | 0.880 ± 0.005 | 0.769 ± 0.011 | 0.371 ± 0.008 | 1.50 |
| ortholog_mixtures | cgre_amac | Small MLP | 4904 | 0.873 ± 0.003 | 0.915 ± 0.002 | 0.834 ± 0.005 | 0.315 ± 0.005 | 3.25 |
| ortholog_mixtures | cgre_amac | Deep MLP | 4904 | 0.871 ± 0.004 | 0.922 ± 0.004 | 0.849 ± 0.006 | 0.300 ± 0.006 | 4.42 |
| ortholog_mixtures | cgre_amac | CNN Jannis | 4904 | 0.864 ± 0.006 | 0.921 ± 0.008 | 0.844 ± 0.016 | 0.304 ± 0.015 | 28.26 |
| ortholog_mixtures | cgre_pplu | Aubin 1–10–1 | 4904 | 0.824 ± 0.006 | 0.869 ± 0.011 | 0.750 ± 0.018 | 0.386 ± 0.014 | 1.64 |
| ortholog_mixtures | cgre_pplu | Small MLP | 4904 | 0.871 ± 0.004 | 0.913 ± 0.005 | 0.831 ± 0.009 | 0.317 ± 0.008 | 3.10 |
| ortholog_mixtures | cgre_pplu | Deep MLP | 4904 | 0.869 ± 0.005 | 0.918 ± 0.006 | 0.842 ± 0.011 | 0.307 ± 0.011 | 3.98 |
| ortholog_mixtures | cgre_pplu | CNN Jannis | 4904 | 0.860 ± 0.008 | 0.914 ± 0.005 | 0.832 ± 0.010 | 0.316 ± 0.009 | 30.80 |
| ortholog_mixtures | amac_pplu | Aubin 1–10–1 | 4904 | -0.075 ± 0.395 | -0.097 ± 0.436 | -2.479 ± 2.009 | 1.399 ± 0.409 | 1.49 |
| ortholog_mixtures | amac_pplu | Small MLP | 4904 | 0.373 ± 0.011 | 0.365 ± 0.033 | -17.871 ± 2.050 | 3.347 ± 0.180 | 3.15 |
| ortholog_mixtures | amac_pplu | Deep MLP | 4904 | 0.291 ± 0.095 | 0.284 ± 0.086 | -14.016 ± 1.535 | 2.986 ± 0.151 | 4.91 |
| ortholog_mixtures | amac_pplu | CNN Jannis | 4904 | -0.084 ± 0.107 | -0.101 ± 0.124 | -1.118 ± 0.021 | 1.122 ± 0.005 | 36.34 |
| ortholog_mixtures | cgre_amac_pplu | Aubin 1–10–1 | 4904 | 0.794 ± 0.006 | 0.828 ± 0.013 | 0.676 ± 0.025 | 0.439 ± 0.017 | 1.48 |
| ortholog_mixtures | cgre_amac_pplu | Small MLP | 4904 | 0.853 ± 0.004 | 0.890 ± 0.002 | 0.789 ± 0.005 | 0.354 ± 0.004 | 3.37 |
| ortholog_mixtures | cgre_amac_pplu | Deep MLP | 4904 | 0.856 ± 0.002 | 0.903 ± 0.002 | 0.810 ± 0.002 | 0.336 ± 0.002 | 4.43 |
| ortholog_mixtures | cgre_amac_pplu | CNN Jannis | 4904 | 0.851 ± 0.006 | 0.901 ± 0.004 | 0.810 ± 0.006 | 0.337 ± 0.005 | 32.99 |
| peak_transfer | artificial_to_artificial | Aubin 1–10–1 | 5382 | 0.759 ± 0.001 | 0.827 ± 0.010 | 0.679 ± 0.020 | 0.384 ± 0.012 | 1.72 |
| peak_transfer | artificial_to_natural | Aubin 1–10–1 | 4904 | 0.843 ± 0.003 | 0.875 ± 0.018 | -1.734 ± 0.352 | 1.274 ± 0.082 | 1.72 |
| peak_transfer | artificial_to_artificial | Small MLP | 5382 | 0.803 ± 0.005 | 0.865 ± 0.001 | 0.742 ± 0.010 | 0.345 ± 0.007 | 2.76 |
| peak_transfer | artificial_to_natural | Small MLP | 4904 | 0.794 ± 0.066 | 0.754 ± 0.089 | -3.233 ± 0.442 | 1.585 ± 0.082 | 2.76 |
| peak_transfer | artificial_to_artificial | Deep MLP | 5382 | 0.801 ± 0.001 | 0.867 ± 0.002 | 0.747 ± 0.000 | 0.342 ± 0.000 | 3.97 |
| peak_transfer | artificial_to_natural | Deep MLP | 4904 | 0.607 ± 0.125 | 0.566 ± 0.128 | -3.521 ± 0.569 | 1.638 ± 0.105 | 3.97 |
| peak_transfer | artificial_to_artificial | CNN Jannis | 5382 | 0.802 ± 0.002 | 0.869 ± 0.002 | 0.751 ± 0.003 | 0.339 ± 0.002 | 32.13 |
| peak_transfer | artificial_to_natural | CNN Jannis | 4904 | 0.824 ± 0.009 | 0.854 ± 0.015 | -1.941 ± 0.051 | 1.323 ± 0.011 | 32.13 |

See per_peak_scores.csv and artificial_macro_scores.csv to distinguish within-peak prediction from pooled rank effects.
A correlation is marked undefined if any repeated fit predicts a constant; individual runs and defined-run counts remain in the CSVs.
Artificial→Artificial is within represented peaks, not leave-one-peak-out generalization.
The natural cgre test set was used in earlier model comparisons; these are exploratory transfer comparisons, not a new blind test.
CPU/GPU hardware and loading costs are retained in runtimes.csv; timing excludes scheduler queue wait.
