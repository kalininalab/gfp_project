# GFP transfer experiment results

5 seeds (42–46), with fixed held-out targets. Values are mean ± run SD, not confidence intervals.
Each fit uses 14,709 training and 4,903 source-validation records.
Natural target: 4,904 test sequences; artificial target: 5,382. Each is 20% of its target landscape.
Natural mixtures have equal source contributions; artificial training is proportional across the four peak datasets.

| Experiment | Training / direction | Model | Test n | Spearman | Pearson | R² | RMSE | Mean fit (min) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ortholog_mixtures | cgre | Aubin | 4904 | 0.887 ± 0.001 | 0.943 ± 0.008 | 0.888 ± 0.015 | 0.258 ± 0.017 | 2.44 |
| peak_transfer | cgre_to_artificial | Aubin | 5382 | 0.712 ± 0.047 | 0.731 ± 0.043 | -3.168 ± 0.386 | 1.384 ± 0.063 | 2.44 |
| peak_transfer | cgre_to_natural | Aubin | 4904 | 0.887 ± 0.001 | 0.943 ± 0.008 | 0.888 ± 0.015 | 0.258 ± 0.017 | 2.44 |
| ortholog_mixtures | cgre | Linear | 4904 | 0.833 ± 0.001 | 0.768 ± 0.000 | 0.587 ± 0.001 | 0.495 ± 0.001 | 1.94 |
| peak_transfer | cgre_to_artificial | Linear | 5382 | 0.569 ± 0.006 | 0.543 ± 0.007 | -9.532 ± 0.571 | 2.202 ± 0.060 | 1.94 |
| peak_transfer | cgre_to_natural | Linear | 4904 | 0.833 ± 0.001 | 0.768 ± 0.000 | 0.587 ± 0.001 | 0.495 ± 0.001 | 1.94 |
| ortholog_mixtures | cgre | MLP | 4904 | 0.889 ± 0.000 | 0.935 ± 0.001 | 0.873 ± 0.001 | 0.275 ± 0.002 | 3.43 |
| peak_transfer | cgre_to_artificial | MLP | 5382 | 0.744 ± 0.005 | 0.775 ± 0.011 | -2.169 ± 0.246 | 1.207 ± 0.047 | 3.43 |
| peak_transfer | cgre_to_natural | MLP | 4904 | 0.889 ± 0.000 | 0.935 ± 0.001 | 0.873 ± 0.001 | 0.275 ± 0.002 | 3.43 |
| ortholog_mixtures | cgre | CNN on OHE | 4904 | 0.893 ± 0.002 | 0.955 ± 0.003 | 0.912 ± 0.005 | 0.229 ± 0.006 | 15.78 |
| peak_transfer | cgre_to_artificial | CNN on OHE | 5382 | 0.762 ± 0.002 | 0.806 ± 0.008 | -1.860 ± 0.193 | 1.147 ± 0.039 | 15.78 |
| peak_transfer | cgre_to_natural | CNN on OHE | 4904 | 0.893 ± 0.002 | 0.955 ± 0.003 | 0.912 ± 0.005 | 0.229 ± 0.006 | 15.78 |
| ortholog_mixtures | amac | Aubin | 4904 | -0.098 ± 0.488 | -0.108 ± 0.512 | -2.773 ± 2.886 | 1.407 ± 0.574 | 2.57 |
| ortholog_mixtures | amac | Linear | 4904 | 0.448 ± 0.014 | 0.471 ± 0.009 | -27.883 ± 4.186 | 4.136 ± 0.304 | 3.06 |
| ortholog_mixtures | amac | MLP | 4904 | 0.321 ± 0.022 | 0.321 ± 0.029 | -13.416 ± 1.795 | 2.924 ± 0.183 | 4.66 |
| ortholog_mixtures | amac | CNN on OHE | 4904 | 0.029 ± 0.231 | -0.004 ± 0.210 | -1.068 ± 0.023 | 1.109 ± 0.006 | 17.13 |
| ortholog_mixtures | pplu | Aubin | 4904 | -0.036 ± 0.204 | -0.016 ± 0.273 | -1.221 ± 1.440 | 1.109 ± 0.339 | 2.90 |
| ortholog_mixtures | pplu | Linear | 4904 | 0.189 ± 0.009 | 0.268 ± 0.005 | -71.969 ± 7.398 | 6.581 ± 0.341 | 2.59 |
| ortholog_mixtures | pplu | MLP | 4904 | 0.099 ± 0.027 | 0.117 ± 0.056 | -23.529 ± 2.942 | 3.814 ± 0.228 | 4.06 |
| ortholog_mixtures | pplu | CNN on OHE | 4904 | -0.025 ± 0.049 | -0.032 ± 0.046 | -1.177 ± 0.054 | 1.138 ± 0.014 | 15.64 |
| ortholog_mixtures | cgre_amac | Aubin | 4904 | 0.837 ± 0.004 | 0.878 ± 0.007 | 0.764 ± 0.015 | 0.374 ± 0.012 | 3.50 |
| ortholog_mixtures | cgre_amac | Linear | 4904 | 0.802 ± 0.004 | 0.741 ± 0.003 | 0.546 ± 0.004 | 0.520 ± 0.002 | 2.85 |
| ortholog_mixtures | cgre_amac | MLP | 4904 | 0.872 ± 0.002 | 0.913 ± 0.003 | 0.833 ± 0.006 | 0.315 ± 0.006 | 4.74 |
| ortholog_mixtures | cgre_amac | CNN on OHE | 4904 | 0.871 ± 0.005 | 0.931 ± 0.003 | 0.865 ± 0.006 | 0.283 ± 0.006 | 15.41 |
| ortholog_mixtures | cgre_pplu | Aubin | 4904 | 0.825 ± 0.005 | 0.874 ± 0.012 | 0.760 ± 0.023 | 0.378 ± 0.018 | 3.60 |
| ortholog_mixtures | cgre_pplu | Linear | 4904 | 0.793 ± 0.004 | 0.735 ± 0.003 | 0.536 ± 0.004 | 0.525 ± 0.002 | 3.54 |
| ortholog_mixtures | cgre_pplu | MLP | 4904 | 0.871 ± 0.003 | 0.912 ± 0.004 | 0.831 ± 0.007 | 0.317 ± 0.007 | 5.14 |
| ortholog_mixtures | cgre_pplu | CNN on OHE | 4904 | 0.867 ± 0.007 | 0.927 ± 0.005 | 0.855 ± 0.012 | 0.293 ± 0.012 | 15.21 |
| ortholog_mixtures | amac_pplu | Aubin | 4904 | -0.050 ± 0.394 | -0.068 ± 0.432 | -2.075 ± 1.564 | 1.322 ± 0.320 | 3.74 |
| ortholog_mixtures | amac_pplu | Linear | 4904 | 0.394 ± 0.020 | 0.430 ± 0.014 | -17.421 ± 4.903 | 3.286 ± 0.449 | 3.86 |
| ortholog_mixtures | amac_pplu | MLP | 4904 | 0.379 ± 0.016 | 0.372 ± 0.030 | -16.426 ± 2.455 | 3.213 ± 0.223 | 5.01 |
| ortholog_mixtures | amac_pplu | CNN on OHE | 4904 | 0.119 ± 0.122 | 0.109 ± 0.125 | -1.107 ± 0.047 | 1.119 ± 0.012 | 17.17 |
| ortholog_mixtures | cgre_amac_pplu | Aubin | 4904 | 0.791 ± 0.006 | 0.826 ± 0.012 | 0.672 ± 0.024 | 0.441 ± 0.016 | 3.36 |
| ortholog_mixtures | cgre_amac_pplu | Linear | 4904 | 0.771 ± 0.006 | 0.716 ± 0.002 | 0.509 ± 0.004 | 0.540 ± 0.002 | 3.68 |
| ortholog_mixtures | cgre_amac_pplu | MLP | 4904 | 0.852 ± 0.003 | 0.889 ± 0.004 | 0.785 ± 0.009 | 0.358 ± 0.008 | 5.32 |
| ortholog_mixtures | cgre_amac_pplu | CNN on OHE | 4904 | 0.854 ± 0.007 | 0.906 ± 0.010 | 0.819 ± 0.018 | 0.328 ± 0.016 | 16.99 |
| peak_transfer | artificial_to_artificial | Aubin | 5382 | 0.759 ± 0.002 | 0.827 ± 0.008 | 0.680 ± 0.015 | 0.384 ± 0.009 | 1.66 |
| peak_transfer | artificial_to_natural | Aubin | 4904 | 0.841 ± 0.007 | 0.869 ± 0.025 | -1.818 ± 0.388 | 1.292 ± 0.089 | 1.66 |
| peak_transfer | artificial_to_artificial | Linear | 5382 | 0.723 ± 0.002 | 0.712 ± 0.001 | 0.506 ± 0.002 | 0.477 ± 0.001 | 1.16 |
| peak_transfer | artificial_to_natural | Linear | 4904 | 0.807 ± 0.002 | 0.736 ± 0.001 | -2.304 ± 0.105 | 1.402 ± 0.022 | 1.16 |
| peak_transfer | artificial_to_artificial | MLP | 5382 | 0.803 ± 0.004 | 0.866 ± 0.001 | 0.745 ± 0.008 | 0.343 ± 0.006 | 2.72 |
| peak_transfer | artificial_to_natural | MLP | 4904 | 0.799 ± 0.048 | 0.766 ± 0.066 | -3.361 ± 0.360 | 1.609 ± 0.067 | 2.72 |
| peak_transfer | artificial_to_artificial | CNN on OHE | 5382 | 0.806 ± 0.005 | 0.879 ± 0.004 | 0.769 ± 0.006 | 0.326 ± 0.004 | 16.62 |
| peak_transfer | artificial_to_natural | CNN on OHE | 4904 | 0.834 ± 0.025 | 0.846 ± 0.038 | -2.248 ± 0.200 | 1.389 ± 0.043 | 16.62 |

See per_peak_scores.csv and artificial_macro_scores.csv to distinguish within-peak prediction from pooled rank effects.
A correlation is marked undefined if any repeated fit predicts a constant; individual runs and defined-run counts remain in the CSVs.
Artificial→Artificial is within represented peaks, not leave-one-peak-out generalization.
The natural cgre test set was used in earlier model comparisons; these are exploratory transfer comparisons, not a new blind test.
CPU/GPU hardware and loading costs are retained in runtimes.csv; timing excludes scheduler queue wait.
