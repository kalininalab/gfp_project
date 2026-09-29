# cgreGFP benchmark results

**Best measured accuracy: CNN_Jannis_OHE.** Across ten folds it has the lowest
mean RMSE (0.227 ± 0.010), highest R² (0.912 ± 0.008), and highest Pearson
(0.956 ± 0.004). Its mean Spearman is 0.900 ± 0.006, only slightly above the
one-hot deep MLP (0.899 ± 0.005); no statistical superiority claim is made.

**Efficient default: one-hot Aubin 1–10–1.** Its CV RMSE is 0.234 and R² is
0.908, with a 64.6-second CPU holdout fit. Jannis OHE reduces mean RMSE by about
2.7%, but its holdout fit takes 1,244 seconds (20.7 minutes) on an RTX 2080 Ti.
These timings use different hardware and are measured whole-fit costs, not
per-sample times or a controlled hardware speed comparison.

**OHE beats ESM for this Jannis architecture in this benchmark.** ESM Jannis has
CV RMSE 0.250, R² 0.894 and Spearman 0.891. OHE needs no embedding extraction;
its input is the native 235-residue cgre sequence encoded in 20 channels. The
architecture and training recipe are otherwise the same, with 4,698,881 rather
than 6,920,961 parameters because the first convolution has fewer input channels.
This does not establish which representation transfers better to other proteins;
Jannis OHE has not been added to the separate transfer experiment.

All **121 ESM fits plus 11 Jannis OHE fits** passed the final prediction checks.
Each CV model has one held-out prediction for every one of the 24,516 sequences.
Holdout train/validation/test is 60/20/20; CV is approximately 72/18/10 per fold.
The new model reuses the original splits and seed 42. Validation alone selects
checkpoints. Mean-pooled ESM models remain weaker under the tested settings.

CV values are mean ± sample SD across ten held-out folds. RMSE is in log10 fluorescence units.
R² measures held-out squared-error reduction relative to the test-target mean; it is not Pearson squared and can be negative.
The one-hot models use native cgre positions in this benchmark, not the alignment used in the separate transfer experiments.
Fit time includes validation/early stopping and hyperparameter selection; inference time covers validation + test.
ESM embedding generation: 732.7 s on Tesla V100-PCIE-32GB (shared by all ESM models).

| Features | Model | Holdout Spearman | Holdout Pearson | Holdout R² | CV Spearman | CV Pearson | CV R² | CV RMSE | Holdout fit (s) | Predict (s) | Including load (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one-hot / Hamming | aubin_linear | 0.831 | 0.767 | 0.585 | 0.847 ± 0.004 | 0.777 ± 0.008 | 0.603 ± 0.013 | 0.485 ± 0.008 | 52.8 | 0.36 | — |
| one-hot / Hamming | aubin_1_10_1 | 0.889 | 0.944 | 0.887 | 0.895 ± 0.004 | 0.953 ± 0.005 | 0.908 ± 0.009 | 0.234 ± 0.011 | 64.6 | 0.38 | — |
| one-hot / Hamming | mean | — | — | -0.001 | — | — | -0.000 ± 0.001 | 0.770 ± 0.003 | 0.0 | 0.00 | — |
| one-hot / Hamming | linear_regression | 0.837 | 0.761 | 0.569 | 0.851 ± 0.004 | 0.773 ± 0.008 | 0.591 ± 0.015 | 0.492 ± 0.009 | 2.9 | 0.01 | — |
| one-hot / Hamming | ridge | 0.850 | 0.772 | 0.596 | 0.862 ± 0.005 | 0.782 ± 0.008 | 0.610 ± 0.013 | 0.480 ± 0.008 | 5.7 | 0.01 | — |
| one-hot / Hamming | mlp_small | 0.890 | 0.935 | 0.874 | 0.897 ± 0.005 | 0.938 ± 0.005 | 0.880 ± 0.009 | 0.267 ± 0.009 | 165.7 | 0.58 | — |
| one-hot / Hamming | mlp_deep | 0.890 | 0.947 | 0.896 | 0.899 ± 0.005 | 0.952 ± 0.005 | 0.904 ± 0.011 | 0.238 ± 0.014 | 215.0 | 0.71 | — |
| one-hot / Hamming | knn | 0.714 | 0.653 | -0.024 | 0.683 ± 0.020 | 0.642 ± 0.014 | 0.061 ± 0.025 | 0.746 ± 0.011 | 257.0 | 84.86 | — |
| ESM mean | mean | — | — | -0.001 | — | — | -0.000 ± 0.001 | 0.770 ± 0.003 | 0.1 | 0.01 | 4.968 |
| ESM mean | linear_regression | 0.686 | 0.640 | 0.407 | 0.684 ± 0.014 | 0.647 ± 0.009 | 0.417 ± 0.012 | 0.588 ± 0.006 | 0.7 | 0.01 | 2.937 |
| ESM mean | ridge | 0.701 | 0.654 | 0.426 | 0.706 ± 0.014 | 0.660 ± 0.016 | 0.434 ± 0.024 | 0.579 ± 0.012 | 0.6 | 0.01 | 2.888 |
| ESM mean | aubin_linear | 0.605 | 0.577 | 0.331 | 0.630 ± 0.019 | 0.603 ± 0.015 | 0.363 ± 0.018 | 0.614 ± 0.008 | 40.0 | 0.03 | 42.361 |
| ESM mean | aubin_1_10_1 | 0.614 | 0.629 | 0.394 | 0.635 ± 0.015 | 0.659 ± 0.017 | 0.432 ± 0.022 | 0.580 ± 0.012 | 18.0 | 0.04 | 20.458 |
| ESM mean | mlp_small | 0.710 | 0.697 | 0.482 | 0.739 ± 0.014 | 0.718 ± 0.011 | 0.510 ± 0.017 | 0.539 ± 0.008 | 18.4 | 0.05 | 20.816 |
| ESM mean | mlp_deep | 0.708 | 0.741 | 0.533 | 0.733 ± 0.008 | 0.758 ± 0.011 | 0.567 ± 0.020 | 0.507 ± 0.010 | 30.9 | 0.09 | 33.389 |
| ESM mean | knn | 0.497 | 0.498 | 0.078 | 0.516 ± 0.021 | 0.515 ± 0.017 | 0.115 ± 0.024 | 0.724 ± 0.011 | 40.1 | 6.35 | 49.738 |
| ESM residues | CNN_old | 0.878 | 0.933 | 0.870 | 0.883 ± 0.004 | 0.937 ± 0.006 | 0.877 ± 0.011 | 0.270 ± 0.011 | 1792.1 | 12.38 | 2101.073 |
| ESM residues | CNN_new | 0.883 | 0.938 | 0.879 | 0.883 ± 0.009 | 0.940 ± 0.005 | 0.883 ± 0.010 | 0.263 ± 0.011 | 3477.0 | 26.41 | 3990.798 |
| ESM residues | CNN_Jannis | 0.887 | 0.945 | 0.893 | 0.891 ± 0.006 | 0.946 ± 0.004 | 0.894 ± 0.009 | 0.250 ± 0.011 | 1754.0 | 8.43 | 2259.236 |
| one-hot / native residues | CNN_Jannis_OHE | 0.892 | 0.954 | 0.909 | 0.900 ± 0.006 | 0.956 ± 0.004 | 0.912 ± 0.008 | 0.227 ± 0.010 | 1244.2 | 4.60 | 1252.134 |

CNNs use GPUs; other regressors use one CPU thread. Hardware is recorded in each metrics.json.
These are measured runtimes, not hardware-independent algorithm speed rankings.
Including load adds shared-storage input loading, checksum checks, and output writing; this is not measured for the old one-hot runs.
The original one-hot CV did not record per-model fit time; this is left missing rather than inferred.

Run instructions, CNN architecture differences, and the repaired legacy bugs are in [Running models](RUNNING_MODELS.md). The pinned environment is [environment.yml](../environment.yml).

Full artifacts: [comparison CSV](../results/esm_cgreGFP/comparison.csv), [CNN holdout plots](../results/esm_cgreGFP/holdout_cnn_true_vs_predicted.png), [CNN out-of-fold plots](../results/esm_cgreGFP/cv_cnn_true_vs_predicted.png), [mean-feature out-of-fold plots](../results/esm_cgreGFP/cv_mean_true_vs_predicted.png). Plot annotations use pooled predictions, while the table reports mean fold metrics; these need not be identical.
