# Training-size learning curves

All 560 fits validated: four native-OHE models × two training sources × 7 sizes × ten folds.
Each checkpoint is evaluated on both target domains. Natural→natural is the same curve in both figure families.
Scores below are mean ± sample SD across ten folds, not confidence intervals or variation across ten random seeds.

Training sizes: [500, 1000, 2500, 5000, 10000, 15000, 17500]. Every fit has 4,413 additional, fixed source-validation labels.
Outer test fraction is 10% of each target domain. Smaller training samples are nested; unused source records stay unused.
At 17,500 training examples, used train/validation/test proportions are about 71.8/18.1/10.1% for natural targets and 71.1/17.9/10.9% for artificial targets.
At smaller sizes these proportions change; per_fold_scores.csv gives exact counts and ratios. The x-axis is not the total labeling budget.
Artificial folds and samples preserve peak proportions. Artificial→artificial tests variants within represented peaks, not unseen peaks.
Training settings are fixed across sizes (30 dense / 60 CNN maximum epochs); curves measure this recipe, not separately optimized models at each data budget.

[Natural cgre learning curve](cgre_learning_curve_spearman.png) · [Four transfer directions](peak_learning_curves_spearman.png)
Pearson, R² and RMSE companion plots use the corresponding filename suffixes. R² may be negative; it is not Pearson squared.

| Training → test | Model | Training n | Spearman | Pearson | R² | RMSE | Mean fit (min) |
|---|---|---:|---:|---:|---:|---:|---:|
| artificial → artificial | Aubin 1–10–1 | 500 | 0.529 ± 0.012 | 0.541 ± 0.014 | 0.023 ± 0.003 | 0.673 ± 0.003 | 0.39 |
| artificial → artificial | Aubin 1–10–1 | 1,000 | 0.547 ± 0.013 | 0.556 ± 0.010 | 0.297 ± 0.010 | 0.571 ± 0.005 | 0.14 |
| artificial → artificial | Aubin 1–10–1 | 2,500 | 0.613 ± 0.015 | 0.626 ± 0.009 | 0.389 ± 0.010 | 0.532 ± 0.004 | 0.23 |
| artificial → artificial | Aubin 1–10–1 | 5,000 | 0.660 ± 0.012 | 0.661 ± 0.008 | 0.432 ± 0.009 | 0.513 ± 0.004 | 0.32 |
| artificial → artificial | Aubin 1–10–1 | 10,000 | 0.744 ± 0.007 | 0.773 ± 0.007 | 0.593 ± 0.013 | 0.434 ± 0.007 | 0.64 |
| artificial → artificial | Aubin 1–10–1 | 15,000 | 0.762 ± 0.009 | 0.827 ± 0.007 | 0.682 ± 0.014 | 0.384 ± 0.009 | 0.89 |
| artificial → artificial | Aubin 1–10–1 | 17,500 | 0.762 ± 0.009 | 0.840 ± 0.006 | 0.704 ± 0.011 | 0.370 ± 0.007 | 1.03 |
| artificial → artificial | Small MLP | 500 | 0.530 ± 0.011 | 0.556 ± 0.011 | 0.306 ± 0.011 | 0.567 ± 0.005 | 0.48 |
| artificial → artificial | Small MLP | 1,000 | 0.560 ± 0.012 | 0.584 ± 0.010 | 0.337 ± 0.013 | 0.554 ± 0.005 | 0.20 |
| artificial → artificial | Small MLP | 2,500 | 0.633 ± 0.010 | 0.646 ± 0.007 | 0.405 ± 0.015 | 0.525 ± 0.005 | 0.37 |
| artificial → artificial | Small MLP | 5,000 | 0.723 ± 0.014 | 0.750 ± 0.015 | 0.548 ± 0.020 | 0.458 ± 0.011 | 0.84 |
| artificial → artificial | Small MLP | 10,000 | 0.783 ± 0.010 | 0.837 ± 0.008 | 0.695 ± 0.012 | 0.376 ± 0.007 | 1.61 |
| artificial → artificial | Small MLP | 15,000 | 0.804 ± 0.009 | 0.865 ± 0.007 | 0.747 ± 0.013 | 0.343 ± 0.009 | 2.25 |
| artificial → artificial | Small MLP | 17,500 | 0.808 ± 0.009 | 0.872 ± 0.007 | 0.757 ± 0.013 | 0.336 ± 0.009 | 2.51 |
| artificial → artificial | Deep MLP | 500 | 0.530 ± 0.014 | 0.555 ± 0.013 | 0.305 ± 0.013 | 0.568 ± 0.006 | 0.48 |
| artificial → artificial | Deep MLP | 1,000 | 0.566 ± 0.011 | 0.589 ± 0.009 | 0.338 ± 0.015 | 0.554 ± 0.006 | 0.29 |
| artificial → artificial | Deep MLP | 2,500 | 0.641 ± 0.015 | 0.671 ± 0.014 | 0.436 ± 0.022 | 0.511 ± 0.009 | 0.70 |
| artificial → artificial | Deep MLP | 5,000 | 0.714 ± 0.013 | 0.761 ± 0.012 | 0.570 ± 0.020 | 0.446 ± 0.011 | 1.26 |
| artificial → artificial | Deep MLP | 10,000 | 0.778 ± 0.011 | 0.838 ± 0.010 | 0.698 ± 0.017 | 0.374 ± 0.011 | 2.39 |
| artificial → artificial | Deep MLP | 15,000 | 0.805 ± 0.010 | 0.868 ± 0.009 | 0.749 ± 0.017 | 0.341 ± 0.012 | 3.81 |
| artificial → artificial | Deep MLP | 17,500 | 0.807 ± 0.008 | 0.874 ± 0.008 | 0.761 ± 0.014 | 0.333 ± 0.009 | 3.78 |
| artificial → artificial | Jannis OHE | 500 | 0.503 ± 0.027 | 0.523 ± 0.020 | 0.269 ± 0.019 | 0.582 ± 0.006 | 1.81 |
| artificial → artificial | Jannis OHE | 1,000 | 0.571 ± 0.055 | 0.592 ± 0.056 | 0.342 ± 0.058 | 0.552 ± 0.025 | 2.86 |
| artificial → artificial | Jannis OHE | 2,500 | 0.673 ± 0.014 | 0.724 ± 0.023 | 0.510 ± 0.035 | 0.477 ± 0.017 | 5.36 |
| artificial → artificial | Jannis OHE | 5,000 | 0.744 ± 0.011 | 0.811 ± 0.014 | 0.652 ± 0.022 | 0.401 ± 0.012 | 7.56 |
| artificial → artificial | Jannis OHE | 10,000 | 0.790 ± 0.011 | 0.864 ± 0.007 | 0.744 ± 0.011 | 0.344 ± 0.008 | 13.78 |
| artificial → artificial | Jannis OHE | 15,000 | 0.806 ± 0.013 | 0.880 ± 0.008 | 0.772 ± 0.014 | 0.325 ± 0.010 | 17.78 |
| artificial → artificial | Jannis OHE | 17,500 | 0.813 ± 0.009 | 0.886 ± 0.009 | 0.782 ± 0.018 | 0.318 ± 0.013 | 21.16 |
| artificial → natural | Aubin 1–10–1 | 500 | 0.415 ± 0.015 | 0.420 ± 0.016 | -2.392 ± 0.130 | 1.417 ± 0.025 | 0.39 |
| artificial → natural | Aubin 1–10–1 | 1,000 | 0.450 ± 0.013 | 0.448 ± 0.013 | -2.080 ± 0.129 | 1.350 ± 0.027 | 0.14 |
| artificial → natural | Aubin 1–10–1 | 2,500 | 0.605 ± 0.015 | 0.584 ± 0.013 | -1.852 ± 0.176 | 1.299 ± 0.036 | 0.23 |
| artificial → natural | Aubin 1–10–1 | 5,000 | 0.698 ± 0.006 | 0.662 ± 0.011 | -1.885 ± 0.203 | 1.306 ± 0.043 | 0.32 |
| artificial → natural | Aubin 1–10–1 | 10,000 | 0.822 ± 0.009 | 0.832 ± 0.013 | -2.434 ± 0.346 | 1.425 ± 0.069 | 0.64 |
| artificial → natural | Aubin 1–10–1 | 15,000 | 0.842 ± 0.012 | 0.853 ± 0.024 | -2.311 ± 0.342 | 1.399 ± 0.069 | 0.89 |
| artificial → natural | Aubin 1–10–1 | 17,500 | 0.841 ± 0.012 | 0.844 ± 0.028 | -2.314 ± 0.300 | 1.400 ± 0.062 | 1.03 |
| artificial → natural | Small MLP | 500 | 0.360 ± 0.019 | 0.372 ± 0.021 | -2.547 ± 0.142 | 1.449 ± 0.028 | 0.48 |
| artificial → natural | Small MLP | 1,000 | 0.454 ± 0.016 | 0.459 ± 0.016 | -2.450 ± 0.155 | 1.429 ± 0.030 | 0.20 |
| artificial → natural | Small MLP | 2,500 | 0.629 ± 0.015 | 0.622 ± 0.013 | -2.475 ± 0.397 | 1.432 ± 0.082 | 0.37 |
| artificial → natural | Small MLP | 5,000 | 0.756 ± 0.019 | 0.727 ± 0.029 | -3.528 ± 0.718 | 1.633 ± 0.126 | 0.84 |
| artificial → natural | Small MLP | 10,000 | 0.803 ± 0.031 | 0.765 ± 0.047 | -3.894 ± 0.468 | 1.701 ± 0.081 | 1.61 |
| artificial → natural | Small MLP | 15,000 | 0.817 ± 0.026 | 0.782 ± 0.038 | -3.676 ± 0.296 | 1.663 ± 0.054 | 2.25 |
| artificial → natural | Small MLP | 17,500 | 0.822 ± 0.027 | 0.784 ± 0.037 | -3.562 ± 0.342 | 1.643 ± 0.060 | 2.51 |
| artificial → natural | Deep MLP | 500 | 0.371 ± 0.024 | 0.381 ± 0.026 | -2.221 ± 0.202 | 1.380 ± 0.042 | 0.48 |
| artificial → natural | Deep MLP | 1,000 | 0.461 ± 0.020 | 0.467 ± 0.020 | -2.139 ± 0.298 | 1.362 ± 0.063 | 0.29 |
| artificial → natural | Deep MLP | 2,500 | 0.643 ± 0.011 | 0.641 ± 0.019 | -2.486 ± 0.436 | 1.434 ± 0.087 | 0.70 |
| artificial → natural | Deep MLP | 5,000 | 0.710 ± 0.055 | 0.703 ± 0.076 | -2.792 ± 0.523 | 1.495 ± 0.101 | 1.26 |
| artificial → natural | Deep MLP | 10,000 | 0.747 ± 0.096 | 0.736 ± 0.119 | -2.768 ± 0.701 | 1.488 ± 0.137 | 2.39 |
| artificial → natural | Deep MLP | 15,000 | 0.772 ± 0.064 | 0.753 ± 0.088 | -2.776 ± 0.497 | 1.493 ± 0.099 | 3.81 |
| artificial → natural | Deep MLP | 17,500 | 0.793 ± 0.056 | 0.776 ± 0.087 | -2.641 ± 0.487 | 1.465 ± 0.095 | 3.78 |
| artificial → natural | Jannis OHE | 500 | 0.264 ± 0.098 | 0.217 ± 0.113 | -2.570 ± 0.269 | 1.453 ± 0.054 | 1.81 |
| artificial → natural | Jannis OHE | 1,000 | 0.453 ± 0.134 | 0.422 ± 0.168 | -2.452 ± 0.567 | 1.425 ± 0.121 | 2.86 |
| artificial → natural | Jannis OHE | 2,500 | 0.711 ± 0.038 | 0.720 ± 0.041 | -2.236 ± 0.506 | 1.380 ± 0.107 | 5.36 |
| artificial → natural | Jannis OHE | 5,000 | 0.808 ± 0.013 | 0.815 ± 0.018 | -2.146 ± 0.441 | 1.362 ± 0.090 | 7.56 |
| artificial → natural | Jannis OHE | 10,000 | 0.828 ± 0.028 | 0.825 ± 0.040 | -2.420 ± 0.337 | 1.422 ± 0.069 | 13.78 |
| artificial → natural | Jannis OHE | 15,000 | 0.848 ± 0.017 | 0.847 ± 0.035 | -2.284 ± 0.364 | 1.393 ± 0.076 | 17.78 |
| artificial → natural | Jannis OHE | 17,500 | 0.842 ± 0.024 | 0.840 ± 0.031 | -2.361 ± 0.191 | 1.410 ± 0.038 | 21.16 |
| natural → artificial | Aubin 1–10–1 | 500 | 0.123 ± 0.021 | 0.195 ± 0.032 | -2.325 ± 0.146 | 1.241 ± 0.024 | 0.40 |
| natural → artificial | Aubin 1–10–1 | 1,000 | 0.177 ± 0.058 | 0.258 ± 0.058 | -2.942 ± 0.083 | 1.352 ± 0.010 | 0.13 |
| natural → artificial | Aubin 1–10–1 | 2,500 | 0.424 ± 0.174 | 0.444 ± 0.145 | -2.947 ± 0.082 | 1.353 ± 0.010 | 0.18 |
| natural → artificial | Aubin 1–10–1 | 5,000 | 0.514 ± 0.076 | 0.462 ± 0.078 | -9.457 ± 1.183 | 2.199 ± 0.131 | 0.32 |
| natural → artificial | Aubin 1–10–1 | 10,000 | 0.716 ± 0.020 | 0.733 ± 0.024 | -4.317 ± 0.549 | 1.568 ± 0.079 | 0.56 |
| natural → artificial | Aubin 1–10–1 | 15,000 | 0.752 ± 0.008 | 0.759 ± 0.017 | -3.050 ± 0.247 | 1.370 ± 0.043 | 0.80 |
| natural → artificial | Aubin 1–10–1 | 17,500 | 0.710 ± 0.021 | 0.743 ± 0.024 | -2.901 ± 0.196 | 1.345 ± 0.031 | 0.92 |
| natural → artificial | Small MLP | 500 | 0.350 ± 0.170 | 0.328 ± 0.203 | -5.547 ± 1.647 | 1.730 ± 0.218 | 0.50 |
| natural → artificial | Small MLP | 1,000 | 0.463 ± 0.083 | 0.441 ± 0.111 | -8.428 ± 2.371 | 2.077 ± 0.254 | 0.23 |
| natural → artificial | Small MLP | 2,500 | 0.583 ± 0.102 | 0.564 ± 0.116 | -8.129 ± 1.993 | 2.046 ± 0.221 | 0.41 |
| natural → artificial | Small MLP | 5,000 | 0.670 ± 0.044 | 0.678 ± 0.041 | -3.135 ± 0.949 | 1.376 ± 0.166 | 0.69 |
| natural → artificial | Small MLP | 10,000 | 0.718 ± 0.034 | 0.745 ± 0.045 | -2.390 ± 0.358 | 1.252 ± 0.066 | 1.34 |
| natural → artificial | Small MLP | 15,000 | 0.723 ± 0.022 | 0.756 ± 0.030 | -2.447 ± 0.220 | 1.264 ± 0.040 | 1.85 |
| natural → artificial | Small MLP | 17,500 | 0.714 ± 0.028 | 0.745 ± 0.038 | -2.272 ± 0.235 | 1.231 ± 0.042 | 2.03 |
| natural → artificial | Deep MLP | 500 | 0.356 ± 0.178 | 0.335 ± 0.207 | -5.380 ± 1.748 | 1.706 ± 0.233 | 0.54 |
| natural → artificial | Deep MLP | 1,000 | 0.481 ± 0.102 | 0.461 ± 0.125 | -7.755 ± 1.750 | 2.006 ± 0.196 | 0.34 |
| natural → artificial | Deep MLP | 2,500 | 0.620 ± 0.074 | 0.574 ± 0.062 | -4.890 ± 0.874 | 1.648 ± 0.126 | 0.74 |
| natural → artificial | Deep MLP | 5,000 | 0.683 ± 0.059 | 0.616 ± 0.040 | -4.153 ± 0.746 | 1.542 ± 0.114 | 1.18 |
| natural → artificial | Deep MLP | 10,000 | 0.706 ± 0.046 | 0.668 ± 0.053 | -3.565 ± 0.466 | 1.453 ± 0.074 | 2.29 |
| natural → artificial | Deep MLP | 15,000 | 0.743 ± 0.017 | 0.708 ± 0.036 | -3.572 ± 0.482 | 1.454 ± 0.081 | 3.27 |
| natural → artificial | Deep MLP | 17,500 | 0.725 ± 0.075 | 0.687 ± 0.076 | -3.615 ± 0.841 | 1.458 ± 0.124 | 3.45 |
| natural → artificial | Jannis OHE | 500 | 0.282 ± 0.177 | 0.286 ± 0.170 | -3.430 ± 0.910 | 1.427 ± 0.134 | 2.36 |
| natural → artificial | Jannis OHE | 1,000 | 0.555 ± 0.095 | 0.528 ± 0.082 | -3.572 ± 1.011 | 1.448 ± 0.153 | 3.40 |
| natural → artificial | Jannis OHE | 2,500 | 0.683 ± 0.023 | 0.699 ± 0.026 | -2.195 ± 0.347 | 1.216 ± 0.063 | 4.72 |
| natural → artificial | Jannis OHE | 5,000 | 0.695 ± 0.023 | 0.732 ± 0.023 | -1.705 ± 0.314 | 1.118 ± 0.063 | 7.01 |
| natural → artificial | Jannis OHE | 10,000 | 0.731 ± 0.025 | 0.771 ± 0.022 | -1.903 ± 0.344 | 1.158 ± 0.067 | 11.08 |
| natural → artificial | Jannis OHE | 15,000 | 0.746 ± 0.009 | 0.796 ± 0.017 | -1.910 ± 0.297 | 1.160 ± 0.059 | 17.94 |
| natural → artificial | Jannis OHE | 17,500 | 0.754 ± 0.011 | 0.809 ± 0.018 | -1.913 ± 0.191 | 1.162 ± 0.035 | 22.08 |
| natural → natural | Aubin 1–10–1 | 500 | 0.392 ± 0.016 | 0.402 ± 0.012 | -0.035 ± 0.013 | 0.783 ± 0.005 | 0.40 |
| natural → natural | Aubin 1–10–1 | 1,000 | 0.464 ± 0.030 | 0.462 ± 0.027 | 0.000 ± 0.001 | 0.770 ± 0.003 | 0.13 |
| natural → natural | Aubin 1–10–1 | 2,500 | 0.656 ± 0.104 | 0.595 ± 0.076 | 0.004 ± 0.002 | 0.768 ± 0.004 | 0.18 |
| natural → natural | Aubin 1–10–1 | 5,000 | 0.771 ± 0.006 | 0.733 ± 0.008 | 0.528 ± 0.013 | 0.529 ± 0.008 | 0.32 |
| natural → natural | Aubin 1–10–1 | 10,000 | 0.877 ± 0.005 | 0.915 ± 0.007 | 0.834 ± 0.013 | 0.314 ± 0.012 | 0.56 |
| natural → natural | Aubin 1–10–1 | 15,000 | 0.893 ± 0.005 | 0.945 ± 0.005 | 0.892 ± 0.010 | 0.253 ± 0.012 | 0.80 |
| natural → natural | Aubin 1–10–1 | 17,500 | 0.894 ± 0.004 | 0.954 ± 0.005 | 0.909 ± 0.009 | 0.232 ± 0.011 | 0.92 |
| natural → natural | Small MLP | 500 | 0.460 ± 0.015 | 0.472 ± 0.012 | 0.220 ± 0.012 | 0.680 ± 0.005 | 0.50 |
| natural → natural | Small MLP | 1,000 | 0.587 ± 0.017 | 0.589 ± 0.017 | 0.343 ± 0.020 | 0.624 ± 0.011 | 0.23 |
| natural → natural | Small MLP | 2,500 | 0.765 ± 0.009 | 0.766 ± 0.015 | 0.563 ± 0.025 | 0.509 ± 0.014 | 0.41 |
| natural → natural | Small MLP | 5,000 | 0.856 ± 0.004 | 0.883 ± 0.005 | 0.774 ± 0.009 | 0.366 ± 0.007 | 0.69 |
| natural → natural | Small MLP | 10,000 | 0.886 ± 0.005 | 0.925 ± 0.006 | 0.854 ± 0.011 | 0.294 ± 0.010 | 1.34 |
| natural → natural | Small MLP | 15,000 | 0.894 ± 0.005 | 0.934 ± 0.005 | 0.873 ± 0.009 | 0.275 ± 0.010 | 1.85 |
| natural → natural | Small MLP | 17,500 | 0.897 ± 0.005 | 0.938 ± 0.005 | 0.879 ± 0.009 | 0.268 ± 0.010 | 2.03 |
| natural → natural | Deep MLP | 500 | 0.465 ± 0.016 | 0.478 ± 0.013 | 0.220 ± 0.010 | 0.680 ± 0.005 | 0.54 |
| natural → natural | Deep MLP | 1,000 | 0.607 ± 0.020 | 0.621 ± 0.020 | 0.359 ± 0.019 | 0.616 ± 0.010 | 0.34 |
| natural → natural | Deep MLP | 2,500 | 0.796 ± 0.008 | 0.832 ± 0.010 | 0.684 ± 0.017 | 0.432 ± 0.011 | 0.74 |
| natural → natural | Deep MLP | 5,000 | 0.856 ± 0.008 | 0.896 ± 0.007 | 0.800 ± 0.012 | 0.344 ± 0.010 | 1.18 |
| natural → natural | Deep MLP | 10,000 | 0.885 ± 0.006 | 0.934 ± 0.006 | 0.870 ± 0.013 | 0.277 ± 0.013 | 2.29 |
| natural → natural | Deep MLP | 15,000 | 0.895 ± 0.005 | 0.947 ± 0.006 | 0.897 ± 0.012 | 0.247 ± 0.014 | 3.27 |
| natural → natural | Deep MLP | 17,500 | 0.898 ± 0.006 | 0.950 ± 0.005 | 0.902 ± 0.009 | 0.241 ± 0.012 | 3.45 |
| natural → natural | Jannis OHE | 500 | 0.476 ± 0.096 | 0.476 ± 0.091 | 0.107 ± 0.139 | 0.725 ± 0.057 | 2.36 |
| natural → natural | Jannis OHE | 1,000 | 0.684 ± 0.034 | 0.694 ± 0.046 | 0.438 ± 0.157 | 0.573 ± 0.073 | 3.40 |
| natural → natural | Jannis OHE | 2,500 | 0.816 ± 0.012 | 0.859 ± 0.010 | 0.730 ± 0.021 | 0.400 ± 0.015 | 4.72 |
| natural → natural | Jannis OHE | 5,000 | 0.861 ± 0.009 | 0.911 ± 0.006 | 0.828 ± 0.012 | 0.319 ± 0.011 | 7.01 |
| natural → natural | Jannis OHE | 10,000 | 0.881 ± 0.006 | 0.939 ± 0.005 | 0.880 ± 0.010 | 0.266 ± 0.011 | 11.08 |
| natural → natural | Jannis OHE | 15,000 | 0.896 ± 0.007 | 0.952 ± 0.005 | 0.905 ± 0.009 | 0.236 ± 0.011 | 17.94 |
| natural → natural | Jannis OHE | 17,500 | 0.900 ± 0.006 | 0.957 ± 0.005 | 0.915 ± 0.010 | 0.224 ± 0.013 | 22.08 |

Runtimes include fitting and validation, exclude scheduler wait and shared context loading, and use different CPU/GPU hardware.
These are exploratory curves on existing data. Previous CV comparisons used these natural folds; no fresh blind dataset is claimed.
Per-peak and macro summaries distinguish within-peak prediction from pooled effects. Undefined correlations are retained rather than set to zero.
