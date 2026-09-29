"""Metrics on log10 fluorescence; correlations are undefined for constants."""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or len(y) < 2:
        raise ValueError('Expected matching one-dimensional arrays of at least two values')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite targets or predictions')
    mse = float(mean_squared_error(y, prediction))
    constant = np.ptp(prediction) == 0 or np.ptp(y) == 0
    return dict(n=len(y), mse=mse, rmse=mse**0.5, r2=float(r2_score(y, prediction)),
                spearman=None if constant else float(spearmanr(y, prediction).statistic),
                pearson=None if constant else float(pearsonr(y, prediction).statistic))
