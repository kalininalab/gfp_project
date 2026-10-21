"""Metrics on log10 fluorescence; correlations are undefined for constants."""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_squared_error, r2_score


def metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or len(y) < 2:
        raise ValueError('Expected matching one-dimensional arrays of at least two values')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite targets or predictions')
    mse = float(mean_squared_error(y, prediction))
    constant = np.ptp(prediction) == 0 or np.ptp(y) == 0
    # Wasserstein is expressed in the target's log10-fluorescence units.  The
    # square of scipy's Jensen-Shannon distance is the bounded divergence.
    lower, upper = min(y.min(), prediction.min()), max(y.max(), prediction.max())
    if lower == upper:
        js_divergence = 0.0
    else:
        edges = np.linspace(lower, upper, 51)
        observed, _ = np.histogram(y, bins=edges)
        predicted, _ = np.histogram(prediction, bins=edges)
        js_divergence = float(jensenshannon(observed, predicted, base=2.0) ** 2)
    return dict(n=len(y), mse=mse, rmse=mse**0.5, r2=float(r2_score(y, prediction)),
                spearman=None if constant else float(spearmanr(y, prediction).statistic),
                kendall_tau=None if constant else float(kendalltau(y, prediction).statistic),
                pearson=None if constant else float(pearsonr(y, prediction).statistic),
                wasserstein=float(wasserstein_distance(y, prediction)),
                js_divergence=js_divergence)
