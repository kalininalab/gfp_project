"""Shared, reproducible Jannis-OHE fitting and publication plots."""
import csv
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from scripts.active_learning.run import Predictor, infer, input_batch
from scripts.regression_metrics import metrics


def fit_jannis(x, y, train, seed, device='cuda', epochs=50, batch_size=32):
    """Fit Jannis OHE from scratch with the Experiment-1 Adam protocol."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda': torch.cuda.manual_seed_all(seed)
    model = Predictor('CNN_Jannis_OHE', x.shape[1], x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for start in range(0, len(train), batch_size):
            ix = rng.permutation(train) if start == 0 else order
            order = ix
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(input_batch(x, batch, device))
            loss = nn.functional.mse_loss(prediction, torch.as_tensor(y[batch], device=device))
            loss.backward(); optimizer.step(); total += loss.item() * len(batch)
        history.append({'epoch': epoch, 'train_mse': total / len(train)})
    return model, history


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def prediction_plots(y_true, y_pred, title, stem):
    """Save distribution and point plots in PNG/PDF/SVG."""
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    bins = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 45)
    axis.hist(y_true, bins=bins, density=True, histtype='step', linewidth=2, label='True', color='#4477AA')
    axis.hist(y_pred, bins=bins, density=True, histtype='step', linewidth=2, label='Predicted', color='#EE6677')
    axis.set(xlabel='log10 fluorescence', ylabel='Density', title=title); axis.legend(frameon=False)
    axis.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf','svg'): figure.savefig(stem.with_name(stem.name + '_distribution').with_suffix('.' + ext), dpi=240)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 4.6), constrained_layout=True)
    axis.scatter(y_true, y_pred, s=8, alpha=.22, linewidths=0, color='#4477AA', rasterized=True)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axis.plot([lo,hi], [lo,hi], color='black', linestyle='--', linewidth=1)
    axis.set(xlabel='True log10 fluorescence', ylabel='Predicted log10 fluorescence', title=title)
    axis.spines[['top','right']].set_visible(False)
    for ext in ('png','pdf','svg'): figure.savefig(stem.with_name(stem.name + '_scatter').with_suffix('.' + ext), dpi=240)
    plt.close(figure)


def evaluate(model, x, y, test, device, batch_size, title, stem):
    prediction, _ = infer(model, x, test, device, batch_size, samples=1)
    prediction_plots(y[test], prediction, title, stem)
    return prediction, metrics(y[test], prediction)
