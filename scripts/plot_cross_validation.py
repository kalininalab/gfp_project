"""Plot fold distributions and pooled out-of-fold predictions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from .evaluate_models import plot_predictions
    from .plot_baselines import LABELS
except ImportError:
    from evaluate_models import plot_predictions
    from plot_baselines import LABELS


def plot(directory):
    run = json.loads((directory/'run.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    with (directory/'fold_metrics.csv').open() as stream:
        scores = list(csv.DictReader(stream))
    names = list(summary)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True, layout='constrained')
    for ax, metric, title in zip(axes, ['spearman','pearson','r2'], ['Spearman','Pearson','R²']):
        for i, name in enumerate(names):
            values = [float(r[metric]) for r in scores if r['model']==name and r[metric]]
            if not values:
                ax.text(0.02, i, 'Undefined within each fold', va='center', fontsize=8)
                continue
            mean = summary[name]['fold_mean'][metric]
            sd = summary[name]['fold_sd'][metric]
            ax.scatter(values, i+np.linspace(-0.14,0.14,len(values)), color='gray', alpha=0.65, s=15)
            ax.errorbar(mean, i, xerr=sd, fmt='o', color='#2166ac', capsize=4, markersize=5)
        ax.set_xlabel(f'{title}: fold mean ± sample SD')
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(len(names)), [LABELS.get(n,n) for n in names])
    axes[0].invert_yaxis()
    fig.suptitle(f"{run['config']['gene']}: 10-fold CV · gray points are individual held-out folds")
    for extension in ['png','pdf']:
        fig.savefig(directory/f'fold_comparison.{extension}', dpi=180)
    plt.close(fig)
    predictions = {}
    for name in names:
        with (directory/f'{name}_oof_predictions.csv').open() as stream:
            predictions[name] = list(csv.DictReader(stream))
    results = {name:{'metrics':{'test':r['pooled_oof']}} for name,r in summary.items()}
    plot_predictions(predictions, results, 'test', f"{run['config']['gene']} (10-fold out-of-fold)", directory)
    for extension in ['png','pdf']:
        (directory/f'true_vs_pred_test.{extension}').rename(directory/f'true_vs_pred_oof.{extension}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    plot(parser.parse_args().directory)
