"""Plot validation and test metrics from the frozen baseline comparison table."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {'aubin_linear': 'Aubin linear', 'aubin_1_10_1': 'Aubin 1–10–1',
          'mean': 'Training mean', 'linear_regression': 'Sklearn linear regression',
          'ridge': 'Ridge', 'mlp_small': 'MLP: 64', 'mlp_deep': 'MLP: 128–64–32', 'knn': 'kNN'}


def plot(run_dir):
    with (run_dir/'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    run = json.loads((run_dir/'run.json').read_text())
    models = list(dict.fromkeys(r['model'] for r in rows))
    scores = {(r['model'], r['split']): r for r in rows}
    metric_names = ['spearman', 'pearson', 'r2'] if 'pearson' in rows[0] else ['spearman', 'r2']
    titles = {'spearman': 'Spearman correlation', 'pearson': 'Pearson correlation', 'r2': 'R²'}
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5.5*len(metric_names), 5.5), sharey=True)
    for ax, metric in zip(axes, metric_names):
        title = titles[metric]
        for split, shift, color in [('validation', -0.15, '#7f8c8d'), ('test', 0.15, '#2471a3')]:
            values = [float(scores[m, split][metric]) if scores[m, split][metric] else np.nan for m in models]
            ax.barh(np.arange(len(models))+shift, values, height=0.28, label=split.capitalize(), color=color)
            for i, value in enumerate(values):
                if np.isfinite(value):
                    ax.text(max(value, 0)+0.012, i+shift, f'{value:.3f}', va='center', fontsize=8)
                elif split == 'test':
                    ax.text(0.02, i, 'Undefined for constant predictions', va='center', fontsize=8)
        ax.set(xlabel=title, xlim=(-0.025, 1.06))
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(models)), [LABELS.get(m, m) for m in models])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    fig.suptitle(f"{run['gene']} · seed {run['seed']} · same frozen splits")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(run_dir/'comparison.png', dpi=180)
    fig.savefig(run_dir/'comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    plot(parser.parse_args().run_dir)
