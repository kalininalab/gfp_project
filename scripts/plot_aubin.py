"""Create a shareable learning-curve and held-out prediction figure."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(run_dir):
    run = json.loads((run_dir/'run.json').read_text())
    architectures = run['architectures']
    fig, axes = plt.subplots(len(architectures), 2, figsize=(10, 4*len(architectures)), squeeze=False)
    for row, name in enumerate(architectures):
        folder = run_dir/name
        with (folder/'history.csv').open() as stream:
            history = list(csv.DictReader(stream))
        with (folder/'predictions.csv').open() as stream:
            predictions = [r for r in csv.DictReader(stream) if r['split']=='test']
        result = run['results'][name]
        ax = axes[row, 0]
        for key, label in [('train_mse', 'Training'), ('validation_mse', 'Validation')]:
            ax.plot([int(r['epoch']) for r in history], [float(r[key]) for r in history], label=label)
        ax.axvline(result['best_epoch'], color='gray', linestyle=':', label='Saved epoch')
        ax.set(xlabel='Epoch', ylabel='MSE (log10 fluorescence)', title=f'Aubin {name}: learning curve')
        ax.legend()
        ax = axes[row, 1]
        actual = [float(r['target_log10']) for r in predictions]
        predicted = [float(r['prediction_log10']) for r in predictions]
        density = ax.hexbin(actual, predicted, gridsize=55, mincnt=1, bins='log', cmap='viridis')
        low, high = min(actual+predicted), max(actual+predicted)
        ax.plot([low, high], [low, high], color='gray', linestyle='--')
        score = result['metrics']['test']
        ax.set(xlabel='Observed log10 fluorescence', ylabel='Predicted log10 fluorescence',
               title=f"Test: Spearman={score['spearman']:.3f}, R²={score['r2']:.3f}, n={score['n']:,}")
        fig.colorbar(density, ax=ax, label='Sequences per bin')
    fig.suptitle(f"{run['gene']} · seed {run['seed']} · fixed 60/20/20 split")
    fig.tight_layout()
    fig.savefig(run_dir/'evaluation.png', dpi=180)
    fig.savefig(run_dir/'evaluation.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    plot(parser.parse_args().run_dir)
