"""Recompute metrics and true-versus-predicted figures without refitting models."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

try:
    from .regression_metrics import metrics
    from .plot_baselines import LABELS, plot as plot_comparison
    from .train_aubin import sha256
except ImportError:
    from regression_metrics import metrics
    from plot_baselines import LABELS, plot as plot_comparison
    from train_aubin import sha256


def read_predictions(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    identities = {r['record_id']: (r['split'], float(r['target_log10'])) for r in rows}
    if len(identities) != len(rows):
        raise ValueError(f'Duplicate record IDs in {path}')
    if {r['split'] for r in rows} != {'train', 'validation', 'test'}:
        raise ValueError(f'Incomplete partitions in {path}')
    return rows, identities


def plot_predictions(predictions, results, split, gene, output):
    names = list(predictions)
    subsets = {name: [r for r in rows if r['split'] == split] for name, rows in predictions.items()}
    arrays = {name: (np.array([float(r['target_log10']) for r in rows]),
                     np.array([float(r['prediction_log10']) for r in rows])) for name, rows in subsets.items()}
    low = min(float(a.min()) for pair in arrays.values() for a in pair)
    high = max(float(a.max()) for pair in arrays.values() for a in pair)
    padding = max((high-low)*0.04, 0.1)
    limits = (low-padding, high+padding)
    fig, axes = plt.subplots(math.ceil(len(names)/4), 4, figsize=(16, 4.4*math.ceil(len(names)/4)),
                             squeeze=False, layout='constrained')
    densities = []
    for ax, name in zip(axes.flat, names):
        y, pred = arrays[name]
        density = ax.hexbin(y, pred, gridsize=55, mincnt=1, extent=(*limits, *limits), cmap='viridis')
        densities.append(density)
        ax.plot(limits, limits, color='gray', linestyle='--', linewidth=1)
        score = results[name]['metrics'][split]
        fmt = lambda v: 'undefined' if v is None else f'{v:.3f}'
        ax.set(title=f"{LABELS.get(name, name)}\nPearson {fmt(score['pearson'])} · Spearman {fmt(score['spearman'])}\nR² {score['r2']:.3f} · n={score['n']:,}",
               xlabel='True log10 fluorescence', ylabel='Predicted log10 fluorescence',
               xlim=limits, ylim=limits)
        ax.set_aspect('equal', adjustable='box')
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    norm = LogNorm(vmin=1, vmax=max(2, max(float(d.get_array().max()) for d in densities)))
    for density in densities:
        density.set_norm(norm)
    fig.colorbar(densities[0], ax=list(axes.flat)[:len(names)], shrink=0.75, label='Sequences per bin')
    suffix = ' · kNN training predictions include self-neighbors' if split == 'train' and 'knn' in names else ''
    fig.suptitle(f'{gene}: {split} predictions{suffix}', fontsize=14)
    for extension in ['png', 'pdf']:
        fig.savefig(output/f'true_vs_pred_{split}.{extension}', dpi=180)
    plt.close(fig)


def evaluate(runs, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Evaluation output directory must be empty')
    reference = None
    metadata = None
    predictions, results, sources = {}, {}, {}
    for run_dir in runs:
        manifest = json.loads((run_dir/'run.json').read_text())
        fingerprint = (manifest['gene'], manifest['dataset_sha256'])
        if metadata is None:
            metadata = fingerprint
        elif metadata != fingerprint:
            raise ValueError('Runs must use the same protein and dataset')
        sources[str(run_dir/'run.json')] = sha256(run_dir/'run.json')
        for model_name in manifest['results']:
            name = 'aubin_'+model_name if model_name in ['linear', '1_10_1'] else model_name
            if name in predictions:
                raise ValueError(f'Duplicate model: {name}')
            path = run_dir/model_name/'predictions.csv'
            rows, identities = read_predictions(path)
            if reference is None:
                reference = identities
            elif reference != identities:
                raise ValueError(f'Prediction IDs/splits/targets differ: {path}')
            sources[str(path)] = sha256(path)
            predictions[name] = rows
            results[name] = {'metrics': {}}
            for split in ['train', 'validation', 'test']:
                subset = [r for r in rows if r['split'] == split]
                results[name]['metrics'][split] = metrics([float(r['target_log10']) for r in subset],
                                                          [float(r['prediction_log10']) for r in subset])
    output.mkdir(parents=True, exist_ok=True)
    report = dict(gene=metadata[0], dataset_sha256=metadata[1], seed=' / '.join(sorted({str(json.loads((p/'run.json').read_text())['seed']) for p in runs})),
                  sources=sources, target='log10 fluorescence', results=results,
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__),
                  scripts={p.name:sha256(p) for p in [Path(__file__), Path(__file__).with_name('regression_metrics.py'), Path(__file__).with_name('plot_baselines.py')]})
    (output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'split', 'n', 'mse', 'rmse', 'r2', 'spearman', 'pearson'])
        writer.writeheader()
        for name, result in results.items():
            for split, scores in result['metrics'].items():
                writer.writerow(dict(model=name, split=split, **scores))
    for split in ['train', 'validation', 'test']:
        plot_predictions(predictions, results, split, metadata[0], output)
    plot_comparison(output)
    print(json.dumps({name: r['metrics']['test'] for name, r in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.runs, args.output)
