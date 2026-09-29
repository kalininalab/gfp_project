"""Plot saved natural-cgre test predictions; no retraining or seed selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import DEFAULT, DATA, MODELS, digest, read_csv, write_csv
from scripts.regression_metrics import metrics

NAMES = ['Aubin 1–10–1', 'Small MLP', 'Deep MLP', 'CNN Jannis']
COLORS = ['#4267AC', '#D6813A', '#865BA6', '#BE577C']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    out = args.directory / 'cgre_true_vs_predicted'
    out.mkdir(exist_ok=True)
    expected = {r['record_id']: float(r['target_log10']) for r in read_csv(args.data)
                if r['gene'] == 'cgreGFP' and r['split'] == 'test'}
    records, scores, inputs = {}, [], {}
    for seed in [42, 43, 44]:
        for model in MODELS:
            folder = args.directory / 'fits' / 'cgre' / f'seed{seed}' / model
            report = json.loads((folder / 'metrics.json').read_text())
            path = folder / 'predictions.csv'
            checksum = digest(path)
            assert checksum == report['predictions_sha256'], path
            rows = [r for r in read_csv(path) if r['partition'] == 'natural']
            assert len(rows) == len(expected) and {r['record_id'] for r in rows} == set(expected)
            assert all(r['gene'] == 'cgreGFP' and np.isclose(float(r['y_true']), expected[r['record_id']], rtol=0, atol=1e-12) for r in rows)
            y, pred = np.array([[float(r['y_true']), float(r['y_pred'])] for r in rows]).T
            assert np.isfinite(pred).all()
            records[seed, model] = (y, pred)
            scores.append(dict(seed=seed, model=model, **metrics(y, pred)))
            inputs[str(path)] = checksum
    values = np.concatenate([a for pair in records.values() for a in pair])
    low, high = float(values.min()), float(values.max())
    margin = (high-low)*.045
    limits = (low-margin, high+margin)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    for seed in [42, 43, 44]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        for ax, model, name, color in zip(axes.flat, MODELS, NAMES, COLORS):
            y, pred = records[seed, model]
            score = metrics(y, pred)
            ax.scatter(y, pred, s=7, alpha=.22, color=color, linewidths=0, rasterized=True)
            ax.plot(limits, limits, '--', color='#626B78', linewidth=1, zorder=0)
            ax.set(xlim=limits, ylim=limits, xlabel='Measured log10 fluorescence', ylabel='Predicted log10 fluorescence')
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(name, color=color, fontweight='bold', loc='left')
            ax.text(.04, .96, f"Spearman ρ = {score['spearman']:.3f}\nPearson r = {score['pearson']:.3f}\nR² = {score['r2']:.3f}\nRMSE = {score['rmse']:.3f}",
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        fig.suptitle(f'Natural cgreGFP: measured vs predicted · seed {seed}', fontsize=17, y=.98)
        fig.text(.08, .055, f'Each dot: one held-out sequence (n = {len(expected):,}); dashed line: perfect prediction.\n'
                 'Train / validation / test: 14,709 / 4,903 / 4,904 ≈ 60% / 20% / 20%.\n'
                 'cgre-only transfer controls: aligned one-hot Aubin/MLPs; full native ESM embeddings for CNN.\n'
                 'Individual run predictions; seeds are shown separately, not averaged.', fontsize=9, linespacing=1.5)
        fig.subplots_adjust(left=.08, right=.98, bottom=.19, top=.92, hspace=.23, wspace=.25)
        for extension in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'cgre_test_seed{seed}.{extension}', dpi=220)
        plt.close(fig)
    write_csv(out / 'metrics.csv', scores)
    (out / 'manifest.json').write_text(json.dumps(dict(prediction_hashes=inputs, data_sha256=digest(args.data),
        script_sha256=digest(Path(__file__)), seeds=[42,43,44], training_source='cgre', test_partition='natural'), indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
