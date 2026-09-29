"""Plot saved amacGFP+ppluGFP -> natural-cgreGFP test predictions."""
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
SEEDS = [42, 43, 44]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    out = args.directory / 'amac_pplu_to_cgre_true_vs_predicted'
    out.mkdir(exist_ok=True)

    expected = {
        r['record_id']: float(r['target_log10'])
        for r in read_csv(args.data)
        if r['gene'] == 'cgreGFP' and r['split'] == 'test'
    }
    records, score_rows, inputs = {}, [], {}
    for seed in SEEDS:
        for model in MODELS:
            folder = args.directory / 'fits' / 'amac_pplu' / f'seed{seed}' / model
            report = json.loads((folder / 'metrics.json').read_text())
            path = folder / 'predictions.csv'
            checksum = digest(path)
            assert checksum == report['predictions_sha256'], path
            rows = [r for r in read_csv(path) if r['partition'] == 'natural']
            assert len(rows) == len(expected)
            assert {r['record_id'] for r in rows} == set(expected)
            assert all(
                r['gene'] == 'cgreGFP'
                and np.isclose(float(r['y_true']), expected[r['record_id']], rtol=0, atol=1e-12)
                for r in rows
            )
            y, pred = np.array([
                [float(r['y_true']), float(r['y_pred'])] for r in rows
            ]).T
            assert np.isfinite(pred).all()
            records[seed, model] = (y, pred)
            score_rows.append(dict(seed=seed, model=model, **metrics(y, pred)))
            inputs[str(path)] = checksum

    values = np.concatenate([a for pair in records.values() for a in pair])
    low, high = float(values.min()), float(values.max())
    margin = (high - low) * .035
    limits = (low - margin, high + margin)
    plt.rcParams.update({
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    fig, axes = plt.subplots(
        len(SEEDS), len(MODELS), figsize=(15.5, 11.5), sharex=True, sharey=True
    )
    for row, seed in enumerate(SEEDS):
        for col, (model, name, color) in enumerate(zip(MODELS, NAMES, COLORS)):
            ax = axes[row, col]
            y, pred = records[seed, model]
            score = metrics(y, pred)
            ax.scatter(
                y, pred, s=6, alpha=.20, color=color, linewidths=0, rasterized=True
            )
            ax.plot(limits, limits, '--', color='#626B78', linewidth=1, zorder=0)
            ax.set(xlim=limits, ylim=limits)
            ax.set_aspect('equal', adjustable='box')
            if row == 0:
                ax.set_title(name, color=color, fontweight='bold', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Seed {seed}\nPredicted log10 fluorescence')
            if row == len(SEEDS) - 1:
                ax.set_xlabel('Measured log10 fluorescence')
            ax.text(
                .04, .96,
                f"ρ = {score['spearman']:.3f}\n"
                f"r = {score['pearson']:.3f}\n"
                f"R² = {score['r2']:.3f}\n"
                f"RMSE = {score['rmse']:.3f}",
                transform=ax.transAxes, va='top', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.86),
            )

    fig.suptitle(
        'Transfer to natural cgreGFP after training on amacGFP + ppluGFP',
        fontsize=17, y=.985,
    )
    fig.text(
        .06, .025,
        f'Each dot is one held-out natural cgreGFP sequence (n = {len(expected):,}); '
        'the dashed line is perfect prediction.\n'
        'Training: 7,355 amacGFP + 7,354 ppluGFP; validation: 2,452 + 2,451; '
        'test: 4,904 cgreGFP. No labeled cgreGFP was used for training or checkpoint selection.\n'
        'Aubin/MLPs: aligned one-hot inputs. CNN Jannis: full residue-level ESM-2 embeddings. '
        'Rows are independent fits and are not averaged.',
        fontsize=9, linespacing=1.45,
    )
    fig.subplots_adjust(left=.06, right=.99, bottom=.105, top=.945, hspace=.17, wspace=.12)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'amac_pplu_to_cgre_all_seeds.{extension}', dpi=220)
    plt.close(fig)

    write_csv(out / 'metrics.csv', score_rows)
    (out / 'manifest.json').write_text(json.dumps({
        'prediction_hashes': inputs,
        'data_sha256': digest(args.data),
        'script_sha256': digest(Path(__file__)),
        'seeds': SEEDS,
        'training_source': 'amac_pplu',
        'test_partition': 'natural',
        'train_counts': {'amacGFP': 7355, 'ppluGFP': 7354},
        'validation_counts': {'amacGFP': 2452, 'ppluGFP': 2451},
        'test_counts': {'cgreGFP': len(expected)},
    }, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
