"""Validate, summarize and publish the rotating-central artificial-peak benchmark."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.regression_metrics import metrics
from .common import (DATA, DEFAULT, MODELS, MODES, PEAKS, PUBLISHED, digest,
                     read_csv, write_csv)

COLORS = {'aubin_1_10_1': '#0072B2', 'aubin_linear': '#E69F00',
          'mlp_small': '#CC79A7', 'CNN_Jannis_OHE': '#56B4E9'}
LABELS = {'aubin_1_10_1': 'Aubin', 'aubin_linear': 'Linear',
          'mlp_small': 'MLP', 'CNN_Jannis_OHE': 'CNN on OHE'}
CONDITIONS = [('central', 'central_test', 'Central → central'),
              ('central', 'other_peaks_test', 'Central → other peaks'),
              ('other_peaks', 'other_peaks_test', 'Other peaks → other peaks'),
              ('other_peaks', 'central_test', 'Other peaks → central')]
METRICS = [('spearman', 'Spearman ρ'), ('pearson', 'Pearson r'), ('r2', 'R²')]


def collect(directory):
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert protocol['excludes_natural_cgreGFP'] and digest(DATA) == protocol['dataset_sha256']
    test_table = read_csv(directory / 'test_records.csv')
    assert digest(directory / 'test_records.csv') == protocol['test_records_sha256']
    rows, hashes = [], {}
    for central in PEAKS:
        expected = {group: {r['record_id'] for r in test_table
                            if r['central'] == central and r['target_group'] == group}
                    for group in ['central', 'other_peaks']}
        for mode in MODES:
            for seed in protocol['seeds']:
                for model in MODELS:
                    out = directory / 'fits' / central / mode / f'seed{seed}' / model
                    report = json.loads((out / 'metrics.json').read_text())
                    assert report['protocol_sha256'] == digest(directory / 'protocol.json')
                    assert report['central'] == central and report['mode'] == mode
                    assert digest(out / 'predictions.csv') == report['predictions_sha256']
                    predictions = read_csv(out / 'predictions.csv')
                    for partition, group in [('central_test', 'central'), ('other_peaks_test', 'other_peaks')]:
                        selected = [r for r in predictions if r['partition'] == partition]
                        assert {r['record_id'] for r in selected} == expected[group]
                        score = metrics([float(r['y_true']) for r in selected],
                                        [float(r['y_pred']) for r in selected])
                        rows.append(dict(central=central, mode=mode, evaluation=partition,
                                         seed=seed, model=model, **score))
                    hashes[str((out / 'metrics.json').relative_to(directory))] = digest(out / 'metrics.json')
    summary = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['central'], row['mode'], row['evaluation'], row['model']].append(row)
    for (central, mode, evaluation, model), values in grouped.items():
        item = dict(central=central, mode=mode, evaluation=evaluation, model=model,
                    n_test=int(values[0]['n']), seeds=len(values))
        for metric in ['spearman', 'pearson', 'r2', 'rmse', 'wasserstein', 'js_divergence']:
            x = np.array([v[metric] for v in values], dtype=float)
            item[metric + '_mean'] = float(x.mean())
            item[metric + '_sd'] = float(x.std(ddof=1))
        summary.append(item)
    return protocol, rows, summary, hashes


def plot(summary, output, metric, ylabel):
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.4), sharey=True)
    width = .18
    x = np.arange(len(CONDITIONS))
    for ax, central in zip(axes.flat, PEAKS):
        for j, model in enumerate(MODELS):
            means, errors = [], []
            for mode, evaluation, _ in CONDITIONS:
                row = next(r for r in summary if r['central'] == central and r['mode'] == mode
                           and r['evaluation'] == evaluation and r['model'] == model)
                means.append(row[metric + '_mean']); errors.append(row[metric + '_sd'])
            ax.bar(x + (j - 1.5) * width, means, width, yerr=errors, capsize=2.5,
                   color=COLORS[model], edgecolor='white', linewidth=.6, label=LABELS[model])
        ax.axhline(0, color='#56616F', linewidth=.8)
        ax.set_title(f'Central peak: {central}', fontsize=15, pad=9)
        ax.set_xticks(x, [label.replace(' → ', '\n→ ') for _, _, label in CONDITIONS], fontsize=10)
        ax.grid(axis='y', color='#DDE2EA', linestyle=':', linewidth=.9)
        ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
        if metric == 'r2':
            ax.set_ylim(-1, 1)
        else:
            ax.set_ylim(-.05, 1.02)
    axes[0, 0].set_ylabel(ylabel, fontsize=13); axes[1, 0].set_ylabel(ylabel, fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(.5, .945), fontsize=12)
    fig.suptitle('Transferability among four artificial cgreGFP peaks', fontsize=20, y=.992)
    fig.subplots_adjust(top=.87, bottom=.09, left=.075, right=.985, hspace=.31, wspace=.12)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'artificial_peak_holdout_{metric}.{suffix}', dpi=260, facecolor='white')
    plt.close(fig)


def archive_previous(output):
    archive = output / 'archive_previous_interpretation'
    if archive.exists():
        return
    archive.mkdir(parents=True)
    for path in list(output.iterdir()):
        if path.is_file():
            shutil.copy2(path, archive / path.name)
    (archive / 'README.md').write_text(
        '# Superseded interpretation\n\nThese files are the earlier experiment that held out three peaks and compared '
        'sources with versus without natural cgreGFP. They are retained for provenance and are not the corrected '
        'rotating-central experiment.\n')


def report(protocol, summary, output):
    lines = ['# Rotating-central transfer among artificial cgreGFP peaks', '',
        '**Corrected design:** natural `cgreGFP` is excluded completely. The four largest artificial landscapes '
        '(`cgre1338`, `cgre4111`, `cgre132`, `cgre9708`) take turns as the pseudo-natural central peak. In each panel, '
        'the remaining three landscapes form the pooled other-peaks domain.', '',
        '## Protocol', '',
        f'- Every fit uses **{protocol["n_train"]:,} train** and **{protocol["n_validation"]:,} validation** records.',
        '- The central-source fit uses only the panel’s central peak.',
        '- The other-peaks fit divides the same budget equally among the remaining three peaks.',
        '- Each fit is evaluated on both frozen domains, producing central→central, central→other peaks, '
        'other peaks→other peaks and other peaks→central.',
        '- The original 20% tests are frozen. Test labels never enter training, validation, checkpoint selection or subsampling.',
        '- Five seeds (42–46) vary source subsampling, initialization and training order. Bars show mean ± run SD.',
        '- Models are Aubin, linear regression on OHE, MLP and CNN on OHE.', '',
        'The common source budget is set by the smallest landscape (`cgre9708`). This prevents peak size from '
        'changing when the designated central peak rotates. Exact source contributions are in `sample_counts.csv`.', '',
        '## Figures', '',
        '![Spearman](artificial_peak_holdout_spearman.png)', '',
        '![Pearson](artificial_peak_holdout_pearson.png)', '',
        '![R2](artificial_peak_holdout_r2.png)', '',
        'R² is displayed on [−1, 1]; exact unclipped values are in the tables.', '',
        '## Main result', '',
        'Within-domain prediction is consistently stronger than transfer from a single central peak to the pooled other peaks. CNN on OHE gives the strongest cross-domain rank correlation in most rotations, especially when `cgre4111`, `cgre132` or `cgre9708` is central. Positive rank correlations can coexist with negative R² because a model may order variants partly correctly while missing the target landscape’s fluorescence level and scale.', '',
        '## Results', '',
        '| Central peak | Direction | Model | Test n | Spearman | Pearson | R² | RMSE |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for central in PEAKS:
        for mode, evaluation, direction in CONDITIONS:
            for model in MODELS:
                row = next(r for r in summary if r['central'] == central and r['mode'] == mode
                           and r['evaluation'] == evaluation and r['model'] == model)
                value = lambda name: f'{row[name + "_mean"]:.3f} ± {row[name + "_sd"]:.3f}'
                lines.append(f'| {central} | {direction} | {LABELS[model]} | {row["n_test"]} | '
                             f'{value("spearman")} | {value("pearson")} | {value("r2")} | {value("rmse")} |')
    lines += ['', 'The former, incorrectly interpreted experiment is retained under '
              '[`archive_previous_interpretation/`](archive_previous_interpretation/).']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=PUBLISHED)
    args = parser.parse_args()
    protocol, rows, summary, hashes = collect(args.directory)
    args.output.mkdir(parents=True, exist_ok=True)
    archive_previous(args.output)
    write_csv(args.output / 'per_run_scores.csv', rows)
    write_csv(args.output / 'summary.csv', summary)
    for metric, ylabel in METRICS:
        plot(summary, args.output, metric, ylabel)
    report(protocol, summary, args.output)
    for name in ['test_records.csv', 'sample_counts.csv']:
        (args.output / name).write_bytes((args.directory / name).read_bytes())
    final = dict(status='complete', fits=len(hashes), protocol_sha256=digest(args.directory / 'protocol.json'),
                 input_metrics=hashes,
                 outputs={p.name: digest(p) for p in args.output.iterdir() if p.is_file() and p.name != 'finalization.json'})
    (args.output / 'finalization.json').write_text(json.dumps(final, indent=2) + '\n')
    print(f'Validated {len(hashes)} fits and wrote {args.output}')


if __name__ == '__main__':
    main()
