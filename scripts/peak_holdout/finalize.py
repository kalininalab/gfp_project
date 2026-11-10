"""Validate all leave-one-artificial-peak-out fits, summarize, and plot."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.regression_metrics import metrics
from .common import (DATA, DEFAULT, MODELS, MODES, PUBLISHED, TARGETS, digest,
                     read_csv, write_csv)

COLORS = {'aubin_1_10_1': '#0072B2', 'aubin_linear': '#E69F00',
          'mlp_small': '#CC79A7', 'CNN_Jannis_OHE': '#56B4E9'}
LABELS = {'aubin_1_10_1': 'Aubin', 'aubin_linear': 'Linear',
          'mlp_small': 'MLP', 'CNN_Jannis_OHE': 'CNN on OHE'}
MODE_LABELS = {'artificial_only': 'Other artificial peaks',
               'with_natural': 'Other peaks + natural cgreGFP'}
METRICS = [('spearman', 'Spearman ρ'), ('pearson', 'Pearson r'), ('r2', 'R²')]


def collect(directory):
    protocol = json.loads((directory / 'protocol.json').read_text())
    assert digest(DATA) == protocol['dataset_sha256']
    table = {row['record_id']: row for row in read_csv(DATA)}
    tests = read_csv(directory / 'test_records.csv')
    assert digest(directory / 'test_records.csv') == protocol['test_records_sha256']
    results = []
    input_hashes = {}
    for target in TARGETS:
        expected = {row['record_id'] for row in tests if row['target'] == target}
        for mode in MODES:
            for seed in protocol['seeds']:
                for model in MODELS:
                    out = directory / 'fits' / target / mode / f'seed{seed}' / model
                    report = json.loads((out / 'metrics.json').read_text())
                    assert report['protocol_sha256'] == digest(directory / 'protocol.json')
                    assert digest(out / 'predictions.csv') == report['predictions_sha256']
                    predictions = [row for row in read_csv(out / 'predictions.csv') if row['partition'] == 'test']
                    assert {row['record_id'] for row in predictions} == expected
                    assert all(row['gene'] == target for row in predictions)
                    assert all(float(row['y_true']) == float(table[row['record_id']]['target_log10']) for row in predictions)
                    score = metrics([float(row['y_true']) for row in predictions],
                                    [float(row['y_pred']) for row in predictions])
                    for name, _ in METRICS:
                        np.testing.assert_allclose(score[name], report['metrics']['test'][name], rtol=1e-10, atol=1e-12)
                    results.append(dict(target=target, mode=mode, seed=seed, model=model,
                                        n_train=protocol['n_train'], n_validation=protocol['n_validation'], **score))
                    input_hashes[str((out / 'metrics.json').relative_to(directory))] = digest(out / 'metrics.json')
    summary = []
    grouped = defaultdict(list)
    for row in results:
        grouped[row['target'], row['mode'], row['model']].append(row)
    for (target, mode, model), rows in grouped.items():
        item = dict(target=target, mode=mode, model=model, n_runs=len(rows), n_test=rows[0]['n'],
                    n_train=rows[0]['n_train'], n_validation=rows[0]['n_validation'])
        for metric in ['spearman', 'pearson', 'r2', 'rmse']:
            values = [row[metric] for row in rows]
            item[f'{metric}_mean'] = float(np.mean(values))
            item[f'{metric}_sd'] = float(np.std(values, ddof=1))
            item[f'{metric}_runs_json'] = json.dumps(values)
        summary.append(item)
    return protocol, results, summary, input_hashes


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#AAB2BF')
    ax.yaxis.grid(True, color='#DDE2EA', linestyle=':', linewidth=.9)
    ax.tick_params(colors='#344052', labelsize=11)
    ax.set_axisbelow(True)
    ax.axhline(0, color='#737E8D', linewidth=.8)


def plot(summary, output, metric, ylabel):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 16, 'axes.labelsize': 13})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    width = .18
    offsets = np.array([-1.5, -.5, .5, 1.5]) * width
    for ax, target in zip(axes, TARGETS):
        for model, offset in zip(MODELS, offsets):
            rows = {row['mode']: row for row in summary if row['target'] == target and row['model'] == model}
            for mode_index, mode in enumerate(MODES):
                row = rows[mode]
                value, error = row[f'{metric}_mean'], row[f'{metric}_sd']
                xpos = mode_index + offset
                alpha = .52 if mode == 'artificial_only' else 1.0
                ax.bar(xpos, value, width=width * .92, color=COLORS[model], alpha=alpha,
                       edgecolor='white', linewidth=.6, zorder=3,
                       label=LABELS[model] if mode_index == 1 else None)
                ax.errorbar(xpos, value, yerr=error, fmt='none', ecolor='#414B5A',
                            capsize=2.5, linewidth=1, zorder=4)
                if metric == 'r2' and value < -1:
                    ax.scatter(xpos, -.965, marker='v', s=28, color='#253244', zorder=6, clip_on=False)
                observations = json.loads(row[f'{metric}_runs_json'])
                for jitter, observation in zip(np.linspace(-.035, .035, len(observations)), observations):
                    ax.scatter(xpos + jitter, observation, s=11, facecolor='white',
                               edgecolor='#303A48', linewidth=.55, zorder=5, clip_on=True)
        ax.set_title(f'Held-out {target}', color='#233044', pad=10)
        ax.set_xticks([0, 1], ['No natural\ncgreGFP', 'Balanced natural\ncgreGFP'], fontsize=11)
        ax.set_xlim(-.58, 1.58)
        ax.set_ylim((-1, 1) if metric == 'r2' else (0, 1))
        style(ax)
    axes[0].set_ylabel(f'Held-out target {ylabel}')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[model]) for model in MODELS]
    fig.legend(handles, [LABELS[model] for model in MODELS], loc='upper center', ncol=4,
               frameon=False, bbox_to_anchor=(.5, .925), fontsize=12)
    fig.suptitle('Transfer to the three largest artificial cgreGFP peaks', fontsize=21,
                 color='#182535', y=.995)
    note = ('Same total source budget in both conditions; natural cgreGFP is limited to 25% in the balanced condition.'
            + ('  ▼ marks a mean R² below −1.' if metric == 'r2' else ''))
    fig.text(.5, .015, note,
             ha='center', fontsize=11, color='#4D596A')
    fig.subplots_adjust(left=.065, right=.99, top=.80, bottom=.20, wspace=.13)
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(output / f'artificial_peak_holdout_{metric}.{suffix}', dpi=240, facecolor='white')
    plt.close(fig)


def report(protocol, summary, output):
    lines = [
        '# Transfer to held-out artificial cgreGFP peaks', '',
        '## Question', '',
        'Can shotgun landscapes from other cgreGFP peaks predict a non-central artificial peak, and does adding the large natural cgreGFP landscape help?', '',
        '## Design', '',
        'The three largest artificial landscapes are tested separately: `cgre1338`, `cgre4111`, and `cgre132`. For each target, the complete target landscape is excluded from training and validation. Its original 20% test partition is the frozen evaluation set.', '',
        'The artificial sources are explicit below. `cgre9708` is the fourth-largest artificial landscape, so it is not one of the three requested targets, but it is included as a source in every comparison:', '',
        '| Held-out target | Artificial source landscapes |',
        '|---|---|',
        '| `cgre1338` | `cgre132`, `cgre4111`, `cgre9708` |',
        '| `cgre4111` | `cgre132`, `cgre1338`, `cgre9708` |',
        '| `cgre132` | `cgre1338`, `cgre4111`, `cgre9708` |', '',
        f'Both source conditions contain exactly **{protocol["n_train"]:,} training** and **{protocol["n_validation"]:,} validation** sequences:', '',
        '- **No natural cgreGFP:** equal contributions from the other three artificial peaks (2,508 train + 836 validation each).',
        '- **Balanced natural cgreGFP:** equal contributions from the other three artificial peaks and natural cgreGFP (1,881 train + 627 validation each). Natural cgreGFP is therefore 25% of the source data and cannot dominate by sample count.', '',
        'Five seeds (42–46) change source subsampling, model initialization, and training order. The frozen target test is identical across all models, source conditions, and seeds. Checkpoints are selected only by source-validation MSE.', '',
        '## Figures', '',
        '![Spearman](artificial_peak_holdout_spearman.png)', '',
        '![Pearson](artificial_peak_holdout_pearson.png)', '',
        '![R2](artificial_peak_holdout_r2.png)', '',
        'R² is displayed on the requested −1 to 1 axis; exact unclipped values are retained below and in `summary.csv`.', '',
        '## Results', '',
        '| Held-out peak | Source condition | Model | Test n | Spearman | Pearson | R² | RMSE |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for target in TARGETS:
        for mode in MODES:
            for model in MODELS:
                row = next(item for item in summary if item['target'] == target and item['mode'] == mode and item['model'] == model)
                value = lambda name: f'{row[name + "_mean"]:.3f} ± {row[name + "_sd"]:.3f}'
                lines.append(f'| {target} | {MODE_LABELS[mode]} | {LABELS[model]} | {row["n_test"]} | {value("spearman")} | {value("pearson")} | {value("r2")} | {value("rmse")} |')
    lines += ['', 'Values are mean ± run SD over five seeds. `per_run_scores.csv` contains every repeat; `sample_counts.csv` in the frozen run directory records every source contribution.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=PUBLISHED)
    args = parser.parse_args()
    protocol, results, summary, hashes = collect(args.directory)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / 'per_run_scores.csv', results)
    write_csv(args.output / 'summary.csv', summary)
    for name, label in METRICS:
        plot(summary, args.output, name, label)
    report(protocol, summary, args.output)
    (args.output / 'test_records.csv').write_bytes((args.directory / 'test_records.csv').read_bytes())
    (args.output / 'sample_counts.csv').write_bytes((args.directory / 'sample_counts.csv').read_bytes())
    final = dict(status='complete', fits=len(hashes), protocol_sha256=digest(args.directory / 'protocol.json'),
                 input_metrics=hashes, outputs={path.name: digest(path) for path in args.output.iterdir() if path.is_file()})
    (args.output / 'finalization.json').write_text(json.dumps(final, indent=2) + '\n')
    print(f'Validated {len(hashes)} fits and wrote {args.output}')


if __name__ == '__main__':
    main()
