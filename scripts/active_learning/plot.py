"""Reusable Pearson plots from CSV tables; no training or torch imports required."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'aubin_1_10_1':'Aubin 1–10–1', 'mlp_small':'Small MLP',
          'mlp_deep':'Deep MLP', 'CNN_Jannis_OHE':'Jannis OHE', 'CNN_Jannis_ESM':'Jannis ESM'}


def collect_results(directory, require_complete=False):
    """Load compatible trajectories and report missing models/rounds explicitly.

    Checks common protocol fields and identical initial splits within each seed.
    A final report requires all five models for every discovered seed. Integrity
    of completed metric files is checked against their completion marker.
    """
    directory = Path(directory)
    rows, issues, protocols, splits = [], [], [], {}
    common_fields = ['rounds','budget','alpha','acquisition_mc_samples','evaluation',
                     'batch_size','effective_training_batch_size','epochs_override',
                     'data_sha256','counts','source_hashes','versions']
    seeds = sorted(directory.glob('seed_*'))
    for seed in seeds:
        for model in LABELS:
            folder = seed/model
            if not (folder/'metrics.csv').exists():
                issues.append(f'{seed.name}/{model}: no completed round yet')
                continue
            protocol = json.loads((folder/'protocol.json').read_text())
            common = {key:protocol[key] for key in common_fields}
            if protocols and common != protocols[0]:
                raise ValueError('Incompatible protocols; do not combine these experiments')
            protocols.append(common)
            split_hash = hashlib.sha256((folder/'splits.csv').read_bytes()).hexdigest()
            if seed.name in splits and split_hash != splits[seed.name]:
                raise ValueError(f'Initial split mismatch within {seed.name}')
            splits[seed.name] = split_hash
            with (folder/'metrics.csv').open(newline='') as stream:
                current = list(csv.DictReader(stream))
            if any(r['model'] != model or int(r['seed']) != protocol['seed'] for r in current):
                raise ValueError('Metric identity mismatch')
            expected = {(r,p) for r in range(protocol['rounds']+1)
                        for p in ['remaining_pool','fixed_test']}
            actual = {(int(r['round']),r['partition']) for r in current}
            if len(actual) != len(current) or not actual <= expected:
                raise ValueError('Duplicate or unexpected round/partition metrics')
            if (folder/'complete.json').exists():
                completed = json.loads((folder/'complete.json').read_text())
                if hashlib.sha256((folder/'metrics.csv').read_bytes()).hexdigest() != completed['metrics_sha256']:
                    raise ValueError('Completed metrics checksum mismatch')
                if hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest() != completed['protocol_sha256']:
                    raise ValueError('Completed protocol checksum mismatch')
            else:
                issues.append(f'{seed.name}/{model}: running or interrupted')
            if actual != expected:
                issues.append(f'{seed.name}/{model}: {len(expected-actual)} metric rows still missing')
            if protocol['epochs_override'] is not None:
                issues.append(f'{seed.name}/{model}: smoke-test epoch override')
            rows.extend(current)
    if not rows:
        raise ValueError('No metrics.csv found')
    if require_complete and issues:
        raise ValueError('Incomplete experiment: ' + '; '.join(issues))
    return rows, issues


def write_report(directory, rows, issues):
    """Write a human-readable, explicitly provisional or completed result table."""
    lines = ['# cgreGFP active learning', '',
             '**Status: incomplete / preliminary.**' if issues else '**Status: all discovered trajectories complete.**', '',
             'Primary Pearson uses deterministic predictions on one shared frozen 20% test.',
             'The test is unavailable to training, validation and acquisition.',
             'Initial pools are shared; subsequent acquisition pools differ by model.', '',
             'This pilot fixes the reference diversity caller to pass sequence strings',
             'instead of OHE tensors; that repair can change selected records.', '',
             '| Model | Seed | Initial Pearson | Final Pearson | Change | Train records | Test records |',
             '|---|---:|---:|---:|---:|---:|---:|']
    groups = sorted({(r['model'],r['seed']) for r in rows})
    for model,seed in groups:
        group = [r for r in rows if r['model']==model and r['seed']==seed and r['partition']=='fixed_test']
        first = min(group,key=lambda r:int(r['round']))
        last = max(group,key=lambda r:int(r['round']))
        initial = float(first['pearson'])
        final = float(last['pearson'])
        lines.append(
            f'| {LABELS[model]} | {seed} | {initial:.4f} | {final:.4f} | '
            f'{final-initial:+.4f} | {last["n_train"]} | {last["n"]} |'
        )
    lines.extend(['', 'One seed is a pilot, not an estimate of between-seed variability.',
                  'There is no random-acquisition control in this model-only comparison.', '',
                  '[Protocol and reproduction](../../docs/ACTIVE_LEARNING.md)', '',
                  '![Primary Pearson curves](pearson_fixed_test.png)', '',
                  '[All metric rows](summary.csv) · [PDF](pearson_fixed_test.pdf) · [SVG](pearson_fixed_test.svg)', ''])
    if issues:
        lines.extend(['## Missing or provisional results', ''] + [f'- {issue}' for issue in issues] + [''])
    (Path(directory)/'RESULTS.md').write_text('\n'.join(lines))


def plot_pearson(rows, partition='fixed_test', ax=None):
    """Plot Pearson versus acquired labels; return (figure, axes).

    Parameters
    ----------
    rows : iterable of dict
        Rows from metrics.csv (or merged summary.csv). Required columns: model,
        seed, round, partition, queried_before_fit, pearson. Empty Pearson values
        are undefined correlations and are omitted, never replaced with zero.
    partition : str
        'fixed_test' is the shared primary evaluation population;
        'remaining_pool' reproduces the student's model-specific diagnostic.
    ax : matplotlib.axes.Axes, optional
        Supply an existing axis to embed the plot in a paper figure.

    Each line is a mean across available seeds at that round. Shading is sample
    SD (ddof=1), only with multiple seeds; it is not a confidence interval. Missing
    runs are not imputed. Callers should check completeness before publication.
    """
    rows = [r for r in rows if r['partition'] == partition and r['pearson'] not in ('',None)]
    if ax is None:
        _, ax = plt.subplots(figsize=(7,4.5), constrained_layout=True)
    for model in LABELS:
        subset = [r for r in rows if r['model'] == model]
        if not subset:
            continue
        rounds = sorted({int(r['round']) for r in subset})
        x, mean, sd = [], [], []
        for round_number in rounds:
            group = [r for r in subset if int(r['round']) == round_number]
            if len({r['seed'] for r in group}) != len(group):
                raise ValueError('Duplicate model/seed/round rows')
            budgets = {int(r['queried_before_fit']) for r in group}
            if len(budgets) != 1:
                raise ValueError('Cannot aggregate different acquisition budgets')
            values = [float(r['pearson']) for r in group]
            x.append(budgets.pop()); mean.append(np.mean(values))
            sd.append(np.std(values,ddof=1) if len(values)>1 else 0.)
        line, = ax.plot(x, mean, marker='o', markersize=4, label=LABELS[model])
        if any(sd):
            ax.fill_between(x,np.array(mean)-sd,np.array(mean)+sd,color=line.get_color(),alpha=.15)
    ax.set(xlabel='Pool labels acquired before fitting', ylabel='Pearson r',
           title='Remaining acquisition pool (MC100)' if partition=='remaining_pool' else 'Fixed test (deterministic)')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
    return ax.figure, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--directory',
        type=Path,
        default=Path('results/active_learning_cgre_fixed_test'),
    )
    parser.add_argument('--require-complete', action='store_true', help='Reject missing models/rounds and smoke tests')
    args = parser.parse_args()
    rows, issues = collect_results(args.directory, args.require_complete)
    with (args.directory/'summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for partition in ['remaining_pool','fixed_test']:
        fig, _ = plot_pearson(rows, partition)
        if issues:
            fig.suptitle('Preliminary: incomplete experiment', fontsize=10)
        for extension in ['png','pdf','svg']:
            fig.savefig(args.directory/f'pearson_{partition}.{extension}',dpi=200)
        plt.close(fig)
    write_report(args.directory, rows, issues)
    print(f'Saved {len(rows)} metric rows and Pearson plots to {args.directory}')


if __name__ == '__main__':
    main()
