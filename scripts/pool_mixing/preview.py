"""Plot a read-only snapshot of completed laboratory-budget fits, including partial seeds.

Run with the combi environment: python -m scripts.pool_mixing.preview.
Outputs are separate from the strict, complete-experiment finalizer.
"""
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data/users/akolchina/gfp_project')
MODELS = ['aubin_1_10_1', 'mlp_small', 'CNN_Jannis_OHE']
LABELS = ['Aubin', 'MLP', 'CNN on OHE']
PLATES = {9: 213, 48: 40, 96: 20, 192: 10, 384: 5}
COLORS = {('random', 'source'): '#0072B2', ('random', 'target'): '#D55E00',
          ('acquisition', 'source'): '#CC79A7', ('acquisition', 'target'): '#6A51A3'}


def kl(path):
    df = pd.read_csv(path, usecols=['evaluation_gene', 'y_true', 'y_pred'])
    result = {}
    for gene, group in df.groupby('evaluation_gene'):
        y, pred = group.y_true.to_numpy(), group.y_pred.to_numpy()
        edges = np.linspace(y.min(), y.max(), 51)
        p = np.histogram(np.clip(y, edges[0], edges[-1]), edges)[0] + .5
        q = np.histogram(np.clip(pred, edges[0], edges[-1]), edges)[0] + .5
        p, q = p / p.sum(), q / q.sum()
        result[gene] = float(np.sum(p * np.log2(p / q)))
    return result


def main():
    stamp = datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds')
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'pdf.fonttype': 42})
    for small in [False, True]:
        base = 'cross_protein_pool_size' + ('_cgre7350' if small else '')
        random = DATA / (base + '_lab_seed42_46_runs')
        acquisition = DATA / (base + '_lab_acquisition_seed42_46_runs')
        output = ROOT / 'results' / (base + '_seed42_46') / 'preliminary'
        output.mkdir(parents=True, exist_ok=True)
        protocol = json.loads((random / 'protocol.json').read_text())
        records, counts = [], defaultdict(lambda: [0, 0])
        for target in ['ppluGFP', 'amacGFP']:
            for model in MODELS:
                sizes = protocol['cnn_random_sizes'] if model == MODELS[-1] else protocol['required_cumulative_sizes']
                def read_random(job):
                    n, seed = job
                    path = random / 'runs' / target / f'n{n:04d}' / f'seed{seed}' / model
                    if not (path / 'complete.json').exists():
                        return n, []
                    frame = pd.read_csv(path / 'metrics.csv')
                    divergence = kl(path / 'predictions.csv')
                    rows = frame.to_dict('records')
                    for row in rows:
                        row['kl_true_to_pred_bits'] = divergence[row['evaluation_gene']]
                    return n, rows

                jobs = [(n, seed) for n in sizes for seed in range(42, 47)]
                counts[('random', model)][1] += len(jobs)
                with ThreadPoolExecutor(max_workers=8) as executor:
                    for n, rows in executor.map(read_random, jobs):
                        if not rows:
                            continue
                        counts[('random', model)][0] += 1
                        for row in rows:
                            for plate, last in PLATES.items():
                                if model == MODELS[-1] and plate not in protocol['cnn_plate_sizes']:
                                    continue
                                if n % plate == 0 and n // plate <= last:
                                    records.append({**row, 'method': 'random', 'plate_size': plate, 'round': n // plate})
                for plate, last in PLATES.items():
                    if model == MODELS[-1] and plate not in protocol['cnn_plate_sizes']:
                        continue
                    for seed in range(42, 47):
                        path = acquisition / 'runs' / target / f'plate{plate:03d}' / f'seed{seed}' / model
                        state = json.loads((path / 'state.json').read_text()) if (path / 'state.json').exists() else {'next_round': 0}
                        for r in range(last + 1):
                            counts[('acquisition', model)][1] += 1
                            if r >= state['next_round']:
                                continue
                            frame = pd.read_csv(path / f'metrics_round_{r:03d}.csv')
                            records.extend({**row, 'method': 'acquisition'} for row in frame.to_dict('records'))
                            counts[('acquisition', model)][0] += 1
                print(base, target, model, 'read', flush=True)
        df = pd.DataFrame(records)
        df.to_csv(output / 'available_seeded_metrics.csv', index=False)
        keys = ['method', 'target_pool', 'model', 'plate_size', 'round', 'evaluation_gene']
        metric_names = ['r2', 'spearman', 'kl_true_to_pred_bits']
        grouped = df.groupby(keys)
        summary = grouped[metric_names].agg(['mean', 'std'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary['n_seeds'] = grouped.seed.nunique()
        summary = summary.reset_index()
        summary.to_csv(output / 'summary.csv', index=False)
        report = [f'# Preliminary laboratory plate results\n\nSnapshot: {stamp}. Source training size: {protocol["base_train_n"]:,}.\n',
                  '**Incomplete experiment.** Each point averages only its available seeds (1–5); hollow points have fewer than five. Shading is ±1 sample SD where at least two seeds exist. Missing rounds are gaps. Changing seed composition can affect the curves; do not treat these as final matched five-seed comparisons. Exact seed counts are in [summary.csv](summary.csv).\n',
                  'Each plate adds target training sequences cumulatively. Models are retrained from new initialization. No target validation labels are used; the source validation set is diagnostic only. Both protein test sets stay frozen. CNN is restricted to plates 96 and 384.\n',
                  '| Sampling | Model | Completed fits/rounds | Total |\n|---|---|---:|---:|']
        for (method, model), (done, total) in counts.items():
            report.append(f'| {method} | {LABELS[MODELS.index(model)]} | {done} | {total} |')
        report.append('\nBlue/orange: random sampling, source/target test. Pink/violet: acquisition sampling, source/target test. Acquisition combines 0.62 scaled hidden-space distance and 0.38 scaled predictive variance.\n')
        for target in ['ppluGFP', 'amacGFP']:
            for metric, ylabel in [('r2', 'R²'), ('spearman', 'Spearman correlation'), ('kl_true_to_pred_bits', 'KL(true || predicted), bits ↓')]:
                fig, axes = plt.subplots(3, 5, figsize=(23, 12), sharex='col', sharey=True, layout='constrained')
                for i, model in enumerate(MODELS):
                    for j, (plate, last) in enumerate(PLATES.items()):
                        ax = axes[i, j]
                        ax.set_title(f'{LABELS[i]} · plate {plate}')
                        if model == MODELS[-1] and plate not in protocol['cnn_plate_sizes']:
                            ax.text(.5, .5, 'Not scheduled', transform=ax.transAxes, ha='center'); ax.set_axis_off(); continue
                        for method in ['random', 'acquisition']:
                            for gene in ['cgreGFP', target]:
                                sub = summary[(summary.target_pool == target) & (summary.model == model) & (summary.plate_size == plate) & (summary.method == method) & (summary.evaluation_gene == gene)].set_index('round').reindex(range(last + 1))
                                x = np.arange(last + 1); y = sub[metric + '_mean'].to_numpy(); sd = sub[metric + '_std'].to_numpy()
                                color = COLORS[(method, 'source' if gene == 'cgreGFP' else 'target')]
                                ax.plot(x, y, color=color, lw=1.8, label=f'{method} · {gene}')
                                ax.fill_between(x, y-sd, y+sd, color=color, alpha=.10)
                                partial = sub.n_seeds.to_numpy() < 5
                                ax.scatter(x[partial], y[partial], facecolors='none', edgecolors=color, s=16)
                        ax.set_xlim(0, last); ax.grid(alpha=.2)
                        if metric == 'r2': ax.set_ylim(-1, 1)
                        if i == 2: ax.set_xlabel('Round')
                        if j == 0: ax.set_ylabel(ylabel)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='outside lower center', ncol=4)
                fig.suptitle(f'Preliminary: cgreGFP + {target} | source train {protocol["base_train_n"]:,}\nAvailable seeds per point: 1–5; hollow markers: incomplete', fontsize=17)
                name = f'cgre_plus_{target}_{metric}'
                for ext in ['png', 'pdf']:
                    fig.savefig(output / f'{name}.{ext}', dpi=180)
                plt.close(fig)
                report.append(f'\n## {target}: {ylabel}\n\n![{ylabel}]({name}.png)\n')
        (output / 'README.md').write_text('\n'.join(report))
        print('COUNTS', base, dict(counts), flush=True)


if __name__ == '__main__':
    main()
