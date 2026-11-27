"""Matched-budget WT/central/other-peak transfer, with source-only validation.

Run `python -m scripts.extended_peak_transfer prepare`, submit the generated
DAG, then use `finalize` (also a dependent DAG job). Each source/model/seed job
fits three nested budgets; completed fits resume without recomputation.
"""
import argparse
import json
import os
from pathlib import Path
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.peak_rotation.common import DATA, PEAKS, ROOT, digest, read_csv, write_csv
from scripts.peak_rotation.run import train_cnn, predict_cnn
from scripts.train_aubin import train, predict_encoded
from scripts.transfer_benchmark.models import AlignedModel, AlignedOHECNN, encode_aligned
from scripts.regression_metrics import metrics

RAW = Path('/data/users/akolchina/gfp_project/extended_peak_transfer_seed42_46')
PUB = ROOT / 'results/extended_peak_transfer_seed42_46'
MODELS = ['mlp_small', 'CNN_Jannis_OHE']
GENES = ['cgreGFP', *PEAKS]


def sources():
    return {'wt': ['cgreGFP'], **{p: [p] for p in PEAKS},
            **{'other_' + p: [g for g in PEAKS if g != p] for p in PEAKS}}


def hashes():
    paths = ['extended_peak_transfer.py', 'peak_rotation/run.py', 'train_aubin.py',
             'transfer_benchmark/models.py', 'aubin_model.py', 'baseline_models.py',
             'esm_benchmark/models.py', 'regression_metrics.py']
    return {p: digest(ROOT / 'scripts' / p) for p in paths}


def allocation(n, k):
    return [n // k + (i < n % k) for i in range(k)]


def prepare(directory):
    if (directory / 'protocol.json').exists():
        raise FileExistsError('Protocol already frozen')
    rows = [r for r in read_csv(DATA) if r['gene'] in GENES]
    assert len({r['sequence'] for r in rows}) == len(rows), 'Duplicate sequences across partitions'
    pools = {(g, s): [r for r in rows if r['gene'] == g and r['split'] == s]
             for g in GENES for s in ['train', 'validation', 'test']}
    maximum = min(len(pools[g, 'train']) for g in GENES)
    validation_n = min(len(pools[g, 'validation']) for g in GENES)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'condor').mkdir(exist_ok=True)
    manifests = {}
    for source, genes in sources().items():
        for seed in range(42, 47):
            ordered = {}
            for split in ['train', 'validation']:
                for g in genes:
                    pool = pools[g, split]
                    rng = np.random.default_rng(np.random.SeedSequence([seed, GENES.index(g), int(split == 'validation'), 2711]))
                    ordered[g, split] = [pool[i] for i in rng.permutation(len(pool))]
            for n in [1000, 2000, maximum]:
                selected = []
                for split, total in [('train', n), ('validation', validation_n)]:
                    for g, budget in zip(genes, allocation(total, len(genes))):
                        assert budget <= len(ordered[g, split])
                        selected += [dict(record_id=r['record_id'], gene=g, split=split)
                                     for r in ordered[g, split][:budget]]
                path = directory / 'subsets' / source / f'{seed}_{n}.csv'
                write_csv(path, selected)
                manifests[str(path.relative_to(directory))] = digest(path)
    tests = [dict(record_id=r['record_id'], gene=r['gene']) for r in rows if r['split'] == 'test']
    write_csv(directory / 'test_records.csv', tests)
    protocol = dict(sources=sources(), sizes=[1000, 2000, maximum], n_validation=validation_n,
                    seeds=list(range(42, 47)), models=MODELS, source_hashes=hashes(),
                    dataset_sha256=digest(DATA), manifests=manifests,
                    tests_sha256=digest(directory / 'test_records.csv'),
                    counts={g: {s: len(pools[g, s]) for s in ['train', 'validation', 'test']} for g in GENES},
                    validation='Fixed source-only validation; same 836-record budget across training sizes and source conditions. Target domains excluded in cross-domain directions.',
                    cnn_settings=dict(epochs=60, patience=10, batch_size=64, lr=.0003, weight_decay=.0001),
                    onehot_settings=dict(epochs=30, patience=10, batch_size=32, lr=.001))
    (directory / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    lines, parents = [], []
    for source in sources():
        for model in MODELS:
            for seed in protocol['seeds']:
                node = f'FIT_{len(parents):03d}'
                parents.append(node)
                gpu = int(model.startswith('CNN'))
                lines += [f'JOB {node} condor/extended_peak_transfer.sub',
                          f'VARS {node} directory="{directory}" source="{source}" model="{model}" seed="{seed}" gpu="{gpu}" memory="{10 if gpu else 6}GB" device="{"cuda" if gpu else "cpu"}"',
                          f'CATEGORY {node} {"GPU" if gpu else "CPU"}']
    lines += ['JOB FINAL condor/extended_peak_finalize.sub',
              f'VARS FINAL directory="{directory}"',
              'PARENT ' + ' '.join(parents) + ' CHILD FINAL', 'MAXJOBS GPU 20', 'MAXJOBS CPU 40']
    (ROOT / 'condor/extended_peak_transfer.dag').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(jobs=len(parents), fits=len(parents)*3, sizes=protocol['sizes'], validation=validation_n)))


def run(args):
    directory = args.directory
    p = json.loads((directory / 'protocol.json').read_text())
    assert p['source_hashes'] == hashes() and p['dataset_sha256'] == digest(DATA)
    assert args.source in p['sources'] and args.seed in p['seeds'] and args.model in MODELS
    assert digest(directory / 'test_records.csv') == p['tests_sha256']
    table = {r['record_id']: r for r in read_csv(DATA)}
    tests = [table[r['record_id']] for r in read_csv(directory / 'test_records.csv')]
    alignment = dict(length=235, mapping={g: list(range(235)) for g in GENES})
    for n in p['sizes']:
        out = directory / 'fits' / args.source / args.model / str(args.seed) / str(n)
        out.mkdir(parents=True, exist_ok=True)
        if (out / 'complete.json').exists():
            assert json.loads((out / 'complete.json').read_text())['protocol_sha256'] == digest(directory / 'protocol.json')
            continue
        path = directory / 'subsets' / args.source / f'{args.seed}_{n}.csv'
        assert digest(path) == p['manifests'][str(path.relative_to(directory))]
        selected = [table[r['record_id']] for r in read_csv(path)]
        training = [r for r in selected if r['split'] == 'train']
        validation = [r for r in selected if r['split'] == 'validation']
        assert len(training) == n and len(validation) == p['n_validation']
        assert all(r['gene'] in p['sources'][args.source] for r in selected)
        assert all(r['split'] == 'test' for r in tests)
        rows = training + validation + tests
        assert len({r['sequence'] for r in rows}) == len(rows)
        ids = dict(train=np.arange(n), validation=np.arange(n, len(selected)), test=np.arange(len(selected), len(rows)))
        x = encode_aligned(rows, alignment)
        y = np.asarray([float(r['target_log10']) for r in rows])
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            if args.model.startswith('CNN'):
                assert args.device == 'cuda'
                model = AlignedOHECNN(235).to(args.device)
                history, best, center, scale = train_cnn(model, x, y, ids, args.seed, args.device, p['cnn_settings'])
                predictions = predict_cnn(model, x, ids['test'], args.device, center, scale)
            else:
                model = AlignedModel(235, args.model)
                s = p['onehot_settings']
                history, best = train(model, x[ids['train']], torch.tensor(y[ids['train']], dtype=torch.float32),
                    x[ids['validation']], torch.tensor(y[ids['validation']], dtype=torch.float32),
                    args.seed, s['epochs'], s['patience'], s['batch_size'], s['lr'])
                center, scale = 0., 1.
                predictions = predict_encoded(model, x[ids['test']]).numpy()
        write_csv(out / 'predictions.csv', [dict(record_id=r['record_id'], gene=r['gene'],
                  y_true=float(r['target_log10']), y_pred=float(v)) for r, v in zip(tests, predictions)])
        write_csv(out / 'history.csv', history)
        torch.save(dict(state_dict=model.cpu().state_dict(), center=center, scale=scale, best_epoch=best), out / 'model.pt')
        (out / 'complete.json').write_text(json.dumps(dict(protocol_sha256=digest(directory / 'protocol.json'),
            prediction_sha256=digest(out / 'predictions.csv'), best_epoch=best)) + '\n')
        print(args.source, args.model, args.seed, n, 'complete', flush=True)


def finalize(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = json.loads((directory / 'protocol.json').read_text())
    rows = []
    expected = {r['record_id'] for r in read_csv(directory / 'test_records.csv')}
    for source in p['sources']:
        for model in MODELS:
            for seed in p['seeds']:
                for n in p['sizes']:
                    out = directory / 'fits' / source / model / str(seed) / str(n)
                    complete = json.loads((out / 'complete.json').read_text())
                    assert complete['protocol_sha256'] == digest(directory / 'protocol.json')
                    assert complete['prediction_sha256'] == digest(out / 'predictions.csv')
                    predictions = read_csv(out / 'predictions.csv')
                    assert {r['record_id'] for r in predictions} == expected
                    for target, genes in sources().items():
                        selected = [r for r in predictions if r['gene'] in genes]
                        score = metrics([float(r['y_true']) for r in selected], [float(r['y_pred']) for r in selected])
                        rows.append(dict(source=source, target=target, model=model, seed=seed, n_train=n, **score))
    PUB.mkdir(parents=True, exist_ok=True)
    write_csv(PUB / 'per_seed_metrics.csv', rows)
    summary = []
    for n in p['sizes']:
        for metric, label in [('pearson', 'Pearson r'), ('spearman', 'Spearman ρ'), ('r2', 'R²')]:
            fig, axes = plt.subplots(2, 4, figsize=(19, 9), sharey=True)
            for col, peak in enumerate(PEAKS):
                conditions = [[('wt', peak), (peak, peak), ('other_'+peak, peak)],
                              [(peak, 'wt'), (peak, peak), (peak, 'other_'+peak)]]
                for row, pairs in enumerate(conditions):
                    ax = axes[row, col]
                    for j, model in enumerate(MODELS):
                        means, sds = [], []
                        for source, target in pairs:
                            values = [r[metric] for r in rows if r['source']==source and r['target']==target and r['model']==model and r['n_train']==n]
                            assert len(values)==5
                            values = np.asarray([np.nan if v is None else v for v in values], float)
                            means.append(float(values.mean())); sds.append(float(values.std(ddof=1)))
                            summary.append(dict(n_train=n, central=peak, source=source, target=target, model=model,
                                                metric=metric, mean=means[-1], sd=sds[-1], n_seeds=5))
                        ax.bar(np.arange(3)+(j-.5)*.34, means, .34, yerr=sds, capsize=3,
                               color=['#CC79A7','#56B4E9'][j], label=['MLP','CNN on OHE'][j])
                        for i, value in enumerate(means):
                            if value < -1: ax.plot(i+(j-.5)*.34, -.97, marker='v', color='#333333', markersize=5)
                    ax.set_title(f'Central: {peak}', fontsize=14)
                    ax.set_xticks(range(3), ['WT → central','Central → central','Other → central'] if row==0 else ['Central → WT','Central → central','Central → other'], rotation=18, ha='right', fontsize=11)
                    ax.set_ylim(-1,1); ax.axhline(0, color='#777777', lw=.7)
                    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
                    ax.spines[['top','right']].set_visible(False)
                    if col==0: ax.set_ylabel(label, fontsize=14)
            fig.suptitle(f'Extended artificial-peak transferability · {n:,} training sequences', fontsize=20)
            fig.legend(*axes[0,0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5,.95), ncol=2, frameon=False)
            fig.tight_layout(rect=(0,.02,1,.9))
            stem = f'extended_transfer_n{n}_{metric}'
            for ext in ['png','pdf','svg']: fig.savefig(PUB / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
            plt.close(fig)
    write_csv(PUB / 'summary.csv', summary)
    report = ['# Extended artificial-peak transferability', '',
              'Completed: 270 fits; MLP and CNN on OHE, five seeds (42–46).', '',
              'WT means the natural cgreGFP landscape. Each artificial peak rotates as central; other means the pooled remaining three artificial peaks, balanced 1:1:1 in training and approximately equally in validation.', '',
              'Training budgets: 1,000, 2,000 and 2,508 records per fit. The maximum is limited by cgre9708 after preserving the original 60/20/20 partitions. Validation is fixed at 836 source-only records at every budget. Cross-domain target landscapes never contribute to validation. All original 20% tests are frozen.', '',
              'Bars show mean ± SD over seeds. Lower-row central → central repeats the same fit as the upper row. Pooled other-peak evaluation uses all three original tests (not equal-sized subsamples). R² below −1 is clipped only in plots and marked with triangles; tables retain full values.', '',
              'This experiment uses the original processed fluorescence values, without the exploratory WT-control rescaling.', '']
    for n in p['sizes']:
        for metric in ['pearson','spearman','r2']:
            report += [f'![n={n}, {metric}](extended_transfer_n{n}_{metric}.png)', '']
    report += ['[Per-seed metrics](per_seed_metrics.csv) · [Summary](summary.csv) · [Protocol](protocol.json)']
    (PUB / 'REPORT.md').write_text('\n'.join(report)+'\n')
    (PUB / 'protocol.json').write_text(json.dumps(p, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','finalize'])
    parser.add_argument('--directory', type=Path, default=RAW)
    parser.add_argument('--source'); parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--seed', type=int); parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.directory)
    elif args.action == 'run': run(args)
    else: finalize(args.directory)
