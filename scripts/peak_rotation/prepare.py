"""Freeze four rotating-central artificial-peak transfer experiments."""
import argparse
from collections import Counter
import datetime
import json

import numpy as np

from .common import (DATA, DEFAULT, MODELS, MODES, PEAKS, digest, read_csv,
                     scientific_sources, write_csv)

N_TRAIN = 2508
N_VALIDATION = 836


def allocation(total, n):
    return [total // n + int(i < total % n) for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=type(DEFAULT), default=DEFAULT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Frozen protocol already exists; choose a new directory')

    rows = [row for row in read_csv(DATA) if row['gene'] in PEAKS]
    assert rows and all(row['gene'] != 'cgreGFP' for row in rows)
    assert len({row['record_id'] for row in rows}) == len(rows)
    assert len({row['sequence'] for row in rows}) == len(rows), 'Duplicate sequences can leak across landscapes'
    assert {len(row['sequence']) for row in rows} == {235}
    pools = {(gene, split): [row for row in rows if row['gene'] == gene and row['split'] == split]
             for gene in PEAKS for split in ['train', 'validation', 'test']}
    alignment = dict(length=235, mapping={gene: list(range(235)) for gene in PEAKS},
                     method='native cgreGFP coordinates; all four artificial landscapes have length 235')
    (out / 'alignment.json').write_text(json.dumps(alignment, indent=2) + '\n')

    subsets, counts, tests, jobs = {}, [], [], []
    for central in PEAKS:
        others = [gene for gene in PEAKS if gene != central]
        tests += [dict(central=central, target_group=('central' if row['gene'] == central else 'other_peaks'),
                       record_id=row['record_id'], gene=row['gene'])
                  for gene in PEAKS for row in pools[gene, 'test']]
        for mode in MODES:
            sources = [central] if mode == 'central' else others
            for seed in args.seeds:
                selected = []
                for split, total in [('train', N_TRAIN), ('validation', N_VALIDATION)]:
                    budgets = allocation(total, len(sources))
                    for gene, budget in zip(sources, budgets):
                        pool = pools[gene, split]
                        assert budget <= len(pool)
                        rng = np.random.default_rng(np.random.SeedSequence([
                            seed, PEAKS.index(gene), int(split == 'validation')]))
                        chosen = sorted(rng.permutation(len(pool))[:budget].tolist())
                        selected += [dict(record_id=pool[i]['record_id'], gene=gene, split=split)
                                     for i in chosen]
                        counts.append(dict(central=central, mode=mode, seed=seed, split=split,
                                           source=gene, n=budget, available=len(pool)))
                path = out / 'subsets' / central / f'{mode}_seed{seed}.csv'
                write_csv(path, selected)
                subsets[str(path.relative_to(out))] = digest(path)
                jobs += [dict(central=central, mode=mode, seed=seed, model=model) for model in MODELS]

    write_csv(out / 'test_records.csv', tests)
    write_csv(out / 'sample_counts.csv', counts)
    write_csv(out / 'jobs.csv', jobs)
    (out / 'cpu_jobs.txt').write_text(''.join(
        f'{job["central"]} {job["mode"]} {job["seed"]} {job["model"]}\n'
        for job in jobs if job['model'] != 'CNN_Jannis_OHE'))
    (out / 'gpu_jobs.txt').write_text(''.join(
        f'{job["central"]} {job["mode"]} {job["seed"]}\n'
        for job in jobs if job['model'] == 'CNN_Jannis_OHE'))
    (out / 'condor').mkdir(exist_ok=True)
    protocol = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        question=('If each of the four largest artificial cgreGFP peaks is treated in turn as the central peak, '
                  'how well do it and the pooled other three peaks transfer to each other?'),
        excludes_natural_cgreGFP=True, dataset_sha256=digest(DATA), seeds=args.seeds,
        central_peaks=PEAKS, modes=MODES, models=MODELS,
        peak_sizes={gene: {split: len(pools[gene, split]) for split in ['train', 'validation', 'test']}
                    for gene in PEAKS},
        n_train=N_TRAIN, n_validation=N_VALIDATION,
        source_balance=('Every fit uses 2,508 train and 836 validation records. The central arm uses only the '
                        'designated central peak. The other-peaks arm divides each budget as equally as possible '
                        'among the remaining three peaks.'),
        evaluation=('Each fitted model is evaluated both on its source-domain frozen test and the opposite-domain '
                    'frozen test. All original 20% test records are retained.'),
        model_selection='source-only validation MSE; test labels never select checkpoints',
        repeats='five seeds vary initialization, shuffling, and source subsampling; error bars are run SD',
        alignment_sha256=digest(out / 'alignment.json'), test_records_sha256=digest(out / 'test_records.csv'),
        subset_hashes=subsets, source_hashes=scientific_sources(),
        onehot_settings=dict(epochs=30, patience=10, batch_size=32, lr=.001),
        cnn_settings=dict(epochs=60, patience=10, batch_size=64, lr=.0003, weight_decay=.0001))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(jobs=len(jobs), cpu_jobs=sum(j['model'] != 'CNN_Jannis_OHE' for j in jobs),
                               gpu_jobs=sum(j['model'] == 'CNN_Jannis_OHE' for j in jobs),
                               test_rows=len(tests), tests=dict(Counter(r['target_group'] for r in tests))), indent=2))


if __name__ == '__main__':
    main()
