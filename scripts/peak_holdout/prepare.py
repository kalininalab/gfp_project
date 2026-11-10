"""Freeze balanced source subsets and held-out artificial-peak tests."""
import argparse
from collections import Counter
import datetime
import json

import numpy as np

from .common import (DATA, DEFAULT, MODELS, MODES, PEAKS, TARGETS, digest,
                     read_csv, scientific_sources, write_csv)

N_TRAIN = 7524
N_VALIDATION = 2508


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=type(DEFAULT), default=DEFAULT)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 47)))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise FileExistsError('Frozen protocol already exists; choose a new directory')

    rows = [row for row in read_csv(DATA) if row['gene'] in ['cgreGFP'] + PEAKS]
    assert len({row['record_id'] for row in rows}) == len(rows)
    assert len({row['sequence'] for row in rows}) == len(rows), 'Duplicate sequences can leak across landscapes'
    assert {len(row['sequence']) for row in rows} == {235}
    pools = {(gene, split): [row for row in rows if row['gene'] == gene and row['split'] == split]
             for gene in ['cgreGFP'] + PEAKS for split in ['train', 'validation', 'test']}

    alignment = dict(length=235, mapping={gene: list(range(235)) for gene in ['cgreGFP'] + PEAKS},
                     method='native cgreGFP coordinates; all five cgre landscapes have length 235')
    (out / 'alignment.json').write_text(json.dumps(alignment, indent=2) + '\n')
    subsets = {}
    counts = []
    tests = []
    jobs = []
    for target in TARGETS:
        target_test = pools[target, 'test']
        tests += [dict(target=target, record_id=row['record_id'], gene=row['gene']) for row in target_test]
        artificial_sources = [gene for gene in PEAKS if gene != target]
        for mode in MODES:
            sources = artificial_sources + (['cgreGFP'] if mode == 'with_natural' else [])
            per_train = N_TRAIN // len(sources)
            per_validation = N_VALIDATION // len(sources)
            assert per_train * len(sources) == N_TRAIN
            assert per_validation * len(sources) == N_VALIDATION
            for seed in args.seeds:
                selected = []
                for split, budget in [('train', per_train), ('validation', per_validation)]:
                    for gene in sources:
                        pool = pools[gene, split]
                        assert budget <= len(pool)
                        rng = np.random.default_rng(np.random.SeedSequence([
                            seed, (['cgreGFP'] + PEAKS).index(gene), int(split == 'validation')]))
                        chosen = sorted(rng.permutation(len(pool))[:budget].tolist())
                        selected += [dict(record_id=pool[i]['record_id'], gene=gene, split=split)
                                     for i in chosen]
                        counts.append(dict(target=target, mode=mode, seed=seed, split=split,
                                           source=gene, n=budget, available=len(pool)))
                path = out / 'subsets' / target / f'{mode}_seed{seed}.csv'
                write_csv(path, selected)
                subsets[str(path.relative_to(out))] = digest(path)
                jobs += [dict(target=target, mode=mode, seed=seed, model=model) for model in MODELS]
    write_csv(out / 'test_records.csv', tests)
    write_csv(out / 'sample_counts.csv', counts)
    write_csv(out / 'jobs.csv', jobs)
    (out / 'cpu_jobs.txt').write_text(''.join(
        f'{job["target"]} {job["mode"]} {job["seed"]} {job["model"]}\n'
        for job in jobs if job['model'] != 'CNN_Jannis_OHE'))
    (out / 'gpu_jobs.txt').write_text(''.join(
        f'{job["target"]} {job["mode"]} {job["seed"]}\n'
        for job in jobs if job['model'] == 'CNN_Jannis_OHE'))
    (out / 'condor').mkdir(exist_ok=True)
    protocol = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        question='Can other cgreGFP shotgun landscapes predict a completely held-out large artificial peak?',
        dataset_sha256=digest(DATA), seeds=args.seeds, targets=TARGETS, modes=MODES, models=MODELS,
        peak_sizes={gene: {split: len(pools[gene, split]) for split in ['train', 'validation', 'test']}
                    for gene in PEAKS},
        target_definition='three artificial landscapes with the largest total record counts',
        n_train=N_TRAIN, n_validation=N_VALIDATION,
        source_balance=('Fixed total budget in both modes. artificial_only: 2508 train and 836 validation '
                        'per remaining artificial peak. with_natural: 1881 train and 627 validation per '
                        'remaining artificial peak and natural cgreGFP; natural therefore contributes 25%.'),
        target_isolation='The target landscape contributes no train or validation labels; its original test split is frozen evaluation.',
        model_selection='source-only validation MSE; target test labels never select checkpoints',
        repeats='five fixed-target runs; seeds vary initialization, shuffling, and source subsampling; error bars are run SD',
        alignment_sha256=digest(out / 'alignment.json'), test_records_sha256=digest(out / 'test_records.csv'),
        subset_hashes=subsets, source_hashes=scientific_sources(),
        onehot_settings=dict(epochs=30, patience=10, batch_size=32, lr=.001),
        cnn_settings=dict(epochs=60, patience=10, batch_size=64, lr=.0003, weight_decay=.0001))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(jobs=len(jobs), n_train=N_TRAIN, n_validation=N_VALIDATION,
                               tests=dict(Counter(row['target'] for row in tests))), indent=2))


if __name__ == '__main__':
    main()
