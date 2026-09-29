"""Freeze ten source folds and nested, peak-proportional training samples."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .common import DATA, DEFAULT, ROOT, MODELS, PEAKS, SIZES, SOURCES, digest, read_csv, write_csv, scientific_sources


def allocate(total, available):
    """Largest-remainder proportional allocation with deterministic tie handling."""
    weights = np.asarray(available, dtype=float)
    if total > sum(available) or total < 1:
        raise ValueError('Sample exceeds available pool or is empty')
    exact = total*weights/weights.sum()
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact-counts), kind='stable')
    counts[order[:total-int(counts.sum())]] += 1
    assert counts.sum() == total and all(counts <= available)
    return counts


def nested_samples(rows, pool, sizes, seed):
    """Use fixed within-gene orders and proportional quotas for all sample sizes."""
    genes = sorted({rows[i]['gene'] for i in pool})
    rng = np.random.default_rng(seed)
    orders = [rng.permutation([i for i in pool if rows[i]['gene'] == gene]) for gene in genes]
    result = {}; previous = set()
    for size in sizes:
        counts = allocate(size, [len(order) for order in orders])
        chosen = np.sort(np.concatenate([order[:n] for order,n in zip(orders,counts)])).astype(int)
        assert previous <= set(chosen), 'Non-nested proportional allocation; adjust size grid'
        result[size] = chosen
        previous = set(chosen)
    return result


def folds_for(rows, stratified=False):
    indices = np.arange(len(rows)); labels = np.array([r['gene'] for r in rows])
    splitter = StratifiedKFold(10, shuffle=True, random_state=42) if stratified else KFold(10, shuffle=True, random_state=42)
    result = []
    for fold,(remaining,test) in enumerate(splitter.split(indices,labels)):
        train, val = train_test_split(remaining, test_size=.2, random_state=42+fold,
                                     stratify=labels[remaining] if stratified else None)
        result.append(dict(train=np.sort(train), validation=np.sort(val), test=np.sort(test)))
    return result


def prepare(data, output, sizes, natural_splits=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('Frozen protocol exists; use another output directory')
    sizes = sorted(set(sizes))
    if not sizes or min(sizes) < 2:
        raise ValueError('At least two training examples are required')
    rows = [r for r in read_csv(data) if r['gene'] in ['cgreGFP']+PEAKS]
    assert len({r['record_id'] for r in rows}) == len(rows)
    assert len({r['sequence'] for r in rows}) == len(rows), 'Group duplicate sequences before splitting'
    assert len({len(r['sequence']) for r in rows}) == 1
    domains = {s:[r for r in rows if (r['gene']=='cgreGFP') == (s=='natural')] for s in SOURCES}
    assert {r['gene'] for r in domains['artificial']} == set(PEAKS)
    folds = {s:folds_for(domains[s], stratified=s=='artificial') for s in SOURCES}
    n_validation = min(len(f['validation']) for f in folds['natural'])
    assert all(len(f['validation']) == n_validation for f in folds['natural'])
    input_folds = {}; files = {}; counts = []; jobs = []
    for source in SOURCES:
        source_rows = domains[source]
        seen = Counter()
        for fold, partitions in enumerate(folds[source]):
            if source=='natural' and natural_splits is not None:
                old = Path(natural_splits)/f'fold_{fold:02d}.csv'
                if old.exists():
                    archived = read_csv(old)
                    assignment = {source_rows[i]['record_id']:part for part,indices in partitions.items() for i in indices}
                    assert len(archived) == len(assignment)
                    assert {r['record_id']:r['split'] for r in archived} == assignment, 'Existing natural fold changed'
                    input_folds[str(old)] = digest(old)
            train_samples = nested_samples(source_rows, partitions['train'], sizes, 42000+fold)
            validation = nested_samples(source_rows, partitions['validation'], [n_validation], 43000+fold)[n_validation]
            assignment = {int(i):'train' for i in partitions['train']}
            assignment.update({int(i):'unused_validation' for i in partitions['validation']})
            assignment.update({int(i):'validation' for i in validation})
            assignment.update({int(i):'test' for i in partitions['test']})
            path = output/'splits'/f'{source}_fold_{fold:02d}.csv'
            write_csv(path, [dict(record_id=r['record_id'],gene=r['gene'],split=assignment[i]) for i,r in enumerate(source_rows)])
            files[str(path.relative_to(output))] = digest(path)
            seen.update(source_rows[i]['record_id'] for i in partitions['test'])
            for size, selected in train_samples.items():
                assert not (set(selected)&set(validation) or set(selected)&set(partitions['test']) or set(validation)&set(partitions['test']))
                path = output/'subsets'/f'{source}_fold_{fold:02d}_n_{size:05d}.csv'
                write_csv(path, [dict(record_id=source_rows[i]['record_id'],gene=source_rows[i]['gene']) for i in selected])
                files[str(path.relative_to(output))] = digest(path)
                for part, indices in [('train',selected), ('validation',validation), ('test',partitions['test'])]:
                    for gene,n in sorted(Counter(source_rows[i]['gene'] for i in indices).items()):
                        counts.append(dict(source=source,fold=fold,training_size=size,partition=part,gene=gene,n=n))
            for model in MODELS:
                jobs.append(dict(source=source,fold=fold,model=model,device='cuda' if model=='CNN_Jannis_OHE' else 'cpu'))
        assert set(seen.values()) == {1} and len(seen)==len(source_rows)
    write_csv(output/'sample_counts.csv',counts)
    write_csv(output/'jobs.csv',jobs)
    for device in ['cpu','cuda']:
        (output/f'{device}_jobs.txt').write_text(''.join(f'{j["source"]} {j["fold"]} {j["model"]}\n' for j in jobs if j['device']==device))
    protocol = dict(data_file=str(Path(data).resolve()),dataset_sha256=digest(data),source_hashes=scientific_sources(),
        sizes=sizes,n_folds=10,training_seed=42,subset_seed_rule='42000+fold; validation 43000+fold',
        n_validation=n_validation,models=MODELS,sources=SOURCES,native_length=len(rows[0]['sequence']),
        domain_counts={s:len(r) for s,r in domains.items()},file_hashes=files,original_natural_folds=input_folds,
        fits=len(jobs)*len(sizes),scheduler_jobs=len(jobs),
        validation_rule='Fixed source-only validation; training-size x excludes validation labels',
        artificial_rule='Stratified outer/inner folds by peak; proportional nested training and fixed validation',
        training=dict(dense=dict(max_epochs=30,patience=10,batch=32,learning_rate=.001,optimizer='Adam',target_scaling=False),
                      cnn=dict(max_epochs=60,patience=10,batch=64,learning_rate=.0003,weight_decay=.0001,optimizer='AdamW',target_scaling='training mean/std')))
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['file_hashes','source_hashes','original_natural_folds']},indent=2))
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--sizes',type=int,nargs='+',default=SIZES)
    parser.add_argument('--natural-splits',type=Path,default=ROOT/'results/cv10_cgreGFP_seed42/splits')
    args = parser.parse_args()
    prepare(args.data,args.output,args.sizes,args.natural_splits)


if __name__=='__main__':
    main()
