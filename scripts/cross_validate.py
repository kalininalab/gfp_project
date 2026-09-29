"""Ten outer folds with fold-local validation for all GFP regression baselines."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import json
import multiprocessing
from pathlib import Path
import pickle
import platform
import random

import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from threadpoolctl import threadpool_limits
import torch

try:
    from .aubin_model import ALPHABET, AubinModel, encode_sequences, load_model
    from .baseline_models import MLPModel, load_baseline, sparse_one_hot
    from .cv_neighbors import make_distance_cache, query_cache
    from .regression_metrics import metrics
    from .train_aubin import ROOT, predict_encoded, sha256, train
    from .train_baselines import fit_ridge, neighbor_predictions, write_csv
except ImportError:
    from aubin_model import ALPHABET, AubinModel, encode_sequences, load_model
    from baseline_models import MLPModel, load_baseline, sparse_one_hot
    from cv_neighbors import make_distance_cache, query_cache
    from regression_metrics import metrics
    from train_aubin import ROOT, predict_encoded, sha256, train
    from train_baselines import fit_ridge, neighbor_predictions, write_csv

MODELS = ['mean', 'linear_regression', 'ridge', 'aubin_linear', 'aubin_1_10_1', 'mlp_small', 'mlp_deep', 'knn']


def build_folds(size, n_splits=10, seed=42):
    folds = []
    for fold, (remaining, test) in enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(size))):
        training, validation = train_test_split(remaining, test_size=0.2, random_state=seed+fold)
        folds.append({'train': np.sort(training), 'validation': np.sort(validation), 'test': np.sort(test)})
    return folds


def read_rows(path, gene):
    with Path(path).open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['gene'] == gene]
    if not rows or len({r['sequence'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty unique sequences within landscape')
    return rows


def one_fold(fold_number, config):
    """Each worker fits fresh models. No outer-test labels enter selection."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        return fit_fold(fold_number, config)


def fit_fold(fold_number, config):
    output = Path(config['output'])/f'fold_{fold_number:02d}'
    output.mkdir(exist_ok=True)
    rows = read_rows(config['data'], config['gene'])
    x = encode_sequences([r['sequence'] for r in rows], len(rows[0]['sequence']))
    y = np.array([float(r['target_log10']) for r in rows])
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite target')
    folds = build_folds(len(rows), config['n_splits'], config['seed'])
    ids = folds[fold_number]
    sparse = {split: sparse_one_hot(x[index].numpy()) for split, index in ids.items()}
    cache = np.load(Path(config['output'])/'hamming_counts.npy', mmap_mode='r')
    summary = {}
    for name in MODELS:
        destination = output/name
        destination.mkdir(exist_ok=True)
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        artifact = dict(gene=config['gene'], alphabet=ALPHABET, sequence_length=x.shape[1],
                        target='target_log10', dataset_sha256=config['dataset_sha256'],
                        seed=config['seed'], fold=fold_number,
                        split_sha256=sha256(Path(config['output'])/'splits'/f'fold_{fold_number:02d}.csv'))
        report = {}
        if name.startswith('aubin_') or name.startswith('mlp_'):
            architecture = name.removeprefix('aubin_')
            model = AubinModel(x.shape[1], architecture) if name.startswith('aubin_') else MLPModel(x.shape[1], architecture)
            with (destination/'training.log').open('w') as log, redirect_stdout(log):
                history, best = train(model, x[ids['train']], torch.tensor(y[ids['train']], dtype=torch.float32),
                                      x[ids['validation']], torch.tensor(y[ids['validation']], dtype=torch.float32),
                                      config['seed'], config['epochs'], 10, 32, 0.001)
            write_csv(destination/'history.csv', history)
            artifact.update(architecture=architecture, state_dict=model.state_dict(), best_epoch=best)
            torch.save(artifact, destination/'model.pt')
            report.update(best_epoch=best, epochs_run=len(history))
            restored, _ = (load_model if name.startswith('aubin_') else load_baseline)(destination/'model.pt')
            prediction = {split: predict_encoded(model, x[ids[split]]).numpy() for split in ['validation', 'test']}
            np.testing.assert_array_equal(prediction['test'], predict_encoded(restored, x[ids['test']]).numpy())
        else:
            if name == 'mean':
                model = DummyRegressor(strategy='mean').fit(sparse['train'], y[ids['train']])
            elif name == 'linear_regression':
                model = LinearRegression(tol=1e-8).fit(sparse['train'], y[ids['train']])
            elif name == 'ridge':
                model, trials = fit_ridge(sparse['train'], y[ids['train']], sparse['validation'], y[ids['validation']], config['ridge_alphas'])
                write_csv(destination/'alpha_search.csv', trials)
                report['selected_alpha'] = float(model.alpha)
            else:
                trials = []
                for k in config['knn_neighbors']:
                    distance, neighbors = query_cache(cache, ids['validation'], ids['train'], k, x.shape[1])
                    for weights in ['uniform', 'distance']:
                        predicted = neighbor_predictions(distance, neighbors, y[ids['train']], weights)
                        trials.append(dict(k=k, weights=weights, validation_mse=float(np.mean((predicted-y[ids['validation']])**2))))
                best = min(trials, key=lambda r:r['validation_mse'])
                write_csv(destination/'neighbor_search.csv', trials)
                report.update(selected_k=best['k'], weights=best['weights'], metric='hamming')
                model = KNeighborsRegressor(n_neighbors=best['k'], weights=best['weights'], metric='hamming',
                                            algorithm='brute', n_jobs=1).fit(x[ids['train']].numpy(), y[ids['train']])
            artifact.update(architecture=name, estimator=model, encoding='hamming' if name == 'knn' else 'one_hot')
            with (destination/'model.pkl').open('wb') as stream:
                pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
            restored, _ = load_baseline(destination/'model.pkl')
            prediction = {}
            for split in ['validation', 'test']:
                if name == 'knn':
                    distance, neighbors = query_cache(cache, ids[split], ids['train'], model.n_neighbors, x.shape[1])
                    prediction[split] = neighbor_predictions(distance, neighbors, y[ids['train']], model.weights)
                else:
                    prediction[split] = model.predict(sparse[split])
            # Check cached-neighbor behavior against the original sklearn predictor,
            # including tied distances, on 64 held-out sequences in every fold.
            sample = ids['test'][:64]
            np.testing.assert_allclose(prediction['test'][:64], restored.predict([rows[i]['sequence'] for i in sample]), rtol=1e-12, atol=1e-12)
        report['metrics'] = {split: metrics(y[ids[split]], prediction[split]) for split in ['validation', 'test']}
        predictions = [dict(record_id=rows[i]['record_id'], fold=fold_number, split=split,
                            target_log10=rows[i]['target_log10'], prediction_log10=float(pred))
                       for split in ['validation', 'test'] for i, pred in zip(ids[split], prediction[split])]
        write_csv(destination/'predictions.csv', predictions)
        (destination/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        summary[name] = report
        print(f'fold {fold_number:02d}: {name} complete', flush=True)
    (output/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return fold_number


def summarize(config):
    output = Path(config['output'])
    rows = read_rows(config['data'], config['gene'])
    expected = {r['record_id']:float(r['target_log10']) for r in rows}
    pooled = {name:[] for name in MODELS}
    fold_scores = []
    for fold in range(config['n_splits']):
        directory = output/f'fold_{fold:02d}'
        scores = json.loads((directory/'metrics.json').read_text())
        for name in MODELS:
            fold_scores.append(dict(model=name, fold=fold, **scores[name]['metrics']['test']))
            with (directory/name/'predictions.csv').open() as stream:
                pooled[name].extend(r for r in csv.DictReader(stream) if r['split']=='test')
    write_csv(output/'fold_metrics.csv', fold_scores)
    results, summary_rows = {}, []
    for name in MODELS:
        observed = {r['record_id']:float(r['target_log10']) for r in pooled[name]}
        if len(pooled[name]) != len(expected) or observed != expected:
            raise ValueError(f'Out-of-fold coverage mismatch: {name}')
        scores = metrics([float(r['target_log10']) for r in pooled[name]], [float(r['prediction_log10']) for r in pooled[name]])
        result = dict(pooled_oof=scores, fold_mean={}, fold_sd={})
        for metric in ['mse','rmse','r2','spearman','pearson']:
            values = [r[metric] for r in fold_scores if r['model']==name and r[metric] is not None]
            result['fold_mean'][metric] = float(np.mean(values)) if values else None
            result['fold_sd'][metric] = float(np.std(values, ddof=1)) if len(values)>1 else None
        results[name] = result
        summary_rows.append(dict(model=name, **{f'{m}_mean':result['fold_mean'][m] for m in result['fold_mean']},
                                 **{f'{m}_sd':result['fold_sd'][m] for m in result['fold_sd']},
                                 **{f'{m}_pooled':v for m,v in scores.items()}))
        write_csv(output/f'{name}_oof_predictions.csv', sorted(pooled[name], key=lambda r:r['record_id']))
    write_csv(output/'summary.csv', summary_rows)
    (output/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    return results


def run(args):
    data = args.data.resolve()
    rows = read_rows(data, args.gene)
    data_manifest = data.parent/'manifest.json'
    if json.loads(data_manifest.read_text())['outputs'][data.name] != sha256(data):
        raise ValueError('Dataset checksum mismatch')
    scripts = ['cross_validate.py','cv_neighbors.py','aubin_model.py','baseline_models.py','train_aubin.py','train_baselines.py','regression_metrics.py']
    config = dict(data=str(data), output=str(args.output.resolve()), gene=args.gene, n_splits=10, seed=args.seed,
                  epochs=30, patience=10, batch_size=32, learning_rate=0.001, inner_validation_fraction=0.2,
                  ridge_alphas=[0.1,1.,10.,100.], knn_neighbors=[1,3,5,11,21,51],
                  dataset_sha256=sha256(data), preprocessing_manifest_sha256=sha256(data_manifest),
                  scripts={p:sha256(Path(__file__).with_name(p)) for p in scripts},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                sklearn=sklearn.__version__, torch=str(torch.__version__)))
    if args.workers < 1:
        raise ValueError('Workers must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume or json.loads((args.output/'config.json').read_text()) != config:
            raise ValueError('Use a new output directory or --resume with identical configuration and code')
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        split_dir = args.output/'splits'
        split_dir.mkdir()
        folds = build_folds(len(rows), 10, args.seed)
        for fold, ids in enumerate(folds):
            membership = {int(i):split for split, indices in ids.items() for i in indices}
            write_csv(split_dir/f'fold_{fold:02d}.csv', [dict(record_id=r['record_id'], split=membership[i]) for i,r in enumerate(rows)])
    cache_path = args.output/'hamming_counts.npy'
    if not cache_path.exists() or not (args.output/'cache.json').exists():
        encoded = encode_sequences([r['sequence'] for r in rows], len(rows[0]['sequence'])).numpy()
        print('Building exact label-free Hamming cache', flush=True)
        cache = make_distance_cache(encoded, cache_path)
        rng = np.random.default_rng(42)
        for i,j in rng.integers(0, len(rows), size=(1000,2)):
            if cache[i,j] != np.count_nonzero(encoded[i]!=encoded[j]):
                raise ValueError('Distance cache audit failed')
        (args.output/'cache.json').write_text(json.dumps(dict(sha256=sha256(cache_path), shape=list(cache.shape)))+'\n')
    else:
        if sha256(cache_path) != json.loads((args.output/'cache.json').read_text())['sha256']:
            raise ValueError('Distance cache checksum mismatch')
    pending = [fold for fold in range(10) if not (args.output/f'fold_{fold:02d}'/'metrics.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(one_fold, fold, config) for fold in pending]
        for future in as_completed(futures):
            print(f'Completed fold {future.result()+1}/10', flush=True)
    results = summarize(config)
    (args.output/'run.json').write_text(json.dumps(dict(config=config, workers=args.workers, status='complete', results=results), indent=2, allow_nan=False)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/processed/baseline_v1/sequences.csv')
    parser.add_argument('--gene', default='cgreGFP')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
