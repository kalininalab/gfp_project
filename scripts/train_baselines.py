"""Fit sklearn and MLP baselines on the same frozen GFP partitions as Aubin."""

import argparse
import csv
import json
from pathlib import Path
import pickle
import platform
import random
import time

import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import torch

try:
    from .aubin_model import ALPHABET
    from .baseline_models import HIDDEN_WIDTHS, MLPModel, load_baseline, sparse_one_hot
    from .train_aubin import ROOT, load_data, predict_encoded, sha256, train
    from .regression_metrics import metrics
except ImportError:
    from aubin_model import ALPHABET
    from baseline_models import HIDDEN_WIDTHS, MLPModel, load_baseline, sparse_one_hot
    from train_aubin import ROOT, load_data, predict_encoded, sha256, train
    from regression_metrics import metrics

MODELS = ['mean', 'linear_regression', 'ridge', 'mlp_small', 'mlp_deep', 'knn']


def neighbor_predictions(distances, indices, targets, weights):
    values = targets[indices]
    if weights == 'uniform':
        return values.mean(axis=1)
    if weights != 'distance':
        raise ValueError('Unknown neighbor weighting')
    zeros = distances == 0
    inverse = 1 / np.where(zeros, 1, distances)
    # Match sklearn: if exact matches exist, all nonzero-distance neighbors get zero weight.
    inverse = np.where(zeros.any(axis=1, keepdims=True), zeros, inverse)
    return (values*inverse).sum(axis=1)/inverse.sum(axis=1)


def fit_knn(x_train, y_train, x_validation, y_validation, neighbors):
    if not neighbors or min(neighbors) < 1 or max(neighbors) > len(x_train):
        raise ValueError('Neighbor counts must be between 1 and training-set size')
    trials = []
    # Query each k separately because boundary ties can depend on requested k.
    # Reuse each neighbor query across both weighting options.
    for k in neighbors:
        candidate = KNeighborsRegressor(n_neighbors=k, metric='hamming', algorithm='brute', n_jobs=1)
        candidate.fit(x_train, y_train)
        distances, indices = candidate.kneighbors(x_validation)
        for weights in ['uniform', 'distance']:
            pred = neighbor_predictions(distances, indices, y_train, weights)
            trials.append(dict(k=k, weights=weights, validation_mse=float(mean_squared_error(y_validation, pred))))
        print(f'knn validation search k={k} complete', flush=True)
    best = min(trials, key=lambda r: r['validation_mse'])
    model = KNeighborsRegressor(n_neighbors=best['k'], weights=best['weights'],
                                metric='hamming', algorithm='brute', n_jobs=1).fit(x_train, y_train)
    return model, trials


def fit_ridge(x_train, y_train, x_validation, y_validation, alphas):
    """Choose alpha using validation MSE; never refit on validation/test."""
    best_model, best_loss = None, float('inf')
    trials = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver='lsqr', tol=1e-8, max_iter=10000)
        model.fit(x_train, y_train)
        mse = float(mean_squared_error(y_validation, model.predict(x_validation)))
        trials.append(dict(alpha=alpha, validation_mse=mse, iterations=int(model.n_iter_[0])))
        if mse < best_loss:
            best_model, best_loss = model, mse
    return best_model, trials


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        raise ValueError('Training settings must be positive')
    if not args.ridge_alphas or any(a <= 0 or not np.isfinite(a) for a in args.ridge_alphas):
        raise ValueError('Ridge alphas must be finite and positive')
    if len(set(args.models)) != len(args.models):
        raise ValueError('Repeated model names')
    data_hash = sha256(args.data)
    manifest_path = args.data.parent/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['outputs'][args.data.name] != data_hash:
        raise ValueError('Dataset does not match preprocessing manifest')
    old_run = None
    if args.aubin_run:
        old_run = json.loads((args.aubin_run/'run.json').read_text())
        if old_run['gene'] != args.gene or old_run['dataset_sha256'] != data_hash:
            raise ValueError('Aubin comparison must use the same gene and prepared dataset')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = load_data(args.data, args.gene)
    length = data['train'][1].shape[1]
    sparse = {split: sparse_one_hot(values[1].numpy()) for split, values in data.items()}
    targets = {split: np.array([float(r['target_log10']) for r in values[0]]) for split, values in data.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    # Model choices and ridge grid are fixed before any new test metrics are seen.
    for name in args.models:
        started = time.monotonic()
        destination = args.output/name
        destination.mkdir()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        artifact = dict(architecture=name, alphabet=ALPHABET, sequence_length=length,
                        gene=args.gene, target='target_log10', dataset_sha256=data_hash, seed=args.seed)
        report = {}
        if name in HIDDEN_WIDTHS:
            model = MLPModel(length, name)
            history, best_epoch = train(model, data['train'][1], data['train'][2],
                                        data['validation'][1], data['validation'][2],
                                        args.seed, args.epochs, args.patience, args.batch_size, args.learning_rate)
            report.update(best_epoch=best_epoch, epochs_run=len(history),
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          hidden_widths=list(HIDDEN_WIDTHS[name]))
            write_csv(destination/'history.csv', history)
            artifact.update(state_dict=model.state_dict(), best_epoch=best_epoch)
            checkpoint = destination/'model.pt'
            torch.save(artifact, checkpoint)
            prediction_fn = lambda split: predict_encoded(model, data[split][1]).numpy()
        else:
            with threadpool_limits(limits=args.threads):
                if name == 'mean':
                    model = DummyRegressor(strategy='mean').fit(sparse['train'], targets['train'])
                elif name == 'linear_regression':
                    # Sparse LSQR avoids unstable dense solutions to redundant one-hot columns.
                    model = LinearRegression(tol=1e-8).fit(sparse['train'], targets['train'])
                    report['solver'] = 'sklearn LinearRegression sparse LSQR, tol=1e-8'
                elif name == 'knn':
                    model, trials = fit_knn(data['train'][1].numpy(), targets['train'],
                                           data['validation'][1].numpy(), targets['validation'], args.knn_neighbors)
                    write_csv(destination/'neighbor_search.csv', trials)
                    artifact['encoding'] = 'hamming'
                    report.update(selected_k=model.n_neighbors, weights=model.weights, metric='hamming',
                                  train_score_includes_self_neighbors=True)
                else:
                    model, trials = fit_ridge(sparse['train'], targets['train'],
                                             sparse['validation'], targets['validation'], args.ridge_alphas)
                    write_csv(destination/'alpha_search.csv', trials)
                    report.update(selected_alpha=float(model.alpha), solver='lsqr', tolerance=1e-8)
            artifact['estimator'] = model
            checkpoint = destination/'model.pkl'
            with checkpoint.open('wb') as stream:
                pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
            prediction_fn = lambda split: model.predict(data[split][1].numpy() if name == 'knn' else sparse[split])
        restored, _ = load_baseline(checkpoint)
        report['metrics'] = {}
        predictions = []
        for split, (rows, _, _) in data.items():
            pred = prediction_fn(split)
            # Full-partition roundtrip check verifies both weights and saved encoding.
            actual = restored.predict([r['sequence'] for r in rows])
            np.testing.assert_array_equal(pred, actual)
            report['metrics'][split] = metrics(targets[split], pred)
            predictions.extend(dict(record_id=r['record_id'], split=split, target_log10=r['target_log10'],
                                    prediction_log10=float(p)) for r, p in zip(rows, pred))
        write_csv(destination/'predictions.csv', predictions)
        report['elapsed_seconds'] = time.monotonic()-started
        (destination/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        results[name] = report
        print(name, json.dumps(report), flush=True)
    scripts = ['train_baselines.py', 'baseline_models.py', 'train_aubin.py', 'aubin_model.py', 'regression_metrics.py']
    run_manifest = dict(gene=args.gene, seed=args.seed, models=args.models, epochs=args.epochs,
                        patience=args.patience, batch_size=args.batch_size, learning_rate=args.learning_rate,
                        adam_epsilon=1e-7, threads=args.threads, device='cpu', ridge_alphas=args.ridge_alphas,
                        knn_neighbors=args.knn_neighbors,
                        dataset_sha256=data_hash, preprocessing_manifest_sha256=sha256(manifest_path),
                        versions=dict(python=platform.python_version(), numpy=np.__version__,
                                      scipy=scipy.__version__, sklearn=sklearn.__version__, torch=str(torch.__version__)),
                        scripts={s: sha256(Path(__file__).with_name(s)) for s in scripts}, results=results)
    comparison = []
    combined = {**({'aubin_'+k: v for k, v in old_run['results'].items()} if old_run else {}), **results}
    if old_run:
        run_manifest['aubin_run_sha256'] = sha256(args.aubin_run/'run.json')
    for name, result in combined.items():
        for split, scores in result['metrics'].items():
            comparison.append(dict(model=name, split=split, **scores))
    write_csv(args.output/'comparison.csv', comparison, ['model', 'split', 'n', 'mse', 'rmse', 'r2', 'spearman', 'pearson'])
    (args.output/'run.json').write_text(json.dumps(run_manifest, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/processed/baseline_v1/sequences.csv')
    parser.add_argument('--gene', default='cgreGFP')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--aubin-run', type=Path)
    parser.add_argument('--ridge-alphas', nargs='+', type=float, default=[0.1, 1., 10., 100.])
    parser.add_argument('--knn-neighbors', nargs='+', type=int, default=[1, 3, 5, 11, 21, 51])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=1)
    run(parser.parse_args())
