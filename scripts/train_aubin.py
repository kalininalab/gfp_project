"""Train reusable Aubin models on a frozen within-landscape split."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import torch

try:
    from .aubin_model import ALPHABET, AubinModel, encode_sequences, load_model
    from .regression_metrics import metrics
except ImportError:
    from aubin_model import ALPHABET, AubinModel, encode_sequences, load_model
    from regression_metrics import metrics

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path, gene):
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['gene'] == gene]
    if not rows:
        raise ValueError(f'No records for {gene}')
    if len({r['sequence'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sequence identities within landscape')
    if {r['split'] for r in rows} != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation, and test partitions')
    result = {}
    for split in ['train', 'validation', 'test']:
        subset = [r for r in rows if r['split'] == split]
        y = torch.tensor([float(r['target_log10']) for r in subset], dtype=torch.float32)
        if not torch.isfinite(y).all():
            raise ValueError('Nonfinite target')
        result[split] = (subset, encode_sequences([r['sequence'] for r in subset], len(rows[0]['sequence'])), y)
    return result


@torch.no_grad()
def predict_encoded(model, x, batch_size=256):
    model.eval()
    return torch.cat([model(x[i:i+batch_size]) for i in range(0, len(x), batch_size)])


def train(model, x_train, y_train, x_validation, y_validation, seed, epochs, patience, batch_size, learning_rate):
    """Only training and validation data enter fitting and checkpoint selection."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, stale = float('inf'), None, None, 0
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for ids in order.split(batch_size):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x_train[ids]), y_train[ids])
            if not torch.isfinite(loss):
                raise ValueError('Training diverged')
            loss.backward()
            optimizer.step()
        train_mse = float(torch.mean((predict_encoded(model, x_train)-y_train)**2))
        val_mse = float(torch.mean((predict_encoded(model, x_validation)-y_validation)**2))
        history.append(dict(epoch=epoch, train_mse=train_mse, validation_mse=val_mse))
        if val_mse < best_loss:
            best_loss, best_epoch, stale = val_mse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f'{model.architecture} epoch={epoch} train_mse={train_mse:.6f} validation_mse={val_mse:.6f}', flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return history, best_epoch


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty; preserve previous experiments')
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        raise ValueError('Training settings must be positive')
    manifest_path = args.data.parent/'manifest.json'
    data_manifest = json.loads(manifest_path.read_text())
    if data_manifest['outputs'][args.data.name] != sha256(args.data):
        raise ValueError('Dataset checksum differs from its preprocessing manifest')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    data = load_data(args.data, args.gene)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for architecture in args.architectures:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = AubinModel(data['train'][1].shape[1], architecture)
        history, best_epoch = train(model, data['train'][1], data['train'][2],
                                    data['validation'][1], data['validation'][2],
                                    args.seed, args.epochs, args.patience, args.batch_size, args.learning_rate)
        destination = args.output/architecture
        destination.mkdir()
        checkpoint_path = destination/'model.pt'
        torch.save(dict(state_dict=model.state_dict(), architecture=architecture,
                        sequence_length=model.sequence_length, alphabet=ALPHABET,
                        gene=args.gene, target='target_log10', seed=args.seed,
                        best_epoch=best_epoch, dataset_sha256=sha256(args.data)), checkpoint_path)
        restored, _ = load_model(checkpoint_path)
        report = dict(best_epoch=best_epoch, epochs_run=len(history), metrics={})
        # Evaluate each fixed checkpoint once; no test-driven tuning.
        with (destination/'predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['record_id', 'split', 'target_log10', 'prediction_log10'])
            writer.writeheader()
            for split, (rows, x, y) in data.items():
                prediction = predict_encoded(restored, x).numpy()
                np.testing.assert_array_equal(prediction, predict_encoded(model, x).numpy())
                report['metrics'][split] = metrics(y.numpy(), prediction)
                writer.writerows(dict(record_id=r['record_id'], split=split,
                                      target_log10=r['target_log10'], prediction_log10=float(p))
                                 for r, p in zip(rows, prediction))
        (destination/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        with (destination/'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        summary[architecture] = report
    run_manifest = dict(gene=args.gene, seed=args.seed, epochs=args.epochs, patience=args.patience,
                        batch_size=args.batch_size, learning_rate=args.learning_rate, adam_epsilon=1e-7,
                        threads=args.threads, device='cpu', architectures=args.architectures,
                        dataset_sha256=sha256(args.data), preprocessing_manifest_sha256=sha256(manifest_path),
                        scripts={p.name:sha256(p) for p in [Path(__file__), Path(__file__).with_name('aubin_model.py'),
                                                          Path(__file__).with_name('regression_metrics.py')]},
                        versions=dict(python=platform.python_version(), numpy=np.__version__,
                                      torch=str(torch.__version__), scipy=scipy.__version__, sklearn=sklearn.__version__),
                        results=summary)
    (args.output/'run.json').write_text(json.dumps(run_manifest, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/processed/baseline_v1/sequences.csv')
    parser.add_argument('--gene', default='cgreGFP')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--architectures', nargs='+', choices=['linear', '1_10_1'], default=['linear', '1_10_1'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--threads', type=int, default=1)
    run(parser.parse_args())
