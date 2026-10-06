"""Run one reproducible cgreGFP active-learning trajectory (see docs/ACTIVE_LEARNING.md)."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split

from scripts.aubin_model import ALPHABET, AubinModel, encode_sequences
from scripts.baseline_models import MLPModel
from scripts.esm_benchmark.models import CNN_Jannis
from scripts.regression_metrics import metrics
from .acquisition import acquisition_scores, select_queries

MODELS = ('aubin_1_10_1', 'mlp_small', 'mlp_deep', 'CNN_Jannis_OHE', 'CNN_Jannis_ESM')
DEFAULT = Path('results/active_learning_cgre_fixed_test')


def digest(path):
    """SHA-256 of a file, streamed so large ESM arrays need not fit in memory."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    """Write an atomic CSV with explicit column names from the first row."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def split_indices(n, seed):
    """Freeze the paper's 20% test, then reproduce the student's AL sizes.

    The test is selected first and is unavailable to training, validation and
    acquisition. From the remaining 80%, use exactly 1%, 10% and 10% of the
    full dataset as initial train, validation and query pool. The remaining 59%
    is unused, preserving the student's manageable 2,452-point spectral pool.
    """
    development, test = train_test_split(np.arange(n), test_size=.2, random_state=seed)
    n_train, n_validation, n_pool = int(.01*n), int(.1*n), int(.1*n)+1
    used, unused = train_test_split(development, train_size=n_train+n_validation+n_pool,
                                    random_state=seed)
    train, remaining = train_test_split(used, train_size=n_train, random_state=seed)
    validation, pool = train_test_split(remaining, train_size=n_validation, random_state=seed)
    return dict(train=train, validation=validation, pool=pool, test=test, unused=unused)


class Predictor(nn.Module):
    """Existing benchmark architecture with access to its last hidden features.

    Dense models take integer amino-acid tokens; CNNs take [N, residues, channels].
    The forward hook reads the input of the output layer without changing the
    architecture, its parameters or its forward calculation.
    """
    def __init__(self, name, length, dim=640):
        super().__init__()
        if name == 'aubin_1_10_1':
            self.base = AubinModel(length, '1_10_1')
        elif name in ('mlp_small', 'mlp_deep'):
            self.base = MLPModel(length, name)
        else:
            self.base = CNN_Jannis(20 if name.endswith('OHE') else dim)
        last = self.base.regressor if name.startswith('CNN') else self.base.network[-1]
        last.register_forward_pre_hook(self._capture)
        self.embedding = None

    def _capture(self, module, inputs):
        self.embedding = inputs[0].detach()

    def forward(self, x):
        return self.base(x)


def input_batch(x, indices, device):
    """Materialize only one batch from integer tokens or memory-mapped ESM."""
    return torch.as_tensor(np.array(x[indices]), device=device).to(
        torch.long if np.issubdtype(x.dtype, np.integer) and x.ndim == 2 else torch.float32)


@torch.no_grad()
def infer(model, x, indices, device, batch_size=64, samples=1, embeddings=False):
    """Return mean predictions and sample variance, or deterministic features.

    samples=1 disables dropout. For models without dropout, MC variance is
    explicitly zero; repeated identical forward passes are unnecessary.
    """
    stochastic = samples > 1 and any(isinstance(m, (nn.Dropout, nn.Dropout1d))
                                    for m in model.modules())
    model.train(stochastic and not embeddings)
    means, variances, hidden = [], [], []
    for start in range(0, len(indices), batch_size):
        xb = input_batch(x, indices[start:start+batch_size], device)
        if embeddings:
            model(xb)
            hidden.append(model.embedding.cpu().numpy())
        else:
            predictions = torch.stack([model(xb) for _ in range(samples if stochastic else 1)])
            means.append(predictions.mean(0).cpu().numpy())
            variances.append((predictions.var(0) if stochastic else torch.zeros_like(predictions[0])).cpu().numpy())
    model.eval()
    return np.concatenate(hidden) if embeddings else (np.concatenate(means), np.concatenate(variances))


def train_round(model, x, y, train, validation, args, round_number):
    """Warm-start one round with the student's common AdamW/Huber recipe.

    Reset optimizer and cosine scheduler each round. Recompute target mean/std
    from labeled training data only. Accumulated microbatches preserve the
    effective batch size of 1024 while bounding CNN GPU memory. Use FP32 and
    cloned checkpoints to avoid the reference CPU checkpoint aliasing bug.
    """
    center, scale = float(y[train].mean()), max(float(y[train].std(ddof=1)), 1e-8)
    target = (y-center)/scale
    epochs = args.epochs if args.epochs is not None else (99 if round_number == 0 else 11)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00063, weight_decay=2.3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss, stale, best_state = float('inf'), 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(train)
        train_loss = 0.
        for start in range(0, len(order), 1024):
            effective = order[start:start+1024]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(effective), args.batch_size):
                ix = effective[offset:offset+args.batch_size]
                prediction = model(input_batch(x, ix, args.device))
                loss = nn.functional.huber_loss(prediction,
                    torch.as_tensor(target[ix], dtype=torch.float32, device=args.device), delta=1.)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss * len(ix)/len(effective)).backward()
                train_loss += loss.item()*len(ix)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        prediction, _ = infer(model, x, validation, args.device, args.batch_size)
        val_loss = nn.functional.huber_loss(torch.from_numpy(prediction),
            torch.as_tensor(target[validation], dtype=torch.float32), delta=1.35).item()
        history.append(dict(round=round_number, epoch=epoch+1,
                            train_huber=train_loss/len(train), validation_huber=val_loss))
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        scheduler.step()
        if stale >= 20:
            break
    model.load_state_dict(best_state)
    return center, scale, history


def run(args):
    """Persist a complete, restartable trajectory, including the acquisition ledger."""
    if args.seed < 0 or args.rounds < 1 or args.budget < 2*args.rounds or args.budget % args.rounds:
        raise ValueError('Positive rounds and divisible budget of at least 2 per round required')
    if args.batch_size < 1 or (args.epochs is not None and args.epochs < 1):
        raise ValueError('Batch size and epochs must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows = [r for r in read_csv(args.data) if r['gene'] == 'cgreGFP']
    if len({r['record_id'] for r in rows}) != len(rows) or len({r['sequence'] for r in rows}) != len(rows):
        raise ValueError('Duplicate cgre record IDs or sequences')
    sequences = [r['sequence'] for r in rows]
    length = len(sequences[0])
    tokens = encode_sequences(sequences, length).numpy()
    y = np.array([float(r['target_log10']) for r in rows], dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite labels')
    ids = split_indices(len(rows), args.seed)
    if args.budget > len(ids['pool']) - 2:
        raise ValueError('Budget must leave at least two pool examples for Pearson')
    out = args.directory / f'seed_{args.seed}' / args.model
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [Path(__file__), Path(__file__).with_name('acquisition.py'),
                   Path('scripts/aubin_model.py'), Path('scripts/baseline_models.py'),
                   Path('scripts/esm_benchmark/models.py'), Path('scripts/regression_metrics.py')]
        protocol = dict(model=args.model, seed=args.seed, rounds=args.rounds, budget=args.budget,
            split='paper random 20% fixed test; student 1%/10%/10% AL subsets inside remaining 80%',
            acquisition_equation='0.38 * scaled_distance + 0.62 * scaled_MC_variance',
            alpha=.38, acquisition_mc_samples=25, evaluation='remaining_pool: MC100; fixed_test: deterministic',
            batch_size=args.batch_size, effective_training_batch_size=1024, threads=args.threads,
            epochs_override=args.epochs, data=str(args.data.resolve()), data_sha256=digest(args.data),
            counts={k:len(v) for k,v in ids.items()}, device=args.device,
            source_hashes={str(p):digest(p) for p in sources},
            versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn']})
        if args.model.endswith('ESM'):
            manifest = json.loads((args.embeddings/'manifest.json').read_text())
            records = read_csv(args.embeddings/'records.csv')
            if manifest['dataset_sha256'] != protocol['data_sha256'] or [(r['record_id'],r['sequence']) for r in rows] != [(r['record_id'],r['sequence']) for r in records]:
                raise ValueError('ESM data identity/order mismatch')
            feature_path = args.embeddings/'residue_embeddings.npy'
            if digest(feature_path) != manifest['files']['residue_embeddings.npy']:
                raise ValueError('ESM feature checksum mismatch')
            x = np.load(feature_path, mmap_mode='r')
            if x.shape[:2] != tokens.shape:
                raise ValueError('ESM feature shape mismatch')
            protocol['embeddings'] = manifest
        else:
            x = np.eye(20, dtype=np.float32)[tokens] if args.model.endswith('OHE') else tokens
        if (out/'protocol.json').exists():
            if json.loads((out/'protocol.json').read_text()) != protocol:
                raise ValueError('Existing protocol differs; use a new output directory')
        else:
            write_json(out/'protocol.json', protocol)
        if (out/'complete.json').exists():
            print(f'Already complete: {out}', flush=True)
            return
        write_csv(out/'splits.csv', [dict(record_id=rows[i]['record_id'], partition=p)
                                   for p, indices in ids.items() for i in indices])
        model = Predictor(args.model, length, x.shape[-1]).to(args.device)
        train, validation, pool = (ids[k].copy() for k in ['train','validation','pool'])
        start_round, summary, ledger, history = 0, [], [], []
        checkpoint = out/'resume.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['state_dict'])
            train, validation, pool = saved['train'], saved['validation'], saved['pool']
            start_round = saved['next_round']
            summary, ledger, history = saved['summary'], saved['ledger'], saved['history']
            random.setstate(saved['random_state']); np.random.set_state(saved['numpy_state'])
            torch.set_rng_state(saved['torch_state'])
            if args.device == 'cuda':
                torch.cuda.set_rng_state_all(saved['cuda_state'])
        for round_number in range(start_round, args.rounds+1):
            started = time.perf_counter()
            center, scale, round_history = train_round(model, x, y, train, validation, args, round_number)
            history.extend(round_history)
            n_train, n_validation, n_pool = len(train), len(validation), len(pool)
            # Paper test is measured before querying and cannot affect acquisition.
            test_prediction, _ = infer(model, x, ids['test'], args.device, args.batch_size)
            test_prediction = test_prediction*scale + center
            query_rows = []
            if round_number < args.rounds:
                labeled_emb = infer(model, x, train, args.device, args.batch_size, embeddings=True)
                pool_emb = infer(model, x, pool, args.device, args.batch_size, embeddings=True)
                _, variance = infer(model, x, pool, args.device, args.batch_size, samples=25)
                scores = acquisition_scores(labeled_emb, pool_emb, variance)
                selected = select_queries(scores, pool_emb, [sequences[i] for i in pool], args.budget//args.rounds)
                queried = pool[selected].tolist()
                random.shuffle(queried)
                new_val = max(1, int(.05*len(queried)))
                validation = np.concatenate([validation, queried[:new_val]])
                train = np.concatenate([train, queried[new_val:]])
                local = {int(i):j for j,i in enumerate(pool)}
                for pos, i in enumerate(queried):
                    query_rows.append(dict(selected_after_round=round_number,
                        first_training_round=round_number+1, record_id=rows[i]['record_id'],
                        destination='validation' if pos < new_val else 'train',
                        acquisition_score=float(scores[local[i]]), variance=float(variance[local[i]])))
                pool = pool[~np.isin(pool, queried)]
                ledger.extend(query_rows)
            pool_prediction, _ = infer(model, x, pool, args.device, args.batch_size, samples=100)
            pool_prediction = pool_prediction*scale + center
            predictions = []
            for partition, indices, prediction in [('fixed_test',ids['test'],test_prediction),
                                                    ('remaining_pool',pool,pool_prediction)]:
                score = metrics(y[indices], prediction)
                summary.append(dict(model=args.model, seed=args.seed, round=round_number,
                    partition=partition, n_train=n_train, n_validation=n_validation,
                    queried_before_fit=round_number*(args.budget//args.rounds),
                    queried_after_fit=len(query_rows), pool_before_query=n_pool,
                    **score, fit_and_query_seconds=time.perf_counter()-started))
                predictions.extend(dict(record_id=rows[i]['record_id'], partition=partition,
                                        y_true=float(y[i]), y_pred=float(p)) for i,p in zip(indices,prediction))
            write_csv(out/f'predictions_round_{round_number:02d}.csv', predictions)
            write_csv(out/'metrics.csv', summary)
            write_csv(out/'history.csv', history)
            if ledger:
                write_csv(out/'queries.csv', ledger)
            artifact = dict(state_dict=model.state_dict(), model=args.model, sequence_length=length,
                target_mean=center, target_std=scale, next_round=round_number+1,
                train=train, validation=validation, pool=pool, summary=summary, ledger=ledger, history=history,
                random_state=random.getstate(), numpy_state=np.random.get_state(), torch_state=torch.get_rng_state(),
                cuda_state=torch.cuda.get_rng_state_all() if args.device == 'cuda' else [])
            torch.save(artifact, out/'resume.pt.tmp')
            (out/'resume.pt.tmp').replace(checkpoint)
            print(f'{args.model} round {round_number}/{args.rounds}: '
                  f'remaining-pool Pearson={summary[-1]["pearson"]}, n_train={n_train}', flush=True)
        write_json(out/'complete.json', dict(host=platform.node(),
            gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
            protocol_sha256=digest(out/'protocol.json'), metrics_sha256=digest(out/'metrics.csv')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--data', type=Path, default=Path('data/processed/baseline_v1/sequences.csv'))
    parser.add_argument('--embeddings', type=Path, default=Path('esm_embeddings/cgreGFP_t30'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--budget', type=int, default=960)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=64, help='Memory microbatch; effective training batch remains 1024')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--epochs', type=int, help='Smoke test only: override epochs in every round')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
