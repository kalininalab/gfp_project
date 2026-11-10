"""Fit one frozen leave-one-peak-out model and evaluate its untouched target test."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.regression_metrics import metrics
from scripts.train_aubin import predict_encoded, train
from scripts.transfer_benchmark.models import AlignedModel, AlignedOHECNN, encode_aligned
from .common import (DATA, DEFAULT, MODELS, MODES, TARGETS, digest, read_csv,
                     scientific_sources, write_csv)


@torch.inference_mode()
def predict_cnn(model, x, indices, device, center, scale, batch_size=256):
    model.eval()
    return np.concatenate([
        model(x[batch].to(device)).float().cpu().numpy() * scale + center
        for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / batch_size))))])


def train_cnn(model, x, y, ids, seed, device, settings):
    center = float(y[ids['train']].mean())
    scale = max(float(y[ids['train']].std()), 1e-8)
    targets = (y - center) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    rng = np.random.default_rng(seed)
    history, best_loss, best_state, best_epoch, stale = [], float('inf'), None, None, 0
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        order = rng.permutation(ids['train'])
        total = 0.0
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / settings['batch_size'])))):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[batch].to(device))
            target = torch.tensor(targets[batch], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CNN loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch)
        validation = predict_cnn(model, x, ids['validation'], device, center, scale)
        validation_mse = float(np.mean((validation - y[ids['validation']]) ** 2))
        history.append(dict(epoch=epoch, train_mse=total / len(order) * scale ** 2,
                            validation_mse=validation_mse))
        print(f'CNN epoch={epoch} validation_mse={validation_mse:.6f}', flush=True)
        if validation_mse < best_loss:
            best_loss, best_epoch, stale = validation_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings['patience']:
                break
    model.load_state_dict(best_state)
    return history, best_epoch, center, scale


def load_partitions(directory, target, mode, seed, protocol):
    table = {row['record_id']: row for row in read_csv(DATA)}
    subset_path = directory / 'subsets' / target / f'{mode}_seed{seed}.csv'
    relative = str(subset_path.relative_to(directory))
    if digest(subset_path) != protocol['subset_hashes'][relative]:
        raise ValueError('Changed source subset')
    subset = read_csv(subset_path)
    test_path = directory / 'test_records.csv'
    if digest(test_path) != protocol['test_records_sha256']:
        raise ValueError('Changed target tests')
    partitions = {split: [table[row['record_id']] for row in subset if row['split'] == split]
                  for split in ['train', 'validation']}
    partitions['test'] = [table[row['record_id']] for row in read_csv(test_path) if row['target'] == target]
    assert len(partitions['train']) == protocol['n_train']
    assert len(partitions['validation']) == protocol['n_validation']
    assert all(row['gene'] != target for split in ['train', 'validation'] for row in partitions[split])
    assert all(row['gene'] == target and row['split'] == 'test' for row in partitions['test'])
    assert len({row['sequence'] for records in partitions.values() for row in records}) == sum(map(len, partitions.values()))
    return partitions


def run(args):
    started = time.perf_counter()
    directory = args.directory
    out = directory / 'fits' / args.target / args.mode / f'seed{args.seed}' / args.model
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = json.loads((directory / 'protocol.json').read_text())
        sources = scientific_sources()
        if sources != protocol['source_hashes'] or digest(DATA) != protocol['dataset_sha256']:
            raise ValueError('Scientific source or dataset changed after protocol freeze')
        if args.target not in protocol['targets'] or args.mode not in protocol['modes'] or args.seed not in protocol['seeds']:
            raise ValueError('Unplanned experiment')
        protocol_hash = digest(directory / 'protocol.json')
        if (out / 'metrics.json').exists():
            report = json.loads((out / 'metrics.json').read_text())
            assert report['protocol_sha256'] == protocol_hash
            print('Verified completed fit:', out)
            return
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        partitions = load_partitions(directory, args.target, args.mode, args.seed, protocol)
        rows, ids = [], {}
        for name, records in partitions.items():
            ids[name] = np.arange(len(rows), len(rows) + len(records))
            rows.extend(records)
        y = np.array([float(row['target_log10']) for row in rows])
        alignment = json.loads((directory / 'alignment.json').read_text())
        if digest(directory / 'alignment.json') != protocol['alignment_sha256']:
            raise ValueError('Changed alignment')
        x = encode_aligned(rows, alignment)
        cnn = args.model == 'CNN_Jannis_OHE'
        if cnn and args.device != 'cuda':
            raise ValueError('CNN on OHE requires CUDA in this benchmark')
        fit_started = time.perf_counter()
        with threadpool_limits(limits=1):
            if cnn:
                model = AlignedOHECNN(alignment['length']).to(args.device)
                history, best_epoch, center, scale = train_cnn(
                    model, x, y, ids, args.seed, args.device, protocol['cnn_settings'])
                predictor = lambda index: predict_cnn(model, x, index, args.device, center, scale)
                torch.cuda.synchronize()
            else:
                model = AlignedModel(alignment['length'], args.model)
                settings = protocol['onehot_settings']
                history, best_epoch = train(
                    model, x[ids['train']], torch.tensor(y[ids['train']], dtype=torch.float32),
                    x[ids['validation']], torch.tensor(y[ids['validation']], dtype=torch.float32),
                    args.seed, settings['epochs'], settings['patience'], settings['batch_size'], settings['lr'])
                center, scale = 0.0, 1.0
                predictor = lambda index: predict_encoded(model, x[index]).numpy()
            fit_seconds = time.perf_counter() - fit_started
            predictions = {name: predictor(index) for name, index in ids.items() if name != 'train'}
        write_csv(out / 'history.csv', history)
        write_csv(out / 'predictions.csv', [
            dict(record_id=rows[i]['record_id'], gene=rows[i]['gene'], partition=partition,
                 y_true=float(y[i]), y_pred=float(prediction))
            for partition, values in predictions.items() for i, prediction in zip(ids[partition], values)])
        torch.save(dict(model=args.model, state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                        alignment=alignment, center=center, scale=scale, target=args.target, mode=args.mode,
                        seed=args.seed, best_epoch=best_epoch, protocol_sha256=protocol_hash), out / 'model.pt')
        report = dict(target=args.target, mode=args.mode, model=args.model, seed=args.seed,
                      metrics={name: metrics(y[ids[name]], values) for name, values in predictions.items()},
                      counts={name: dict(Counter(row['gene'] for row in records)) for name, records in partitions.items()},
                      best_epoch=best_epoch, epochs_run=len(history), fit_seconds=fit_seconds,
                      elapsed_seconds=time.perf_counter() - started, device=args.device if cnn else 'cpu',
                      gpu=torch.cuda.get_device_name() if cnn else None, host=platform.node(),
                      protocol_sha256=protocol_hash, source_hashes=sources,
                      predictions_sha256=digest(out / 'predictions.csv'), model_sha256=digest(out / 'model.pt'))
        temporary = out / 'metrics.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(out / 'metrics.json')
        print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--target', choices=TARGETS, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
