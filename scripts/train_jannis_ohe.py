"""Jannis CNN on native 20-channel one-hot cgre sequences and frozen benchmark splits."""
import argparse
import fcntl
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
from scripts.aubin_model import ALPHABET, encode_sequences
from scripts.esm_benchmark.embed import digest
from scripts.esm_benchmark.run import neural_fit, sync, write_csv
from scripts.regression_metrics import metrics
from scripts.transfer_benchmark.common import read_csv

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    names = ['scripts/train_jannis_ohe.py', 'scripts/aubin_model.py',
             'scripts/esm_benchmark/run.py', 'scripts/esm_benchmark/models.py',
             'scripts/esm_benchmark/embed.py', 'scripts/transfer_benchmark/common.py',
             'scripts/regression_metrics.py']
    return {name: digest(ROOT/name) for name in names}


def load_ohe(data, split):
    rows = [r for r in read_csv(data) if r['gene'] == 'cgreGFP']
    assert len({r['record_id'] for r in rows}) == len(rows)
    assert len({r['sequence'] for r in rows}) == len(rows)
    assignments = rows if split == 'holdout' else read_csv(split)
    assignment = {r['record_id']: r['split'] for r in assignments}
    assert len(assignments) == len(assignment) and set(assignment) == {r['record_id'] for r in rows}
    ids = {s: np.array([i for i,r in enumerate(rows) if assignment[r['record_id']] == s])
           for s in ['train', 'validation', 'test']}
    assert sum(map(len,ids.values())) == len(rows) and all(len(ix) >= 2 for ix in ids.values())
    tokens = encode_sequences([r['sequence'] for r in rows], len(rows[0]['sequence'])).numpy()
    # Native positions, 20 channels, no gaps, terminal padding or ESM features.
    x = np.eye(len(ALPHABET), dtype=np.uint8)[tokens]
    y = np.array([float(r['target_log10']) for r in rows])
    return rows, ids, x, y


def run(a):
    out = Path(a.output)/a.split_name/'CNN_Jannis_OHE'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = dict(dataset_sha256=digest(a.data), split_sha256=digest(a.data if a.split=='holdout' else a.split),
                          source_hashes=source_hashes(), seed=a.seed, device=a.device, epochs_override=a.epochs)
        if (out/'metrics.json').exists():
            old = json.loads((out/'metrics.json').read_text())
            assert all(old[k] == v for k,v in provenance.items()), 'Changed inputs; use a new output directory'
            assert digest(out/'predictions.csv') == old['predictions_sha256']
            assert digest(out/'model.pt') == old['model_sha256']
            print('Verified completed run:', out)
            return
        started = time.perf_counter()
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        rows, ids, x, y = load_ohe(a.data, a.split)
        sync(a.device); fit_start = time.perf_counter()
        fn, report = neural_fit('CNN_Jannis', x, y, ids, a, out)
        sync(a.device); fit_seconds = time.perf_counter()-fit_start
        inference_start = time.perf_counter()
        predictions = {s: fn(ids[s]) for s in ['validation','test']}
        sync(a.device); inference_seconds = time.perf_counter()-inference_start
        write_csv(out/'predictions.csv', [dict(record_id=rows[i]['record_id'],split=s,y_true=float(y[i]),y_pred=float(p))
            for s, pred in predictions.items() for i,p in zip(ids[s],pred)])
        report.update(provenance, model='CNN_Jannis_OHE', representation='one-hot / native residues',
            architecture='CNN_Jannis', alphabet=ALPHABET, input_shape=list(x.shape), split=a.split_name,
            counts={s:len(ix) for s,ix in ids.items()}, metrics={s:metrics(y[ids[s]],pred) for s,pred in predictions.items()},
            fit_seconds=fit_seconds,inference_seconds=inference_seconds,elapsed_seconds=time.perf_counter()-started,
            gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,host=platform.node(),
            precision=dict(tensor_dtype='float32',matmul_tf32=False,cudnn_tf32=torch.backends.cudnn.allow_tf32),
            versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn']},
            predictions_sha256=digest(out/'predictions.csv'),model_sha256=digest(out/'model.pt'))
        assert source_hashes() == provenance['source_hashes'], 'Code changed during fit'
        assert digest(a.data) == provenance['dataset_sha256']
        temporary=out/'metrics.json.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(out/'metrics.json')
        print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/processed/baseline_v1/sequences.csv')
    p.add_argument('--output',default='results/jannis_ohe_cgreGFP')
    p.add_argument('--split',default='holdout')
    p.add_argument('--split-name',default='holdout')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,help='Smoke tests only; default same 60-epoch CNN recipe')
    run(p.parse_args())


if __name__=='__main__':
    main()
