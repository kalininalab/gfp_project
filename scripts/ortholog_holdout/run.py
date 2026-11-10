"""Fit one four-protein leave-one-out source combination."""
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

from scripts.peak_holdout.run import predict_cnn, train_cnn
from scripts.regression_metrics import metrics
from scripts.train_aubin import predict_encoded, train
from scripts.transfer_benchmark.models import AlignedModel, AlignedOHECNN, encode_aligned
from .common import (DATA, DEFAULT, GENES, MODELS, condition_id, digest, read_csv,
                     scientific_sources, sources_for, write_csv)


def load_partitions(directory, target, condition, seed, protocol):
    table = {row['record_id']: row for row in read_csv(DATA)}
    subset_path = directory / 'subsets' / target / f'{condition}_seed{seed}.csv'
    relative = str(subset_path.relative_to(directory))
    if digest(subset_path) != protocol['subset_hashes'][relative]: raise ValueError('Changed source subset')
    subset = read_csv(subset_path)
    test_path = directory / 'test_records.csv'
    if digest(test_path) != protocol['test_records_sha256']: raise ValueError('Changed tests')
    parts = {split:[table[row['record_id']] for row in subset if row['split'] == split]
             for split in ['train','validation']}
    parts['test'] = [table[row['record_id']] for row in read_csv(test_path) if row['target'] == target]
    assert len(parts['train']) == protocol['n_train'] and len(parts['validation']) == protocol['n_validation']
    assert all(row['gene'] != target for split in ['train','validation'] for row in parts[split])
    assert all(row['gene'] == target and row['split'] == 'test' for row in parts['test'])
    assert len({row['sequence'] for values in parts.values() for row in values}) == sum(map(len, parts.values()))
    return parts


def run(args):
    started = time.perf_counter(); directory = args.directory
    out = directory/'fits'/args.target/args.condition/f'seed{args.seed}'/args.model
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol = json.loads((directory/'protocol.json').read_text()); sources = scientific_sources()
        if sources != protocol['source_hashes'] or digest(DATA) != protocol['dataset_sha256']:
            raise ValueError('Scientific source or dataset changed after freeze')
        if args.condition not in protocol['conditions'][args.target] or args.seed not in protocol['seeds']:
            raise ValueError('Unplanned experiment')
        protocol_hash = digest(directory/'protocol.json')
        if (out/'metrics.json').exists():
            assert json.loads((out/'metrics.json').read_text())['protocol_sha256'] == protocol_hash; return
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False
        partitions = load_partitions(directory,args.target,args.condition,args.seed,protocol)
        rows=[]; ids={}
        for name,records in partitions.items(): ids[name]=np.arange(len(rows),len(rows)+len(records)); rows.extend(records)
        y=np.array([float(row['target_log10']) for row in rows])
        alignment=json.loads((directory/'alignment.json').read_text())
        if digest(directory/'alignment.json') != protocol['alignment_sha256']: raise ValueError('Changed alignment')
        x=encode_aligned(rows,alignment); cnn=args.model=='CNN_Jannis_OHE'
        if cnn and args.device!='cuda': raise ValueError('CNN requires CUDA')
        fit_started=time.perf_counter()
        with threadpool_limits(limits=1):
            if cnn:
                model=AlignedOHECNN(alignment['length']).to(args.device)
                history,best,center,scale=train_cnn(model,x,y,ids,args.seed,args.device,protocol['cnn_settings'])
                predictor=lambda index:predict_cnn(model,x,index,args.device,center,scale)
                torch.cuda.synchronize()
            else:
                model=AlignedModel(alignment['length'],args.model); s=protocol['onehot_settings']
                history,best=train(model,x[ids['train']],torch.tensor(y[ids['train']],dtype=torch.float32),
                    x[ids['validation']],torch.tensor(y[ids['validation']],dtype=torch.float32),
                    args.seed,s['epochs'],s['patience'],s['batch_size'],s['lr'])
                center,scale=0.,1.; predictor=lambda index:predict_encoded(model,x[index]).numpy()
            fit_seconds=time.perf_counter()-fit_started
            predictions={name:predictor(index) for name,index in ids.items() if name!='train'}
        write_csv(out/'history.csv',history)
        write_csv(out/'predictions.csv',[dict(record_id=rows[i]['record_id'],gene=rows[i]['gene'],partition=part,
            y_true=float(y[i]),y_pred=float(pred)) for part,values in predictions.items() for i,pred in zip(ids[part],values)])
        torch.save(dict(model=args.model,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            alignment=alignment,center=center,scale=scale,target=args.target,condition=args.condition,
            seed=args.seed,best_epoch=best,protocol_sha256=protocol_hash),out/'model.pt')
        report=dict(target=args.target,condition=args.condition,model=args.model,seed=args.seed,
            metrics={name:metrics(y[ids[name]],values) for name,values in predictions.items()},
            counts={name:dict(Counter(row['gene'] for row in records)) for name,records in partitions.items()},
            best_epoch=best,epochs_run=len(history),fit_seconds=fit_seconds,elapsed_seconds=time.perf_counter()-started,
            device=args.device if cnn else 'cpu',gpu=torch.cuda.get_device_name() if cnn else None,host=platform.node(),
            protocol_sha256=protocol_hash,source_hashes=sources,predictions_sha256=digest(out/'predictions.csv'),model_sha256=digest(out/'model.pt'))
        temp=out/'metrics.json.tmp'; temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); temp.replace(out/'metrics.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--target',choices=GENES,required=True); parser.add_argument('--condition',required=True)
    parser.add_argument('--seed',type=int,required=True); parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu'); run(parser.parse_args())


if __name__=='__main__': main()
