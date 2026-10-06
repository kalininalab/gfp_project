"""Fit one requested base model on the manuscript's random 80/20 cgre split.

The manuscript specifies a random 80/20 split, 50 epochs and Adam for its CNN.
It omits the seed, learning rate and batch size. We freeze seed 42 and use the
legacy code's Adam learning rate 1e-3 and batch size 32, recording this gap in
the protocol. The held-out test is evaluated once after all 50 epochs.
"""
import argparse
import csv
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
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from scripts.active_learning.run import Predictor, input_batch, infer
from scripts.aubin_model import encode_sequences
from scripts.regression_metrics import metrics

MODELS = ('aubin_1_10_1','mlp_small','mlp_deep','CNN_Jannis_OHE','CNN_Jannis_ESM')
DEFAULT = Path('results/paper_reproduction_cgre_80_20')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def split_indices(size,seed=42):
    """Return exhaustive, disjoint 80/20 indices with sklearn's stable split."""
    train,test=train_test_split(np.arange(size),test_size=.2,random_state=seed,shuffle=True)
    return np.asarray(train),np.asarray(test)


def run(args):
    if args.epochs != 50:
        raise ValueError('The manuscript-described run requires exactly 50 epochs')
    if args.batch_size != 32 or args.learning_rate != .001:
        raise ValueError('Use the frozen legacy-derived batch size and learning rate')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='cuda': torch.cuda.manual_seed_all(args.seed)

    rows=[r for r in read_csv(args.data) if r['gene']=='cgreGFP']
    if len(rows)!=len({r['record_id'] for r in rows}) or len(rows)!=len({r['sequence'] for r in rows}):
        raise ValueError('Duplicate record IDs or sequences')
    sequences=[r['sequence'] for r in rows];length=len(sequences[0])
    tokens=encode_sequences(sequences,length).numpy()
    y=np.asarray([float(r['target_log10']) for r in rows],dtype=np.float32)
    train,test=split_indices(len(rows),args.seed)
    if len(train)!=int(.8*len(rows)) or len(test)!=len(rows)-len(train) or set(train)&set(test):
        raise RuntimeError('Invalid 80/20 split')

    if args.model=='CNN_Jannis_ESM':
        manifest=json.loads((args.embeddings/'manifest.json').read_text())
        records=read_csv(args.embeddings/'records.csv')
        if manifest['dataset_sha256']!=digest(args.data):
            raise ValueError('ESM manifest/data hash mismatch')
        if [(r['record_id'],r['sequence']) for r in rows] != [(r['record_id'],r['sequence']) for r in records]:
            raise ValueError('ESM record order mismatch')
        feature_path=args.embeddings/'residue_embeddings.npy'
        if digest(feature_path)!=manifest['files']['residue_embeddings.npy']:
            raise ValueError('ESM feature checksum mismatch')
        x=np.load(feature_path,mmap_mode='r')
    elif args.model=='CNN_Jannis_OHE':
        x=np.eye(20,dtype=np.float32)[tokens]
    else:
        x=tokens

    out=args.directory/f'seed_{args.seed}'/args.model
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(model=args.model,seed=args.seed,split='random 80% train / 20% held-out test',
        counts=dict(train=len(train),test=len(test)),epochs=args.epochs,optimizer='Adam',
        learning_rate=args.learning_rate,batch_size=args.batch_size,loss='MSE',
        test_used_for_training_or_model_selection=False,target='log10 fluorescence',device=args.device,
        manuscript_specified=['random 80/20 split','50 epochs','Adam optimizer'],
        manuscript_omitted=['random seed','learning rate','batch size'],
        omitted_values_source='CNN/model_legacy.py',data=str(args.data.resolve()),
        data_sha256=digest(args.data),source_hashes={str(Path(__file__)):digest(__file__),
        'scripts/active_learning/run.py':digest('scripts/active_learning/run.py'),
        'scripts/aubin_model.py':digest('scripts/aubin_model.py'),
        'scripts/baseline_models.py':digest('scripts/baseline_models.py'),
        'scripts/esm_benchmark/models.py':digest('scripts/esm_benchmark/models.py')},
        versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn']})
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Existing output has a different protocol')
    write_json(protocol_path,protocol)
    write_csv(out/'split.csv',[dict(record_id=rows[i]['record_id'],split=part)
                              for part,indices in [('train',train),('test',test)] for i in indices])
    if (out/'complete.json').exists():
        print('Already complete:',out,flush=True);return

    model=Predictor(args.model,length,x.shape[-1]).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    history=[];started=time.perf_counter()
    rng=np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train();order=rng.permutation(train);total=0.
        for start in range(0,len(order),args.batch_size):
            ix=order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction=model(input_batch(x,ix,args.device))
            target=torch.as_tensor(y[ix],dtype=torch.float32,device=args.device)
            loss=nn.functional.mse_loss(prediction,target)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=loss.item()*len(ix)
        history.append(dict(epoch=epoch,train_mse=total/len(train)))
        print(f'{args.model} epoch {epoch}/{args.epochs}: train MSE={history[-1]["train_mse"]:.6f}',flush=True)
    fit_seconds=time.perf_counter()-started
    prediction,_=infer(model,x,test,args.device,args.batch_size,samples=1)
    score=metrics(y[test],prediction)
    write_csv(out/'history.csv',history)
    write_csv(out/'predictions.csv',[dict(record_id=rows[i]['record_id'],y_true=float(y[i]),y_pred=float(p))
                                     for i,p in zip(test,prediction)])
    torch.save(dict(model=args.model,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    sequence_length=length,seed=args.seed),out/'model.pt')
    report=dict(**protocol,metrics=score,fit_seconds=fit_seconds,
                inference_seconds=time.perf_counter()-started-fit_seconds,host=platform.node(),
                gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
                predictions_sha256=digest(out/'predictions.csv'),model_sha256=digest(out/'model.pt'))
    write_json(out/'metrics.json',report)
    write_json(out/'complete.json',dict(protocol_sha256=digest(protocol_path),metrics_sha256=digest(out/'metrics.json')))
    print(json.dumps(score,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--directory',type=Path,default=DEFAULT)
    p.add_argument('--data',type=Path,default=Path('data/processed/baseline_v1/sequences.csv'))
    p.add_argument('--embeddings',type=Path,default=Path('esm_embeddings/cgreGFP_t30'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(p.parse_args())


if __name__=='__main__':main()
