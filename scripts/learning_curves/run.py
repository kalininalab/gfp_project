"""Fit independent models at nested sizes; score both target domains per checkpoint."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from scripts.aubin_model import ALPHABET, AubinModel, encode_sequences
from scripts.baseline_models import MLPModel
from scripts.train_aubin import train, predict_encoded
from scripts.esm_benchmark.run import neural_fit, predict, sync
from scripts.esm_benchmark.models import CNN_Jannis
from scripts.regression_metrics import metrics
from .common import DEFAULT, MODELS, PEAKS, SOURCES, digest, read_csv, write_csv, scientific_sources, fit_directory


def checked_csv(directory, protocol, relative):
    path = directory/relative
    assert digest(path) == protocol['file_hashes'][relative], f'Changed split/subset: {path}'
    return read_csv(path)


def load_context(directory, source, fold, cnn):
    directory = Path(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    assert protocol['source_hashes'] == scientific_sources(), 'Scientific code changed; prepare a new experiment'
    assert digest(protocol['data_file']) == protocol['dataset_sha256'], 'Prepared data changed'
    assert source in SOURCES and 0 <= fold < protocol['n_folds']
    rows = [r for r in read_csv(protocol['data_file']) if r['gene'] in ['cgreGFP']+PEAKS]
    index = {r['record_id']:i for i,r in enumerate(rows)}
    assert len(index) == len(rows) == len({r['sequence'] for r in rows})
    assignments = {s:checked_csv(directory,protocol,f'splits/{s}_fold_{fold:02d}.csv') for s in SOURCES}
    ids = {s:np.array([index[r['record_id']] for r in assignments[s] if r['split']=='test']) for s in SOURCES}
    ids['validation'] = np.array([index[r['record_id']] for r in assignments[source] if r['split']=='validation'])
    pool = {r['record_id'] for r in assignments[source] if r['split']=='train'}
    subsets = {}
    for size in protocol['sizes']:
        chosen = checked_csv(directory,protocol,f'subsets/{source}_fold_{fold:02d}_n_{size:05d}.csv')
        assert len(chosen) == size and len({r['record_id'] for r in chosen}) == size
        assert {r['record_id'] for r in chosen} <= pool
        subsets[size] = np.array([index[r['record_id']] for r in chosen])
        groups = [set(subsets[size])] + [set(ix) for ix in ids.values()]
        assert all(not groups[i]&groups[j] for i in range(len(groups)) for j in range(i))
    tokens = encode_sequences([r['sequence'] for r in rows],protocol['native_length'])
    x = np.eye(len(ALPHABET),dtype=np.uint8)[tokens.numpy()] if cnn else tokens
    y = np.array([float(r['target_log10']) for r in rows])
    assert np.isfinite(y).all()
    return SimpleNamespace(directory=directory,protocol=protocol,rows=rows,ids=ids,subsets=subsets,x=x,y=y,
                           protocol_sha256=digest(directory/'protocol.json'),source=source,fold=fold)


def fit(context, model_name, size, device, epochs=None):
    c = context; protocol = c.protocol
    out = fit_directory(c.directory,c.source,c.fold,size,model_name)
    out.mkdir(parents=True,exist_ok=True)
    provenance = dict(protocol_sha256=c.protocol_sha256,source_hashes=protocol['source_hashes'],
                      model=model_name,source=c.source,fold=c.fold,training_size=size,
                      seed=protocol['training_seed'],device=device,epochs_override=epochs)
    with (out/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'metrics.json').exists():
            saved = json.loads((out/'metrics.json').read_text())
            assert all(saved[k]==v for k,v in provenance.items()), 'Changed run settings'
            assert digest(out/'predictions.csv')==saved['predictions_sha256']
            assert digest(out/'model.pt')==saved['model_sha256']
            print('Reusing',out,flush=True)
            return
        started = time.perf_counter()
        random.seed(provenance['seed']); np.random.seed(provenance['seed']); torch.manual_seed(provenance['seed'])
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False
        cnn = model_name=='CNN_Jannis_OHE'
        if not cnn and device!='cpu':
            raise ValueError('Use CPU for the original dense training recipe')
        ids = dict(c.ids,train=c.subsets[size]); x=c.x; y=c.y
        sync(device); fit_start=time.perf_counter()
        if cnn:
            args=SimpleNamespace(device=device,seed=provenance['seed'],epochs=epochs)
            fn,report=neural_fit('CNN_Jannis',x,y,ids,args,out)
        else:
            model=AubinModel(protocol['native_length'],'1_10_1') if model_name=='aubin_1_10_1' else MLPModel(protocol['native_length'],model_name)
            history,best=train(model,x[ids['train']],torch.tensor(y[ids['train']],dtype=torch.float32),
                               x[ids['validation']],torch.tensor(y[ids['validation']],dtype=torch.float32),
                               provenance['seed'],30 if epochs is None else epochs,10,32,.001)
            write_csv(out/'history.csv',history)
            torch.save(dict(state_dict=model.state_dict(),model=model_name,sequence_length=protocol['native_length'],
                            alphabet=ALPHABET,best_epoch=best,seed=provenance['seed']),out/'model.pt')
            restored=AubinModel(protocol['native_length'],'1_10_1') if model_name=='aubin_1_10_1' else MLPModel(protocol['native_length'],model_name)
            restored.load_state_dict(torch.load(out/'model.pt',weights_only=True,map_location='cpu')['state_dict'])
            np.testing.assert_array_equal(predict_encoded(model,x[ids['validation']]).numpy(),
                                          predict_encoded(restored,x[ids['validation']]).numpy())
            fn=lambda ix:predict_encoded(restored,x[ix]).numpy()
            report=dict(best_epoch=best,epochs_run=len(history),max_epochs=30 if epochs is None else epochs,
                        patience=10,batch_size=32,learning_rate=.001,parameters=sum(p.numel() for p in model.parameters()),
                        target_mean=0.,target_std=1.)
        sync(device); fit_seconds=time.perf_counter()-fit_start
        inference_start=time.perf_counter()
        predictions={s:fn(ids[s]) for s in ['validation']+SOURCES}
        sync(device); inference_seconds=time.perf_counter()-inference_start
        write_csv(out/'predictions.csv',[dict(record_id=c.rows[i]['record_id'],gene=c.rows[i]['gene'],
                  partition=s,y_true=float(y[i]),y_pred=float(p)) for s,pred in predictions.items() for i,p in zip(ids[s],pred)])
        report.update(provenance,metrics={s:metrics(y[ids[s]],p) for s,p in predictions.items()},
            counts={s:len(ix) for s,ix in ids.items()},fit_seconds=fit_seconds,inference_seconds=inference_seconds,
            elapsed_seconds=time.perf_counter()-started,representation='native OHE',
            gpu=torch.cuda.get_device_name() if device=='cuda' else None,host=platform.node(),
            precision=dict(tensor_dtype='float32',matmul_tf32=False,cudnn_tf32=torch.backends.cudnn.allow_tf32),
            versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn']},
            predictions_sha256=digest(out/'predictions.csv'),model_sha256=digest(out/'model.pt'))
        assert scientific_sources()==protocol['source_hashes'], 'Scientific code changed during fit'
        temporary=out/'metrics.json.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(out/'metrics.json')
        print(f'Completed {c.source} fold {c.fold} n={size} {model_name}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=DEFAULT)
    p.add_argument('--source',choices=SOURCES,required=True)
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--size',type=int,help='Omit to run every frozen size, with independent fresh initialization')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,help='Smoke tests only; final publication rejects overridden epochs')
    a=p.parse_args()
    c=load_context(a.directory,a.source,a.fold,a.model=='CNN_Jannis_OHE')
    for size in c.protocol['sizes'] if a.size is None else [a.size]:
        if size not in c.subsets:
            raise ValueError('Size not in frozen protocol')
        fit(c,a.model,size,a.device,a.epochs)


if __name__=='__main__':
    main()
