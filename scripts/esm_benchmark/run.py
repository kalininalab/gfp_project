"""Fit a single frozen-feature regressor on an existing holdout or CV split."""
import argparse
import csv
import json
import importlib.metadata
import os
from pathlib import Path
import pickle
import platform
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from scripts.regression_metrics import metrics
from .embed import digest
from .models import make_model

MEAN_MODELS = ['mean','linear_regression','ridge','aubin_linear','aubin_1_10_1','mlp_small','mlp_deep','knn']


def write_csv(path, rows):
    with Path(path).open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def load_data(data, embeddings, split):
    manifest = json.loads((embeddings/'manifest.json').read_text())
    if digest(data) != manifest['dataset_sha256']:
        raise ValueError('Prepared data changed since embedding generation')
    with Path(data).open() as f:
        rows = [r for r in csv.DictReader(f) if r['gene']==manifest['gene']]
    with (embeddings/'records.csv').open() as f:
        records = list(csv.DictReader(f))
    if [(r['record_id'],r['sequence']) for r in rows] != [(r['record_id'],r['sequence']) for r in records]:
        raise ValueError('Embedding order/sequence mismatch')
    assignment = {r['record_id']:r['split'] for r in rows}
    if split != 'holdout':
        with Path(split).open() as f:
            split_rows = list(csv.DictReader(f))
        assignment = {r['record_id']:r['split'] for r in split_rows}
        if len(assignment)!=len(split_rows) or set(assignment)!={r['record_id'] for r in rows}:
            raise ValueError('Split must contain every record exactly once')
    ids = {s:np.array([i for i,r in enumerate(rows) if assignment[r['record_id']]==s])
           for s in ['train','validation','test']}
    if sum(map(len,ids.values()))!=len(rows) or any(len(v)<2 for v in ids.values()):
        raise ValueError('Invalid split')
    seqs = {s:{rows[i]['sequence'] for i in ix} for s,ix in ids.items()}
    assert not (seqs['train']&seqs['validation'] or seqs['train']&seqs['test'] or seqs['validation']&seqs['test'])
    return rows, ids, manifest


def sync(device):
    if device=='cuda':
        torch.cuda.synchronize()


@torch.inference_mode()
def predict(model, x, ids, device, batch_size, center=0., scale=1.):
    model.eval()
    predictions = []
    for start in range(0,len(ids),batch_size):
        xb = torch.from_numpy(np.array(x[ids[start:start+batch_size]],dtype=np.float32)).to(device)
        predictions.append(model(xb).float().cpu().numpy()*scale+center)
    return np.concatenate(predictions)


def neural_fit(name,x,y,ids,a,out):
    cnn = name.startswith('CNN_')
    model = make_model(name,x.shape[-1]).to(a.device)
    center,scale = (float(y[ids['train']].mean()),float(y[ids['train']].std())) if cnn else (0.,1.)
    scale = max(scale,1e-8)
    target = (y-center)/scale
    epochs,batch,lr = (60,64,3e-4) if cnn else (30,32,.001)
    if a.epochs is not None:
        epochs = a.epochs
    optimizer = (torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4) if cnn
                 else torch.optim.Adam(model.parameters(),lr=lr,eps=1e-7))
    best_loss,best_epoch,best_state,stale = float('inf'),0,None,0
    history = []
    rng = np.random.default_rng(a.seed)
    for epoch in range(1,epochs+1):
        model.train()
        order = rng.permutation(ids['train'])
        total = 0.
        for start in range(0,len(order),batch):
            idx = order[start:start+batch]
            xb = torch.from_numpy(np.array(x[idx],dtype=np.float32)).to(a.device)
            yb = torch.tensor(target[idx],dtype=torch.float32,device=a.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb),yb)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if cnn:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total += loss.item()*len(idx)
        pred = predict(model,x,ids['validation'],a.device,batch,center,scale)
        val_loss = float(np.mean((pred-y[ids['validation']])**2))
        history.append(dict(epoch=epoch,train_mse=total/len(order)*scale**2,validation_mse=val_loss))
        print(f'{name} epoch {epoch}: validation MSE {val_loss:.6f}',flush=True)
        if val_loss < best_loss:
            best_loss,best_epoch,stale = val_loss,epoch,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state)
    artifact = dict(model=name,input_dim=x.shape[-1],state_dict=best_state,target_mean=center,target_std=scale,
                    best_epoch=best_epoch,seed=a.seed)
    torch.save(artifact,out/'model.pt')
    write_csv(out/'history.csv',history)
    restored = make_model(name,x.shape[-1]).to(a.device)
    restored.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
    np.testing.assert_array_equal(predict(model,x,ids['validation'][:32],a.device,batch,center,scale),
                                  predict(restored,x,ids['validation'][:32],a.device,batch,center,scale))
    report = dict(best_epoch=best_epoch,epochs_run=len(history),max_epochs=epochs,patience=10,batch_size=batch,
                  learning_rate=lr,parameters=sum(p.numel() for p in model.parameters()),target_mean=center,target_std=scale)
    return lambda idx:predict(model,x,idx,a.device,batch,center,scale),report


def sklearn_fit(name,x,y,ids,out):
    xt,yt = x[ids['train']],y[ids['train']]
    xv,yv = x[ids['validation']],y[ids['validation']]
    trials = []
    if name=='mean':
        model = DummyRegressor().fit(xt,yt)
    elif name=='linear_regression':
        model = LinearRegression().fit(xt,yt)
    elif name=='ridge':
        candidates = []
        for alpha in [.1,1.,10.,100.]:
            m = Ridge(alpha=alpha).fit(xt,yt)
            loss = float(np.mean((m.predict(xv)-yv)**2))
            trials.append(dict(alpha=alpha,validation_mse=loss)); candidates.append(m)
        model = candidates[int(np.argmin([t['validation_mse'] for t in trials]))]
    else:
        candidates = []
        for k in [1,3,5,11,21,51]:
            for weights in ['uniform','distance']:
                m = KNeighborsRegressor(n_neighbors=k,weights=weights,metric='euclidean',algorithm='brute',n_jobs=1).fit(xt,yt)
                loss = float(np.mean((m.predict(xv)-yv)**2))
                trials.append(dict(k=k,weights=weights,validation_mse=loss)); candidates.append(m)
        model = candidates[int(np.argmin([t['validation_mse'] for t in trials]))]
    with (out/'model.pkl').open('wb') as f:
        pickle.dump(model,f)
    with (out/'model.pkl').open('rb') as f:
        restored = pickle.load(f)
    np.testing.assert_allclose(model.predict(xv[:32]),restored.predict(xv[:32]),rtol=0,atol=0)
    return lambda idx:model.predict(x[idx]),dict(trials=trials,selected_parameters=model.get_params())


def run(a):
    out = Path(a.output)/a.split_name/a.model
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        old = json.loads((out/'metrics.json').read_text())
        current_scripts = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        if (old['scripts'] != current_scripts or old['dataset_sha256'] != digest(a.data)
            or old['split_sha256'] != digest(a.data if a.split=='holdout' else a.split)
            or old['embedding_manifest_sha256'] != digest(Path(a.embeddings)/'manifest.json')
            or old['seed'] != a.seed or old['device'] != a.device or old.get('epochs_override') != a.epochs):
            raise FileExistsError(f'{out}: completed with different inputs/settings; use a new output root')
        print(f'{out}: verified completed run; reusing',flush=True)
        return
    started = time.perf_counter()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows,ids,manifest = load_data(a.data,Path(a.embeddings),a.split)
    y = np.array([float(r['target_log10']) for r in rows])
    cnn = a.model.startswith('CNN_')
    feature_file = Path(a.embeddings)/('residue_embeddings.npy' if cnn else 'mean_embeddings.npy')
    # One RAM copy per CNN worker avoids repeated random reads from the network filesystem.
    x = np.load(feature_file)
    expected_shape = ((len(rows),len(rows[0]['sequence']),640) if cnn else (len(rows),640))
    if x.shape != expected_shape or digest(feature_file) != manifest['files'][feature_file.name]:
        raise ValueError('Embedding shape or checksum mismatch')
    for start in range(0,len(x),128):
        if not np.isfinite(x[start:start+128]).all():
            raise ValueError('Nonfinite embeddings')
    if not cnn:
        scaler = StandardScaler().fit(x[ids['train']])
        x = scaler.transform(x).astype(np.float32)
        with (out/'feature_scaler.pkl').open('wb') as f:
            pickle.dump(scaler,f)
    sync(a.device)
    fit_start = time.perf_counter()
    with threadpool_limits(limits=1):
        fn,report = (sklearn_fit(a.model,x,y,ids,out) if a.model in ['mean','linear_regression','ridge','knn']
                     else neural_fit(a.model,x,y,ids,a,out))
        sync(a.device)
        fit_seconds = time.perf_counter()-fit_start
        inference_start = time.perf_counter()
        predictions = {s:fn(ids[s]) for s in ['validation','test']}
        sync(a.device)
        inference_seconds = time.perf_counter()-inference_start
    pred_rows = [dict(record_id=rows[i]['record_id'],split=s,y_true=float(y[i]),y_pred=float(p))
                 for s in predictions for i,p in zip(ids[s],predictions[s])]
    write_csv(out/'predictions.csv',pred_rows)
    report.update(model=a.model,representation='ESM residues' if cnn else 'ESM mean',split=a.split_name,
        metrics={s:metrics(y[ids[s]],p) for s,p in predictions.items()},seed=a.seed,epochs_override=a.epochs,
        fit_seconds=fit_seconds,inference_seconds=inference_seconds,elapsed_seconds=time.perf_counter()-started,
        device=a.device,gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,host=platform.node(),
        precision=dict(tensor_dtype='float32',matmul_tf32=False,cudnn_tf32=torch.backends.cudnn.allow_tf32),
        versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','scikit-learn','transformers']},
        dataset_sha256=manifest['dataset_sha256'],embedding_manifest_sha256=digest(Path(a.embeddings)/'manifest.json'),
        split_sha256=digest(a.data if a.split=='holdout' else a.split),
        scripts={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/processed/baseline_v1/sequences.csv')
    p.add_argument('--embeddings',default='esm_embeddings/cgreGFP_t30')
    p.add_argument('--output',default='results/esm_cgreGFP')
    p.add_argument('--split',default='holdout',help='holdout or record_id,split CSV')
    p.add_argument('--split-name',default='holdout')
    p.add_argument('--model',required=True,choices=MEAN_MODELS+['CNN_old','CNN_new','CNN_Jannis','all_mean'])
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,help='Override preset; only for explicit experiments/smoke tests')
    a = p.parse_args()
    for model in MEAN_MODELS if a.model=='all_mean' else [a.model]:
        a.model = model
        run(a)


if __name__=='__main__':
    main()
