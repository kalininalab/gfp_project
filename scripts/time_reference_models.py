"""Repeat fixed one-hot holdout fits to measure fit/inference time consistently."""
import csv
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
import torch

from scripts.aubin_model import AubinModel
from scripts.baseline_models import MLPModel, sparse_one_hot
from scripts.train_aubin import load_data, train, predict_encoded, sha256
from scripts.train_baselines import fit_ridge, fit_knn
from scripts.regression_metrics import metrics


def main():
    destination=Path('results/esm_cgreGFP/reference_timing.json')
    if destination.exists():
        raise FileExistsError(destination)
    data_path=Path('data/processed/baseline_v1/sequences.csv')
    data=load_data(data_path,'cgreGFP')
    x={s:sparse_one_hot(v[1].numpy()) for s,v in data.items()}
    y={s:np.array([float(r['target_log10']) for r in v[0]]) for s,v in data.items()}
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    reports={}
    for name in ['mean','linear_regression','ridge','aubin_linear','aubin_1_10_1','mlp_small','mlp_deep','knn']:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        with threadpool_limits(limits=1):
            started=time.perf_counter()
            if name.startswith(('aubin_','mlp_')):
                model=(AubinModel(data['train'][1].shape[1],name.removeprefix('aubin_')) if name.startswith('aubin_')
                       else MLPModel(data['train'][1].shape[1],name))
                history,best=train(model,data['train'][1],data['train'][2],data['validation'][1],data['validation'][2],42,30,10,32,.001)
                fn=lambda s:predict_encoded(model,data[s][1]).numpy()
            else:
                if name=='mean':
                    model=DummyRegressor().fit(x['train'],y['train'])
                elif name=='linear_regression':
                    model=LinearRegression(tol=1e-8).fit(x['train'],y['train'])
                elif name=='ridge':
                    model,_=fit_ridge(x['train'],y['train'],x['validation'],y['validation'],[.1,1.,10.,100.])
                else:
                    model,_=fit_knn(data['train'][1].numpy(),y['train'],data['validation'][1].numpy(),y['validation'],[1,3,5,11,21,51])
                fn=lambda s:model.predict(data[s][1].numpy() if name=='knn' else x[s])
            fit_seconds=time.perf_counter()-started
            started=time.perf_counter()
            predictions={s:fn(s) for s in ['validation','test']}
            inference_seconds=time.perf_counter()-started
        if name.startswith('aubin_'):
            original=Path('results/aubin_cgreGFP_seed42')/name.removeprefix('aubin_')/'predictions.csv'
        else:
            original=Path('results/knn_cgreGFP_seed42' if name=='knn' else 'results/baselines_cgreGFP_seed42')/name/'predictions.csv'
        records=list(csv.DictReader(original.open()))
        reference={r['record_id']:float(r['prediction_log10']) for r in records if r['split']=='test'}
        expected=np.array([reference[r['record_id']] for r in data['test'][0]])
        np.testing.assert_allclose(predictions['test'],expected,rtol=1e-5,atol=1e-6)
        reports[name]=dict(fit_seconds=fit_seconds,inference_seconds=inference_seconds,
                          test_metrics=metrics(y['test'],predictions['test']),host=platform.node(),device='cpu',threads=1)
        print(name,reports[name],flush=True)
    destination.write_text(json.dumps(dict(results=reports,dataset_sha256=sha256(data_path),
                                           script_sha256=sha256(Path(__file__))),indent=2)+'\n')


if __name__=='__main__':
    main()
