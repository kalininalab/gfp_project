"""Separate MLP classifier and low/high experts using a train-only KDE valley."""
import argparse
import copy
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from threadpoolctl import threadpool_limits
from scripts.baseline_models import MLPModel
from scripts.train_aubin import load_data, train, predict_encoded
from scripts.regression_metrics import metrics
from scripts.peak_rotation.common import DATA, ROOT, digest, read_csv, write_csv

RAW = Path('/data/users/akolchina/gfp_project/valley_mlp_cgre_seed42_46')
PUB = ROOT / 'results/valley_mlp_cgre_seed42_46'


def hashes():
    return {s: digest(ROOT / 'scripts' / s) for s in ['hurdle_predictor/valley_mlp.py',
        'baseline_models.py', 'aubin_model.py', 'train_aubin.py', 'regression_metrics.py']}


def savefig(fig, stem):
    for ext in ['png', 'pdf', 'svg']: fig.savefig(stem.with_suffix('.'+ext), dpi=300, bbox_inches='tight')
    plt.close(fig)


def prepare():
    if (RAW/'protocol.json').exists(): raise FileExistsError('Protocol already frozen')
    data = load_data(DATA, 'cgreGFP')
    y = data['train'][2].numpy().astype(float)
    grid = np.linspace(y.min()-.2, y.max()+.2, 2048)
    checks = []
    for multiplier in [.8, 1., 1.2]:
        kde = gaussian_kde(y); kde.set_bandwidth(kde.factor*multiplier)
        density = kde(grid)
        peaks, _ = find_peaks(density, prominence=.05*density.max())
        if len(peaks) < 2: raise ValueError('No clear two-mode structure: no jobs prepared')
        left, right = sorted(sorted(peaks, key=lambda i: density[i], reverse=True)[:2])
        valley = left + int(np.argmin(density[left:right+1]))
        ratio = float(density[valley]/min(density[left], density[right]))
        if ratio > .8: raise ValueError('Valley too shallow: no jobs prepared')
        checks.append(dict(bandwidth_multiplier=multiplier, threshold=float(grid[valley]), valley_to_smaller_mode=ratio))
        if multiplier == 1.: chosen_density = density.copy()
    if np.ptp([c['threshold'] for c in checks]) > .15: raise ValueError('Unstable valley: no jobs prepared')
    threshold = checks[1]['threshold']
    RAW.mkdir(parents=True); (RAW/'condor').mkdir(); PUB.mkdir(parents=True, exist_ok=True)
    protocol = dict(gene='cgreGFP', threshold=threshold, threshold_method='Scott-bandwidth Gaussian KDE; minimum between two highest prominent modes; training labels only',
        sensitivity=checks, seeds=list(range(42,47)), dataset_sha256=digest(DATA), source_hashes=hashes(),
        counts={s: len(v[0]) for s,v in data.items()}, train_low_fraction=float((y<=threshold).mean()),
        architecture='Separate OHE → 64 ReLU → 1 MLPs for classifier, low expert, high expert and direct comparator',
        epochs=30, patience=10, batch_size=32, learning_rate=.001,
        validation='Classifier: all validation records, BCE. Experts: respective true validation group, MSE. Direct: all validation records, MSE.',
        prediction='No clipping. Hard selects high expert iff p >= 0.5; soft weights same experts. Oracle uses true test class only as diagnostic, never deployable.',
        target='Original log10 fluorescence, no target normalization; matches existing small MLP recipe')
    (RAW/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (PUB/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    write_csv(RAW/'split_records.csv',[dict(record_id=r['record_id'], split=s) for s,v in data.items() for r in v[0]])
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(y,bins=60,density=True,color='#56B4E9',alpha=.4,label='Training fluorescence')
    ax.plot(grid,chosen_density,color='#8254A2',lw=2,label='KDE')
    ax.axvline(threshold,color='#CC79A7',lw=2,ls='--',label=f'Valley = {threshold:.3f}')
    ax.set(xlabel='log10 fluorescence',ylabel='Density',title='Training-only low/high boundary: natural cgreGFP'); ax.legend()
    savefig(fig,PUB/'training_threshold')
    nodes=[f'S{seed}' for seed in protocol['seeds']]
    lines=[]
    for seed,node in zip(protocol['seeds'],nodes):
        lines += [f'JOB {node} condor/valley_mlp.sub',f'VARS {node} seed="{seed}"']
    lines += ['JOB FINAL condor/valley_mlp_finalize.sub','PARENT '+' '.join(nodes)+' CHILD FINAL']
    (ROOT/'condor/valley_mlp.dag').write_text('\n'.join(lines)+'\n')
    print(json.dumps(protocol,indent=2))


def classifier_fit(model, xt, yt, xv, yv, seed, p):
    optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],eps=1e-7)
    generator=torch.Generator().manual_seed(seed)
    best=float('inf'); state=None; stale=0; history=[]; best_epoch=0
    for epoch in range(1,p['epochs']+1):
        model.train()
        for ids in torch.randperm(len(xt),generator=generator).split(p['batch_size']):
            optimizer.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xt[ids]),yt[ids])
            if not torch.isfinite(loss): raise ValueError('Nonfinite classifier loss')
            loss.backward(); optimizer.step()
        val=float(torch.nn.functional.binary_cross_entropy_with_logits(predict_encoded(model,xv),yv))
        history.append(dict(epoch=epoch,validation_bce=val))
        if val<best: best=val; state=copy.deepcopy(model.state_dict()); best_epoch=epoch; stale=0
        else: stale+=1
        if stale>=p['patience']: break
    model.load_state_dict(state)
    return history,best_epoch


def run(seed):
    p=json.loads((RAW/'protocol.json').read_text())
    assert hashes()==p['source_hashes'] and digest(DATA)==p['dataset_sha256'] and seed in p['seeds']
    out=RAW/f'seed{seed}'; out.mkdir(exist_ok=True)
    ph=digest(RAW/'protocol.json')
    if (out/'complete.json').exists():
        assert json.loads((out/'complete.json').read_text())['protocol_sha256']==ph
        return
    data=load_data(DATA,'cgreGFP'); xt,yt=data['train'][1:]; xv,yv=data['validation'][1:]; xe,ye=data['test'][1:]
    threshold=p['threshold']; labels_t=yt>threshold; labels_v=yv>threshold
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    predictions={}
    with threadpool_limits(limits=1):
        for name in ['direct','classifier','low','high']:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model=MLPModel(xt.shape[1],'mlp_small')
            checkpoint=out/f'{name}.pt'
            if checkpoint.exists():
                ck=torch.load(checkpoint,weights_only=True)
                assert ck['protocol_sha256']==ph
                model.load_state_dict(ck['state_dict'])
            else:
                if name=='classifier':
                    history,best=classifier_fit(model,xt,labels_t.float(),xv,labels_v.float(),seed,p)
                else:
                    ti=torch.ones(len(yt),dtype=torch.bool) if name=='direct' else (labels_t if name=='high' else ~labels_t)
                    vi=torch.ones(len(yv),dtype=torch.bool) if name=='direct' else (labels_v if name=='high' else ~labels_v)
                    history,best=train(model,xt[ti],yt[ti],xv[vi],yv[vi],seed,p['epochs'],p['patience'],p['batch_size'],p['learning_rate'])
                write_csv(out/f'{name}_history.csv',history)
                torch.save(dict(state_dict=model.state_dict(),best_epoch=best,protocol_sha256=ph),checkpoint)
            values=predict_encoded(model,xe)
            predictions[name]=(torch.sigmoid(values) if name=='classifier' else values).numpy()
    prob=predictions['classifier']; low=predictions['low']; high=predictions['high']; truth=ye.numpy(); actual=truth>threshold
    outputs=dict(direct=predictions['direct'],hard=np.where(prob>=.5,high,low),
                 soft=(1-prob)*low+prob*high,oracle_diagnostic=np.where(actual,high,low))
    scores=[dict(seed=seed,method=k,**metrics(truth,v)) for k,v in outputs.items()]
    write_csv(out/'metrics.csv',scores)
    classes=dict(roc_auc=float(roc_auc_score(actual,prob)),balanced_accuracy=float(balanced_accuracy_score(actual,prob>=.5)),
        confusion_matrix=confusion_matrix(actual,prob>=.5).tolist(),
        low_expert_true_low=metrics(truth[~actual],low[~actual]),high_expert_true_high=metrics(truth[actual],high[actual]))
    (out/'classifier_and_experts.json').write_text(json.dumps(classes,indent=2)+'\n')
    write_csv(out/'predictions.csv',[dict(record_id=r['record_id'],y_true=float(truth[i]),p_high=float(prob[i]),
        low=float(low[i]),high=float(high[i]),**{k:float(v[i]) for k,v in outputs.items()}) for i,r in enumerate(data['test'][0])])
    (out/'complete.json').write_text(json.dumps(dict(protocol_sha256=ph,predictions_sha256=digest(out/'predictions.csv')))+'\n')
    print(seed,'complete',flush=True)


def finalize():
    p=json.loads((RAW/'protocol.json').read_text()); rows=[]; diagnostics=[]
    for seed in p['seeds']:
        out=RAW/f'seed{seed}'; complete=json.loads((out/'complete.json').read_text())
        assert complete['protocol_sha256']==digest(RAW/'protocol.json') and complete['predictions_sha256']==digest(out/'predictions.csv')
        rows+=read_csv(out/'metrics.csv')
        diagnostics.append(dict(seed=seed,**json.loads((out/'classifier_and_experts.json').read_text())))
        pred=read_csv(out/'predictions.csv'); y=np.array([float(r['y_true']) for r in pred])
        fig,axes=plt.subplots(2,4,figsize=(16,7))
        for j,method in enumerate(['direct','hard','soft','oracle_diagnostic']):
            values=np.array([float(r[method]) for r in pred]); lim=[min(y.min(),values.min()),max(y.max(),values.max())]
            axes[0,j].scatter(y,values,s=3,alpha=.15,color='#0072B2'); axes[0,j].plot(lim,lim,color='#333333',lw=1)
            axes[0,j].set(title=method.replace('_',' '),xlabel='Measured log10 fluorescence',ylabel='Predicted log10 fluorescence')
            bins=np.linspace(*lim,51)
            axes[1,j].hist(y,bins=bins,density=True,histtype='step',color='#0072B2',label='True')
            axes[1,j].hist(values,bins=bins,density=True,histtype='step',color='#CC79A7',label='Predicted')
            axes[1,j].set(xlabel='log10 fluorescence',ylabel='Density'); axes[1,j].legend()
        fig.suptitle(f'cgreGFP MLP classification–regression · seed {seed}'); fig.tight_layout()
        savefig(fig,PUB/f'diagnostics_seed{seed}')
    write_csv(PUB/'all_metrics.csv',rows)
    (PUB/'classifier_and_experts.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    summary=[]
    for method in ['direct','hard','soft','oracle_diagnostic']:
        selected=[r for r in rows if r['method']==method]; item=dict(method=method,n_seeds=len(selected))
        for metric in ['pearson','spearman','r2','rmse']:
            vals=np.array([float(r[metric]) for r in selected]); item[metric+'_mean']=float(vals.mean());item[metric+'_sd']=float(vals.std(ddof=1))
        summary.append(item)
    write_csv(PUB/'summary.csv',summary)
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,metric in zip(axes,['pearson','spearman','r2','rmse']):
        ax.bar(range(4),[r[metric+'_mean'] for r in summary],yerr=[r[metric+'_sd'] for r in summary],capsize=3,color=['#0072B2','#CC79A7','#56B4E9','#999999'])
        ax.set_xticks(range(4),['Direct','Hard','Soft','Oracle*'],rotation=25); ax.set_title(metric)
        if metric=='r2':ax.set_ylim(-1,1)
    fig.suptitle('cgreGFP: separate MLP classifier and regressors (*oracle is diagnostic only)');fig.tight_layout();savefig(fig,PUB/'comparison')
    report=(PUB/'REPORT.md').read_text()
    report=report.replace('Status: submitted; results pending.', 'Status: complete, all five seeds.')
    report+='\n## Results\n\n![Comparison](comparison.png)\n\n| Method | Pearson | Spearman | R² | RMSE |\n|---|---:|---:|---:|---:|\n'
    for r in summary:report+='| '+r['method']+' | '+' | '.join(f'{r[m+"_mean"]:.3f} ± {r[m+"_sd"]:.3f}' for m in ['pearson','spearman','r2','rmse'])+' |\n'
    report+='\n[Classifier and expert diagnostics](classifier_and_experts.json) · [All seeded metrics](all_metrics.csv)\n'
    for seed in p['seeds']:report+=f'\n[Seed {seed}: scatter plots and distributions](diagnostics_seed{seed}.png)\n'
    (PUB/'REPORT.md').write_text(report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','finalize']);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args.seed)
    else:finalize()
