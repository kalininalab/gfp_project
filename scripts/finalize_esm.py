"""Validate every holdout/CV prediction and build the complete GFP benchmark tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from scripts.esm_benchmark.run import MEAN_MODELS, load_data, write_csv
from scripts.regression_metrics import metrics

MODELS=MEAN_MODELS+['CNN_old','CNN_new','CNN_Jannis']
METRICS=['spearman','pearson','r2','rmse']


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def scatter_panels(data,path):
    cols=min(4,len(data)); rows=(len(data)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(4*cols,3.8*rows),squeeze=False,layout='constrained')
    lower=min(float(np.min(a)) for v in data.values() for a in v)
    upper=max(float(np.max(a)) for v in data.values() for a in v)
    collections=[]
    for ax,(name,(true,pred)) in zip(axes.flat,data.items()):
        lo,hi=lower,upper
        collections.append(ax.hexbin(true,pred,gridsize=55,mincnt=1,cmap='viridis',extent=(lo,hi,lo,hi)))
        ax.plot([lo,hi],[lo,hi],'k--',lw=1)
        ax.set(xlabel='Measured log10 fluorescence',ylabel='Predicted log10 fluorescence',xlim=(lo,hi),ylim=(lo,hi))
        ax.set_aspect('equal',adjustable='box')
        m=metrics(true,pred)
        f=lambda v:'undefined' if v is None else f'{v:.3f}'
        ax.set_title(f'{name}\nSpearman {f(m["spearman"])} | Pearson {f(m["pearson"])}\nRMSE {m["rmse"]:.3f}')
    for ax in list(axes.flat)[len(data):]:
        ax.set_visible(False)
    norm=LogNorm(vmin=1,vmax=max(2,max(float(c.get_array().max()) for c in collections)))
    for collection in collections:
        collection.set_norm(norm)
    fig.colorbar(collections[0],ax=list(axes.flat)[:len(data)],label='Sequences per hexagon',shrink=.8)
    fig.suptitle('cgreGFP: pooled out-of-fold predictions' if path.name.startswith('cv_') else 'cgreGFP: held-out test predictions')
    fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('results/esm_cgreGFP'))
    a=p.parse_args(); root=a.directory
    rows,holdout,manifest=load_data('data/processed/baseline_v1/sequences.csv',Path('esm_embeddings/cgreGFP_t30'),'holdout')
    row_map={r['record_id']:r for r in rows}
    timing=json.loads((root/'reference_timing.json').read_text())['results']
    table=[]
    old_holdout={r['model']:r for r in read_csv('results/evaluation_cgreGFP_seed42/comparison.csv') if r['split']=='test'}
    old_cv={r['model']:r for r in read_csv('results/cv10_cgreGFP_seed42/summary.csv')}
    for name,h in old_holdout.items():
        item=dict(representation='one-hot / Hamming',model=name,holdout_fit_seconds=timing[name]['fit_seconds'],
                  holdout_inference_seconds=timing[name]['inference_seconds'],device='CPU (1 thread)',cv_fit_seconds_mean=None,
                  holdout_total_seconds=None,cv_total_seconds_mean=None)
        for metric in METRICS:
            item[f'holdout_{metric}']=float(h[metric]) if h[metric] else None
            item[f'cv_{metric}_mean']=float(old_cv[name][metric+'_mean']) if old_cv[name][metric+'_mean'] else None
            item[f'cv_{metric}_sd']=float(old_cv[name][metric+'_sd']) if old_cv[name][metric+'_sd'] else None
        table.append(item)
    plot_data={s:{'mean':{},'cnn':{}} for s in ['holdout','cv']}
    for name in MODELS:
        fold_reports=[]; oof={}
        for split_name in ['holdout']+[f'{i:02d}' for i in range(10)]:
            out=root/split_name/name
            report=json.loads((out/'metrics.json').read_text())
            if report['dataset_sha256']!=manifest['dataset_sha256']:
                raise ValueError('Dataset mismatch')
            predictions=read_csv(out/'predictions.csv')
            assignment=({r['record_id']:r['split'] for r in rows} if split_name=='holdout' else
                        {r['record_id']:r['split'] for r in read_csv(f'results/cv10_cgreGFP_seed42/splits/fold_{split_name}.csv')})
            expected={rid for rid,s in assignment.items() if s in ['validation','test']}
            if len(predictions)!=len(expected) or {r['record_id'] for r in predictions}!=expected:
                raise ValueError('Missing/repeated predictions')
            for r in predictions:
                rid=r['record_id']
                if r['split']!=assignment[rid] or float(r['y_true'])!=float(row_map[rid]['target_log10']):
                    raise ValueError('Prediction/label/split misalignment')
            for s in ['validation','test']:
                subset=[r for r in predictions if r['split']==s]
                m=metrics([float(r['y_true']) for r in subset],[float(r['y_pred']) for r in subset])
                for metric in METRICS:
                    if m[metric] is None:
                        assert report['metrics'][s][metric] is None
                    else:
                        np.testing.assert_allclose(m[metric],report['metrics'][s][metric],rtol=1e-10,atol=1e-12)
            tests=[r for r in predictions if r['split']=='test']
            if split_name=='holdout':
                h=report
                plot_data['holdout']['cnn' if name.startswith('CNN_') else 'mean'][name]=(
                    np.array([float(r['y_true']) for r in tests]),np.array([float(r['y_pred']) for r in tests]))
            else:
                fold_reports.append(report)
                for r in tests:
                    if r['record_id'] in oof:
                        raise ValueError('Record tested in more than one fold')
                    oof[r['record_id']]=dict(r,fold=split_name)
        if set(oof)!=set(row_map):
            raise ValueError('Incomplete out-of-fold coverage')
        ordered=[oof[r['record_id']] for r in rows]
        write_csv(root/f'{name}_oof_predictions.csv',ordered)
        plot_data['cv']['cnn' if name.startswith('CNN_') else 'mean'][name]=(
            np.array([float(r['y_true']) for r in ordered]),np.array([float(r['y_pred']) for r in ordered]))
        item=dict(representation=h['representation'],model=name,holdout_fit_seconds=h['fit_seconds'],
                  holdout_inference_seconds=h['inference_seconds'],device=h['gpu'] or 'CPU (1 thread)',
                  cv_fit_seconds_mean=float(np.mean([r['fit_seconds'] for r in fold_reports])),
                  holdout_total_seconds=h['elapsed_seconds'],
                  cv_total_seconds_mean=float(np.mean([r['elapsed_seconds'] for r in fold_reports])))
        for metric in METRICS:
            item[f'holdout_{metric}']=h['metrics']['test'][metric]
            values=[r['metrics']['test'][metric] for r in fold_reports]
            item[f'cv_{metric}_mean']=float(np.mean(values)) if all(v is not None for v in values) else None
            item[f'cv_{metric}_sd']=float(np.std(values,ddof=1)) if all(v is not None for v in values) else None
        table.append(item)
    write_csv(root/'comparison.csv',table)
    lines=['# cgreGFP benchmark results','',
           'CV values are mean ± sample SD across ten held-out folds. RMSE is in log10 fluorescence units.',
           'Fit time includes validation/early stopping and hyperparameter selection; inference time covers validation + test.',
           f'ESM embedding generation: {manifest["elapsed_seconds"]:.1f} s on {manifest["gpu"]} (shared by all ESM models).','',
           '| Features | Model | Holdout Spearman | Holdout Pearson | CV Spearman | CV Pearson | CV RMSE | Holdout fit (s) | Predict (s) | Including load (s) |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    for r in table:
        cv=lambda m:'—' if r[f'cv_{m}_mean'] is None else f'{r[f"cv_{m}_mean"]:.3f} ± {r[f"cv_{m}_sd"]:.3f}'
        lines.append(f'| {r["representation"]} | {r["model"]} | {fmt(r["holdout_spearman"])} | {fmt(r["holdout_pearson"])} | {cv("spearman")} | {cv("pearson")} | {cv("rmse")} | {r["holdout_fit_seconds"]:.1f} | {r["holdout_inference_seconds"]:.2f} | {fmt(r["holdout_total_seconds"])} |')
    lines.extend(['','CNNs use GPUs; other regressors use one CPU thread. Hardware is recorded in each metrics.json.',
                  'These are measured runtimes, not hardware-independent algorithm speed rankings.',
                  'Including load adds shared-storage input loading, checksum checks, and output writing; this is not measured for the old one-hot runs.',
                  'The original one-hot CV did not record per-model fit time; this is left missing rather than inferred.'])
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    for protocol,groups in plot_data.items():
        for group,data in groups.items():
            scatter_panels(data,root/f'{protocol}_{group}_true_vs_predicted')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for r in table:
        if r['cv_spearman_mean'] is None:
            continue
        color={'one-hot / Hamming':'tab:blue','ESM mean':'tab:orange','ESM residues':'tab:green'}[r['representation']]
        for ax,metric in zip(axes,['spearman','rmse']):
            ax.scatter(r['holdout_fit_seconds'],r[f'cv_{metric}_mean'],color=color)
            ax.annotate(r['model'],(r['holdout_fit_seconds'],r[f'cv_{metric}_mean']),fontsize=7,xytext=(3,3),textcoords='offset points')
            ax.set(xscale='log',xlabel='Holdout fit time (s; CPU/GPU differ)',ylabel=f'10-fold mean {metric}')
    fig.savefig(root/'accuracy_vs_runtime.png',dpi=180); plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':
    main()
