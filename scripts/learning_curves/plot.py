"""Validate all learning-curve fits and plot mean ± SD across ten held-out folds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.regression_metrics import metrics
from .common import DEFAULT, MODELS, SOURCES, PEAKS, COLORS, LABELS, digest, read_csv, write_csv, scientific_sources, fit_directory

METRICS = ['spearman','pearson','r2','rmse']
TITLES = {'spearman':'Spearman ρ','pearson':'Pearson r','r2':'R²','rmse':'RMSE (log10 fluorescence)'}
DIRECTIONS = [('artificial','artificial'),('artificial','natural'),('natural','artificial'),('natural','natural')]


def summarize(records, keys, values=METRICS):
    groups=defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,rows in groups.items():
        assert len(rows)==10 and {r['fold'] for r in rows}==set(range(10))
        item=dict(zip(keys,key),n_folds=len(rows))
        for metric in values:
            nums=[r[metric] for r in rows]
            valid=all(v is not None for v in nums)
            item[metric+'_mean']=float(np.mean(nums)) if valid else None
            item[metric+'_sd']=float(np.std(nums,ddof=1)) if valid else None
            item[metric+'_defined_folds']=sum(v is not None for v in nums)
        result.append(item)
    return result


def collect(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    assert scientific_sources()==protocol['source_hashes']
    assert digest(protocol['data_file'])==protocol['dataset_sha256']
    protocol_hash=digest(directory/'protocol.json')
    all_rows={r['record_id']:r for r in read_csv(protocol['data_file'])}
    for relative,checksum in protocol['file_hashes'].items():
        assert digest(directory/relative)==checksum, f'Changed {relative}'
    split_rows={(source,fold):read_csv(directory/'splits'/f'{source}_fold_{fold:02d}.csv')
                for source in SOURCES for fold in range(10)}
    target_ids={source:{r['record_id'] for r in split_rows[source,0]} for source in SOURCES}
    fold_scores=[]; per_peak=[]; macro=[]; runtimes=[]; pooled=[]; inputs={}
    assert len(list((directory/'fits').glob('*/*/*/*/metrics.json')))==protocol['fits'], 'Incomplete or unexpected fits'
    for source in SOURCES:
        for model in MODELS:
            for size in protocol['sizes']:
                oof={target:{} for target in SOURCES}
                for fold in range(10):
                    folder=fit_directory(directory,source,fold,size,model)
                    report=json.loads((folder/'metrics.json').read_text())
                    assert report['protocol_sha256']==protocol_hash and report['source_hashes']==protocol['source_hashes']
                    assert (report['source'],report['model'],report['fold'],report['training_size'])==(source,model,fold,size)
                    assert report['epochs_override'] is None and report['seed']==protocol['training_seed']
                    assert digest(folder/'predictions.csv')==report['predictions_sha256']
                    assert digest(folder/'model.pt')==report['model_sha256']
                    predictions=read_csv(folder/'predictions.csv')
                    groups={'validation':{r['record_id'] for r in split_rows[source,fold] if r['split']=='validation'}}
                    groups.update({target:{r['record_id'] for r in split_rows[target,fold] if r['split']=='test'} for target in SOURCES})
                    assert len(predictions)==sum(len(ids) for ids in groups.values())
                    for partition,ids in groups.items():
                        selected=[r for r in predictions if r['partition']==partition]
                        assert len(selected)==len(ids) and {r['record_id'] for r in selected}==ids
                        for r in selected:
                            original=all_rows[r['record_id']]
                            assert r['gene']==original['gene'] and float(r['y_true'])==float(original['target_log10'])
                        score=metrics([float(r['y_true']) for r in selected],[float(r['y_pred']) for r in selected])
                        for name in METRICS:
                            if score[name] is None:
                                assert report['metrics'][partition][name] is None
                            else:
                                np.testing.assert_allclose(score[name],report['metrics'][partition][name],rtol=1e-10,atol=1e-12)
                        if partition=='validation':
                            continue
                        base=dict(source=source,target=partition,model=model,training_size=size,fold=fold)
                        total=size+protocol['n_validation']+len(selected)
                        fold_scores.append(dict(base,n_train=size,n_validation=protocol['n_validation'],n_test=len(selected),
                            used_train_fraction=size/total,used_validation_fraction=protocol['n_validation']/total,
                            used_test_fraction=len(selected)/total,**score))
                        for r in selected:
                            assert r['record_id'] not in oof[partition]
                            oof[partition][r['record_id']]=(float(r['y_true']),float(r['y_pred']))
                        if partition=='artificial':
                            peak_scores=[]
                            for peak in PEAKS:
                                chosen=[r for r in selected if r['gene']==peak]
                                ps=metrics([float(r['y_true']) for r in chosen],[float(r['y_pred']) for r in chosen])
                                per_peak.append(dict(base,target_peak=peak,**ps));peak_scores.append(ps)
                            macro.append(dict(base,**{m:float(np.mean([p[m] for p in peak_scores])) if all(p[m] is not None for p in peak_scores) else None for m in METRICS}))
                    runtimes.append(dict(source=source,model=model,training_size=size,fold=fold,fit_seconds=report['fit_seconds'],
                        inference_seconds=report['inference_seconds'],gpu=report['gpu'],host=report['host'],
                        epochs_run=report['epochs_run'],best_epoch=report['best_epoch']))
                    inputs[str((folder/'metrics.json').relative_to(directory))]=digest(folder/'metrics.json')
                for target,values in oof.items():
                    assert set(values)==target_ids[target], 'Incomplete outer-fold coverage'
                    array=np.array(list(values.values()))
                    pooled.append(dict(source=source,target=target,model=model,training_size=size,
                                       **metrics(array[:,0],array[:,1])))
    summary=summarize(fold_scores,['source','target','model','training_size'])
    tables={'per_fold_scores.csv':fold_scores,'summary.csv':summary,'pooled_oof_scores.csv':pooled,
            'per_peak_fold_scores.csv':per_peak,'per_peak_summary.csv':summarize(per_peak,['source','target','target_peak','model','training_size']),
            'artificial_macro_summary.csv':summarize(macro,['source','target','model','training_size']),
            'runtimes.csv':runtimes,'runtime_summary.csv':summarize(runtimes,['source','model','training_size'],['fit_seconds','inference_seconds'])}
    for name,rows in tables.items():
        write_csv(directory/name,rows)
    return protocol,summary,tables['runtime_summary.csv'],inputs


def y_limits(rows,metric):
    values=[(r[metric+'_mean'],r[metric+'_sd']) for r in rows if r[metric+'_mean'] is not None]
    low=min(m-s for m,s in values);high=max(m+s for m,s in values)
    pad=max(.04,(high-low)*.07)
    if metric in ['spearman','pearson','r2']:
        return min(0,low-pad),max(1.02,high+pad)
    return max(0,low-pad),high+pad


def panel(ax,rows,metric,limits):
    for model,label,color,marker in zip(MODELS,LABELS,COLORS,['o','s','^','D']):
        selected=sorted([r for r in rows if r['model']==model],key=lambda r:r['training_size'])
        y=[np.nan if r[metric+'_mean'] is None else r[metric+'_mean'] for r in selected]
        sd=[0 if r[metric+'_sd'] is None else r[metric+'_sd'] for r in selected]
        ax.errorbar([r['training_size'] for r in selected],y,yerr=sd,color=color,marker=marker,
                    markersize=4.5,linewidth=1.5,elinewidth=1,capsize=3,label=label)
    ax.set(xlabel='Training sequences (validation excluded)',ylabel=TITLES[metric],ylim=limits)
    ax.set_xticks([0,2500,5000,10000,15000,17500])
    ax.tick_params(axis='x',labelsize=9)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(True,color='#E1E5EC',linestyle=':',linewidth=.8)
    ax.set_axisbelow(True)
    if limits[0]<0:
        ax.axhline(0,color='#9099A5',linewidth=.8)


def save(fig,directory,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(directory/f'{name}.{ext}',dpi=220,facecolor='white')
    plt.close(fig)


def plots(protocol,summary,directory):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    for metric in METRICS:
        natural=[r for r in summary if r['source']=='natural' and r['target']=='natural']
        fig,ax=plt.subplots(figsize=(10,6.5))
        panel(ax,natural,metric,y_limits(natural,metric))
        ax.set_title('Natural cgreGFP → natural cgreGFP',fontsize=17,pad=16)
        ax.legend(loc='best',frameon=False)
        fig.subplots_adjust(left=.10,right=.97,top=.89,bottom=.25)
        fig.text(.10,.14,f'Mean ± sample SD across 10 held-out folds; nested training sizes; {protocol["n_validation"]:,} fixed source-validation examples.',fontsize=9)
        fig.text(.10,.10,'Test: 2,451–2,452 natural cgre sequences per fold (10%); each sequence tested once per model and size.',fontsize=9)
        fig.text(.10,.06,'The x-axis counts training labels only; total source-label budget = training size + validation size.',fontsize=9)
        save(fig,directory,f'cgre_learning_curve_{metric}')
        fig,axes=plt.subplots(2,2,figsize=(13,10))
        limits=y_limits(summary,metric)
        for ax,(source,target) in zip(axes.flat,DIRECTIONS):
            selected=[r for r in summary if r['source']==source and r['target']==target]
            panel(ax,selected,metric,limits)
            ax.set_title(f'{source.capitalize()} → {target.capitalize()}',fontweight='bold')
        fig.suptitle('Training size and natural/artificial cgre transfer',fontsize=18,y=.975)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.105),ncol=4,frameon=False)
        fig.subplots_adjust(left=.09,right=.98,top=.91,bottom=.24,hspace=.34,wspace=.24)
        fig.text(.09,.087,f'10 paired source/target folds; mean ± fold SD. Each fit uses n training + {protocol["n_validation"]:,} source-validation sequences.',fontsize=9)
        fig.text(.09,.058,'Test: natural 2,451–2,452 or artificial 2,690–2,691 (10% of target data per fold). Exact used ratios are in per_fold_scores.csv.',fontsize=9)
        fig.text(.09,.029,'Artificial = four peaks pooled, with proportional sampling; all peaks represented in each fold. All models use native OHE.',fontsize=9)
        save(fig,directory,f'peak_learning_curves_{metric}')


def report(protocol,summary,runtimes,directory):
    largest=max(protocol['sizes']); val=protocol['n_validation']
    ratios={}
    for target in SOURCES:
        test=int(np.ceil(protocol['domain_counts'][target]/10))
        total=largest+val+test
        ratios[target]='/'.join(f'{100*n/total:.1f}' for n in [largest,val,test])
    lines=['# Training-size learning curves','',
        f'All {protocol["fits"]} fits validated: four native-OHE models × two training sources × {len(protocol["sizes"])} sizes × ten folds.',
        'Each checkpoint is evaluated on both target domains. Natural→natural is the same curve in both figure families.',
        'Scores below are mean ± sample SD across ten folds, not confidence intervals or variation across ten random seeds.','',
        f'Training sizes: {protocol["sizes"]}. Every fit has {protocol["n_validation"]:,} additional, fixed source-validation labels.',
        'Outer test fraction is 10% of each target domain. Smaller training samples are nested; unused source records stay unused.',
        f'At {largest:,} training examples, used train/validation/test proportions are about {ratios["natural"]}% for natural targets and {ratios["artificial"]}% for artificial targets.',
        'At smaller sizes these proportions change; per_fold_scores.csv gives exact counts and ratios. The x-axis is not the total labeling budget.',
        'Artificial folds and samples preserve peak proportions. Artificial→artificial tests variants within represented peaks, not unseen peaks.',
        'Training settings are fixed across sizes (30 dense / 60 CNN maximum epochs); curves measure this recipe, not separately optimized models at each data budget.','',
        '[Natural cgre learning curve](cgre_learning_curve_spearman.png) · [Four transfer directions](peak_learning_curves_spearman.png)',
        'Pearson, R² and RMSE companion plots use the corresponding filename suffixes. R² may be negative; it is not Pearson squared.','',
        '| Training → test | Model | Training n | Spearman | Pearson | R² | RMSE | Mean fit (min) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    times={(r['source'],r['model'],r['training_size']):r['fit_seconds_mean']/60 for r in runtimes}
    for source,target in DIRECTIONS:
        for model,label in zip(MODELS,LABELS):
            for r in sorted([r for r in summary if (r['source'],r['target'],r['model'])==(source,target,model)],key=lambda r:r['training_size']):
                fmt=lambda m:'undefined' if r[m+'_mean'] is None else f'{r[m+"_mean"]:.3f} ± {r[m+"_sd"]:.3f}'
                lines.append(f'| {source} → {target} | {label} | {r["training_size"]:,} | {fmt("spearman")} | {fmt("pearson")} | {fmt("r2")} | {fmt("rmse")} | {times[source,model,r["training_size"]]:.2f} |')
    lines += ['', 'Runtimes include fitting and validation, exclude scheduler wait and shared context loading, and use different CPU/GPU hardware.',
              'These are exploratory curves on existing data. Previous CV comparisons used these natural folds; no fresh blind dataset is claimed.',
              'Per-peak and macro summaries distinguish within-peak prediction from pooled effects. Undefined correlations are retained rather than set to zero.']
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=DEFAULT)
    a=p.parse_args()
    protocol,summary,runtimes,inputs=collect(a.directory)
    plots(protocol,summary,a.directory)
    report(protocol,summary,runtimes,a.directory)
    result=dict(status='complete',fits=len(inputs),protocol_sha256=digest(a.directory/'protocol.json'),
                plot_script_sha256=digest(__file__),input_metrics=inputs,
                outputs={p.name:digest(p) for p in a.directory.iterdir() if p.is_file() and p.suffix in ['.csv','.png','.pdf','.svg','.md']})
    (a.directory/'finalization.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Validated all fits and produced learning curves:',a.directory)


if __name__=='__main__':
    main()
