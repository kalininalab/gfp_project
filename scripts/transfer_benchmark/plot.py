"""Validate completed transfer experiments and draw source-mixture/transfer plots."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.regression_metrics import metrics
from .common import DEFAULT,DATA,MODELS,MIXES,NATURAL,PEAKS,digest,read_csv,write_csv,scientific_sources

COLORS={'aubin_1_10_1':'#4267AC','mlp_small':'#D6813A','mlp_deep':'#865BA6','CNN_Jannis':'#BE577C'}
LABELS={'aubin_1_10_1':'Aubin 1–10–1','mlp_small':'Small MLP','mlp_deep':'Deep MLP','CNN_Jannis':'CNN Jannis'}
MIX_ORDER=[m for m in MIXES if m!='artificial']
DIRECTIONS=[('artificial','artificial'),('artificial','natural'),('cgre','artificial'),('cgre','natural')]
SCORES=['spearman','pearson','rmse','r2']


def style_axis(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ABB3C0')
    ax.tick_params(colors='#394352',labelsize=10)
    ax.yaxis.grid(True,color='#E1E5EC',linestyle=':',linewidth=.9)
    ax.set_axisbelow(True);ax.axhline(0,color='#7D8798',lw=.8)


def draw_bars(ax,values,errors,color,ylim,samples=None):
    x=np.arange(len(values))
    plotted=[np.nan if v is None else v for v in values]
    ax.bar(x,plotted,width=.65,color=color,edgecolor='white',linewidth=.8,zorder=3)
    ax.errorbar(x,plotted,yerr=[0 if e is None else e for e in errors],fmt='none',ecolor='#4B5565',capsize=3,lw=1.15,zorder=4)
    ax.set_ylim(*ylim);ax.set_xlim(-.65,len(values)-.35);style_axis(ax)
    for i,observations in enumerate(samples or []):
        for offset,value in zip(np.linspace(-.10,.10,len(observations)),observations):
            if value is not None:
                ax.scatter(i+offset,value,s=16,facecolor='white',edgecolor='#354052',linewidth=.8,zorder=5)
    for i,(v,e) in enumerate(zip(values,errors)):
        if v is None:
            ax.text(i,.035,'undefined',ha='center',va='bottom',fontsize=9,rotation=90,color='#596779')
            continue
        ax.text(i,v+e+.024 if v>=0 else v-e-.024,f'{v:.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=10,color='#283344')


def limits(summary,experiment,metric):
    vals=[r for r in summary if r['experiment']==experiment]
    lo=min(r[metric+'_mean']-r[metric+'_sd'] for r in vals if r[metric+'_mean'] is not None)
    observations=[v for r in vals for v in json.loads(r.get(metric+'_runs_json','[]')) if v is not None]
    if observations:
        lo=min(lo,min(observations))
    return min(-.10,np.floor((lo-.075)*10)/10),1.03


def save(fig,directory,name):
    for suffix in ['png','pdf','svg']:
        fig.savefig(directory/f'{name}.{suffix}',dpi=220,facecolor='white')
    plt.close(fig)


def mixture_plot(summary,protocol,directory,metric='spearman'):
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':14,'axes.labelsize':11})
    fig=plt.figure(figsize=(14,11.5))
    grid=fig.add_gridspec(2,2,left=.12,right=.98,top=.89,bottom=.15,hspace=.40,wspace=.22)
    ylim=limits(summary,'ortholog_mixtures',metric)
    for panel,model in enumerate(MODELS):
        inner=grid[panel//2,panel%2].subgridspec(2,1,height_ratios=[3.2,1.05],hspace=.05)
        ax=fig.add_subplot(inner[0]);membership=fig.add_subplot(inner[1],sharex=ax)
        lookup={r['condition']:r for r in summary if r['experiment']=='ortholog_mixtures' and r['model']==model}
        values=[lookup[m][metric+'_mean'] for m in MIX_ORDER];errors=[lookup[m][metric+'_sd'] for m in MIX_ORDER]
        draw_bars(ax,values,errors,COLORS[model],ylim,[json.loads(lookup[m].get(metric+'_runs_json','[]')) for m in MIX_ORDER])
        ax.set_title(LABELS[model],loc='left',color=COLORS[model],fontweight='bold',pad=12)
        ax.set_ylabel('Test Spearman ρ' if metric=='spearman' else 'Test Pearson r')
        ax.tick_params(axis='x',bottom=False,labelbottom=False)
        for row,gene in enumerate(NATURAL):
            for col,mix in enumerate(MIX_ORDER):
                included=gene in MIXES[mix]
                membership.scatter(col,row,s=100,facecolor=COLORS[model] if included else 'white',
                                   edgecolor=COLORS[model] if included else '#BEC6D2',linewidth=1.3)
        membership.set_yticks(range(3),NATURAL,fontsize=10)
        membership.set_ylim(2.6,-.6);membership.set_xticks([])
        membership.tick_params(axis='y',length=0,pad=9)
        membership.spines[:].set_visible(False)
    fig.suptitle('Which training sources transfer to natural cgreGFP?',fontsize=21,x=.12,ha='left',y=.976,color='#202C3C')
    fig.text(.12,.936,'Same test sequences and same training budget in every column; filled dots identify training sources.',fontsize=11,color='#596779')
    fig.text(.12,.085,f'Used per fit: {protocol["n_train"]:,} train / {protocol["n_validation"]:,} validation / {protocol["test_counts"]["natural"]:,} test  ≈ 60% / 20% / 20%.',fontsize=11,color='#394352')
    fig.text(.12,.059,'Training-source ratios: single source 100%; pairs 50:50; triple 1:1:1. Validation has the same source ratios.',fontsize=10,color='#596779')
    fig.text(.12,.034,f'Bars: mean of {len(protocol["seeds"])} runs; small points: individual runs; error bars: run SD (not a confidence interval). Source-only checkpoint selection.',fontsize=10,color='#596779')
    save(fig,directory,f'ortholog_mixtures_{metric}')


def peaks_plot(summary,protocol,directory,metric='spearman'):
    fig,axes=plt.subplots(2,2,figsize=(14,10.5))
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.22,hspace=.5,wspace=.24)
    ylim=limits(summary,'peak_transfer',metric)
    labels=['Artificial\n→ Artificial','Artificial\n→ Natural','Natural\n→ Artificial','Natural\n→ Natural']
    for ax,model in zip(axes.flat,MODELS):
        lookup={r['condition']:r for r in summary if r['experiment']=='peak_transfer' and r['model']==model}
        names=[f'{source}_to_{target}' for source,target in DIRECTIONS]
        draw_bars(ax,[lookup[n][metric+'_mean'] for n in names],[lookup[n][metric+'_sd'] for n in names],COLORS[model],ylim,
                  [json.loads(lookup[n].get(metric+'_runs_json','[]')) for n in names])
        ax.set_title(LABELS[model],loc='left',color=COLORS[model],fontweight='bold',pad=10)
        ax.set_ylabel('Pooled test Spearman ρ' if metric=='spearman' else 'Pooled test Pearson r')
        ax.set_xticks(range(4),labels,fontsize=10)
    fig.suptitle('Transfer between natural cgreGFP and artificial peaks',fontsize=21,x=.09,ha='left',y=.966,color='#202C3C')
    fig.text(.09,.922,'Natural = the original cgreGFP landscape (including mutants). Artificial = the four approved peaks pooled.',fontsize=11,color='#596779')
    fig.text(.09,.137,f'Every fit: {protocol["n_train"]:,} training + {protocol["n_validation"]:,} validation sequences (3:1); validation comes only from training sources.',fontsize=10.5,color='#394352')
    fig.text(.09,.109,f'Test: natural {protocol["test_counts"]["natural"]:,} or artificial {protocol["test_counts"]["artificial"]:,} sequences — fixed 20% of each target landscape.',fontsize=10.5,color='#394352')
    fig.text(.09,.080,'Used train/validation/test ratio: 60/20/20 for natural targets; 58.8/19.6/21.5 for artificial targets (rounding).',fontsize=10,color='#596779')
    fig.text(.09,.053,'Artificial training/validation is proportional across peaks. Per-peak scores accompany these pooled scores.',fontsize=10,color='#596779')
    fig.text(.09,.028,f'Bars: mean of {len(protocol["seeds"])} runs; points: runs; error bars: SD. Artificial→Artificial tests variants within represented peaks, not an unseen peak.',fontsize=10,color='#596779')
    save(fig,directory,f'artificial_peak_transfer_{metric}')


def collect(directory):
    protocol=json.loads((directory/'protocol.json').read_text());code=scientific_sources()
    assert protocol['source_hashes']==code and digest(DATA)==protocol['dataset_sha256']
    table={r['record_id']:r for r in read_csv(DATA)}
    tests=read_csv(directory/'test_records.csv')
    assert digest(directory/'test_records.csv')==protocol['test_records_sha256']
    test_groups={g:{r['record_id'] for r in tests if r['target_group']==g} for g in ['natural','artificial']}
    rows=[];per_peak=[];fits=[];inputs={}
    for mix in MIXES:
        for seed in protocol['seeds']:
            subset=read_csv(directory/'subsets'/f'{mix}_seed{seed}.csv')
            validation={r['record_id'] for r in subset if r['split']=='validation'}
            for model in MODELS:
                out=directory/'fits'/mix/f'seed{seed}'/model
                report=json.loads((out/'metrics.json').read_text())
                assert report['source_hashes']==code and report['protocol_sha256']==digest(directory/'protocol.json')
                assert digest(out/'predictions.csv')==report['predictions_sha256']
                assert digest(out/'model.pt')==report['model_sha256']
                predictions=read_csv(out/'predictions.csv')
                expected={'validation':validation,'natural':test_groups['natural']}
                if mix in ['cgre','artificial']:
                    expected['artificial']=test_groups['artificial']
                assert set(r['partition'] for r in predictions)==set(expected)
                scores={}
                for group,expected_ids in expected.items():
                    chosen=[r for r in predictions if r['partition']==group]
                    assert len(chosen)==len(expected_ids) and {r['record_id'] for r in chosen}==expected_ids
                    for r in chosen:
                        original=table[r['record_id']]
                        assert r['gene']==original['gene'] and float(r['y_true'])==float(original['target_log10'])
                    score=metrics([float(r['y_true']) for r in chosen],[float(r['y_pred']) for r in chosen])
                    for k in SCORES:
                        if score[k] is None:
                            assert report['metrics'][group][k] is None
                        else:
                            np.testing.assert_allclose(score[k],report['metrics'][group][k],atol=1e-12,rtol=1e-10)
                    scores[group]=score
                    if group=='artificial':
                        for gene in PEAKS:
                            selected=[r for r in chosen if r['gene']==gene]
                            per_peak.append(dict(model=model,source=mix,seed=seed,target_peak=gene,
                                **metrics([float(r['y_true']) for r in selected],[float(r['y_pred']) for r in selected])))
                base=dict(model=model,seed=seed,fit_seconds=report['fit_seconds'],inference_seconds=report['inference_seconds'],
                          n_train=protocol['n_train'],n_validation=protocol['n_validation'])
                if mix!='artificial':
                    rows.append(dict(experiment='ortholog_mixtures',condition=mix,**base,**scores['natural']))
                if mix in ['cgre','artificial']:
                    for target in ['artificial','natural']:
                        rows.append(dict(experiment='peak_transfer',condition=f'{mix}_to_{target}',**base,**scores[target]))
                fits.append(dict(mix=mix,seed=seed,model=model,device=report['device'],gpu=report['gpu'],host=report['host'],
                                 fit_seconds=report['fit_seconds'],inference_seconds=report['inference_seconds'],elapsed_seconds=report['elapsed_seconds']))
                inputs[str(out.relative_to(directory)/'metrics.json')]=digest(out/'metrics.json')
    summary=[];grouped=defaultdict(list)
    for r in rows:
        grouped[r['experiment'],r['condition'],r['model']].append(r)
    for (experiment,condition,model),records in grouped.items():
        assert len(records)==len(protocol['seeds'])
        item=dict(experiment=experiment,condition=condition,model=model,n_runs=len(records),n_train=records[0]['n_train'],n_validation=records[0]['n_validation'],n_test=records[0]['n'])
        for metric in SCORES+['fit_seconds','inference_seconds']:
            values=[r[metric] for r in records]
            valid=all(v is not None for v in values)
            item[metric+'_mean']=float(np.mean(values)) if valid else None
            item[metric+'_sd']=float(np.std(values,ddof=1)) if valid else None
            item[metric+'_defined_runs']=sum(v is not None for v in values)
            item[metric+'_runs_json']=json.dumps(values)
        summary.append(item)
    write_csv(directory/'per_run_scores.csv',rows);write_csv(directory/'summary.csv',summary)
    write_csv(directory/'per_peak_scores.csv',per_peak);write_csv(directory/'runtimes.csv',fits)
    macro=[]
    for model in MODELS:
        for source in ['cgre','artificial']:
            safe_mean=lambda values:float(np.mean(values)) if all(v is not None for v in values) else None
            runs=[{m:safe_mean([r[m] for r in per_peak if r['model']==model and r['source']==source and r['seed']==seed]) for m in SCORES} for seed in protocol['seeds']]
            macro.append(dict(model=model,source=source,**{f'{m}_{stat}':float(fn([r[m] for r in runs])) if all(r[m] is not None for r in runs) else None for m in SCORES for stat,fn in [('mean',np.mean),('sd',lambda x:np.std(x,ddof=1))]}))
    write_csv(directory/'artificial_macro_scores.csv',macro)
    return protocol,summary,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=DEFAULT)
    a=p.parse_args();protocol,summary,inputs=collect(a.directory)
    for metric in ['spearman','pearson']:
        mixture_plot(summary,protocol,a.directory,metric);peaks_plot(summary,protocol,a.directory,metric)
    lines=['# GFP transfer experiment results','',
        'Three seeds (42, 43, 44), with fixed held-out targets. Values are mean ± run SD, not confidence intervals.',
        f'Each fit uses {protocol["n_train"]:,} training and {protocol["n_validation"]:,} source-validation records.',
        f'Natural target: {protocol["test_counts"]["natural"]:,} test sequences; artificial target: {protocol["test_counts"]["artificial"]:,}. Each is 20% of its target landscape.',
        'Natural mixtures have equal source contributions; artificial training is proportional across the four peak datasets.','',
        '| Experiment | Training / direction | Model | Test n | Spearman | Pearson | RMSE | Mean fit (min) |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda metric:'undefined' if r[metric+'_mean'] is None else f'{r[metric+"_mean"]:.3f} ± {r[metric+"_sd"]:.3f}'
        lines.append(f'| {r["experiment"]} | {r["condition"]} | {LABELS[r["model"]]} | {r["n_test"]} | {fmt("spearman")} | {fmt("pearson")} | {fmt("rmse")} | {r["fit_seconds_mean"]/60:.2f} |')
    lines.extend(['','See per_peak_scores.csv and artificial_macro_scores.csv to distinguish within-peak prediction from pooled rank effects.',
                  'A correlation is marked undefined if any repeated fit predicts a constant; individual runs and defined-run counts remain in the CSVs.',
                  'Artificial→Artificial is within represented peaks, not leave-one-peak-out generalization.',
                  'The natural cgre test set was used in earlier model comparisons; these are exploratory transfer comparisons, not a new blind test.',
                  'CPU/GPU hardware and loading costs are retained in runtimes.csv; timing excludes scheduler queue wait.'])
    (a.directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    report=dict(status='complete',fits=len(inputs),protocol_sha256=digest(a.directory/'protocol.json'),
                plot_script_sha256=digest(__file__),input_metrics=inputs,
                outputs={p.name:digest(p) for p in a.directory.iterdir() if p.is_file() and p.suffix in ['.png','.pdf','.svg','.csv']})
    (a.directory/'finalization.json').write_text(json.dumps(report,indent=2)+'\n')
    print('All 96 fits validated; source-mixture and artificial-transfer plots saved.')


if __name__=='__main__':
    main()
