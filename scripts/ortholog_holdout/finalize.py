"""Validate, summarize, and plot the four-protein leave-one-out benchmark."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices

from scripts.regression_metrics import metrics
from scripts.prepare_data import read_fasta
from .common import (ROOT, DATA, DEFAULT, GENES, MODELS, PUBLISHED, condition_id,
                     digest, read_csv, sources_for, write_csv)

COLORS={'aubin_1_10_1':'#0072B2','aubin_linear':'#E69F00','mlp_small':'#CC79A7','CNN_Jannis_OHE':'#56B4E9'}
LABELS={'aubin_1_10_1':'Aubin','aubin_linear':'Linear','mlp_small':'MLP','CNN_Jannis_OHE':'CNN on OHE'}
SHORT={'cgreGFP':'cgre','amacGFP':'amac','ppluGFP':'pplu','avGFP':'av'}
METRICS=[('spearman','Spearman ρ'),('pearson','Pearson r'),('r2','R²')]


def collect(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    assert digest(DATA)==protocol['dataset_sha256']
    table={row['record_id']:row for row in read_csv(DATA)}
    tests=read_csv(directory/'test_records.csv')
    assert digest(directory/'test_records.csv')==protocol['test_records_sha256']
    per_run=[]; inputs={}
    for target in GENES:
        expected={row['record_id'] for row in tests if row['target']==target}
        for condition in protocol['conditions'][target]:
            for seed in protocol['seeds']:
                for model in MODELS:
                    out=directory/'fits'/target/condition/f'seed{seed}'/model
                    report=json.loads((out/'metrics.json').read_text())
                    assert report['protocol_sha256']==digest(directory/'protocol.json')
                    assert report['source_hashes']==protocol['source_hashes']
                    assert digest(out/'predictions.csv')==report['predictions_sha256']
                    predictions=[row for row in read_csv(out/'predictions.csv') if row['partition']=='test']
                    assert {row['record_id'] for row in predictions}==expected
                    assert all(row['gene']==target for row in predictions)
                    assert all(float(row['y_true'])==float(table[row['record_id']]['target_log10']) for row in predictions)
                    score=metrics([float(row['y_true']) for row in predictions],
                                  [float(row['y_pred']) for row in predictions])
                    for name,_ in METRICS:
                        np.testing.assert_allclose(score[name],report['metrics']['test'][name],rtol=1e-10,atol=1e-12)
                    per_run.append(dict(target=target,condition=condition,seed=seed,model=model,
                        n_train=protocol['n_train'],n_validation=protocol['n_validation'],**score))
                    inputs[str((out/'metrics.json').relative_to(directory))]=digest(out/'metrics.json')
    grouped=defaultdict(list)
    for row in per_run: grouped[row['target'],row['condition'],row['model']].append(row)
    summary=[]
    for (target,condition,model),rows in grouped.items():
        item=dict(target=target,condition=condition,n_sources=condition.count('+')+1,model=model,
                  n_runs=len(rows),n_test=rows[0]['n'],n_train=rows[0]['n_train'],n_validation=rows[0]['n_validation'])
        for metric in ['spearman','pearson','r2','rmse']:
            values=[row[metric] for row in rows]
            item[metric+'_mean']=float(np.mean(values)); item[metric+'_sd']=float(np.std(values,ddof=1))
            item[metric+'_runs_json']=json.dumps(values)
        summary.append(item)
    return protocol,per_run,summary,inputs


def style(ax):
    ax.spines[['top','right']].set_visible(False); ax.spines[['left','bottom']].set_color('#AAB2BF')
    ax.yaxis.grid(True,color='#DDE2EA',linestyle=':',linewidth=.9); ax.set_axisbelow(True)
    ax.axhline(0,color='#737E8D',linewidth=.8); ax.tick_params(colors='#344052',labelsize=10)


def plot(summary,output,metric,label):
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':16,'axes.labelsize':13})
    fig,axes=plt.subplots(2,2,figsize=(18,10),sharey=True)
    width=.19; offsets=np.array([-1.5,-.5,.5,1.5])*width
    all_values=[row[metric+'_mean'] for row in summary]
    lower=-1 if metric=='r2' else max(-1,min(-.1,np.floor((min(all_values)-.08)*10)/10))
    for ax,target in zip(axes.flat,GENES):
        conditions=[condition_id(x) for x in sources_for(target)]
        for model,offset in zip(MODELS,offsets):
            lookup={row['condition']:row for row in summary if row['target']==target and row['model']==model}
            values=[lookup[c][metric+'_mean'] for c in conditions]; errors=[lookup[c][metric+'_sd'] for c in conditions]
            x=np.arange(len(conditions))+offset
            ax.bar(x,values,width=width*.92,color=COLORS[model],edgecolor='white',linewidth=.5,zorder=3,label=LABELS[model])
            ax.errorbar(x,values,yerr=errors,fmt='none',ecolor='#414B5A',capsize=2,linewidth=.9,zorder=4)
            for xpos,value in zip(x,values):
                if metric=='r2' and value < -1: ax.scatter(xpos,-.965,marker='v',s=24,color='#253244',zorder=6,clip_on=False)
        ax.set_title(f'Predict held-out {target}',color='#233044',pad=8)
        ax.set_xticks(range(len(conditions)),[' + '.join(SHORT[g] for g in c.split('+')) for c in conditions],rotation=27,ha='right')
        ax.set_ylim(lower,1); style(ax)
    axes[0,0].set_ylabel('Held-out test '+label); axes[1,0].set_ylabel('Held-out test '+label)
    handles=[plt.Rectangle((0,0),1,1,color=COLORS[m]) for m in MODELS]
    fig.legend(handles,[LABELS[m] for m in MODELS],loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.94),fontsize=12)
    fig.suptitle('Predicting each GFP protein from combinations of the other three',fontsize=22,color='#182535',y=.995)
    note='Fixed 14,709-train / 4,903-validation source budget; target protein is absent from fitting.'
    if metric=='r2': note+='  ▼ marks a mean R² below −1.'
    fig.text(.5,.018,note,ha='center',fontsize=11,color='#4D596A')
    fig.subplots_adjust(left=.065,right=.99,top=.87,bottom=.15,hspace=.34,wspace=.12)
    for suffix in ['png','pdf','svg']: fig.savefig(output/f'four_protein_holdout_{metric}.{suffix}',dpi=240,facecolor='white')
    plt.close(fig)


def report(protocol,summary,output):
    lines=['# Four-protein leave-one-out transfer','',
        '## Question','',
        'Can any one of cgreGFP, amacGFP, ppluGFP and avGFP be predicted using each possible non-empty combination of the other three proteins?','',
        '## Design','',
        f'For every target, all seven source combinations are tested. Every fit uses **{protocol["n_train"]:,} training** and **{protocol["n_validation"]:,} source-validation** records. Multiple sources contribute equally, rounded by at most one record. The target protein contributes no training or validation records. Its original 20% test split is frozen across all models, combinations and seeds.','',
        'Five seeds (42–46) vary source subsampling, initialization and training order. Checkpoints are selected only by source-validation MSE. Two cross-landscape duplicate sequences in amacGFP/avGFP are assigned to one source, and any source sequence identical to a target-test sequence is excluded.','',
        '## Reference-sequence identity','',
        'Identity is calculated from the reference proteins using global pairwise alignment with BLOSUM62, gap-open −10 and gap-extension −0.5, matching the transfer benchmark. The percentage is identical residues divided by total alignment columns, including columns containing a gap.','',
        '| | cgreGFP | amacGFP | ppluGFP | avGFP |',
        '|---|---:|---:|---:|---:|',
        '| **cgreGFP** | 100.00% | 43.62% | 19.76% | 42.15% |',
        '| **amacGFP** | 43.62% | 100.00% | 19.69% | 82.77% |',
        '| **ppluGFP** | 19.76% | 19.69% | 100.00% | 20.72% |',
        '| **avGFP** | 42.15% | 82.77% | 20.72% | 100.00% |','',
        'The machine-readable matrix is in `sequence_identity_matrix.csv`.','',
        '## Main figures','',
        '![Spearman](four_protein_holdout_spearman.png)','',
        '![Pearson](four_protein_holdout_pearson.png)','',
        '![R2](four_protein_holdout_r2.png)','',
        'R² uses the requested −1 to 1 display range; exact values remain in the table and CSV.','',
        '## Main findings','',
        '- Linear gives the strongest mean Spearman and Pearson result for every held-out protein. Its best Spearman values are 0.525 for cgreGFP, 0.450 for amacGFP, 0.195 for ppluGFP and 0.630 for avGFP.',
        '- ppluGFP is the hardest target: no source combination exceeds mean Spearman 0.195.',
        '- avGFP is the most transferable target by rank. Training Linear on cgreGFP + amacGFP gives mean Spearman 0.630 and Pearson 0.646.',
        '- Correlation and absolute calibration answer different questions. Many fits have useful positive rank correlation but negative R², meaning that they order variants better than chance while transferring the fluorescence scale poorly. The best mean R² values are −0.746 for cgreGFP, −1.613 for amacGFP, −1.305 for ppluGFP and 0.125 for avGFP.','',
        '## Results','',
        '| Target | Sources | Model | Test n | Spearman | Pearson | R² | RMSE |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for target in GENES:
        for condition in protocol['conditions'][target]:
            for model in MODELS:
                row=next(x for x in summary if x['target']==target and x['condition']==condition and x['model']==model)
                value=lambda m:f'{row[m+"_mean"]:.3f} ± {row[m+"_sd"]:.3f}'
                lines.append(f'| {target} | {condition} | {LABELS[model]} | {row["n_test"]} | {value("spearman")} | {value("pearson")} | {value("r2")} | {value("rmse")} |')
    lines += ['','Values are mean ± run SD across five seeds. `per_run_scores.csv` contains every repeat and `sample_counts.csv` preserves exact source allocations.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=PUBLISHED); args=parser.parse_args()
    protocol,per_run,summary,inputs=collect(args.directory); args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'per_run_scores.csv',per_run); write_csv(args.output/'summary.csv',summary)
    references={name:sequence.rstrip('*') for name,sequence in
                read_fasta(ROOT/'data/raw/fasta_sequences/protein_seqs.fa').items() if name in GENES}
    aligner=PairwiseAligner(mode='global',substitution_matrix=substitution_matrices.load('BLOSUM62'),
                            open_gap_score=-10.,extend_gap_score=-.5)
    def identity(first,second):
        if first==second:return 100.
        alignment=aligner.align(references[first],references[second])[0]
        a,b=alignment[0],alignment[1]
        return 100*sum(x==y for x,y in zip(a,b))/len(a)
    write_csv(args.output/'sequence_identity_matrix.csv',[
        {'sequence':name,**{other:f'{identity(name,other):.6f}' for other in GENES}} for name in GENES])
    for metric,label in METRICS: plot(summary,args.output,metric,label)
    report(protocol,summary,args.output)
    for name in ['test_records.csv','sample_counts.csv']:
        (args.output/name).write_bytes((args.directory/name).read_bytes())
    final=dict(status='complete',fits=len(inputs),protocol_sha256=digest(args.directory/'protocol.json'),input_metrics=inputs,
               outputs={p.name:digest(p) for p in args.output.iterdir()
                        if p.is_file() and p.name!='finalization.json'})
    (args.output/'finalization.json').write_text(json.dumps(final,indent=2)+'\n')
    print(f'Validated {len(inputs)} fits and wrote {args.output}')


if __name__=='__main__': main()
