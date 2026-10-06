"""Plot the saved Experiment 1 metrics without importing training code."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(rows, ax=None):
    """Plot Pearson, Spearman and R² for summary.csv rows.

    Supply an existing three-axis array to embed the panel in another figure;
    otherwise a new publication-sized figure is created. The dotted Pearson
    line is the manuscript's reported CNN value, not a confidence interval.
    """
    if ax is None:
        figure, ax = plt.subplots(1,3,figsize=(11,4),sharey=True,constrained_layout=True)
    else:
        figure=ax[0].figure
    labels=list(dict.fromkeys(r['label'] for r in rows));x=np.arange(len(labels))
    colors=['#4477AA','#66CCEE','#228833','#CCBB44','#AA3377']
    for axis,metric,title in zip(ax,['pearson','spearman','r2'],['Pearson r','Spearman ρ','R²']):
        groups=[[float(r[metric]) for r in rows if r['label']==label] for label in labels]
        axis.bar(x,[np.mean(v) for v in groups],yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in groups],capsize=3,color=colors)
        axis.set_xticks(x,labels,rotation=35,ha='right')
        axis.set_title(title);axis.set_ylim(.8,1.0);axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.2)
    ax[0].axhline(.943,color='black',linestyle=':',linewidth=1.5,label='Manuscript CNN: 0.943')
    ax[0].legend(frameon=False,fontsize=8)
    ax[0].set_ylabel('Held-out test score')
    seeds=sorted({int(r['seed']) for r in rows})
    figure.suptitle(f'Experiment 1: cgreGFP random 80/20 split (seeds {seeds[0]}–{seeds[-1]})')
    return figure,ax


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('results/paper_reproduction_cgre_80_20'))
    args=parser.parse_args()
    with (args.directory/'summary.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    figure,_=plot_metrics(rows)
    for extension in ['png','pdf','svg']:
        figure.savefig(args.directory/f'model_comparison.{extension}',dpi=240)
    print(f'Saved Experiment 1 plots to {args.directory}')


if __name__=='__main__':main()
