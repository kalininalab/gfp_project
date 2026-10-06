"""Validate Experiments 2/3 and build multi-seed reports and figures."""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import write_csv

METRICS=(('r2','R²'),('pearson','Pearson r'),('spearman','Spearman ρ'),('kendall_tau','Kendall τ'))


def load(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def curve(rows, methods, title, output):
    fig,axes=plt.subplots(2,2,figsize=(9,7),sharex=True,constrained_layout=True); axes=axes.ravel()
    colors={'random':'#4477AA','fancy':'#EE6677'}
    for axis,(metric,label) in zip(axes,METRICS):
        for method in methods:
            subset=[r for r in rows if r['method']==method]
            fractions=sorted({float(r['train_fraction']) for r in subset}); mean=[];sd=[]
            for fraction in fractions:
                values=[float(r[metric]) for r in subset if float(r['train_fraction'])==fraction]
                mean.append(np.mean(values));sd.append(np.std(values,ddof=1))
            axis.errorbar(fractions,mean,yerr=sd,marker='o',capsize=3,label=method.capitalize(),color=colors[method])
        axis.set(ylabel=label);axis.grid(alpha=.2);axis.spines[['top','right']].set_visible(False)
    for axis in axes[2:]:axis.set_xlabel('Training fraction of full cgreGFP dataset')
    axes[0].legend(frameon=False);fig.suptitle(title)
    output=Path(output)
    for ext in ('png','pdf','svg'):fig.savefig(output.with_suffix('.'+ext),dpi=240)


def main():
    p=argparse.ArgumentParser();p.add_argument('--exp2',type=Path,default=Path('results/experiment_2_training_size'))
    p.add_argument('--exp3',type=Path,default=Path('results/experiment_3_sampling'));a=p.parse_args()
    random=[];fancy=[]
    for seed in range(42,47):
        if not (a.exp2/f'seed_{seed}'/'complete.json').exists():raise FileNotFoundError(f'Experiment 2 seed {seed} incomplete')
        if not (a.exp3/f'seed_{seed}'/'fancy'/'complete.json').exists():raise FileNotFoundError(f'Experiment 3 seed {seed} incomplete')
        for row in load(a.exp2/f'seed_{seed}'/'metrics.csv'):row['method']='random';random.append(row)
        fancy.extend(load(a.exp3/f'seed_{seed}'/'fancy'/'metrics.csv'))
    write_csv(a.exp2/'summary.csv',random);write_csv(a.exp3/'summary.csv',random+fancy)
    curve(random,['random'],'Experiment 2: Jannis OHE training-size ablation',a.exp2/'training_size_metrics')
    curve(random+fancy,['random','fancy'],'Experiment 2: acquisition versus random sampling',a.exp3/'sampling_comparison')
    (a.exp2/'RESULTS.md').write_text('# Experiment 2: Jannis OHE training-size ablation\n\nFive seeds; nested random 10/20/40/60/80% training subsets; common held-out 20% test within each seed. Error bars are SD.\n\n![Metrics](training_size_metrics.png)\n')
    (a.exp3/'RESULTS.md').write_text('# Experiment 2: acquisition versus random sampling\n\nFive matched seeds. Each fancy/random pair uses identical test IDs, initial 10% train, target sizes and training protocol. Error bars are SD.\n\n![Metrics](sampling_comparison.png)\n')

if __name__=='__main__':main()
