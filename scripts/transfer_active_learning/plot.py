"""Validate transfer-AL trajectories and render AL-only and paired figures."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from scripts.transfer_benchmark.common import MODELS,MIXES,NATURAL,digest,read_csv,write_csv
from scripts.transfer_benchmark.plot import COLORS,LABELS,MIX_ORDER,DIRECTIONS,mixture_plot,peaks_plot,style_axis

def dark(color):return mcolors.to_hex(np.asarray(mcolors.to_rgb(color))*.68)

def collect(directory):
 protocol=json.loads((directory/'protocol.json').read_text());rows=[];inputs={}
 for mix in MIXES:
  for seed in protocol['seeds']:
   for model in MODELS:
    folder=directory/'fits'/mix/f'seed{seed}'/model;complete=json.loads((folder/'complete.json').read_text())
    if complete['protocol_sha256']!=digest(directory/'protocol.json') or complete['test_records_sha256']!=protocol['test_records_sha256'] or complete['metrics_sha256']!=digest(folder/'metrics.csv'):raise ValueError(f'Integrity failure: {mix}/{seed}/{model}')
    current=read_csv(folder/'metrics.csv');targets={'natural'}|({'artificial'} if mix in ('cgre','artificial') else set())
    expected={(r,t) for r in range(protocol['rounds']+1) for t in targets};actual={(int(r['round']),r['target']) for r in current}
    if actual!=expected or len(actual)!=len(current):raise ValueError(f'Incomplete rounds: {mix}/{seed}/{model}')
    rows.extend(current);inputs[str(folder/'metrics.csv')]=digest(folder/'metrics.csv')
 write_csv(directory/'per_round_scores.csv',rows)
 summary=[];grouped=defaultdict(list)
 for r in rows:
  if int(r['round'])!=protocol['rounds']:continue
  mix=r['mix'];target=r['target']
  if mix!='artificial' and target=='natural':key=('ortholog_mixtures',mix,r['model'])
  elif mix in ('cgre','artificial'):key=('peak_transfer',f'{mix}_to_{target}',r['model'])
  else:continue
  grouped[key].append(r)
 for (experiment,condition,model),records in grouped.items():
  if len(records)!=len(protocol['seeds']):raise ValueError('Missing seed in final AL summary')
  item={'experiment':experiment,'condition':condition,'model':model,'n_runs':len(records),'n_train':int(records[0]['n_train']),'n_validation':protocol['validation_count'],'n_test':int(records[0]['n'])}
  for metric in ('spearman','pearson','rmse','r2'):
   values=[float(r[metric]) for r in records];item[metric+'_mean']=float(np.mean(values));item[metric+'_sd']=float(np.std(values,ddof=1));item[metric+'_defined_runs']=len(values);item[metric+'_runs_json']=json.dumps(values)
  item['fit_seconds_mean']=float(np.mean([float(r['fit_seconds']) for r in records]));item['fit_seconds_sd']=float(np.std([float(r['fit_seconds']) for r in records],ddof=1));item['fit_seconds_defined_runs']=len(records);item['fit_seconds_runs_json']=json.dumps([float(r['fit_seconds']) for r in records]);item['inference_seconds_mean']=item['inference_seconds_sd']=0.;item['inference_seconds_defined_runs']=len(records);item['inference_seconds_runs_json']=json.dumps([0.]*len(records));summary.append(item)
 write_csv(directory/'summary.csv',summary);return protocol,summary,inputs

def paired(summary_non,summary_al,experiment,directory,metric):
 fig=plt.figure(figsize=(13,9.5 if experiment=='ortholog_mixtures' else 8.8));grid=fig.add_gridspec(2,2,left=.09,right=.985,top=.89,bottom=.08,hspace=.24,wspace=.18)
 conditions=MIX_ORDER if experiment=='ortholog_mixtures' else [f'{a}_to_{b}' for a,b in DIRECTIONS]
 labels=None if experiment=='ortholog_mixtures' else ['Artificial\n→ Artificial','Artificial\n→ Natural','Natural\n→ Artificial','Natural\n→ Natural']
 for panel,model in enumerate(MODELS):
  ax=fig.add_subplot(grid[panel//2,panel%2]);non={(r['condition']):r for r in summary_non if r['experiment']==experiment and r['model']==model};al={(r['condition']):r for r in summary_al if r['experiment']==experiment and r['model']==model};x=np.arange(len(conditions));width=.36
  nv=[float(non[c][metric+'_mean']) for c in conditions];av=[float(al[c][metric+'_mean']) for c in conditions]
  ax.bar(x-width/2,nv,width,yerr=[float(non[c][metric+'_sd']) for c in conditions],capsize=3,color=COLORS[model],label='Non-AL')
  ax.bar(x+width/2,av,width,yerr=[float(al[c][metric+'_sd']) for c in conditions],capsize=3,color=dark(COLORS[model]),label='AL')
  ax.set_title(LABELS[model],loc='center',color=COLORS[model],fontweight='bold',fontsize=18);ax.set_ylabel('Test Spearman ρ' if experiment=='ortholog_mixtures' else 'Pooled test Spearman ρ',fontsize=15);ax.set_xticks(x,labels or ['']*len(x),fontsize=12);style_axis(ax)
  if experiment=='ortholog_mixtures':
   ax.set_xticks(x,[m.replace('_',' + ') for m in conditions],rotation=32,ha='right')
  if panel==0:ax.legend(frameon=False,fontsize=13)
 title=('Transferability between GFP proteins' if experiment=='ortholog_mixtures' else 'Transferability between natural cgreGFP and artificial peaks')+' — non-AL vs AL'
 fig.suptitle(title,fontsize=23,y=.975);fig.savefig(directory/f'{"ortholog_mixtures" if experiment=="ortholog_mixtures" else "artificial_peak_transfer"}_{metric}_non_al_vs_al.png',dpi=240);fig.savefig(directory/f'{"ortholog_mixtures" if experiment=="ortholog_mixtures" else "artificial_peak_transfer"}_{metric}_non_al_vs_al.pdf');fig.savefig(directory/f'{"ortholog_mixtures" if experiment=="ortholog_mixtures" else "artificial_peak_transfer"}_{metric}_non_al_vs_al.svg');plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('results/transfer_al_cgreGFP_seed42_46'));p.add_argument('--non-al',type=Path,default=Path('results/transfer_cgreGFP_seed42_46'));a=p.parse_args();protocol,summary,inputs=collect(a.directory);non=read_csv(a.non_al/'summary.csv')
 mixture_plot(summary,protocol,a.directory,'spearman',' with active learning');peaks_plot(summary,protocol,a.directory,'spearman',' with active learning');paired(non,summary,'ortholog_mixtures',a.directory,'spearman');paired(non,summary,'peak_transfer',a.directory,'spearman')
 (a.directory/'RESULTS.md').write_text('# Transferability with active learning\n\nFive seeds, ten acquisition rounds, and the exact frozen test records used by the non-AL transfer experiment.\n\n![AL GFP transfer](ortholog_mixtures_spearman.png)\n\n![AL peak transfer](artificial_peak_transfer_spearman.png)\n\n![Paired GFP transfer](ortholog_mixtures_spearman_non_al_vs_al.png)\n\n![Paired peak transfer](artificial_peak_transfer_spearman_non_al_vs_al.png)\n')
 (a.directory/'finalization.json').write_text(json.dumps({'status':'complete','fits':len(inputs),'protocol_sha256':digest(a.directory/'protocol.json'),'test_records_sha256':protocol['test_records_sha256'],'inputs':inputs},indent=2)+'\n')
if __name__=='__main__':main()
