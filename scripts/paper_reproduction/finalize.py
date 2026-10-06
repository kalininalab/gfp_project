"""Validate and aggregate Experiment 1 over seeds 42–46."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scripts.regression_metrics import metrics
MODELS=('aubin_1_10_1','mlp_small','mlp_deep','CNN_Jannis_OHE','CNN_Jannis_ESM')
LABELS=dict(zip(MODELS,('Aubin 1–10–1','Small MLP','Deep MLP','Jannis OHE','Jannis ESM')))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('results/paper_reproduction_cgre_80_20'));p.add_argument('--seeds',type=int,nargs='+',default=list(range(42,47)));a=p.parse_args();rows=[]
 for seed in a.seeds:
  split_hash=None
  for model in MODELS:
   folder=a.directory/f'seed_{seed}'/model
   if not (folder/'complete.json').exists():raise FileNotFoundError(f'Incomplete: seed {seed}, {model}')
   complete=json.loads((folder/'complete.json').read_text())
   if digest(folder/'protocol.json')!=complete['protocol_sha256'] or digest(folder/'metrics.json')!=complete['metrics_sha256']:raise ValueError(f'Checksum mismatch: {seed}/{model}')
   current=digest(folder/'split.csv')
   if split_hash is not None and current!=split_hash:raise ValueError(f'Model split mismatch: seed {seed}')
   split_hash=current
   with (folder/'predictions.csv').open(newline='') as stream:pred=list(csv.DictReader(stream))
   score=metrics([float(r['y_true']) for r in pred],[float(r['y_pred']) for r in pred]);saved=json.loads((folder/'metrics.json').read_text())['metrics']
   for key,value in saved.items():
    if value is not None and not math.isclose(float(value),float(score[key]),rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f'Metric mismatch: {seed}/{model}/{key}')
   rows.append({'model':model,'label':LABELS[model],'seed':seed,**score})
 with (a.directory/'summary.csv').open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Experiment 1: base-model comparison without active learning','',f'Random 80/20 cgreGFP split; seeds {min(a.seeds)}–{max(a.seeds)}; 50 Adam epochs. Values are mean ± sample SD.','', '| Model | Pearson | Spearman | Kendall τ | R² | RMSE |','|---|---:|---:|---:|---:|---:|']
 for model in MODELS:
  group=[r for r in rows if r['model']==model];values=[]
  for key in ('pearson','spearman','kendall_tau','r2','rmse'):
   x=np.asarray([r[key] for r in group]);values.append(f'{x.mean():.3f} ± {x.std(ddof=1):.3f}')
  lines.append(f'| {LABELS[model]} | '+' | '.join(values)+' |')
 lines+=['','The held-out test is never used for fitting, checkpoint selection, or model selection.','','![Model comparison](model_comparison.png)',''];(a.directory/'RESULTS.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
