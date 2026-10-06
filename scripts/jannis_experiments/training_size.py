"""Experiment 2: Jannis OHE ablation over nested random training sizes."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from scripts.aubin_model import encode_sequences
from scripts.paper_reproduction.run import read_csv, write_json
from .common import evaluate, fit_jannis, write_csv

FRACTIONS = (.1, .2, .4, .6, .8)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--data',type=Path,default=Path('data/processed/baseline_v1/sequences.csv'))
    p.add_argument('--directory',type=Path,default=Path('results/experiment_2_training_size')); a=p.parse_args()
    rows=[r for r in read_csv(a.data) if r['gene']=='cgreGFP']; seq=[r['sequence'] for r in rows]
    tokens=encode_sequences(seq,len(seq[0])).numpy(); x=np.eye(20,dtype=np.float32)[tokens]
    y=np.asarray([float(r['target_log10']) for r in rows],dtype=np.float32)
    development,test=train_test_split(np.arange(len(rows)),test_size=.2,random_state=a.seed,shuffle=True)
    order=np.random.default_rng(a.seed).permutation(development); out=a.directory/f'seed_{a.seed}'; out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for fraction in FRACTIONS:
        count=int(fraction*len(rows)); train=order[:count]; folder=out/f'train_{fraction:.1f}'; folder.mkdir(exist_ok=True)
        model,history=fit_jannis(x,y,train,a.seed+count,a.device)
        pred,score=evaluate(model,x,y,test,a.device,32,f'Jannis OHE — {fraction:.0%} train',folder/'test')
        write_csv(folder/'history.csv',history); write_csv(folder/'predictions.csv',[{'record_id':rows[i]['record_id'],'y_true':float(y[i]),'y_pred':float(v)} for i,v in zip(test,pred)])
        summary.append({'seed':a.seed,'train_fraction':fraction,'n_train':len(train),'n_test':len(test),**score})
        torch_path=folder/'model.pt'; import torch; torch.save(model.state_dict(),torch_path)
    write_csv(out/'metrics.csv',summary); write_json(out/'complete.json',{'seed':a.seed,'fractions':FRACTIONS})
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
