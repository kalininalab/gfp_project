"""Experiment 3 fancy arm: Jannis OHE score-based acquisition versus Exp-2 random."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from scripts.active_learning.acquisition import acquisition_scores
from scripts.active_learning.run import infer
from scripts.aubin_model import encode_sequences
from scripts.paper_reproduction.run import read_csv, write_json
from .common import evaluate, fit_jannis, write_csv

FRACTIONS = (.1, .2, .4, .6, .8)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--device',default='cuda')
    p.add_argument('--data',type=Path,default=Path('data/processed/baseline_v1/sequences.csv'))
    p.add_argument('--directory',type=Path,default=Path('results/experiment_3_sampling')); a=p.parse_args()
    rows=[r for r in read_csv(a.data) if r['gene']=='cgreGFP']; seq=[r['sequence'] for r in rows]
    tokens=encode_sequences(seq,len(seq[0])).numpy(); x=np.eye(20,dtype=np.float32)[tokens]
    y=np.asarray([float(r['target_log10']) for r in rows],dtype=np.float32)
    development,test=train_test_split(np.arange(len(rows)),test_size=.2,random_state=a.seed,shuffle=True)
    order=np.random.default_rng(a.seed).permutation(development); train=order[:int(.1*len(rows))]; pool=order[int(.1*len(rows)):]
    out=a.directory/f'seed_{a.seed}'/'fancy'; out.mkdir(parents=True,exist_ok=True); summary=[]; queries=[]
    for step,fraction in enumerate(FRACTIONS):
        expected=int(fraction*len(rows))
        if len(train)!=expected: raise RuntimeError('Acquisition did not reach target train size')
        folder=out/f'train_{fraction:.1f}'; folder.mkdir(exist_ok=True)
        model,history=fit_jannis(x,y,train,a.seed+expected,a.device)
        pred,score=evaluate(model,x,y,test,a.device,32,f'Fancy acquisition — {fraction:.0%} train',folder/'test')
        write_csv(folder/'history.csv',history); write_csv(folder/'predictions.csv',[{'record_id':rows[i]['record_id'],'y_true':float(y[i]),'y_pred':float(v)} for i,v in zip(test,pred)])
        summary.append({'method':'fancy','seed':a.seed,'train_fraction':fraction,'n_train':len(train),'n_test':len(test),**score})
        if step+1 < len(FRACTIONS):
            labeled_hidden=infer(model,x,train,a.device,64,embeddings=True)
            pool_hidden=infer(model,x,pool,a.device,64,embeddings=True)
            _,variance=infer(model,x,pool,a.device,64,samples=25)
            # Match the student's executable default: alpha weights uncertainty
            # in its code, so distance receives 1-alpha = 0.62.
            score_values=acquisition_scores(labeled_hidden,pool_hidden,variance,alpha=.62)
            count=int(FRACTIONS[step+1]*len(rows))-len(train)
            selected=np.argsort(-score_values,kind='stable')[:count]
            chosen=pool[selected]
            queries.extend({'selected_after_fraction':fraction,'record_id':rows[i]['record_id'],
                            'acquisition_score':float(score_values[j]),'mc_variance':float(variance[j])}
                           for j,i in zip(selected,chosen))
            train=np.concatenate([train,chosen]); pool=np.delete(pool,selected)
    write_csv(out/'metrics.csv',summary); write_csv(out/'queries.csv',queries)
    write_json(out/'complete.json',{'seed':a.seed,'fractions':FRACTIONS,
        'score':'0.62 * scaled hidden-space distance + 0.38 * scaled MC-dropout variance',
        'selection':'global descending score; spectral clustering omitted because its dense affinity is quadratic at this pool size'})
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
