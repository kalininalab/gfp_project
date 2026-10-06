"""Run one 10-round transfer active-learning trajectory."""
import argparse,csv,fcntl,json,random,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.active_learning.acquisition import acquisition_scores
from scripts.regression_metrics import metrics
from scripts.train_aubin import train,predict_encoded
from scripts.transfer_benchmark.common import DATA,MIXES,digest,read_csv,write_csv
from scripts.transfer_benchmark.models import AlignedModel,AlignedOHECNN,encode_aligned
from scripts.transfer_benchmark.prepare import allocate
from scripts.transfer_benchmark.run import train_cnn,predict_cnn

def atomic_json(path,value):
 tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

class TokenFeatures:
 def __init__(self,x):self.x=x
 def forward(self,model,index,device):return model(self.x[index].to(device))

def fit(model_name,x,y,ids,seed,device,protocol):
 if model_name=='CNN_Jannis_OHE':
  model=AlignedOHECNN(x.shape[1]).to(device);features=TokenFeatures(x)
  model,history,best,center,scale=train_cnn(model,features,y,ids,seed,device,None,protocol['cnn_settings'])
  predict=lambda index:predict_cnn(model,features,index,device,center,scale)
 else:
  model=AlignedModel(x.shape[1],model_name);s=protocol['onehot_settings']
  history,best=train(model,x[ids['train']],torch.tensor(y[ids['train']],dtype=torch.float32),x[ids['validation']],torch.tensor(y[ids['validation']],dtype=torch.float32),seed,s['epochs'],s['patience'],s['batch_size'],s['lr'])
  center,scale=0.,1.;predict=lambda index:predict_encoded(model,x[index]).numpy()
 return model,history,best,center,scale,predict

@torch.inference_mode()
def representations(model,x,index,device,batch=128):
 captured=[]
 layer=model.base.regressor if isinstance(model,AlignedOHECNN) else model.network[-1]
 hook=layer.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu()))
 model.eval()
 for start in range(0,len(index),batch):model(x[index[start:start+batch]].to(device))
 hook.remove();return torch.cat(captured).numpy()

@torch.inference_mode()
def uncertainty(model,x,index,device,samples=25,batch=128):
 if not isinstance(model,AlignedOHECNN):return np.zeros(len(index),dtype=np.float32)
 values=[];model.train()
 for start in range(0,len(index),batch):
  xb=x[index[start:start+batch]].to(device);pred=torch.stack([model(xb) for _ in range(samples)])
  values.append(pred.var(0).cpu().numpy())
 model.eval();return np.concatenate(values)

def ensemble_uncertainty(model_name,x,y,ids,seed,device,protocol,index,base_predict,members=5):
 """Prediction variance from a five-member ensemble for non-CNN models.

 Aubin and Linear auxiliary members use bootstrap-resampled labelled records;
 MLP members use the same records with independent initialization and batch
 order. The already fitted primary model is the first ensemble member.
 """
 if members < 2:raise ValueError('Ensemble uncertainty requires at least two members')
 predictions=[np.asarray(base_predict(index),dtype=np.float32)]
 for member in range(1,members):
  member_ids={k:np.asarray(v).copy() for k,v in ids.items()}
  member_seed=seed+100003*member
  if model_name in ('aubin_1_10_1','aubin_linear'):
   rng=np.random.default_rng(np.random.SeedSequence([seed,member,7919]));member_ids['train']=rng.choice(member_ids['train'],len(member_ids['train']),replace=True)
  _,_,_,_,_,member_predict=fit(model_name,x,y,member_ids,member_seed,device,protocol)
  predictions.append(np.asarray(member_predict(index),dtype=np.float32))
 return np.var(np.stack(predictions),axis=0,ddof=1).astype(np.float32)

def projected(values,seed,dimensions):
 if values.shape[1]<=dimensions:return values.astype(np.float32)
 rng=np.random.default_rng(seed);matrix=rng.normal(size=(values.shape[1],dimensions)).astype(np.float32)/np.sqrt(dimensions)
 return values.astype(np.float32)@matrix

def round_targets(final_budget,rounds=10):
 """Return strictly increasing sizes from 10% to the matched final budget."""
 values=np.rint(np.linspace(.1,1,rounds+1)*final_budget).astype(int)
 if values[-1]!=final_budget or np.any(np.diff(values)<=0):raise ValueError('Invalid AL budget schedule')
 return values

def save_plots(y_true,y_pred,title,stem):
 stem=Path(stem);bins=np.linspace(min(y_true.min(),y_pred.min()),max(y_true.max(),y_pred.max()),45)
 fig,ax=plt.subplots(figsize=(5.2,4.2),constrained_layout=True);ax.hist(y_true,bins=bins,density=True,histtype='step',lw=2,label='True',color='#0072B2');ax.hist(y_pred,bins=bins,density=True,histtype='step',lw=2,label='Predicted',color='#E69F00');ax.set(xlabel='log10 fluorescence',ylabel='Density',title=title);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);fig.savefig(stem.with_name(stem.name+'_distribution.png'),dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,4.6),constrained_layout=True);ax.scatter(y_true,y_pred,s=8,alpha=.22,lw=0,color='#0072B2',rasterized=True);lo,hi=min(y_true.min(),y_pred.min()),max(y_true.max(),y_pred.max());ax.plot([lo,hi],[lo,hi],'--',color='black',lw=1);ax.set(xlabel='True log10 fluorescence',ylabel='Predicted log10 fluorescence',title=title);ax.spines[['top','right']].set_visible(False);fig.savefig(stem.with_name(stem.name+'_scatter.png'),dpi=220);plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('results/transfer_al_cgreGFP_seed42_46'));p.add_argument('--mix',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--model',required=True);p.add_argument('--device',default='cpu');p.add_argument('--method',choices=('fancy','random'),default='fancy');p.add_argument('--ensemble-members',type=int,default=1);a=p.parse_args()
 protocol=json.loads((a.directory/'protocol.json').read_text());reference=Path(protocol['reference_directory'])
 if a.seed not in protocol['seeds'] or a.mix not in protocol['mixes'] or a.model not in protocol['models']:raise ValueError('Unplanned trajectory')
 out=a.directory/'fits'/a.mix/f'seed{a.seed}'/a.model;out.mkdir(parents=True,exist_ok=True)
 with (out/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (out/'complete.json').exists():print('Already complete',out);return
  rows_all=read_csv(DATA);table={r['record_id']:r for r in rows_all};genes=set(MIXES[a.mix]);tests=read_csv(a.directory/'test_records.csv')
  subset=read_csv(reference/'subsets'/f'{a.mix}_seed{a.seed}.csv');validation_rows=[table[r['record_id']] for r in subset if r['split']=='validation'];ordered=[table[r['record_id']] for r in subset if r['split']=='train']
  source_pool=[r for r in rows_all if r['gene'] in genes and r['split']=='train'];ordered_set={x['record_id'] for x in ordered}
  by_gene={gene:[r['record_id'] for r in ordered if r['gene']==gene] for gene in MIXES[a.mix]};initial_counts=allocate(round_targets(protocol['final_training_budget'],protocol['rounds'])[0],[len(by_gene[g]) for g in MIXES[a.mix]])
  initial_ids=[rid for gene,count in zip(MIXES[a.mix],initial_counts) for rid in by_gene[gene][:count]];initial_set=set(initial_ids)
  ordered_ids=initial_ids+[r['record_id'] for r in ordered if r['record_id'] not in initial_set]+[r['record_id'] for r in source_pool if r['record_id'] not in ordered_set]
  test_groups={'natural':[table[r['record_id']] for r in tests if r['target_group']=='natural']}
  if a.mix in ('cgre','artificial'):test_groups['artificial']=[table[r['record_id']] for r in tests if r['target_group']=='artificial']
  rows=source_pool+validation_rows+sum(test_groups.values(),[]);index={r['record_id']:i for i,r in enumerate(rows)};alignment=json.loads((reference/'alignment.json').read_text());x=encode_aligned(rows,alignment);y=np.asarray([float(r['target_log10']) for r in rows])
  validation=np.asarray([index[r['record_id']] for r in validation_rows]);pool=np.asarray([index[rid] for rid in ordered_ids]);targets=round_targets(protocol['final_training_budget'],protocol['rounds']);train=pool[:targets[0]];pool=pool[targets[0]:];summary=[];queries=[]
  test_ids={name:np.asarray([index[r['record_id']] for r in records]) for name,records in test_groups.items()}
  if set(train)&set(validation) or any(set(train)&set(v) for v in test_ids.values()):raise ValueError('Partition leakage')
  for round_number,target_size in enumerate(targets):
   started=time.perf_counter();ids={'train':train,'validation':validation};model,history,best,center,scale,predict=fit(a.model,x,y,ids,a.seed,a.device,protocol)
   write_csv(out/f'history_round_{round_number:02d}.csv',history);pred_rows=[]
   for target,indexes in test_ids.items():
    pred=predict(indexes);score=metrics(y[indexes],pred);summary.append({'mix':a.mix,'model':a.model,'seed':a.seed,'round':round_number,'target':target,'n_train':len(train),**score,'fit_seconds':time.perf_counter()-started});pred_rows.extend({'record_id':rows[i]['record_id'],'gene':rows[i]['gene'],'target':target,'y_true':float(y[i]),'y_pred':float(v)} for i,v in zip(indexes,pred));save_plots(y[indexes],pred,f'{a.model} · {a.mix} · round {round_number}',out/f'round_{round_number:02d}_{target}')
   write_csv(out/f'predictions_round_{round_number:02d}.csv',pred_rows);write_csv(out/'metrics.csv',summary)
   torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'round':round_number,'train_record_ids':[rows[i]['record_id'] for i in train]},out/f'model_round_{round_number:02d}.pt')
   if round_number==10:break
   need=targets[round_number+1]-len(train);rng=np.random.default_rng(np.random.SeedSequence([a.seed,round_number]))
   if a.method=='random':
    chosen=rng.choice(len(pool),need,replace=False);scores=np.full(len(pool),np.nan);variance=np.full(len(pool),np.nan)
   else:
    hidden_train=representations(model,x,train,a.device);hidden_pool=representations(model,x,pool,a.device);anchors=rng.choice(len(hidden_train),min(protocol['distance_anchors'],len(hidden_train)),replace=False);hp=projected(hidden_pool,a.seed*1000+round_number,protocol['projection_dimensions']);ht=projected(hidden_train[anchors],a.seed*1000+round_number,protocol['projection_dimensions'])
    variance=(ensemble_uncertainty(a.model,x,y,ids,a.seed,a.device,protocol,pool,predict,a.ensemble_members) if a.model!='CNN_Jannis_OHE' and a.ensemble_members>1 else uncertainty(model,x,pool,a.device,protocol['uncertainty_samples']))
    scores=acquisition_scores(ht,hp,variance,alpha=.62);chosen=np.argsort(-scores,kind='stable')[:need]
   selected=pool[chosen];queries.extend({'method':a.method,'selected_after_round':round_number,'first_fit_round':round_number+1,'record_id':rows[i]['record_id'],'score':None if a.method=='random' else float(scores[j]),'variance':None if a.method=='random' else float(variance[j])} for j,i in zip(chosen,selected));train=np.concatenate([train,selected]);pool=np.delete(pool,chosen);write_csv(out/'queries.csv',queries)
  atomic_json(out/'complete.json',{'protocol_sha256':digest(a.directory/'protocol.json'),'metrics_sha256':digest(out/'metrics.csv'),'test_records_sha256':digest(a.directory/'test_records.csv')})
if __name__=='__main__':main()
