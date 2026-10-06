"""Train a frozen-source mixture once; evaluate only after source-only selection."""
import argparse
from collections import Counter
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.train_aubin import train,predict_encoded
from scripts.regression_metrics import metrics
from .common import ROOT,DEFAULT,DATA,GENES,PEAKS,MODELS,MIXES,digest,read_csv,write_csv,scientific_sources
from .models import AlignedModel,AlignedOHECNN,encode_aligned


def get_rows(directory,mix,seed,protocol):
    table={r['record_id']:r for r in read_csv(DATA) if r['gene'] in GENES}
    subset_path=directory/'subsets'/f'{mix}_seed{seed}.csv'
    if digest(subset_path)!=protocol['subset_hashes'][str(subset_path.relative_to(directory))]:
        raise ValueError('Changed training subset')
    subset=read_csv(subset_path)
    tests=read_csv(directory/'test_records.csv')
    if digest(directory/'test_records.csv')!=protocol['test_records_sha256']:
        raise ValueError('Changed test records')
    partitions={s:[table[r['record_id']] for r in subset if r['split']==s] for s in ['train','validation']}
    partitions['natural']=[table[r['record_id']] for r in tests if r['target_group']=='natural']
    if mix in ['cgre','artificial']:
        partitions['artificial']=[table[r['record_id']] for r in tests if r['target_group']=='artificial']
    for split in ['train','validation']:
        if len(partitions[split])!=protocol['n_train' if split=='train' else 'n_validation']:
            raise ValueError('Incorrect sample budget')
        if {r['gene'] for r in partitions[split]}!=set(MIXES[mix]):
            raise ValueError('Target-domain labels entered source-only fitting')
        assert all(r['split']==split for r in partitions[split])
    for group in set(partitions)-{'train','validation'}:
        assert all(r['split']=='test' for r in partitions[group])
    sequences=[r['sequence'] for records in partitions.values() for r in records]
    assert len(sequences)==len(set(sequences)), 'Training/validation/test sequence overlap'
    return partitions


class ResidueFeatures:
    """Only selected records are loaded; each protein retains its own full length.

    Mixed minibatches are forwarded in same-length gene groups, with one optimizer
    step on the sample-weighted combined loss. GroupNorm has no batch statistics.
    """
    def __init__(self,rows,directory):
        audit=json.loads((directory/'features_audit.json').read_text())
        self.features={};self.lookup={}
        for gene in GENES:
            needed=[i for i,r in enumerate(rows) if r['gene']==gene]
            if not needed:
                continue
            info=audit[gene];folder=Path(info['directory'])
            if digest(folder/'manifest.json')!=info['manifest_sha256']:
                raise ValueError('Changed embedding manifest')
            for name,fingerprint in info['files'].items():
                stat=(folder/name).stat()
                if stat.st_size!=fingerprint['size'] or stat.st_mtime_ns!=fingerprint['mtime_ns']:
                    raise ValueError('Embedding file changed after checksum audit; audit again')
            records=read_csv(folder/'records.csv');index={r['record_id']:i for i,r in enumerate(records)}
            ordered=sorted(needed,key=lambda i:index[rows[i]['record_id']])
            source_ids=np.array([index[rows[i]['record_id']] for i in ordered])
            assert all(rows[i]['sequence']==records[index[rows[i]['record_id']]]['sequence'] for i in ordered)
            raw=np.load(folder/'residue_embeddings.npy',mmap_mode='r')
            selected=np.empty((len(ordered),*raw.shape[1:]),dtype=np.float16)
            for start in range(0,len(ordered),128):
                block=raw[source_ids[start:start+128]]
                if not np.isfinite(block).all():
                    raise ValueError('Nonfinite features')
                selected[start:start+len(block)]=block
            self.features[gene]=selected
            lookup=np.full(len(rows),-1,dtype=np.int64);lookup[ordered]=np.arange(len(ordered))
            self.lookup[gene]=lookup
            print(f'Loaded {gene}: {selected.shape}, {selected.nbytes/1e9:.2f} GB',flush=True)
            del raw

    def forward(self,model,ids,device):
        pieces=[];locations=[]
        for gene,features in self.features.items():
            local=self.lookup[gene][ids]
            positions=np.flatnonzero(local>=0)
            if not len(positions):
                continue
            x=torch.from_numpy(features[local[positions]].astype(np.float32)).to(device)
            pieces.append(model(x));locations.extend(positions.tolist())
        return torch.cat(pieces)[torch.tensor(np.argsort(locations),device=device)]


@torch.inference_mode()
def predict_cnn(model,features,ids,device,center,scale,batch=64):
    model.eval()
    return np.concatenate([features.forward(model,ids[i:i+batch],device).float().cpu().numpy()*scale+center
                           for i in range(0,len(ids),batch)])


def train_cnn(model,features,y,ids,seed,device,out,settings):
    center=float(y[ids['train']].mean());scale=max(float(y[ids['train']].std()),1e-8)
    targets=(y-center)/scale
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    rng=np.random.default_rng(seed);history=[];best=float('inf');best_state=None;best_epoch=0;stale=0
    for epoch in range(1,settings['epochs']+1):
        model.train();order=rng.permutation(ids['train']);total=0.
        for start in range(0,len(order),settings['batch_size']):
            ix=order[start:start+settings['batch_size']]
            target=torch.tensor(targets[ix],dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True)
            pred=features.forward(model,ix,device)
            loss=torch.nn.functional.mse_loss(pred,target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.item())*len(ix)
        pred=predict_cnn(model,features,ids['validation'],device,center,scale)
        val=float(np.mean((pred-y[ids['validation']])**2))
        history.append(dict(epoch=epoch,train_mse=total/len(order)*scale**2,validation_mse=val))
        print(f'CNN_Jannis epoch {epoch}: validation MSE {val:.6f}',flush=True)
        if val<best:
            best=val;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=settings['patience']:
                break
    model.load_state_dict(best_state)
    return model,history,best_epoch,center,scale


def run(a):
    start=time.perf_counter();directory=a.directory;out=directory/'fits'/a.mix/f'seed{a.seed}'/a.model
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=json.loads((directory/'protocol.json').read_text())
        code=scientific_sources()
        if code!=protocol['source_hashes'] or digest(DATA)!=protocol['dataset_sha256']:
            raise ValueError('Source code/data changed since protocol was frozen')
        if a.seed not in protocol['seeds'] or a.mix not in protocol['mixes']:
            raise ValueError('Unplanned experiment')
        protocol_hash=digest(directory/'protocol.json')
        if (out/'metrics.json').exists():
            old=json.loads((out/'metrics.json').read_text())
            assert old['protocol_sha256']==protocol_hash
            print('Verified completed fit:',out,flush=True);return
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        partitions=get_rows(directory,a.mix,a.seed,protocol)
        rows=[];ids={}
        for name,records in partitions.items():
            ids[name]=np.arange(len(rows),len(rows)+len(records));rows.extend(records)
        y=np.array([float(r['target_log10']) for r in rows])
        alignment=json.loads((directory/'alignment.json').read_text())
        if digest(directory/'alignment.json')!=protocol['alignment_sha256']:
            raise ValueError('Changed alignment')
        cnn=a.model=='CNN_Jannis_OHE'
        x=encode_aligned(rows,alignment)
        if cnn:
            if a.device!='cuda':
                raise ValueError('Jannis OHE benchmark requires a GPU')
            class TokenFeatures:
                def forward(self,model,index,device):return model(x[index].to(device))
            features=TokenFeatures()
            torch.cuda.synchronize()
        fit_start=time.perf_counter()
        with threadpool_limits(limits=1):
            if cnn:
                model=AlignedOHECNN(alignment['length']).to(a.device)
                model,history,best,center,scale=train_cnn(model,features,y,ids,a.seed,a.device,out,protocol['cnn_settings'])
                fn=lambda index:predict_cnn(model,features,index,a.device,center,scale)
            else:
                model=AlignedModel(alignment['length'],a.model)
                s=protocol['onehot_settings']
                history,best=train(model,x[ids['train']],torch.tensor(y[ids['train']],dtype=torch.float32),
                    x[ids['validation']],torch.tensor(y[ids['validation']],dtype=torch.float32),a.seed,s['epochs'],s['patience'],s['batch_size'],s['lr'])
                center,scale=0.,1.
                fn=lambda index:predict_encoded(model,x[index]).numpy()
            artifact=dict(model=a.model,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                          alignment=alignment,center=center,scale=scale,seed=a.seed,mix=a.mix,best_epoch=best,protocol_sha256=protocol_hash)
            torch.save(artifact,out/'model.pt');write_csv(out/'history.csv',history)
            # A separate model verifies that the reusable checkpoint restores the same predictor.
            restored=AlignedOHECNN(alignment['length']).to(a.device) if cnn else AlignedModel(alignment['length'],a.model)
            restored.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
            check=ids['validation'][:32]
            saved=(predict_cnn(restored,features,check,a.device,center,scale) if cnn else predict_encoded(restored,x[check]).numpy())
            np.testing.assert_array_equal(saved,fn(check))
            del restored
            if cnn:
                torch.cuda.synchronize()
            fit_seconds=time.perf_counter()-fit_start;infer_start=time.perf_counter()
            predictions={group:fn(index) for group,index in ids.items() if group!='train'}
            if cnn:
                torch.cuda.synchronize()
            inference_seconds=time.perf_counter()-infer_start
        result={group:metrics(y[ids[group]],pred) for group,pred in predictions.items()}
        per_peak={}
        if 'artificial' in predictions:
            for gene in PEAKS:
                mask=np.array([rows[i]['gene']==gene for i in ids['artificial']])
                per_peak[gene]=metrics(y[ids['artificial']][mask],predictions['artificial'][mask])
        write_csv(out/'predictions.csv',[dict(record_id=rows[i]['record_id'],gene=rows[i]['gene'],partition=group,
                    y_true=float(y[i]),y_pred=float(pred)) for group,values in predictions.items() for i,pred in zip(ids[group],values)])
        if scientific_sources()!=code:
            raise ValueError('Scientific code was modified during this run')
        report=dict(model=a.model,mix=a.mix,seed=a.seed,metrics=result,per_peak=per_peak,
            counts={s:dict(Counter(r['gene'] for r in records)) for s,records in partitions.items()},
            epochs_run=len(history),best_epoch=best,fit_seconds=fit_seconds,inference_seconds=inference_seconds,
            elapsed_seconds=time.perf_counter()-start,device=a.device if cnn else 'cpu',
            gpu=torch.cuda.get_device_name() if cnn else None,host=platform.node(),parameters=sum(p.numel() for p in model.parameters()),
            target_center=center,target_scale=scale,protocol_sha256=protocol_hash,
            features_audit_sha256=None,
            predictions_sha256=digest(out/'predictions.csv'),model_sha256=digest(out/'model.pt'),source_hashes=code,
            versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn']})
        temp=out/'metrics.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(out/'metrics.json')
        print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=DEFAULT)
    p.add_argument('--mix',choices=MIXES,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(p.parse_args())


if __name__=='__main__':
    main()
