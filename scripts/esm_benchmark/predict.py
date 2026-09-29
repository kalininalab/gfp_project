"""Apply a saved regressor to embeddings from the same ESM checkpoint."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .models import make_model
from .run import predict


def load_predictor(directory, device='cpu'):
    directory = Path(directory)
    report = json.loads((directory/'metrics.json').read_text())
    scaler = None
    if (directory/'feature_scaler.pkl').exists():
        with (directory/'feature_scaler.pkl').open('rb') as f:
            scaler = pickle.load(f)
    if (directory/'model.pt').exists():
        artifact = torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
        model = make_model(artifact['model'],artifact['input_dim']).to(device)
        model.load_state_dict(artifact['state_dict'])
        def fn(x):
            if scaler is not None:
                x = scaler.transform(x).astype(np.float32)
            return predict(model,x,np.arange(len(x)),device,64,artifact['target_mean'],artifact['target_std'])
    else:
        with (directory/'model.pkl').open('rb') as f:
            model = pickle.load(f)
        def fn(x):
            return model.predict(scaler.transform(x).astype(np.float32))
    return fn,report


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--features',required=True,help='mean [N,640] or full [N,L,640] .npy')
    p.add_argument('--output',required=True,help='Output .npy of predicted log10 fluorescence')
    p.add_argument('--device',default='cpu')
    a = p.parse_args()
    torch.set_num_threads(1)
    fn,_ = load_predictor(a.model_dir,a.device)
    np.save(a.output,fn(np.load(a.features,mmap_mode='r')))
