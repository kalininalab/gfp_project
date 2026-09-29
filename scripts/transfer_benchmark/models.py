import numpy as np
import torch
from torch import nn

from scripts.aubin_model import ALPHABET,AubinModel
from scripts.baseline_models import MLPModel


def encode_aligned(rows,alignment):
    lookup={aa:i for i,aa in enumerate(ALPHABET)}
    x=np.full((len(rows),alignment['length']),-1,dtype=np.int64)
    for i,r in enumerate(rows):
        positions=alignment['mapping'][r['gene']]
        if len(r['sequence'])!=len(positions):
            raise ValueError('Sequence length differs from its WT map')
        x[i,positions]=[lookup[aa] for aa in r['sequence']]
    return torch.from_numpy(x)


class AlignedModel(nn.Module):
    """Same Aubin/MLP head; homologous columns shared, gaps encoded as all zeros."""
    def __init__(self,length,name):
        super().__init__()
        self.sequence_length=length
        self.architecture=name
        base=AubinModel(length,'1_10_1') if name=='aubin_1_10_1' else MLPModel(length,name)
        self.network=base.network

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=self.sequence_length:
            raise ValueError('Incorrect aligned input shape')
        valid=x>=0
        features=nn.functional.one_hot(x.clamp(min=0),len(ALPHABET)).float()*valid.unsqueeze(-1)
        features=nn.functional.pad(features,(0,0,0,1))
        return self.network(features.flatten(1)).squeeze(-1)
