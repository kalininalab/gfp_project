"""CNN architectures accept [batch, residues, embedding channels]."""
import torch
from torch import nn


class GlobalMaxPool(nn.Module):
    """Same global maximum/first-tie gradient as AdaptiveMaxPool1d(1).

    Explicit reduction avoids the nondeterministic CUDA adaptive-pool backward.
    """
    def forward(self,x):
        return x.max(dim=-1,keepdim=True).values


class CNN_old(nn.Module):
    """Original architecture, including its intentional two-residue length reduction.

    Training/data bugs are fixed by run.py; topology is preserved for a fair reference.
    """
    def __init__(self, dim=640):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(dim,64,5,padding=2), nn.ReLU(),
                                 nn.Conv1d(64,32,3,padding=1,dilation=2), nn.ReLU(),
                                 GlobalMaxPool(), nn.Flatten(),
                                 nn.Linear(32,64), nn.ReLU(), nn.Linear(64,1))

    def forward(self, x):
        return self.net(x.transpose(1,2)).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8,channels), nn.GELU(),
            nn.Conv1d(channels,channels,3,padding=dilation,dilation=dilation),
            nn.GroupNorm(8,channels), nn.GELU(), nn.Dropout(.1),
            nn.Conv1d(channels,channels,1))

    def forward(self, x):
        return x+self.net(x)


class CNN_new(nn.Module):
    """Compact residual CNN: 128 channels; dilations 1,2,4; mean+max pooling."""
    def __init__(self, dim=640):
        super().__init__()
        self.projection = nn.Conv1d(dim,128,1)
        self.blocks = nn.Sequential(*(ResidualBlock(128,d) for d in (1,2,4)))
        self.head = nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,1))

    def forward(self, x):
        x = self.blocks(self.projection(x.transpose(1,2)))
        return self.head(torch.cat([x.mean(-1),x.amax(-1)],dim=1)).squeeze(-1)


class CNN_Jannis(nn.Module):
    """Port of Jannis's ProtCNN, using NON_AL defaults (4 conv / 4 head layers).

    Source: master_thesis_jaca00001/src/model.py and src/configs.py.
    Preserve module names and operations to permit exact weight/forward comparison.
    """
    def __init__(self, dim=640):
        super().__init__()
        self.gelu = nn.GELU()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(dim if i==0 else 512,512,7 if i==0 else 5,
                padding=3 if i==0 else 6,dilation=1 if i==0 else 3,bias=False),
                nn.GroupNorm(8,512),nn.GELU(),nn.Dropout1d(.15)) for i in range(4)])
        self.ff_layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(a,b),nn.LayerNorm(b),nn.GELU(),nn.Dropout(.13249619898782194))
            for a,b in [(1024,512),(512,256),(256,128)]])
        self.regressor = nn.Linear(128,1)
        for m in self.modules():
            if isinstance(m,(nn.Conv1d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.transpose(1,2)
        for block in self.conv_layers:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x+residual
        x = self.gelu(x)
        # Jannis concatenates average first, maximum second.
        x = torch.cat([x.mean(-1),x.max(dim=-1).values],dim=1)
        return self.regressor(self.ff_layers(x)).squeeze(-1)


def vector_model(name, dim=640):
    if name == 'aubin_linear':
        layers = [nn.Linear(dim,1)]
    elif name == 'aubin_1_10_1':
        layers = [nn.Linear(dim,1),nn.Linear(1,10),nn.Sigmoid(),nn.Linear(10,1)]
    else:
        widths = [dim,64,1] if name=='mlp_small' else [dim,128,64,32,1]
        layers = []
        for i,(a,b) in enumerate(zip(widths,widths[1:])):
            layers.append(nn.Linear(a,b))
            if i<len(widths)-2:
                layers.append(nn.ReLU())
    model = nn.Sequential(*layers,nn.Flatten(0))
    for m in model.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    return model


def make_model(name, dim=640):
    return globals()[name](dim) if name.startswith('CNN_') else vector_model(name,dim)
