"""Reusable positional one-hot regressors and conventional ReLU MLPs."""

import pickle

import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch import nn

try:
    from .aubin_model import ALPHABET, AubinModel, encode_sequences
except ImportError:
    from aubin_model import ALPHABET, AubinModel, encode_sequences


HIDDEN_WIDTHS = {'mlp_small': (64,), 'mlp_deep': (128, 64, 32)}


def sparse_one_hot(encoded):
    """Same positional features as Aubin, stored sparsely for sklearn."""
    array = np.asarray(encoded)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError('Expected a matrix of integer amino-acid indices')
    n, length = array.shape
    if np.any(array < 0) or np.any(array >= len(ALPHABET)):
        raise ValueError('Amino-acid index outside the fixed alphabet')
    columns = (array + np.arange(length) * len(ALPHABET)).ravel()
    pointers = np.arange(n+1) * length
    return csr_matrix((np.ones(n*length, dtype=np.float64), columns, pointers),
                      shape=(n, (length+1)*len(ALPHABET)))


class MLPModel(AubinModel):
    """Conventional dense ReLU network, without the scalar Aubin bottleneck."""

    def __init__(self, sequence_length, architecture='mlp_small'):
        nn.Module.__init__(self)
        self.sequence_length = sequence_length
        self.architecture = architecture
        if architecture not in HIDDEN_WIDTHS:
            raise ValueError(f'Unknown MLP architecture: {architecture}')
        layers = []
        width = (sequence_length+1)*len(ALPHABET)
        for next_width in HIDDEN_WIDTHS[architecture]:
            layers.extend([nn.Linear(width, next_width), nn.ReLU()])
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


class SequenceRegressor:
    """Attach fixed sequence encoding to a fitted sklearn estimator."""

    def __init__(self, estimator, sequence_length, encoding='one_hot'):
        self.estimator = estimator
        self.sequence_length = sequence_length
        self.encoding = encoding

    def predict(self, sequences):
        if not sequences:
            return np.empty(0)
        encoded = encode_sequences(sequences, self.sequence_length).numpy()
        if self.encoding == 'one_hot':
            x = sparse_one_hot(encoded)
        elif self.encoding == 'hamming':
            x = encoded
        else:
            raise ValueError(f'Unknown encoding: {self.encoding}')
        return self.estimator.predict(x)


def load_baseline(path):
    """Load our local artifacts. Pickle files must come from a trusted source."""
    from pathlib import Path
    path = Path(path)
    if path.suffix == '.pt':
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if artifact['alphabet'] != ALPHABET:
            raise ValueError('Checkpoint alphabet mismatch')
        model = MLPModel(artifact['sequence_length'], artifact['architecture'])
        model.load_state_dict(artifact['state_dict'])
        model.eval()
    elif path.suffix == '.pkl':
        with path.open('rb') as stream:
            artifact = pickle.load(stream)
        if artifact['alphabet'] != ALPHABET:
            raise ValueError('Checkpoint alphabet mismatch')
        model = SequenceRegressor(artifact['estimator'], artifact['sequence_length'], artifact.get('encoding', 'one_hot'))
    else:
        raise ValueError('Expected .pt or .pkl artifact')
    return model, artifact
