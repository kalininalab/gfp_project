"""Aubin's one-hot linear and 1–10–1 fluorescence models, implemented in PyTorch."""

import numpy as np
import torch
from torch import nn

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'


def encode_sequences(sequences, length):
    """Fixed alphabet; no vocabulary fitting or silent truncation."""
    lookup = {aa: i for i, aa in enumerate(ALPHABET)}
    encoded = []
    for sequence in sequences:
        if len(sequence) != length:
            raise ValueError(f'Expected {length} residues, got {len(sequence)}')
        try:
            encoded.append([lookup[aa] for aa in sequence])
        except KeyError as exc:
            raise ValueError(f'Unsupported amino acid: {exc.args[0]}') from exc
    return torch.tensor(encoded, dtype=torch.long).reshape(-1, length)


class AubinModel(nn.Module):
    """Dense one-hot model with optional nonlinear output subnetwork.

    A zero-valued terminal position preserves the original notebook's input
    width: it retained the stop position but removed all '*' feature columns.
    """

    def __init__(self, sequence_length, architecture='linear'):
        super().__init__()
        self.sequence_length = sequence_length
        self.architecture = architecture
        width = (sequence_length + 1) * len(ALPHABET)
        if architecture == 'linear':
            self.network = nn.Sequential(nn.Linear(width, 1))
        elif architecture == '1_10_1':
            self.network = nn.Sequential(nn.Linear(width, 1), nn.Linear(1, 10),
                                         nn.Sigmoid(), nn.Linear(10, 1))
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, encoded):
        if encoded.ndim != 2 or encoded.shape[1] != self.sequence_length:
            raise ValueError('Incorrect sequence tensor shape')
        features = nn.functional.one_hot(encoded, len(ALPHABET)).to(torch.float32)
        features = nn.functional.pad(features, (0, 0, 0, 1))
        return self.network(features.flatten(start_dim=1)).squeeze(-1)

    @torch.no_grad()
    def predict(self, sequences, batch_size=256):
        self.eval()
        encoded = encode_sequences(sequences, self.sequence_length)
        if len(encoded) == 0:
            return np.empty(0, dtype=np.float32)
        return torch.cat([self(encoded[i:i+batch_size])
                          for i in range(0, len(encoded), batch_size)]).numpy()


def load_model(path):
    """Load a local training artifact and its metadata, on CPU."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['alphabet'] != ALPHABET:
        raise ValueError('Checkpoint alphabet differs from this implementation')
    model = AubinModel(checkpoint['sequence_length'], checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
