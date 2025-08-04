# model.py (обновлённый)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import random
import argparse
import pandas as pd
import os
import gc

# Argument parser
parser = argparse.ArgumentParser(description="Run CNN model with user-defined genes and seed")
parser.add_argument("--train", type=str, nargs='+', required=True)
parser.add_argument("--test", type=str, nargs='+', required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--train_points", type=int, required=True)
parser.add_argument("--test_points", type=int, required=True)
args = parser.parse_args()

train_genes = args.train
test_genes = args.test
SEED = args.seed
PATIENCE = 50

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, dim_embedding):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=dim_embedding, out_channels=64, kernel_size=5, padding=2)
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1, dilation=2)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1d_1(x))
        x = self.relu(self.conv1d_2(x))
        x = self.global_max_pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def load_embeddings(gene):
    if gene == "ppluGFP":
        path = f'/wibicomfs/STBS/anastasia/GFP/esm_embeddings_full/ppluGFP_ESM_embeddings.pt'
    elif gene == "amacGFP":
        path = f'/wibicomfs/STBS/anastasia/GFP/esm_embeddings_full/amacGFP_ESM_embeddings.pt'
    else:
        path = f'esm_embeddings/{gene}_ESM_embeddings.pt'
    return torch.load(path)

x_train_raw, y_train_raw = [], []
for gene in train_genes:
    emb = torch.stack(load_embeddings(gene)).squeeze(1)
    lab = np.load(f'esm_embeddings/{gene}_ESM_labels.npy')
    x_train_raw.append(emb)
    y_train_raw.append(lab)
    gc.collect()
    torch.cuda.empty_cache()

x_test_raw, y_test_raw = [], []
for gene in test_genes:
    emb = torch.stack(load_embeddings(gene)).squeeze(1)
    lab = np.load(f'esm_embeddings/{gene}_ESM_labels.npy')
    x_test_raw.append(emb)
    y_test_raw.append(lab)
    gc.collect()
    torch.cuda.empty_cache()

min_seq_len = min([t.shape[1] for t in x_train_raw + x_test_raw])
x_train_raw = [t[:, :min_seq_len, :] for t in x_train_raw]
x_test_raw = [t[:, :min_seq_len, :] for t in x_test_raw]

def split_and_sample(genes, x_raw, y_raw, total_n):
    if "cgreGFP" in genes and len(genes) > 1:
        gfp_idx = genes.index("cgreGFP")
        x1 = x_raw[gfp_idx][np.random.choice(len(x_raw[gfp_idx]), total_n // 2, replace=False)]
        y1 = y_raw[gfp_idx][np.random.choice(len(y_raw[gfp_idx]), total_n // 2, replace=False)]
        x2_all = torch.cat([x for i,x in enumerate(x_raw) if i != gfp_idx], dim=0)
        y2_all = np.concatenate([y for i,y in enumerate(y_raw) if i != gfp_idx])
        idx = np.random.choice(len(x2_all), total_n - len(x1), replace=False)
        return torch.cat([x1, x2_all[idx]]), np.concatenate([y1, y2_all[idx]])
    else:
        x_all = torch.cat(x_raw, dim=0)
        y_all = np.concatenate(y_raw, axis=0)
        idx = np.random.choice(len(x_all), total_n, replace=False)
        return x_all[idx], y_all[idx]

if '_'.join(train_genes) == '_'.join(test_genes):
    lengths = [x.shape[0] for x in x_train_raw]
    cum_lengths = np.cumsum([0] + lengths)
    total_size = sum(lengths)
    indices = np.random.permutation(total_size)
    train_idx = indices[:args.train_points]
    test_idx = indices[args.train_points:args.train_points + args.test_points]
    def extract(x_raw, y_raw, idxs):
        xs, ys = [], []
        for i in idxs:
            for j in range(len(cum_lengths)-1):
                if cum_lengths[j] <= i < cum_lengths[j+1]:
                    xs.append(x_raw[j][i - cum_lengths[j]])
                    ys.append(y_raw[j][i - cum_lengths[j]])
                    break
        return torch.stack(xs), np.array(ys)
    x_train, y_train = extract(x_train_raw, y_train_raw, train_idx)
    x_test, y_test = extract(x_train_raw, y_train_raw, test_idx)
else:
    x_train, y_train = split_and_sample(train_genes, x_train_raw, y_train_raw, args.train_points)
    x_test, y_test = split_and_sample(test_genes, x_test_raw, y_test_raw, args.test_points)

del x_train_raw, y_train_raw, x_test_raw, y_test_raw
gc.collect()
torch.cuda.empty_cache()

_, _, dim_embedding = x_train.shape
x_train = x_train.permute(0, 2, 1).to(device)
x_test = x_test.permute(0, 2, 1).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32)

model = ConvNet(dim_embedding).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_state, patience = None, 0
min_loss = float('inf')
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    model.eval()
    val_loss = sum(criterion(model(xb), yb).item() for xb, yb in test_loader) / len(test_loader)
    if val_loss < min_loss:
        min_loss = val_loss
        best_state = model.state_dict()
        patience = 0
    else:
        patience += 1
        if patience >= PATIENCE:
            break

model.load_state_dict(best_state)
torch.save(model, f"{args.train_points}_{'_'.join(train_genes)}_{'_'.join(test_genes)}_model_{SEED}.pt")

model.eval()
y_pred = torch.cat([model(xb).cpu() for xb, _ in test_loader])
rho = spearmanr(y_pred.flatten(), y_test.cpu().flatten())[0]

np.save(f"preds_{SEED}.npy", y_pred.numpy())

pd.DataFrame([{
    'train': '_'.join(train_genes),
    'test': '_'.join(test_genes),
    'seed': SEED,
    'rho': rho
}]).to_csv(args.output_file, mode='a', index=False, header=not os.path.exists(args.output_file))
