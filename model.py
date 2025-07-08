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

# Argument parser
parser = argparse.ArgumentParser(description="Run CNN model with user-defined genes and seed")
parser.add_argument("--train", type=str, nargs='+', required=True, help="Training genes (space-separated)")
parser.add_argument("--test", type=str, nargs='+', required=True, help="Testing genes (space-separated)")
parser.add_argument("--seed", type=int, required=True, help="Random seed value")
parser.add_argument("--output_file", type=str, required=True, help="Output CSV file to append results")
parser.add_argument("--train_points", type=int, required=True, help="Number of train points to use")
parser.add_argument("--test_points", type=int, required=True, help="Number of test points to use")
args = parser.parse_args()

# Variables
train_genes = args.train
test_genes = args.test
SEED = args.seed
PATIENCE = 50

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
class ConvNet(nn.Module):
    def __init__(self, dim_embedding):
        super(ConvNet, self).__init__()
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
        x = self.fc2(x)
        return x

# Load train data
x_train_raw, y_train_raw = [], []
for gene in train_genes:
    embeddings = torch.load(f'esm_embeddings/{gene}_ESM_embeddings.pt')
    embeddings = torch.stack(embeddings).squeeze(1)
    labels = np.load(f'esm_embeddings/{gene}_ESM_labels.npy')
    x_train_raw.append(embeddings)
    y_train_raw.append(labels)

# Load test data
x_test_raw, y_test_raw = [], []
for gene in test_genes:
    embeddings = torch.load(f'esm_embeddings/{gene}_ESM_embeddings.pt')
    embeddings = torch.stack(embeddings).squeeze(1)
    labels = np.load(f'esm_embeddings/{gene}_ESM_labels.npy')
    x_test_raw.append(embeddings)
    y_test_raw.append(labels)

# Crop all to the same global min sequence length
all_seq_lens = [t.shape[1] for t in x_train_raw + x_test_raw]
min_seq_len = min(all_seq_lens)
x_train_raw = [t[:, :min_seq_len, :] for t in x_train_raw]
x_test_raw = [t[:, :min_seq_len, :] for t in x_test_raw]

# Custom sample function
def split_and_sample(genes, x_raw, y_raw, total_n):
    if "cgreGFP" in genes and len(genes) > 1:
        gfp_idx = genes.index("cgreGFP")
        x_gfp = x_raw[gfp_idx]
        y_gfp = y_raw[gfp_idx]

        x_others = [x for i, x in enumerate(x_raw) if i != gfp_idx]
        y_others = [y for i, y in enumerate(y_raw) if i != gfp_idx]

        n_half = total_n // 2

        gfp_indices = np.random.choice(len(x_gfp), size=n_half, replace=False)
        x1 = x_gfp[gfp_indices]
        y1 = y_gfp[gfp_indices]

        x2_all = torch.cat(x_others, dim=0)
        y2_all = np.concatenate(y_others, axis=0)
        x2_indices = np.random.choice(len(x2_all), size=total_n - n_half, replace=False)
        x2 = x2_all[x2_indices]
        y2 = y2_all[x2_indices]

        x_comb = torch.cat([x1, x2], dim=0)
        y_comb = np.concatenate([y1, y2], axis=0)
        return x_comb, y_comb
    else:
        x_all = torch.cat(x_raw, dim=0)
        y_all = np.concatenate(y_raw, axis=0)
        indices = np.random.choice(len(x_all), size=total_n, replace=False)
        return x_all[indices], y_all[indices]

# Subsampling
if '_'.join(train_genes) == '_'.join(test_genes):
    # Same set → split without overlap
    all_x = torch.cat(x_train_raw, dim=0)
    all_y = np.concatenate(y_train_raw, axis=0)
    total_indices = np.random.permutation(len(all_x))
    n_train = args.train_points
    n_test = args.test_points
    assert n_train + n_test <= len(total_indices), "Not enough data to split!"
    train_indices = total_indices[:n_train]
    test_indices = total_indices[n_train:n_train + n_test]
    x_train = all_x[train_indices]
    y_train = all_y[train_indices]
    x_test = all_x[test_indices]
    y_test = all_y[test_indices]

else:
    x_train, y_train = split_and_sample(train_genes, x_train_raw, y_train_raw, args.train_points)
    x_test, y_test = split_and_sample(test_genes, x_test_raw, y_test_raw, args.test_points)

# Prepare datasets
_, _, dim_embedding = x_train.shape
x_train = torch.tensor(x_train, dtype=torch.float32).permute(0, 2, 1).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).permute(0, 2, 1).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

# Init model
model = ConvNet(dim_embedding).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
patience_counter, min_val_loss = 0, float('inf')
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(test_loader)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break

# Evaluate
model.load_state_dict(best_model_state)
test_set_predictions = []
model.eval()
with torch.no_grad():
    for inputs in test_loader:
        outputs = model(inputs[0])
        test_set_predictions.append(outputs.cpu().numpy())
test_set_predictions = np.concatenate(test_set_predictions, axis=0)
rho = spearmanr(test_set_predictions.flatten(), y_test.cpu().numpy().flatten())[0]

# Save result
result_row = pd.DataFrame([{
    'train': '_'.join(train_genes),
    'test': '_'.join(test_genes),
    'seed': SEED,
    'rho': rho
}])
file_exists = os.path.isfile(args.output_file)
result_row.to_csv(args.output_file, mode='a', index=False, header=not file_exists)