"""Matched direct and shared-trunk hurdle neural predictors on one landscape."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from torch import nn

from scripts.aubin_model import ALPHABET
from scripts.esm_benchmark.models import CNN_Jannis
from scripts.hurdle_predictor.run import DATA, FEATURES, GENES
from scripts.regression_metrics import metrics

OUTPUT = Path("results/hurdle_neural_seed42_46")


class ESMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(640, 128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.ReLU())
        self.output_dim = 64

    def forward(self, x):
        return self.network(x)


class OHECNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = CNN_Jannis(20)
        self.gelu, self.conv_layers, self.ff_layers = base.gelu, base.conv_layers, base.ff_layers
        self.output_dim = 128

    def forward(self, tokens):
        x = nn.functional.one_hot(tokens, 20).float().transpose(1, 2)
        for block in self.conv_layers:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x + residual
        x = self.gelu(x)
        return self.ff_layers(torch.cat([x.mean(-1), x.max(dim=-1).values], dim=1))


class DirectModel(nn.Module):
    def __init__(self, architecture):
        super().__init__(); self.backbone = ESMBackbone() if architecture == "mlp_esm" else OHECNNBackbone()
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class HurdleModel(nn.Module):
    def __init__(self, architecture):
        super().__init__(); self.backbone = ESMBackbone() if architecture == "mlp_esm" else OHECNNBackbone()
        self.classifier = nn.Linear(self.backbone.output_dim, 1); self.regressor = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x):
        hidden = self.backbone(x)
        return self.classifier(hidden).squeeze(-1), self.regressor(hidden).squeeze(-1)


def load(gene, architecture):
    rows = [row for row in csv.DictReader(DATA.open()) if row["gene"] == gene]
    y = np.asarray([float(row["target_log10"]) for row in rows], dtype=np.float32)
    ids = {name: np.asarray([i for i, row in enumerate(rows) if row["split"] == name]) for name in ("train", "validation", "test")}
    if architecture == "mlp_esm":
        directory = FEATURES / f"{gene}_t30"; records = list(csv.DictReader((directory / "records.csv").open()))
        embeddings = np.load(directory / "mean_embeddings.npy", mmap_mode="r")
        index = {row["record_id"]: i for i, row in enumerate(records)}
        x = np.asarray(embeddings[[index[row["record_id"]] for row in rows]], dtype=np.float32)
        mean, scale = x[ids["train"]].mean(0), x[ids["train"]].std(0); scale[scale < 1e-8] = 1
        x = ((x - mean) / scale).astype(np.float32)
    else:
        lookup = {aa: i for i, aa in enumerate(ALPHABET)}
        x = np.asarray([[lookup[aa] for aa in row["sequence"]] for row in rows], dtype=np.int64)
    return rows, x, y, ids


@torch.inference_mode()
def predict(model, x, indexes, architecture, device, center, scale, floor, batch_size):
    model.eval(); direct, probability, conditional = [], [], []
    for start in range(0, len(indexes), batch_size):
        values = torch.from_numpy(x[indexes[start:start + batch_size]]).to(device)
        if architecture == "cnn_ohe": values = values.long()
        output = model(values)
        if isinstance(output, tuple):
            logits, regression = output; probability.append(torch.sigmoid(logits).cpu().numpy())
            conditional.append((regression * scale + center).cpu().numpy())
        else: direct.append((output * scale + center).cpu().numpy())
    if direct: return np.concatenate(direct)
    p, regression = np.concatenate(probability), np.maximum(np.concatenate(conditional), floor)
    return p, regression, (1 - p) * floor + p * regression


def fit(kind, architecture, x, y, ids, floor, seed, device, epochs_override=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = (DirectModel if kind == "direct" else HurdleModel)(architecture).to(device)
    center, scale = float(y[ids["train"]].mean()), float(y[ids["train"]].std()); scale = max(scale, 1e-8)
    normalized = (y - center) / scale; measurable = ~np.isclose(y, floor, atol=1e-10)
    batch_size, epochs, patience = ((128, 50, 8) if architecture == "mlp_esm" else (64, 60, 10))
    if epochs_override is not None: epochs, patience = epochs_override, epochs_override + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed); best_loss, stale, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(ids["train"]); total = 0
        for start in range(0, len(order), batch_size):
            indexes = order[start:start + batch_size]; xb = torch.from_numpy(x[indexes]).to(device)
            if architecture == "cnn_ohe": xb = xb.long()
            target = torch.tensor(normalized[indexes], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            if kind == "direct": loss = nn.functional.mse_loss(output, target)
            else:
                logits, regression = output
                labels = torch.tensor(measurable[indexes], dtype=torch.float32, device=device)
                classification = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                mask = labels.bool(); regression_loss = nn.functional.mse_loss(regression[mask], target[mask])
                loss = classification + regression_loss
            if not torch.isfinite(loss): raise ValueError("Non-finite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step(); total += loss.item() * len(indexes)
        validation = predict(model, x, ids["validation"], architecture, device, center, scale, floor, batch_size)
        values = validation if kind == "direct" else validation[2]
        validation_mse = float(np.mean((values - y[ids["validation"]]) ** 2))
        history.append({"epoch": epoch, "training_loss": total / len(order), "validation_mse": validation_mse})
        if validation_mse < best_loss:
            best_loss, stale = validation_mse, 0; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best_state)
    return model, center, scale, batch_size, history, best_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", choices=GENES, required=True); parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--architecture", choices=["mlp_esm", "cnn_ohe"], required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT); args = parser.parse_args()
    output = args.output / args.architecture / args.gene / f"seed{args.seed}"; output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists(): return
    rows, x, y, ids = load(args.gene, args.architecture); floor = float(y[ids["train"]].min())
    results, prediction_columns, histories = [], {}, {}
    for kind in ("direct", "hurdle"):
        model, center, scale, batch, history, best = fit(kind, args.architecture, x, y, ids, floor, args.seed, args.device, args.epochs)
        histories[kind] = history
        outcome = predict(model, x, ids["test"], args.architecture, args.device, center, scale, floor, batch)
        prediction = outcome if kind == "direct" else outcome[2]
        prediction_columns[kind] = prediction
        if kind == "hurdle": prediction_columns["p_measurable"], prediction_columns["conditional"] = outcome[0], outcome[1]
        results.append({"architecture": args.architecture, "gene": args.gene, "seed": args.seed, "model": kind, "best_validation_mse": best, **metrics(y[ids["test"]], prediction)})
        torch.save({"state_dict": model.state_dict(), "center": center, "scale": scale, "floor": floor}, output / f"{kind}.pt")
    labels = ~np.isclose(y[ids["test"]], floor, atol=1e-10); probability = prediction_columns["p_measurable"]
    classifier = {"roc_auc": roc_auc_score(labels, probability), "floor_pr_auc": average_precision_score(~labels, 1-probability),
                  "brier": brier_score_loss(labels, probability), "balanced_accuracy": balanced_accuracy_score(labels, probability >= .5),
                  "mcc": matthews_corrcoef(labels, probability >= .5), "floor_prevalence": float((~labels).mean())}
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    (output / "classifier.json").write_text(json.dumps(classifier, indent=2) + "\n")
    for kind, history in histories.items():
        with (output / f"history_{kind}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    with (output / "predictions.csv").open("w", newline="") as handle:
        fields = ["record_id", "y_true", "direct", "hurdle", "p_measurable", "conditional"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({"record_id": rows[index]["record_id"], "y_true": y[index], **{key: prediction_columns[key][position] for key in fields[2:]}} for position, index in enumerate(ids["test"]))
    (output / "complete.json").write_text(json.dumps({"architecture": args.architecture, "gene": args.gene, "seed": args.seed, "floor": floor, "frozen_test_used_for_fitting": False}, indent=2) + "\n")


if __name__ == "__main__": main()
