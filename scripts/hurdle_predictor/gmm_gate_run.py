"""Train a two-GMM-gated CNN mixture of low/high fluorescence experts."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.mixture import GaussianMixture
from torch import nn

from scripts.hurdle_predictor.neural_run import OHECNNBackbone, load
from scripts.hurdle_predictor.run import GENES
from scripts.mixture_analysis.analyze import two_component_diagnostics
from scripts.regression_metrics import metrics

OUTPUT = Path("results/gmm_gated_cnn_seed42_46")


class GMMGatedCNN(nn.Module):
    """Shared CNN with a low/high gate and one regression head per group."""

    def __init__(self):
        super().__init__()
        self.backbone = OHECNNBackbone()
        self.gate = nn.Linear(self.backbone.output_dim, 1)
        self.low = nn.Linear(self.backbone.output_dim, 1)
        self.high = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x):
        hidden = self.backbone(x)
        return self.gate(hidden).squeeze(-1), self.low(hidden).squeeze(-1), self.high(hidden).squeeze(-1)


@torch.inference_mode()
def predict(model, x, indexes, device, center, scale, threshold, batch_size=64):
    model.eval(); probabilities, low_values, high_values = [], [], []
    for start in range(0, len(indexes), batch_size):
        xb = torch.from_numpy(x[indexes[start:start + batch_size]]).long().to(device)
        logits, low, high = model(xb)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        low_values.append((low * scale + center).cpu().numpy())
        high_values.append((high * scale + center).cpu().numpy())
    probability = np.concatenate(probabilities)
    low_prediction = np.minimum(np.concatenate(low_values), threshold)
    high_prediction = np.maximum(np.concatenate(high_values), threshold)
    combined = (1 - probability) * low_prediction + probability * high_prediction
    return probability, low_prediction, high_prediction, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", choices=GENES, required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    destination = args.output / args.gene / f"seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists(): return

    rows, x, y, ids = load(args.gene, "cnn_ohe")
    # The split boundary is fitted to training labels only. Validation labels
    # select the neural checkpoint; test labels are untouched until evaluation.
    gmm = GaussianMixture(2, n_init=10, random_state=args.seed, reg_covar=1e-6).fit(y[ids["train"], None])
    diagnostics = two_component_diagnostics(gmm, y[ids["train"]])
    threshold = float(diagnostics["threshold"])
    high = y > threshold

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = GMMGatedCNN().to(args.device)
    center = float(y[ids["train"]].mean()); scale = max(float(y[ids["train"]].std()), 1e-8)
    normalized = (y - center) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(ids["train"]); total = 0.0
        for start in range(0, len(order), 64):
            indexes = order[start:start + 64]
            xb = torch.from_numpy(x[indexes]).long().to(args.device)
            target = torch.from_numpy(normalized[indexes]).float().to(args.device)
            label = torch.from_numpy(high[indexes].astype(np.float32)).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, low_output, high_output = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, label)
            low_mask, high_mask = ~label.bool(), label.bool()
            if low_mask.any(): loss = loss + nn.functional.mse_loss(low_output[low_mask], target[low_mask])
            if high_mask.any(): loss = loss + nn.functional.mse_loss(high_output[high_mask], target[high_mask])
            if not torch.isfinite(loss): raise ValueError("Non-finite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            total += loss.item() * len(indexes)
        validation = predict(model, x, ids["validation"], args.device, center, scale, threshold)[3]
        validation_mse = float(np.mean((validation - y[ids["validation"]]) ** 2))
        history.append({"epoch": epoch, "training_loss": total / len(order), "validation_mse": validation_mse})
        if validation_mse < best:
            best, stale = validation_mse, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10: break
    model.load_state_dict(best_state)
    probability, low_prediction, high_prediction, prediction = predict(
        model, x, ids["test"], args.device, center, scale, threshold
    )
    result = {"model": "two_gmm_gated_cnn", "gene": args.gene, "seed": args.seed,
              "threshold": threshold, "best_validation_mse": best, **metrics(y[ids["test"]], prediction)}
    with (destination / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result)); writer.writeheader(); writer.writerow(result)
    labels = high[ids["test"]]
    classifier = {"roc_auc": float(roc_auc_score(labels, probability)),
                  "pr_auc_high": float(average_precision_score(labels, probability)),
                  "brier": float(brier_score_loss(labels, probability)),
                  "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= .5)),
                  "mcc": float(matthews_corrcoef(labels, probability >= .5)),
                  "test_high_fraction": float(labels.mean())}
    (destination / "classifier.json").write_text(json.dumps(classifier, indent=2) + "\n")
    with (destination / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    with (destination / "predictions.csv").open("w", newline="") as handle:
        fields = ["record_id", "y_true", "is_high", "p_high", "low_expert", "high_expert", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({"record_id": rows[index]["record_id"], "y_true": y[index], "is_high": int(high[index]),
                          "p_high": probability[position], "low_expert": low_prediction[position],
                          "high_expert": high_prediction[position], "prediction": prediction[position]}
                         for position, index in enumerate(ids["test"]))
    (destination / "complete.json").write_text(json.dumps({
        "gene": args.gene, "seed": args.seed, "threshold": threshold,
        "threshold_source": "two-component Gaussian mixture fitted to training labels only",
        "gmm": {key: float(value) for key, value in diagnostics.items()},
        "frozen_test_used_for_fitting": False,
    }, indent=2) + "\n")


if __name__ == "__main__": main()
