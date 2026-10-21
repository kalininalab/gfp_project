"""Compare direct and soft-hurdle models on frozen ProteinBERT embeddings."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn

from scripts.hurdle_predictor.run import DATA, GENES
from scripts.regression_metrics import metrics

FEATURES = Path("/data/users/akolchina/gfp_project_artifacts/proteinbert_embeddings")
OUTPUT = Path("results/proteinbert_models_seed42_46")


def load(gene):
    rows = [row for row in csv.DictReader(DATA.open()) if row["gene"] == gene]
    folder = FEATURES / gene
    records = list(csv.DictReader((folder / "records.csv").open()))
    index = {row["record_id"]: i for i, row in enumerate(records)}
    if set(index) != {row["record_id"] for row in rows}:
        raise ValueError(f"ProteinBERT record mismatch for {gene}")
    stored = np.load(folder / "global_embeddings.npy", mmap_mode="r")
    x = np.asarray(stored[[index[row["record_id"]] for row in rows]], dtype=np.float32)
    y = np.asarray([float(row["target_log10"]) for row in rows], dtype=np.float32)
    ids = {name: np.asarray([i for i, row in enumerate(rows) if row["split"] == name]) for name in ("train", "validation", "test")}
    mean, scale = x[ids["train"]].mean(0), x[ids["train"]].std(0)
    scale[scale < 1e-8] = 1
    return rows, ((x - mean) / scale).astype(np.float32), y, ids


class DirectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


class HurdleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.15), nn.Linear(128, 64), nn.ReLU())
        self.classifier, self.regressor = nn.Linear(64, 1), nn.Linear(64, 1)

    def forward(self, x):
        hidden = self.backbone(x)
        return self.classifier(hidden).squeeze(-1), self.regressor(hidden).squeeze(-1)


@torch.inference_mode()
def mlp_predict(model, x, indexes, device, center, scale, floor, batch=128):
    model.eval(); outputs = []
    for start in range(0, len(indexes), batch):
        output = model(torch.from_numpy(x[indexes[start:start + batch]]).to(device))
        if isinstance(output, tuple):
            outputs.append((torch.sigmoid(output[0]).cpu().numpy(), (output[1] * scale + center).cpu().numpy()))
        else:
            outputs.append((output * scale + center).cpu().numpy())
    if outputs and isinstance(outputs[0], tuple):
        p = np.concatenate([item[0] for item in outputs]); conditional = np.maximum(np.concatenate([item[1] for item in outputs]), floor)
        return p, conditional, (1 - p) * floor + p * conditional
    return np.concatenate(outputs)


def fit_mlp(kind, x, y, ids, floor, seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = (DirectMLP if kind == "direct" else HurdleMLP)().to(device)
    center, scale = float(y[ids["train"]].mean()), max(float(y[ids["train"]].std()), 1e-8)
    normalized = (y - center) / scale; measurable = ~np.isclose(y, floor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed); best, best_state, stale = float("inf"), None, 0
    for _epoch in range(50):
        model.train()
        order = rng.permutation(ids["train"])
        for start in range(0, len(order), 128):
            indexes = order[start:start + 128]
            xb = torch.from_numpy(x[indexes]).to(device); target = torch.from_numpy(normalized[indexes]).to(device)
            optimizer.zero_grad(set_to_none=True); output = model(xb)
            if kind == "direct": loss = nn.functional.mse_loss(output, target)
            else:
                labels = torch.from_numpy(measurable[indexes].astype(np.float32)).to(device)
                mask = labels.bool(); loss = nn.functional.binary_cross_entropy_with_logits(output[0], labels)
                loss = loss + nn.functional.mse_loss(output[1][mask], target[mask])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        predicted = mlp_predict(model, x, ids["validation"], device, center, scale, floor)
        if kind == "hurdle": predicted = predicted[2]
        score = float(np.mean((predicted - y[ids["validation"]]) ** 2))
        if score < best:
            best, stale = score, 0; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8: break
    model.load_state_dict(best_state)
    return model, center, scale


def run_xgboost(x, y, ids, floor, seed):
    import xgboost as xgb
    common = dict(n_estimators=2000, learning_rate=.03, max_depth=6, subsample=.8, colsample_bytree=.8,
                  tree_method="hist", n_jobs=4, random_state=seed, early_stopping_rounds=50)
    measurable = ~np.isclose(y, floor)
    direct = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common).fit(
        x[ids["train"]], y[ids["train"]], eval_set=[(x[ids["validation"]], y[ids["validation"]])], verbose=False)
    classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common).fit(
        x[ids["train"]], measurable[ids["train"]].astype(np.int8),
        eval_set=[(x[ids["validation"]], measurable[ids["validation"]].astype(np.int8))], verbose=False)
    active_train = ids["train"][measurable[ids["train"]]]
    regressor = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common).fit(
        x[active_train], y[active_train], eval_set=[(x[ids["validation"]][measurable[ids["validation"]]], y[ids["validation"]][measurable[ids["validation"]]])], verbose=False)
    direct_prediction = direct.predict(x[ids["test"]]); probability = classifier.predict_proba(x[ids["test"]])[:, 1]
    conditional = np.maximum(regressor.predict(x[ids["test"]]), floor)
    return direct_prediction, probability, conditional, (1 - probability) * floor + probability * conditional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", choices=GENES, required=True); parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--model", choices=["xgboost", "mlp"], required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=OUTPUT); args = parser.parse_args()
    destination = args.output / args.model / args.gene / f"seed{args.seed}"; destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists(): return
    rows, x, y, ids = load(args.gene); floor = float(y[ids["train"]].min())
    if args.model == "xgboost":
        direct, probability, conditional, hurdle = run_xgboost(x, y, ids, floor, args.seed)
    else:
        direct_model, center, scale = fit_mlp("direct", x, y, ids, floor, args.seed, args.device)
        direct = mlp_predict(direct_model, x, ids["test"], args.device, center, scale, floor)
        hurdle_model, center, scale = fit_mlp("hurdle", x, y, ids, floor, args.seed, args.device)
        probability, conditional, hurdle = mlp_predict(hurdle_model, x, ids["test"], args.device, center, scale, floor)
    score_rows = [{"representation": "proteinbert", "model_family": args.model, "gene": args.gene, "seed": args.seed,
                   "prediction": kind, **metrics(y[ids["test"]], prediction)} for kind, prediction in (("direct", direct), ("hurdle", hurdle))]
    with (destination / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0])); writer.writeheader(); writer.writerows(score_rows)
    labels = ~np.isclose(y[ids["test"]], floor)
    (destination / "classifier.json").write_text(json.dumps({"roc_auc": float(roc_auc_score(labels, probability)),
        "floor_pr_auc": float(average_precision_score(~labels, 1 - probability)),
        "brier": float(brier_score_loss(labels, probability))}, indent=2) + "\n")
    with (destination / "predictions.csv").open("w", newline="") as handle:
        fields = ["record_id", "y_true", "direct", "hurdle", "p_measurable", "conditional"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({"record_id": rows[index]["record_id"], "y_true": y[index], "direct": direct[pos], "hurdle": hurdle[pos],
                          "p_measurable": probability[pos], "conditional": conditional[pos]} for pos, index in enumerate(ids["test"]))
    (destination / "complete.json").write_text(json.dumps({"frozen_test_used_for_fitting": False, "representation": "ProteinBERT final global 512D",
        "model_family": args.model, "gene": args.gene, "seed": args.seed}, indent=2) + "\n")


if __name__ == "__main__": main()
