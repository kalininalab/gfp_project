"""Matched direct and soft-hurdle XGBoost predictors on ESM or positional OHE."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score

from scripts.aubin_model import ALPHABET
from scripts.hurdle_predictor.run import DATA, FEATURES, GENES
from scripts.regression_metrics import metrics

OUTPUT = Path("results/hurdle_xgboost_seed42_46")


def positional_ohe(rows):
    lookup = {aa: index for index, aa in enumerate(ALPHABET)}; length = len(rows[0]["sequence"])
    encoded = np.asarray([[lookup[aa] for aa in row["sequence"]] for row in rows], dtype=np.int32)
    columns = (encoded + np.arange(length) * len(ALPHABET)).ravel()
    pointers = np.arange(len(rows) + 1) * length
    return csr_matrix((np.ones(len(rows) * length, dtype=np.float32), columns, pointers), shape=(len(rows), length * len(ALPHABET)))


def load(gene, representation):
    rows = [row for row in csv.DictReader(DATA.open()) if row["gene"] == gene]
    y = np.asarray([float(row["target_log10"]) for row in rows], dtype=np.float32)
    ids = {name: np.asarray([i for i, row in enumerate(rows) if row["split"] == name]) for name in ("train", "validation", "test")}
    if representation == "ohe": x = positional_ohe(rows)
    else:
        directory = FEATURES / f"{gene}_t30"; records = list(csv.DictReader((directory / "records.csv").open()))
        embeddings = np.load(directory / "mean_embeddings.npy", mmap_mode="r"); index = {row["record_id"]: i for i, row in enumerate(records)}
        x = np.asarray(embeddings[[index[row["record_id"]] for row in rows]], dtype=np.float32)
    return rows, x, y, ids


def parameters(seed, n_estimators=2000):
    return dict(n_estimators=n_estimators, learning_rate=.03, max_depth=6, min_child_weight=1,
                subsample=.8, colsample_bytree=.8, reg_lambda=1, tree_method="hist",
                n_jobs=4, random_state=seed, early_stopping_rounds=50)


def main():
    import xgboost as xgb
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", choices=GENES, required=True); parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--representation", choices=["esm", "ohe"], required=True); parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--n-estimators", type=int, default=2000)
    args = parser.parse_args(); output = args.output / args.representation / args.gene / f"seed{args.seed}"; output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists(): return
    rows, x, y, ids = load(args.gene, args.representation); floor = float(y[ids["train"]].min())
    measurable = ~np.isclose(y, floor, atol=1e-10); common = parameters(args.seed, args.n_estimators)
    direct = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    direct.fit(x[ids["train"]], y[ids["train"]], eval_set=[(x[ids["validation"]], y[ids["validation"]])], verbose=False)
    classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    classifier.fit(x[ids["train"]], measurable[ids["train"]].astype(np.int8),
                   eval_set=[(x[ids["validation"]], measurable[ids["validation"]].astype(np.int8))], verbose=False)
    train_measurable = ids["train"][measurable[ids["train"]]]; validation_measurable = ids["validation"][measurable[ids["validation"]]]
    conditional = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    conditional.fit(x[train_measurable], y[train_measurable], eval_set=[(x[validation_measurable], y[validation_measurable])], verbose=False)
    direct_prediction = direct.predict(x[ids["test"]]); probability = classifier.predict_proba(x[ids["test"]])[:, 1]
    conditional_prediction = np.maximum(conditional.predict(x[ids["test"]]), floor)
    hurdle_prediction = (1 - probability) * floor + probability * conditional_prediction
    score_rows = [
        {"representation": args.representation, "gene": args.gene, "seed": args.seed, "model": model, **metrics(y[ids["test"]], prediction)}
        for model, prediction in (("direct", direct_prediction), ("hurdle", hurdle_prediction))
    ]
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0])); writer.writeheader(); writer.writerows(score_rows)
    labels = measurable[ids["test"]]
    (output / "classifier.json").write_text(json.dumps({
        "roc_auc": roc_auc_score(labels, probability), "floor_pr_auc": average_precision_score(~labels, 1-probability),
        "brier": brier_score_loss(labels, probability), "balanced_accuracy": balanced_accuracy_score(labels, probability >= .5),
        "mcc": matthews_corrcoef(labels, probability >= .5), "floor_prevalence": float((~labels).mean()),
    }, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as handle:
        fields = ["record_id", "y_true", "direct", "hurdle", "p_measurable", "conditional"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows({"record_id": rows[index]["record_id"], "y_true": y[index], "direct": direct_prediction[position],
                          "hurdle": hurdle_prediction[position], "p_measurable": probability[position], "conditional": conditional_prediction[position]}
                         for position, index in enumerate(ids["test"]))
    (output / "complete.json").write_text(json.dumps({
        "xgboost": xgb.__version__, "representation": args.representation, "gene": args.gene, "seed": args.seed,
        "floor": floor, "direct_best_iteration": direct.best_iteration, "classifier_best_iteration": classifier.best_iteration,
        "conditional_best_iteration": conditional.best_iteration, "frozen_test_used_for_fitting": False,
    }, indent=2) + "\n")


if __name__ == "__main__": main()
