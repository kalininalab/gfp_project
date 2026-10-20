"""Compare direct and hurdle prediction on one frozen GFP landscape split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.regression_metrics import metrics

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/baseline_v1/sequences.csv"
FEATURES = ROOT / "esm_embeddings"
OUTPUT = ROOT / "results/hurdle_esm_ridge_seed42_46"
GENES = ["cgreGFP", "cgre132", "cgre1338", "cgre4111", "cgre9708"]


def load(gene):
    rows = [row for row in csv.DictReader(DATA.open()) if row["gene"] == gene]
    directory = FEATURES / f"{gene}_t30"
    records = list(csv.DictReader((directory / "records.csv").open()))
    embeddings = np.load(directory / "mean_embeddings.npy", mmap_mode="r")
    if len(records) != len(embeddings) or {row["record_id"] for row in rows} != {row["record_id"] for row in records}:
        raise ValueError(f"Feature manifest mismatch for {gene}")
    index = {row["record_id"]: position for position, row in enumerate(records)}
    x = np.asarray(embeddings[[index[row["record_id"]] for row in rows]], dtype=np.float32)
    y = np.asarray([float(row["target_log10"]) for row in rows])
    development = np.asarray([row["split"] != "test" for row in rows])
    test = ~development
    return rows, x, y, development, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", choices=GENES, required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, x, y, development, test = load(args.gene)
    floor = float(y[development].min())
    active = ~np.isclose(y, floor, atol=1e-10)
    if active[development].min() == active[development].max():
        raise ValueError("Development data need both floor and measurable records")

    direct = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(x[development], y[development])
    direct_prediction = direct.predict(x[test])
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
    ).fit(x[development], active[development])
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        x[development & active], y[development & active]
    )
    probability = classifier.predict_proba(x[test])[:, 1]
    measurable_prediction = np.maximum(regressor.predict(x[test]), floor)
    soft_prediction = (1.0 - probability) * floor + probability * measurable_prediction
    hard_prediction = np.where(probability >= 0.5, measurable_prediction, floor)

    output = args.output / args.gene / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for model, prediction in (("direct", direct_prediction), ("hurdle_soft", soft_prediction), ("hurdle_hard", hard_prediction)):
        score_rows.append({"gene": args.gene, "seed": args.seed, "model": model, **metrics(y[test], prediction)})
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader(); writer.writerows(score_rows)
    labels = active[test]
    classifier_metrics = {
        "gene": args.gene, "seed": args.seed, "floor": floor,
        "development_n": int(development.sum()), "development_floor_n": int((development & ~active).sum()),
        "test_n": int(test.sum()), "test_floor_n": int((test & ~active).sum()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= 0.5)),
        "mcc": float(matthews_corrcoef(labels, probability >= 0.5)),
    }
    (output / "classifier_metrics.json").write_text(json.dumps(classifier_metrics, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as handle:
        fields = ["record_id", "y_true", "is_measurable", "p_measurable", "direct", "hurdle_soft", "hurdle_hard"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        test_rows = np.flatnonzero(test)
        writer.writerows({
            "record_id": rows[index]["record_id"], "y_true": y[index], "is_measurable": int(active[index]),
            "p_measurable": probability[position], "direct": direct_prediction[position],
            "hurdle_soft": soft_prediction[position], "hurdle_hard": hard_prediction[position],
        } for position, index in enumerate(test_rows))
    (output / "complete.json").write_text(json.dumps({
        "feature_model": "facebook/esm2_t30_150M_UR50D mean embeddings", "classifier": "standardized logistic regression",
        "regressor": "standardized ridge alpha=10", "floor_defined_on": "train+validation only",
        "frozen_test_used_for_fitting": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
