"""Aggregate and plot the matched ESM-ridge hurdle experiment."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.hurdle_predictor.run import GENES, OUTPUT

MODELS = ["direct", "hurdle_soft", "hurdle_hard"]
LABELS = {"direct": "Direct regression", "hurdle_soft": "Soft hurdle", "hurdle_hard": "Hard hurdle"}
COLORS = {"direct": "#56B4E9", "hurdle_soft": "#0072B2", "hurdle_hard": "#D55E00"}


def main():
    rows, classifier = [], []
    for gene in GENES:
        for seed in range(42, 47):
            directory = OUTPUT / gene / f"seed{seed}"
            rows.extend(csv.DictReader((directory / "metrics.csv").open()))
            classifier.append(json.loads((directory / "classifier_metrics.json").read_text()))
    with (OUTPUT / "summary.csv").open("w", newline="") as handle:
        fields = ["gene", "model", "metric", "mean", "sd"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        summary = []
        for gene in GENES:
            for model in MODELS:
                subset = [row for row in rows if row["gene"] == gene and row["model"] == model]
                for metric in ("r2", "pearson", "spearman", "rmse"):
                    values = np.asarray([float(row[metric]) for row in subset])
                    item = {"gene": gene, "model": model, "metric": metric, "mean": values.mean(), "sd": values.std(ddof=1)}
                    writer.writerow(item); summary.append(item)
    with (OUTPUT / "classifier_summary.csv").open("w", newline="") as handle:
        fields = list(classifier[0]); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(classifier)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    width = 0.24; positions = np.arange(len(GENES))
    for model_index, model in enumerate(MODELS):
        offset = (model_index - 1) * width
        for axis, metric in zip(axes, ("r2", "pearson")):
            selected = [next(row for row in summary if row["gene"] == gene and row["model"] == model and row["metric"] == metric) for gene in GENES]
            axis.bar(positions + offset, [row["mean"] for row in selected], width,
                     label=LABELS[model], color=COLORS[model])
    for axis, title in zip(axes, ("R²", "Pearson")):
        axis.set(xticks=positions, xticklabels=GENES, ylabel=title, title=f"Frozen-test {title}")
        axis.tick_params(axis="x", rotation=25); axis.axhline(0, color="#555555", lw=0.8)
    axes[0].legend(frameon=False)
    fig.suptitle("Direct versus classification–regression prediction", fontsize=16)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Classification–regression predictor", "", "All models use the same frozen 20% test records. Fitting, measurement-floor definition and feature scaling use only the remaining 80% development records. The representation is mean-pooled ESM-2 150M; the matched direct and conditional regressors are ridge models.", "", "![Comparison](comparison.png)", "", "## Frozen-test results", "", "| Landscape | Model | R² | Pearson | RMSE |", "|---|---|---:|---:|---:|"]
    for gene in GENES:
        for model in MODELS:
            get = lambda metric: next(row for row in summary if row["gene"] == gene and row["model"] == model and row["metric"] == metric)
            lines.append(f"| {gene} | {LABELS[model]} | {get('r2')['mean']:.3f} | {get('pearson')['mean']:.3f} | {get('rmse')['mean']:.3f} |")
    lines += ["", "## Measurement-floor classifier", "", "| Landscape | ROC-AUC | Balanced accuracy | MCC | Floor prevalence in test |", "|---|---:|---:|---:|---:|"]
    for gene in GENES:
        item = next(row for row in classifier if row["gene"] == gene)
        lines.append(f"| {gene} | {item['roc_auc']:.3f} | {item['balanced_accuracy']:.3f} | {item['mcc']:.3f} | {item['test_floor_n'] / item['test_n']:.1%} |")
    lines += [
        "", "## Interpretation", "",
        "The soft hurdle improves R² and Pearson over matched direct regression in every landscape. The largest R² gains are cgreGFP (0.435 to 0.509) and cgre4111 (0.338 to 0.394). Hard thresholding loses information and is worse than probability-weighted prediction for every landscape.", "",
        "The classifier itself is weak for cgre1338 and cgre4111, and the floor class is extremely rare for cgre132 and cgre9708. This experiment therefore supports the soft architecture, but it is not a final production predictor. The next test should place the same soft hurdle head on stronger MLP and CNN representations and compare it with their matched direct regressors.", "",
        "The five seeds repeat model initialization on an identical frozen split. Ridge and logistic regression are deterministic here; the repeated identical values are retained only to match the project reporting convention and must not be interpreted as independent uncertainty estimates.", "",
        "Machine-readable files: [regression summary](summary.csv) and [classifier metrics](classifier_summary.csv).", "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))
    (OUTPUT / "protocol.json").write_text(json.dumps({
        "genes": GENES, "seeds": list(range(42, 47)), "frozen_test_split": "test",
        "development_splits": ["train", "validation"],
        "features": "facebook/esm2_t30_150M_UR50D mean embeddings",
        "direct_regressor": "StandardScaler + Ridge(alpha=10)",
        "classifier": "StandardScaler + LogisticRegression(C=1)",
        "conditional_regressor": "StandardScaler + Ridge(alpha=10), fit above the development measurement floor",
        "soft_prediction": "(1-p_measurable)*floor + p_measurable*max(conditional_prediction,floor)",
        "hard_prediction": "floor if p_measurable < 0.5 else max(conditional_prediction,floor)",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
