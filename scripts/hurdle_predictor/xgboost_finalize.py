"""Aggregate XGBoost direct-versus-hurdle results into a concise report."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from scripts.hurdle_predictor.run import GENES
from scripts.hurdle_predictor.xgboost_run import OUTPUT

REPRESENTATIONS = ["esm", "ohe"]
LABELS = {"esm": "XGBoost on ESM", "ohe": "XGBoost on OHE"}


def main():
    rows, classifier = [], []
    for representation in REPRESENTATIONS:
        for gene in GENES:
            for seed in range(42, 47):
                directory = OUTPUT / representation / gene / f"seed{seed}"
                if not (directory / "complete.json").exists(): raise FileNotFoundError(directory)
                rows.extend(csv.DictReader((directory / "metrics.csv").open()))
                classifier.append({"representation": representation, "gene": gene, "seed": seed, **json.loads((directory / "classifier.json").read_text())})
    summary = []
    for representation in REPRESENTATIONS:
        for gene in GENES:
            for model in ("direct", "hurdle"):
                subset = [row for row in rows if row["representation"] == representation and row["gene"] == gene and row["model"] == model]
                for metric in ("r2", "pearson", "spearman", "rmse"):
                    values = np.asarray([float(row[metric]) for row in subset])
                    summary.append({"representation": representation, "gene": gene, "model": model, "metric": metric, "mean": values.mean(), "sd": values.std(ddof=1)})
    with (OUTPUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with (OUTPUT / "classifier_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(classifier[0])); writer.writeheader(); writer.writerows(classifier)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True); positions = np.arange(len(GENES)); width = .36
    for row_index, representation in enumerate(REPRESENTATIONS):
        for column_index, metric in enumerate(("r2", "pearson")):
            axis = axes[row_index, column_index]
            for model_index, model in enumerate(("direct", "hurdle")):
                chosen = [next(item for item in summary if item["representation"] == representation and item["gene"] == gene and item["model"] == model and item["metric"] == metric) for gene in GENES]
                axis.bar(positions + (model_index-.5)*width, [item["mean"] for item in chosen], width, yerr=[item["sd"] for item in chosen],
                         color="#56B4E9" if model == "direct" else "#0072B2", capsize=3, label="Direct" if model == "direct" else "Soft hurdle")
            axis.set(xticks=positions, xticklabels=GENES, ylabel="R²" if metric == "r2" else "Pearson",
                     title=f"{LABELS[representation]} · {'R²' if metric == 'r2' else 'Pearson'}")
            axis.tick_params(axis="x", rotation=25); axis.axhline(0, color="#555555", lw=.8)
    axes[0, 0].legend(frameon=False); fig.suptitle("XGBoost fluorescence prediction", fontsize=17)
    for suffix in ("png", "pdf", "svg"): fig.savefig(OUTPUT / f"comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    lines = ["# XGBoost for cgreGFP landscapes", "", "## Question", "", "Can boosted decision trees improve fluorescence prediction, and does the soft classification–regression design still help?", "", "## Design", "", "Every comparison uses the same fixed 60% training, 20% validation and 20% frozen test records. Five stochastic fits (seeds 42–46) are run for positional one-hot features and mean ESM-2 embeddings. Early stopping uses validation only.", "", "- **Direct:** one XGBoost regressor predicts log10 fluorescence.", "- **Soft hurdle:** an XGBoost classifier estimates the probability of measurable fluorescence and a second regressor predicts fluorescence among measurable variants.", "- **OHE:** 4,700 sparse `position × amino acid` features; suitable for position-specific rules and mutation interactions.", "- **ESM:** 640 dense protein-language-model features.", "", "![XGBoost comparison](comparison.png)", "", "## Results", "", "| Representation | Landscape | Direct R² | Hurdle R² | ΔR² | Direct Pearson | Hurdle Pearson |", "|---|---|---:|---:|---:|---:|---:|"]
    for representation in REPRESENTATIONS:
        for gene in GENES:
            get = lambda model, metric: next(item for item in summary if item["representation"] == representation and item["gene"] == gene and item["model"] == model and item["metric"] == metric)
            dr, hr, dp, hp = get("direct", "r2"), get("hurdle", "r2"), get("direct", "pearson"), get("hurdle", "pearson")
            lines.append(f"| {LABELS[representation]} | {gene} | {dr['mean']:.3f} ± {dr['sd']:.3f} | {hr['mean']:.3f} ± {hr['sd']:.3f} | {hr['mean']-dr['mean']:+.3f} | {dp['mean']:.3f} | {hp['mean']:.3f} |")
    lines += ["", "## Decision rule", "", "XGBoost is retained only if its frozen-test metrics improve over the corresponding MLP/CNN baseline. The hurdle version is retained only if it consistently improves final fluorescence metrics; classifier performance alone is insufficient.", "", "Machine-readable outputs: [summary](summary.csv) and [classifier metrics](classifier_metrics.csv).", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
