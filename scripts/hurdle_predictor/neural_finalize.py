"""Validate and summarize all neural hurdle runs for a management-facing report."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.hurdle_predictor.neural_run import OUTPUT
from scripts.hurdle_predictor.run import GENES

ARCHITECTURES = ["mlp_esm", "cnn_ohe"]
LABELS = {"mlp_esm": "MLP on ESM", "cnn_ohe": "CNN on OHE"}
COLORS = {"direct": "#56B4E9", "hurdle": "#0072B2"}


def main():
    rows, classifier = [], []
    for architecture in ARCHITECTURES:
        for gene in GENES:
            for seed in range(42, 47):
                directory = OUTPUT / architecture / gene / f"seed{seed}"
                if not (directory / "complete.json").exists(): raise FileNotFoundError(directory)
                rows.extend(csv.DictReader((directory / "metrics.csv").open()))
                classifier.append({"architecture": architecture, "gene": gene, "seed": seed, **json.loads((directory / "classifier.json").read_text())})
    summary = []
    for architecture in ARCHITECTURES:
        for gene in GENES:
            for model in ("direct", "hurdle"):
                subset = [row for row in rows if row["architecture"] == architecture and row["gene"] == gene and row["model"] == model]
                for metric in ("r2", "pearson", "spearman", "rmse"):
                    values = np.asarray([float(row[metric]) for row in subset])
                    summary.append({"architecture": architecture, "gene": gene, "model": model, "metric": metric, "mean": values.mean(), "sd": values.std(ddof=1)})
    with (OUTPUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with (OUTPUT / "classifier_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(classifier[0])); writer.writeheader(); writer.writerows(classifier)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    positions = np.arange(len(GENES)); width = .36
    for row_index, architecture in enumerate(ARCHITECTURES):
        for column_index, metric in enumerate(("r2", "pearson")):
            axis = axes[row_index, column_index]
            for model_index, model in enumerate(("direct", "hurdle")):
                chosen = [next(item for item in summary if item["architecture"] == architecture and item["gene"] == gene and item["model"] == model and item["metric"] == metric) for gene in GENES]
                axis.bar(positions + (model_index - .5) * width, [item["mean"] for item in chosen], width,
                         yerr=[item["sd"] for item in chosen], color=COLORS[model], capsize=3,
                         label="Direct" if model == "direct" else "Soft hurdle")
            axis.set(xticks=positions, xticklabels=GENES, ylabel="R²" if metric == "r2" else "Pearson",
                     title=f"{LABELS[architecture]} · {'R²' if metric == 'r2' else 'Pearson'}")
            axis.tick_params(axis="x", rotation=25); axis.axhline(0, color="#555555", lw=.8)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Does classification–regression improve fluorescence prediction?", fontsize=17)
    for suffix in ("png", "pdf", "svg"): fig.savefig(OUTPUT / f"comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Classification–regression models for cgreGFP landscapes", "",
        "## Question", "",
        "Does explicitly modelling assay-floor variants improve sequence-to-fluorescence prediction for natural cgreGFP and its four artificial-peak landscapes?", "",
        "## Experimental design", "",
        "Each comparison uses the same fixed 60% training, 20% validation and 20% test records. The test set is never used for fitting, threshold selection or checkpoint selection. Five independently initialized runs (seeds 42–46) are trained for every landscape and architecture.", "",
        "- **Direct:** one neural network predicts log10 fluorescence.",
        "- **Soft hurdle:** a shared network has a classification head for `measurement floor versus measurable` and a regression head fitted on measurable variants. The final prediction is a probability-weighted combination of the assay floor and conditional fluorescence.",
        "- **MLP on ESM:** mean-pooled ESM-2 150M embeddings.",
        "- **CNN on OHE:** native 235-residue one-hot sequence and the project's Jannis-style CNN backbone.", "",
        "![Matched neural comparison](comparison.png)", "", "## Results", "",
        "| Architecture | Landscape | Direct R² | Hurdle R² | ΔR² | Direct Pearson | Hurdle Pearson |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        for gene in GENES:
            get = lambda model, metric: next(item for item in summary if item["architecture"] == architecture and item["gene"] == gene and item["model"] == model and item["metric"] == metric)
            dr, hr, dp, hp = get("direct", "r2"), get("hurdle", "r2"), get("direct", "pearson"), get("hurdle", "pearson")
            lines.append(f"| {LABELS[architecture]} | {gene} | {dr['mean']:.3f} ± {dr['sd']:.3f} | {hr['mean']:.3f} ± {hr['sd']:.3f} | {hr['mean']-dr['mean']:+.3f} | {dp['mean']:.3f} | {hp['mean']:.3f} |")
    gains = [item for item in summary if item["metric"] == "r2" and item["model"] == "hurdle"]
    lines += ["", "## Decision", "", "The hurdle architecture should be retained only where its five-seed test improvement is consistent and practically meaningful. A small classifier score alone is not evidence of a better fluorescence predictor; the decision is based on final frozen-test R², Pearson and RMSE.", "", "## Earlier distribution analysis in plain language", "", "A Gaussian mixture model approximates the observed fluorescence histogram with several bell-shaped curves. We compared one through four curves. **AIC** and **BIC** are fit scores that penalize adding extra curves; lower values are better. Among the tested choices, four curves described every landscape better than one or two. This does not mean that the protein has four biological states: many observations are exactly at the instrument's lower limit, which is not bell-shaped. That finding motivated the measurement-floor classifier used here.", "", "Related evidence: [distribution analysis](../cgre_peak_mixture_analysis/REPORT.md) and [linear soft-hurdle pilot](../hurdle_esm_ridge_seed42_46/REPORT.md). Machine-readable outputs: [neural summary](summary.csv) and [classifier metrics](classifier_metrics.csv).", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
