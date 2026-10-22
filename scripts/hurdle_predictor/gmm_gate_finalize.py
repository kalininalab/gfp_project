"""Compare direct, floor-hurdle and two-GMM-gated CNN predictors."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.hurdle_predictor.gmm_gate_run import OUTPUT
from scripts.hurdle_predictor.run import GENES
from scripts.regression_metrics import metrics

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "results/hurdle_neural_seed42_46/cnn_ohe"
LABELS = {"direct": "Direct CNN", "floor_hurdle": "Floor hurdle", "gmm_gate": "2-GMM gate"}
COLORS = {"direct": "#56B4E9", "floor_hurdle": "#0072B2", "gmm_gate": "#D55E00"}


def main():
    records, classifier, thresholds = [], [], []
    for gene in GENES:
        for seed in range(42, 47):
            reference = list(csv.DictReader((REFERENCE / gene / f"seed{seed}" / "predictions.csv").open()))
            y = np.asarray([float(row["y_true"]) for row in reference])
            for model, column in (("direct", "direct"), ("floor_hurdle", "hurdle")):
                records.append({"model": model, "gene": gene, "seed": seed,
                                **metrics(y, np.asarray([float(row[column]) for row in reference]))})
            directory = OUTPUT / gene / f"seed{seed}"
            if not (directory / "complete.json").exists(): raise FileNotFoundError(directory)
            rows = list(csv.DictReader((directory / "predictions.csv").open()))
            y_gate = np.asarray([float(row["y_true"]) for row in rows])
            if not np.allclose(y, y_gate): raise ValueError(f"Frozen-test mismatch for {gene} seed {seed}")
            records.append({"model": "gmm_gate", "gene": gene, "seed": seed,
                            **metrics(y, np.asarray([float(row["prediction"]) for row in rows]))})
            complete = json.loads((directory / "complete.json").read_text())
            thresholds.append({"gene": gene, "seed": seed, "threshold": complete["threshold"]})
            classifier.append({"gene": gene, "seed": seed, **json.loads((directory / "classifier.json").read_text())})
    summary = []
    for gene in GENES:
        for model in LABELS:
            subset = [row for row in records if row["gene"] == gene and row["model"] == model]
            row = {"gene": gene, "model": model}
            for metric in ("r2", "pearson", "spearman", "rmse", "wasserstein", "js_divergence"):
                values = np.asarray([item[metric] for item in subset], dtype=float)
                row[f"{metric}_mean"], row[f"{metric}_sd"] = values.mean(), values.std(ddof=1)
            summary.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("all_metrics.csv", records), ("summary.csv", summary),
                           ("classifier_metrics.csv", classifier), ("thresholds.csv", thresholds)):
        with (OUTPUT / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    positions, width = np.arange(len(GENES)), .24
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("r2", "wasserstein"), ("R² ↑", "Wasserstein distance ↓")):
        for index, model in enumerate(LABELS):
            chosen = [next(row for row in summary if row["gene"] == gene and row["model"] == model) for gene in GENES]
            axis.bar(positions + (index - 1) * width, [row[f"{metric}_mean"] for row in chosen], width,
                     yerr=[row[f"{metric}_sd"] for row in chosen], color=COLORS[model], capsize=3, label=LABELS[model])
        axis.set(xticks=positions, xticklabels=GENES, ylabel=title); axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=.2); axis.axhline(0, color="#555555", lw=.8)
    axes[0].legend(frameon=False); fig.suptitle("Does a train-only two-Gaussian split improve CNN prediction?", fontsize=16)
    for suffix in ("png", "pdf", "svg"): fig.savefig(OUTPUT / f"comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Two-GMM-gated CNN predictor", "", "## Question", "",
             "Can the low/high boundary suggested by a two-component Gaussian mixture provide a better classification–regression predictor than separating only the exact measurement-floor values?", "",
             "## Model", "",
             "For each seed and landscape, a two-component Gaussian mixture is fitted to the **training fluorescence labels only**. The intersection of its weighted component densities defines a low/high threshold. A shared CNN on positional one-hot sequence has three outputs: `p_high`, a low-fluorescence regression expert, and a high-fluorescence regression expert. Its final prediction is `prediction = (1 - p_high) × low_prediction + p_high × high_prediction`.", "",
             "This is a two-expert mixture, not a classical floor hurdle. The validation split selects the neural checkpoint, and the frozen test split is used only once for final evaluation. The fact that four Gaussians fit the marginal histogram better by AIC/BIC does not invalidate this supervised test: here the question is whether a simpler low/high boundary helps sequence-to-fluorescence prediction.", "",
             "![Comparison](comparison.png)", "", "## Results", "",
             "| Landscape | Direct CNN R² | Floor-hurdle R² | 2-GMM-gated R² | 2-GMM-gated Pearson | 2-GMM-gated Wasserstein |", "|---|---:|---:|---:|---:|---:|"]
    for gene in GENES:
        get = lambda model: next(row for row in summary if row["gene"] == gene and row["model"] == model)
        direct, floor, gate = get("direct"), get("floor_hurdle"), get("gmm_gate")
        lines.append(f"| {gene} | {direct['r2_mean']:.3f} ± {direct['r2_sd']:.3f} | {floor['r2_mean']:.3f} ± {floor['r2_sd']:.3f} | {gate['r2_mean']:.3f} ± {gate['r2_sd']:.3f} | {gate['pearson_mean']:.3f} ± {gate['pearson_sd']:.3f} | {gate['wasserstein_mean']:.3f} ± {gate['wasserstein_sd']:.3f} |")
    lines += ["", "The model should be retained only if its five-seed frozen-test improvement is consistent. Machine-readable outputs: [summary](summary.csv), [all seeded metrics](all_metrics.csv), [train-only thresholds](thresholds.csv), and [gate-classifier metrics](classifier_metrics.csv).", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__": main()
