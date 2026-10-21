"""Build the two-part distribution and predictor study report."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.hurdle_predictor.run import GENES
from scripts.regression_metrics import metrics

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/cgre_predictor_study"
SOURCES = [
    ("ESM-2", "MLP", ROOT / "results/hurdle_neural_seed42_46/mlp_esm"),
    ("OHE", "CNN", ROOT / "results/hurdle_neural_seed42_46/cnn_ohe"),
    ("ESM-2", "XGBoost", ROOT / "results/hurdle_xgboost_seed42_46/esm"),
    ("OHE", "XGBoost", ROOT / "results/hurdle_xgboost_seed42_46/ohe"),
    ("ProteinBERT", "MLP", ROOT / "results/proteinbert_models_seed42_46/mlp"),
    ("ProteinBERT", "XGBoost", ROOT / "results/proteinbert_models_seed42_46/xgboost"),
]
COLORS = {"ESM-2": "#0072B2", "OHE": "#D55E00", "ProteinBERT": "#CC79A7"}


def read_predictions(path):
    rows = list(csv.DictReader(path.open()))
    return rows, np.asarray([float(row["y_true"]) for row in rows])


def gather():
    records, predictions = [], {}
    for encoding, family, base in SOURCES:
        for gene in GENES:
            for seed in range(42, 47):
                path = base / gene / f"seed{seed}" / "predictions.csv"
                if not path.exists(): raise FileNotFoundError(path)
                rows, y = read_predictions(path)
                for variant in ("direct", "hurdle"):
                    prediction = np.asarray([float(row[variant]) for row in rows])
                    row = {"encoding": encoding, "model": family, "prediction": variant, "gene": gene, "seed": seed, **metrics(y, prediction)}
                    records.append(row); predictions[(encoding, family, variant, gene, seed)] = (y, prediction)
    return records, predictions


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["encoding"], row["model"], row["prediction"], row["gene"])].append(row)
    summary = []
    for key, values in grouped.items():
        row = dict(zip(("encoding", "model", "prediction", "gene"), key))
        for metric in ("r2", "pearson", "spearman", "rmse", "wasserstein", "js_divergence"):
            scores = np.asarray([value[metric] for value in values])
            row[f"{metric}_mean"], row[f"{metric}_sd"] = float(scores.mean()), float(scores.std(ddof=1))
        summary.append(row)
    return summary


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def comparison_plot(summary):
    models = [(encoding, family, variant) for encoding, family, _ in SOURCES for variant in ("direct", "hurdle")]
    labels = [f"{encoding}\n{family} ({'soft hurdle' if variant == 'hurdle' else 'direct'})" for encoding, family, variant in models]
    fig, axes = plt.subplots(2, len(GENES), figsize=(18, 7), sharex="col")
    for column, gene in enumerate(GENES):
        for row_index, (metric, title) in enumerate((("r2", "R² ↑"), ("wasserstein", "Wasserstein ↓"))):
            ax = axes[row_index, column]
            values = [next(row for row in summary if (row["encoding"], row["model"], row["prediction"], row["gene"]) == (*model, gene)) for model in models]
            ax.bar(np.arange(len(models)), [row[f"{metric}_mean"] for row in values],
                   yerr=[row[f"{metric}_sd"] for row in values], color=[COLORS[model[0]] for model in models], capsize=2)
            ax.axhline(0, color="#555555", lw=.8); ax.set_title(gene if row_index == 0 else "")
            if column == 0: ax.set_ylabel(title)
            ax.grid(axis="y", alpha=.2)
            if row_index == 1:
                ax.set_xticks(np.arange(len(models)), labels, rotation=65, ha="right", fontsize=7)
    fig.suptitle("Encoding and predictor comparison on frozen test sets", fontsize=16)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"): fig.savefig(OUTPUT / f"model_comparison.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def diagnostic_plot(summary, predictions):
    best = {}
    for gene in GENES:
        candidates = [row for row in summary if row["gene"] == gene]
        best[gene] = max(candidates, key=lambda row: row["r2_mean"])
    fig, axes = plt.subplots(2, len(GENES), figsize=(17, 6))
    for column, gene in enumerate(GENES):
        chosen = best[gene]; key = (chosen["encoding"], chosen["model"], chosen["prediction"], gene, 42)
        y, prediction = predictions[key]; label = f"{chosen['encoding']} + {chosen['model']} ({chosen['prediction']})"
        edges = np.linspace(min(y.min(), prediction.min()), max(y.max(), prediction.max()), 40)
        axes[0, column].hist(y, bins=edges, density=True, histtype="step", lw=2, color="#0072B2", label="Observed")
        axes[0, column].hist(prediction, bins=edges, density=True, histtype="step", lw=2, color="#D55E00", label="Predicted")
        axes[0, column].set_title(f"{gene}\n{label}", fontsize=10); axes[0, column].set_xlabel("log₁₀ fluorescence")
        axes[1, column].scatter(y, prediction, s=5, alpha=.25, color="#0072B2", rasterized=True)
        limits = [min(y.min(), prediction.min()), max(y.max(), prediction.max())]
        axes[1, column].plot(limits, limits, color="#D55E00", lw=1); axes[1, column].set_xlim(limits); axes[1, column].set_ylim(limits)
        axes[1, column].set_xlabel("Observed"); axes[1, column].set_ylabel("Predicted" if column == 0 else "")
    axes[0, 0].set_ylabel("Density"); axes[0, 0].legend(frameon=False)
    fig.suptitle("Observed and predicted fluorescence for the strongest tested model", fontsize=16); fig.tight_layout()
    for extension in ("png", "pdf", "svg"): fig.savefig(OUTPUT / f"best_model_diagnostics.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return best


def report(summary, best):
    lines = ["# cgreGFP landscape distributions and predictor comparison", "",
             "## Executive summary", "",
             "This study first asks whether the fluorescence values form two clean Gaussian groups. It then tests whether pretrained protein encodings improve prediction when the data split and downstream models are held constant.", "",
             "## Part 1 — What does the fluorescence distribution look like?", "",
             "The [distribution analysis](../cgre_peak_mixture_analysis/REPORT.md) fits one to four Gaussian components. AIC and BIC reward fit but penalize unnecessary components; lower values are better. Both prefer four components over two for every tested landscape. This means that two Gaussians are too simple as a statistical description. It does not prove four biological states, because the assay floor creates a spike that is not Gaussian.", "",
             "A **hurdle model** treats the spike at the measurement floor and the continuous fluorescence above it as two linked prediction tasks. A classifier estimates `p`, the probability that a sequence is above the floor. A regressor estimates `bright_prediction`, its fluorescence conditional on being above the floor. The soft-hurdle output is `prediction = (1 - p) × floor + p × bright_prediction`. For example, if the floor is 2, the conditional prediction is 5 and `p = 0.8`, the final prediction is 4.4. This probability weighting avoids forcing uncertain sequences into a hard dark/bright class.", "",
             "Every soft-hurdle result below is compared with a direct regressor using the same encoding, split and seed. The hurdle construction is expected to help most when many observations are at the floor; it may add little where the floor fraction is close to zero.", "",
             "## Part 2 — Do ProteinBERT encodings improve prediction?", "",
             "### Design", "",
             "- Five landscapes: natural cgreGFP and four artificial peak datasets.",
             "- Existing frozen 60% train / 20% validation / 20% test splits; the test records are never used for fitting or early stopping.",
             "- Five seeds (42–46). Error bars are standard deviations across seeds.",
             "- Encodings: 512-dimensional final ProteinBERT global state, 640-dimensional mean ESM-2 state, and positional one-hot sequence.",
             "- Predictors: MLP, XGBoost and the current CNN-on-OHE reference. Each is evaluated as direct regression and soft hurdle.",
             "- Main accuracy metric: R² (higher is better). Distribution metric: Wasserstein distance in log₁₀-fluorescence units (lower is better). Pearson, Spearman, RMSE and Jensen–Shannon divergence are retained in the CSV.", "",
             "![Model comparison](model_comparison.png)", "", "### Best mean R² in each landscape", "",
             "| Landscape | Encoding | Predictor | Form | R² | Pearson | Wasserstein |", "|---|---|---|---|---:|---:|---:|"]
    for gene in GENES:
        chosen = best[gene]
        lines.append(f"| {gene} | {chosen['encoding']} | {chosen['model']} | {chosen['prediction']} | {chosen['r2_mean']:.3f} ± {chosen['r2_sd']:.3f} | {chosen['pearson_mean']:.3f} ± {chosen['pearson_sd']:.3f} | {chosen['wasserstein_mean']:.3f} ± {chosen['wasserstein_sd']:.3f} |")
    lines += ["", "![Best-model distributions and dot plots](best_model_diagnostics.png)", "",
              "### How to read the metrics", "",
              "R² measures explained squared error: 1 is perfect, 0 matches predicting the test mean, and negative values are worse. Pearson measures linear agreement and Spearman measures rank agreement. RMSE reports typical error in log₁₀ fluorescence. Wasserstein distance measures the average horizontal movement needed to align the predicted and observed distributions, in the same units as the target. Jensen–Shannon divergence is a bounded, symmetric histogram comparison; 0 means identical distributions.", "",
              "Machine-readable results: [all seeded metrics](all_metrics.csv) and [five-seed summary](summary.csv). ProteinBERT embeddings and the 192 MB checkpoint live under `/data/users/akolchina`, outside Git.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records, predictions = gather(); summary = aggregate(records)
    write_csv(OUTPUT / "all_metrics.csv", records); write_csv(OUTPUT / "summary.csv", summary)
    comparison_plot(summary); best = diagnostic_plot(summary, predictions); report(summary, best)


if __name__ == "__main__": main()
