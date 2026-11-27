"""Aggregate and plot the artificial-peak saturation benchmark."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import MODELS, PEAKS, PUBLISHED, RAW, SEEDS, write_csv

LABELS = {"mlp_small": "MLP", "CNN_Jannis_OHE": "CNN on OHE"}
COLORS = {"mlp_small": "#0072B2", "CNN_Jannis_OHE": "#D55E00"}
METRICS = {"spearman": "Spearman ↑", "pearson": "Pearson ↑", "r2": "R² ↑",
           "rmse": "RMSE ↓", "mae": "MAE ↓", "kl_true_to_pred_bits": "KL divergence (bits) ↓",
           "wasserstein": "Wasserstein distance ↓"}


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=RAW)
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    rows = []
    for peak in PEAKS:
        for model in MODELS:
            for seed in SEEDS:
                for n_train in protocol["sizes"][peak]:
                    path = args.directory / "fits" / peak / f"seed{seed}" / model / f"n{n_train:05d}" / "complete.json"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    report = json.loads(path.read_text())
                    measured = dict(report["metrics"])
                    if "mae" not in measured:
                        with (path.parent / "predictions.csv").open(newline="") as handle:
                            predictions = list(csv.DictReader(handle))
                        measured["mae"] = float(np.mean([
                            abs(float(row["y_true"]) - float(row["y_pred"])) for row in predictions]))
                    rows.append({"peak": peak, "model": model, "seed": seed, "n_train": n_train,
                                 "n_validation": report["n_validation"], "n_test": report["n_test"], **measured})
    args.published.mkdir(parents=True, exist_ok=True)
    write_csv(args.published / "all_seeded_metrics.csv", rows)
    summaries = []
    for peak in PEAKS:
        for model in MODELS:
            for n_train in protocol["sizes"][peak]:
                subset = [row for row in rows if row["peak"] == peak and row["model"] == model
                          and row["n_train"] == n_train]
                item = {"peak": peak, "model": model, "n_train": n_train, "n_seeds": len(subset),
                        "n_validation": subset[0]["n_validation"], "n_test": subset[0]["n_test"]}
                for metric in METRICS:
                    values = np.asarray([float(row[metric]) for row in subset])
                    item[metric + "_mean"] = values.mean()
                    item[metric + "_sd"] = values.std(ddof=1)
                summaries.append(item)
    write_csv(args.published / "summary.csv", summaries)

    for metric, ylabel in METRICS.items():
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        for axis, peak in zip(axes.flat, PEAKS):
            for model in MODELS:
                subset = [row for row in summaries if row["peak"] == peak and row["model"] == model]
                x = np.asarray([row["n_train"] for row in subset])
                mean = np.asarray([row[metric + "_mean"] for row in subset])
                sd = np.asarray([row[metric + "_sd"] for row in subset])
                axis.plot(x, mean, marker="o", lw=2.2, color=COLORS[model], label=LABELS[model])
                axis.fill_between(x, mean - sd, mean + sd, color=COLORS[model], alpha=.16)
            axis.set_xscale("log", base=2)
            axis.set(title=f"Source: {peak}", xlabel="Artificial-peak training sequences", ylabel=ylabel)
            axis.grid(alpha=.2)
            if metric in ("spearman", "pearson", "r2"):
                axis.set_ylim(-1.05, 1.05)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.03))
        fig.suptitle("Prediction of natural cgreGFP from artificial peaks", fontsize=19, y=1.06)
        save(fig, args.published / f"performance_vs_train_size_{metric}")
    shutil.copy2(args.directory / "protocol.json", args.published / "protocol.json")
    report = ["# Prediction of natural cgreGFP from increasing artificial-peak training sets", "",
              "Each panel trains MLP and CNN on OHE on nested subsets from one artificial peak and evaluates prediction on natural cgreGFP. All sequences available for that source peak are eligible for training. The natural cgreGFP validation set is fixed and selects checkpoints; its frozen test set is plotted and never selects models.", "",
              "Five seeds (42–46) change source-subset order and model initialization. The complete source spaces determine the final points: cgre1338 = 10,241; cgre4111 = 8,214; cgre132 = 4,267; cgre9708 = 4,180.", "", "## Figures", ""]
    for metric, label in METRICS.items():
        report += [f"### {label}", "", f"![{label}](performance_vs_train_size_{metric}.png)", ""]
    report += ["## Files", "", "[Summary](summary.csv) · [All seeded metrics](all_seeded_metrics.csv) · [Protocol](protocol.json)", ""]
    (args.published / "REPORT.md").write_text("\n".join(report))
    (args.published / "finalization.json").write_text(
        json.dumps({"complete": True, "fits": len(rows)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
