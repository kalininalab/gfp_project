"""Create provisional AL figures from the three completed CPU models."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from scripts.transfer_benchmark.plot import COLORS, LABELS, MIX_ORDER, DIRECTIONS, style_axis

ROOT = Path("results/transfer_al_cgreGFP_seed42_46")
NON_AL = Path("results/transfer_cgreGFP_seed42_46")
MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small"]


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ROOT / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def final_rows():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    rows = []
    for mix in protocol["mixes"]:
        for seed in protocol["seeds"]:
            for model in MODELS:
                folder = ROOT / "fits" / mix / f"seed{seed}" / model
                if not (folder / "complete.json").exists():
                    raise RuntimeError(f"Missing completed CPU trajectory: {folder}")
                rows.extend(
                    row for row in read_csv(folder / "metrics.csv")
                    if int(row["round"]) == protocol["rounds"]
                )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        mix, target = row["mix"], row["target"]
        if mix != "artificial" and target == "natural":
            grouped[("ortholog", mix, row["model"])].append(float(row["spearman"]))
        if mix in ("cgre", "artificial"):
            grouped[("peaks", f"{mix}_to_{target}", row["model"])].append(float(row["spearman"]))
    return {
        key: (float(np.mean(values)), float(np.std(values, ddof=1)), values)
        for key, values in grouped.items()
    }


def transfer_panel(scores, experiment, stem, title):
    conditions = MIX_ORDER if experiment == "ortholog" else [f"{a}_to_{b}" for a, b in DIRECTIONS]
    labels = ([name.replace("_", " + ") for name in conditions]
              if experiment == "ortholog" else
              ["Artificial\n→ Artificial", "Artificial\n→ Natural", "Natural\n→ Artificial", "Natural\n→ Natural"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharey=True)
    for axis, model in zip(axes, MODELS):
        values = [scores[(experiment, condition, model)][0] for condition in conditions]
        errors = [scores[(experiment, condition, model)][1] for condition in conditions]
        x = np.arange(len(conditions))
        axis.bar(x, values, yerr=errors, color=COLORS[model], capsize=3)
        for position, condition in enumerate(conditions):
            runs = scores[(experiment, condition, model)][2]
            axis.scatter(position + np.linspace(-0.09, 0.09, len(runs)), runs,
                         s=16, facecolor="white", edgecolor="#354052", zorder=3)
        axis.set_title(LABELS[model], loc="center", color=COLORS[model], fontweight="bold", fontsize=19)
        axis.set_xticks(x, labels, rotation=30 if experiment == "ortholog" else 0,
                        ha="right" if experiment == "ortholog" else "center")
        axis.set_ylabel("Test Spearman ρ")
        style_axis(axis)
    fig.suptitle(title, fontsize=23, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.4)
    save(fig, stem)


def paired_panel(scores, experiment, stem, title):
    non_al = read_csv(NON_AL / "summary.csv")
    conditions = MIX_ORDER if experiment == "ortholog" else [f"{a}_to_{b}" for a, b in DIRECTIONS]
    labels = ([name.replace("_", " + ") for name in conditions]
              if experiment == "ortholog" else
              ["Artificial\n→ Artificial", "Artificial\n→ Natural", "Natural\n→ Artificial", "Natural\n→ Natural"])
    old_experiment = "ortholog_mixtures" if experiment == "ortholog" else "peak_transfer"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharey=True)
    for axis, model in zip(axes, MODELS):
        lookup = {row["condition"]: row for row in non_al if row["experiment"] == old_experiment and row["model"] == model}
        x = np.arange(len(conditions)); width = 0.36
        light = COLORS[model]
        dark = mcolors.to_hex(np.asarray(mcolors.to_rgb(light)) * 0.68)
        old = [float(lookup[condition]["spearman_mean"]) for condition in conditions]
        old_sd = [float(lookup[condition]["spearman_sd"]) for condition in conditions]
        new = [scores[(experiment, condition, model)][0] for condition in conditions]
        new_sd = [scores[(experiment, condition, model)][1] for condition in conditions]
        axis.bar(x - width / 2, old, width, yerr=old_sd, color=light, capsize=3, label="Non-AL")
        axis.bar(x + width / 2, new, width, yerr=new_sd, color=dark, capsize=3, label="AL")
        axis.set_title(LABELS[model], loc="center", color=COLORS[model], fontweight="bold", fontsize=19)
        axis.set_xticks(x, labels, rotation=30 if experiment == "ortholog" else 0,
                        ha="right" if experiment == "ortholog" else "center")
        axis.set_ylabel("Test Spearman ρ")
        style_axis(axis)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(title, fontsize=23, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.4)
    save(fig, stem)


def model_selection(rows):
    chosen = [row for row in rows if row["mix"] == "cgre" and row["target"] == "natural"]
    metrics = [("pearson", "Pearson r"), ("spearman", "Spearman ρ"), ("r2", "R²")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.7), sharey=True)
    x = np.arange(len(MODELS))
    for axis, (metric, title) in zip(axes, metrics):
        groups = [[float(row[metric]) for row in chosen if row["model"] == model] for model in MODELS]
        axis.bar(x, [np.mean(values) for values in groups],
                 yerr=[np.std(values, ddof=1) for values in groups],
                 color=[COLORS[model] for model in MODELS], capsize=4)
        axis.set_xticks(x, [LABELS[model] for model in MODELS], rotation=22, ha="right")
        axis.set_title(title, loc="center", fontsize=17)
        axis.set_ylim(0.5, 1.01)
        style_axis(axis)
    axes[0].set_ylabel("Frozen cgreGFP test score")
    fig.suptitle("Base-model comparison with active learning — CPU models", fontsize=22, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.1)
    save(fig, "model_comparison_al_cpu_preview")


def main():
    rows = final_rows()
    scores = summarize(rows)
    transfer_panel(scores, "ortholog", "ortholog_mixtures_spearman_al_cpu_preview",
                   "Transferability between GFP proteins with active learning — CPU models")
    transfer_panel(scores, "peaks", "artificial_peak_transfer_spearman_al_cpu_preview",
                   "Transferability between natural cgreGFP and artificial peaks with active learning — CPU models")
    paired_panel(scores, "ortholog", "ortholog_mixtures_spearman_non_al_vs_al_cpu_preview",
                 "Transferability between GFP proteins — non-AL vs AL, CPU models")
    paired_panel(scores, "peaks", "artificial_peak_transfer_spearman_non_al_vs_al_cpu_preview",
                 "Natural cgreGFP and artificial peaks — non-AL vs AL, CPU models")
    model_selection(rows)


if __name__ == "__main__":
    main()
