"""Build the canonical target-adaptation report, metric tables and R2 plots."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.target_adaptation.finalize import ARMS, ARM_COLORS, ARM_LABELS
from scripts.transfer_benchmark.common import read_csv, write_csv
from scripts.transfer_benchmark.plot import COLORS, LABELS, style_axis

MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small", "CNN_Jannis_OHE"]
SEEDS = list(range(42, 47))
NATURAL_PAIRS = [
    "cgreGFP_to_amacGFP", "cgreGFP_to_ppluGFP", "amacGFP_to_cgreGFP",
    "amacGFP_to_ppluGFP", "ppluGFP_to_cgreGFP", "ppluGFP_to_amacGFP",
]
PEAK_PAIRS = ["cgreGFP_to_artificial", "artificial_to_cgreGFP"]
METRICS = ["mse", "rmse", "r2", "pearson", "spearman", "kendall_tau"]
OUTPUT = Path("results/target_adaptation_complete_report")
ALL_SCENARIOS = [
    ("source_only", 0, "No AL", "#999999"),
    ("iterative_fancy", 1, "AL Fancy\n96", "#E69F00"),
    ("iterative_fancy", 10, "AL Fancy\n960", "#D55E00"),
    ("iterative_random", 1, "AL Random\n96", "#56B4E9"),
    ("iterative_random", 10, "AL Random\n960", "#0072B2"),
    ("oneshot_fancy", 1, "One-shot Fancy\n960", "#CC79A7"),
    ("oneshot_random", 1, "One-shot Random\n960", "#009E73"),
]


def root(model, peaks):
    if model == "CNN_Jannis_OHE":
        return Path("results/target_adaptation_peaks_cnn_seed42_46" if peaks else
                    "results/target_adaptation_cnn_seed42_46")
    return Path("results/target_adaptation_peaks_seed42_46" if peaks else
                "results/target_adaptation_seed42_46")


def collect():
    rows = []
    for pair in NATURAL_PAIRS + PEAK_PAIRS:
        peaks = pair in PEAK_PAIRS
        for model in MODELS:
            for seed in SEEDS:
                path = root(model, peaks) / "fits" / pair / f"seed{seed}" / model / "metrics.csv"
                for row in read_csv(path):
                    rows.append({"domain": "natural_artificial" if peaks else "natural_proteins",
                                 "pair": pair, "model": model, "seed": seed, **row})
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["domain"], row["pair"], row["model"], row["arm"],
               int(row["round"]), int(row["n_target"]), int(row["n"]))
        grouped[key].append(row)
    output = []
    for key, values in grouped.items():
        domain, pair, model, arm, round_number, n_target, n_test = key
        result = {"domain": domain, "pair": pair, "model": model, "arm": arm,
                  "round": round_number, "n_target": n_target, "n_test": n_test,
                  "n_seeds": len(values)}
        for metric in METRICS:
            numbers = np.asarray([float(row[metric]) if row[metric] != "" else np.nan
                                  for row in values])
            valid = numbers[np.isfinite(numbers)]
            result[f"{metric}_n"] = len(valid)
            result[f"{metric}_mean"] = "" if not len(valid) else float(valid.mean())
            result[f"{metric}_sd"] = "" if len(valid) < 2 else float(valid.std(ddof=1))
        output.append(result)
    return sorted(output, key=lambda x: (x["domain"], x["pair"], x["model"], x["arm"], x["round"]))


def plot_r2(summary, pairs, stem, title):
    final = [row for row in summary if row["pair"] in pairs and
             (row["arm"].startswith("oneshot") or
              (row["arm"].startswith("iterative") and row["round"] == 10))]
    lookup = {(row["pair"], row["model"], row["arm"]): row for row in final}
    fig, axes = plt.subplots(4, 1, figsize=(15.5, 17.5), sharex=True, sharey=True)
    x = np.arange(len(pairs)); width = .19
    for axis, model in zip(axes, MODELS):
        for position, (arm, label, color) in enumerate(zip(ARMS, ARM_LABELS, ARM_COLORS)):
            means = [lookup[(pair, model, arm)]["r2_mean"] for pair in pairs]
            errors = [lookup[(pair, model, arm)]["r2_sd"] for pair in pairs]
            axis.bar(x + (position - 1.5) * width, means, width, yerr=errors,
                     capsize=3, color=color, label=label)
        axis.axhline(0, color="#333333", linewidth=.9)
        axis.set_title(LABELS[model], color=COLORS[model], fontweight="bold", fontsize=17)
        axis.set_ylabel("Target-test R²")
        style_axis(axis)
    labels = [pair.replace("_to_", " → ").replace("GFP", "") for pair in pairs]
    axes[-1].set_xticks(x, labels, rotation=22, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, .966))
    fig.suptitle(title, fontsize=22, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=1.3)
    figures = OUTPUT / "barplots"; figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def plot_all_scenarios(summary, pairs, metric, stem, title):
    lookup = {(row["pair"], row["model"], row["arm"], row["round"]): row for row in summary}
    fig, axes = plt.subplots(4, 1, figsize=(16.5, 18), sharex=True, sharey=True)
    x = np.arange(len(pairs)); width = .115
    for axis, model in zip(axes, MODELS):
        for position, (arm, round_number, label, color) in enumerate(ALL_SCENARIOS):
            rows = [lookup[(pair, model, arm, round_number)] for pair in pairs]
            means = [row[f"{metric}_mean"] for row in rows]
            errors = [0 if row[f"{metric}_sd"] == "" else row[f"{metric}_sd"] for row in rows]
            axis.bar(x + (position - 3) * width, means, width, yerr=errors,
                     capsize=2, color=color, label=label)
        if metric == "r2":
            axis.axhline(0, color="#333333", linewidth=.9)
            axis.set_yscale("symlog", linthresh=1, linscale=1)
        axis.set_title(LABELS[model], color=COLORS[model], fontweight="bold", fontsize=17)
        axis.set_ylabel("Target-test R² (symlog)" if metric == "r2" else "Target-test Spearman ρ")
        style_axis(axis)
    pair_labels = [pair.replace("_to_", " → ").replace("GFP", "") for pair in pairs]
    axes[-1].set_xticks(x, pair_labels, rotation=22, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(.5, .974), fontsize=10)
    fig.suptitle(title, fontsize=22, y=.997)
    if metric == "r2":
        fig.text(.99, .008, "Symmetric-log y-axis retains extreme negative source-only R² values.",
                 ha="right", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .93), h_pad=1.3)
    figures = OUTPUT / "barplots"; figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(figures / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def figure_index():
    rows = []
    atlas = Path("results/target_adaptation_diagnostic_atlas")
    for density in sorted((atlas / "distributions").glob("*.png")):
        dotplot = atlas / "dotplots" / density.name
        rows.append({"filename_stem": density.stem,
                     "distribution": density.as_posix(), "dotplot": dotplot.as_posix()})
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    summary = summarize(rows)
    write_csv(OUTPUT / "metrics_per_seed.csv", rows)
    write_csv(OUTPUT / "metrics_summary.csv", summary)
    write_csv(OUTPUT / "figure_index.csv", figure_index())
    plot_r2(summary, NATURAL_PAIRS, "target_adaptation_proteins_r2",
            "Target adaptation between GFP proteins — R²")
    plot_r2(summary, PEAK_PAIRS, "target_adaptation_peaks_r2",
            "Target adaptation between natural cgreGFP and artificial peaks — R²")
    plot_all_scenarios(summary, NATURAL_PAIRS, "r2", "all_scenarios_proteins_r2",
                       "All target-adaptation scenarios between GFP proteins — R²")
    plot_all_scenarios(summary, PEAK_PAIRS, "r2", "all_scenarios_peaks_r2",
                       "All target-adaptation scenarios: natural cgreGFP and artificial peaks — R²")
    plot_all_scenarios(summary, NATURAL_PAIRS, "spearman", "all_scenarios_proteins_spearman",
                       "All target-adaptation scenarios between GFP proteins — Spearman")
    plot_all_scenarios(summary, PEAK_PAIRS, "spearman", "all_scenarios_peaks_spearman",
                       "All target-adaptation scenarios: natural cgreGFP and artificial peaks — Spearman")
    (OUTPUT / "README.md").write_text("""# Complete target-adaptation results

This is the canonical entry point for the target-adaptation experiments. It links the complete metric tables and figures without duplicating the underlying predictions or per-round artifacts.

## What is covered

- 8 directed transfer scenarios: 6 between natural GFP proteins and 2 between natural cgreGFP and artificial peaks;
- Aubin, Linear, MLP and CNN on OHE;
- seeds 42–46 and one shared frozen target test per direction;
- source-only, iterative Fancy, iterative Random, one-shot Fancy and one-shot Random;
- iterative metrics for every round from 96 through 960 added target sequences;
- MSE, RMSE, R², Pearson, Spearman and Kendall tau.

## Metric tables

- [Every per-seed result](metrics_per_seed.csv): one row per direction, model, seed, arm and evaluated round.
- [Mean and sample SD across five seeds](metrics_summary.csv): includes every metric, especially `r2_mean` and `r2_sd`.
- [Figure index](figure_index.csv): machine-readable paths matching every distribution with its dotplot.

## Final-budget bar plots

These compare the four 960-target-sequence arms: one-shot Random, iterative Random, one-shot Fancy and iterative Fancy.

### Spearman

- [All four models between GFP proteins](../target_adaptation_combined_seed42_46/target_adaptation_all_models.png)
- [All four models between natural cgreGFP and artificial peaks](../target_adaptation_combined_seed42_46/target_adaptation_peaks_all_models.png)

### R²

![R² between GFP proteins](barplots/target_adaptation_proteins_r2.png)

![R² between natural cgreGFP and artificial peaks](barplots/target_adaptation_peaks_r2.png)

## All scenarios, including no AL and 96 target sequences

These figures include source-only models with zero target sequences, both
iterative methods after the first 96-sequence round and after all ten rounds,
and both one-shot 960-sequence controls.

### R² — all scenarios

![All scenarios between GFP proteins — R²](barplots/all_scenarios_proteins_r2.png)

![All scenarios between natural cgreGFP and artificial peaks — R²](barplots/all_scenarios_peaks_r2.png)

### Spearman — all scenarios

![All scenarios between GFP proteins — Spearman](barplots/all_scenarios_proteins_spearman.png)

![All scenarios between natural cgreGFP and artificial peaks — Spearman](barplots/all_scenarios_peaks_spearman.png)

## Prediction diagnostics

- [All 56 true/predicted distribution panels](../target_adaptation_diagnostic_atlas/distributions/)
- [All 56 true-vs-predicted dotplot panels](../target_adaptation_diagnostic_atlas/dotplots/)
- [Human-readable scenario index](../target_adaptation_diagnostic_atlas/README.md)

Large raw predictions, model checkpoints and per-round diagnostics remain under the experiment `fits/` directories, linked to `/data/users/akolchina/gfp_project_artifacts`.
""")


if __name__ == "__main__":
    main()
