"""Build four-model density and true-vs-predicted panels for every main arm."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from scripts.transfer_benchmark.plot import COLORS, LABELS

MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small", "CNN_Jannis_OHE"]
SEEDS = range(42, 47)
OUTPUT = Path("results/target_adaptation_diagnostic_atlas")
SCENARIOS = [
    ("source_only", 0, "source_only__0_target", "Source only", "0 target sequences; no acquisition"),
    ("iterative_fancy", 1, "iterative_fancy__round01__96_target", "Iterative Fancy · round 1", "96 target sequences added"),
    ("iterative_fancy", 10, "iterative_fancy__round10__960_target", "Iterative Fancy · round 10", "960 target sequences added over 10 rounds"),
    ("iterative_random", 1, "iterative_random__round01__96_target", "Iterative Random · round 1", "96 random target sequences added"),
    ("iterative_random", 10, "iterative_random__round10__960_target", "Iterative Random · round 10", "960 random target sequences added over 10 rounds"),
    ("oneshot_fancy", 1, "oneshot_fancy__960_target", "One-shot Fancy", "960 target sequences selected once; no iterative rounds"),
    ("oneshot_random", 1, "oneshot_random__960_target__no_al", "One-shot Random · no AL", "960 random target sequences added once"),
]
NATURAL_PAIRS = [
    "cgreGFP_to_amacGFP", "cgreGFP_to_ppluGFP", "amacGFP_to_cgreGFP",
    "amacGFP_to_ppluGFP", "ppluGFP_to_cgreGFP", "ppluGFP_to_amacGFP",
]
PEAK_PAIRS = ["cgreGFP_to_artificial", "artificial_to_cgreGFP"]


def roots(peaks):
    cpu = Path("results/target_adaptation_peaks_seed42_46" if peaks else
               "results/target_adaptation_seed42_46")
    cnn = Path("results/target_adaptation_peaks_cnn_seed42_46" if peaks else
               "results/target_adaptation_cnn_seed42_46")
    return {model: cnn if model == "CNN_Jannis_OHE" else cpu for model in MODELS}


def read_selected(path):
    wanted = {(arm, str(round_number)): key for arm, round_number, key, _, _ in SCENARIOS}
    values = defaultdict(lambda: ([], []))
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = wanted.get((row["arm"], row["round"]))
            if key:
                values[key][0].append(float(row["y_true"]))
                values[key][1].append(float(row["y_pred"]))
    return {key: (np.asarray(truth), np.asarray(prediction))
            for key, (truth, prediction) in values.items()}


def load_pair(pair, peaks):
    data = defaultdict(list)
    model_roots = roots(peaks)
    for model in MODELS:
        reference_truth = {}
        for seed in SEEDS:
            path = model_roots[model] / "fits" / pair / f"seed{seed}" / model / "predictions.csv"
            selected = read_selected(path)
            for _, _, key, _, _ in SCENARIOS:
                truth, prediction = selected[key]
                if key in reference_truth:
                    np.testing.assert_allclose(reference_truth[key], truth)
                else:
                    reference_truth[key] = truth
                data[(model, key)].append((truth, prediction))
    return data


def pair_label(pair):
    return pair.replace("_to_", " → ").replace("artificial", "artificial peaks")


def decorate(axis, model, correlations):
    axis.set_title(LABELS[model], color=COLORS[model], fontsize=16,
                   fontweight="bold", pad=7)
    axis.text(.97, .93,
              f"Spearman ρ = {correlations.mean():.3f} ± {correlations.std(ddof=1):.3f}",
              transform=axis.transAxes, ha="right", va="top", fontsize=10.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.20, linestyle=":")
    axis.tick_params(labelsize=10.5)


def plot_density(pair, data, scenario):
    _, _, key, label, budget = scenario
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), sharex=True, sharey=True)
    all_values = []
    for model in MODELS:
        for truth, prediction in data[(model, key)]:
            all_values.extend((truth, prediction))
    low = min(array.min() for array in all_values); high = max(array.max() for array in all_values)
    bins = np.linspace(low - .03, high + .03, 48)
    for axis, model in zip(axes.flat, MODELS):
        runs = data[(model, key)]
        truth = runs[0][0]
        prediction = np.concatenate([run[1] for run in runs])
        correlations = np.asarray([spearmanr(run[0], run[1]).statistic for run in runs])
        axis.hist(truth, bins=bins, density=True, histtype="step", linewidth=2.3,
                  color="#0072B2", label="True")
        axis.hist(prediction, bins=bins, density=True, histtype="step", linewidth=2.3,
                  color="#E69F00", label="Predicted")
        decorate(axis, model, correlations)
    for axis in axes[:, 0]: axis.set_ylabel("Density", fontsize=12)
    for axis in axes[-1, :]: axis.set_xlabel("log10 fluorescence", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, .925))
    fig.suptitle(f"{pair_label(pair)} · {label}", fontsize=20, fontweight="bold", y=.995)
    fig.text(.5, .948, f"{budget} · frozen target test · 5 seeds", ha="center", va="top", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, .89), h_pad=1.8, w_pad=1.5)
    path = OUTPUT / "distributions" / f"{pair}__{key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def plot_scatter(pair, data, scenario):
    _, _, key, label, budget = scenario
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), sharex=True, sharey=True)
    all_values = []
    for model in MODELS:
        for truth, prediction in data[(model, key)]: all_values.extend((truth, prediction))
    low = min(array.min() for array in all_values); high = max(array.max() for array in all_values)
    pad = .04 * (high - low); limits = (low - pad, high + pad)
    for axis, model in zip(axes.flat, MODELS):
        runs = data[(model, key)]
        truth = np.concatenate([run[0] for run in runs])
        prediction = np.concatenate([run[1] for run in runs])
        correlations = np.asarray([spearmanr(run[0], run[1]).statistic for run in runs])
        axis.scatter(truth, prediction, s=5, alpha=.075, linewidth=0,
                     color="#0072B2", rasterized=True)
        axis.plot(limits, limits, linestyle="--", linewidth=1.2, color="#333333")
        axis.set_xlim(limits); axis.set_ylim(limits)
        decorate(axis, model, correlations)
    for axis in axes[:, 0]: axis.set_ylabel("Predicted log10 fluorescence", fontsize=12)
    for axis in axes[-1, :]: axis.set_xlabel("True log10 fluorescence", fontsize=12)
    fig.suptitle(f"{pair_label(pair)} · {label}", fontsize=20, fontweight="bold", y=.995)
    fig.text(.5, .958, f"{budget} · frozen target test · 5 seeds", ha="center", va="top", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, .92), h_pad=1.8, w_pad=1.5)
    path = OUTPUT / "dotplots" / f"{pair}__{key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
    return path


def main():
    outputs = []
    for pair in NATURAL_PAIRS + PEAK_PAIRS:
        data = load_pair(pair, pair in PEAK_PAIRS)
        for scenario in SCENARIOS:
            density = plot_density(pair, data, scenario)
            scatter = plot_scatter(pair, data, scenario)
            outputs.append((pair, scenario[3], scenario[4], density, scatter))
    lines = ["# Target-adaptation diagnostic atlas", "",
             "Every panel contains Aubin, Linear, MLP and CNN on OHE on the same frozen target test. Error annotations are mean ± sample SD across seeds 42–46.", "",
             "| Direction | Scenario | Label budget | Density | Dotplot |", "|---|---|---|---|---|"]
    for pair, scenario, budget, density, scatter in outputs:
        lines.append(f"| {pair_label(pair)} | {scenario} | {budget} | [density]({density.relative_to(OUTPUT)}) | [dotplot]({scatter.relative_to(OUTPUT)}) |")
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
