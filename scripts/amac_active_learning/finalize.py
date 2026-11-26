"""Validate, aggregate, and plot the completed amacGFP AL benchmark."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import MODELS, RAW, RESULTS, SEEDS, STRATEGIES, read_csv, write_csv, write_json

LABELS = {"random": "Random", "uncertainty": "Uncertainty", "distance": "Distance",
          "fitness_high": "High fitness", "fitness_low": "Low fitness",
          "uncertainty_fitness_high": "Uncertainty + high fitness",
          "uncertainty_fitness_low": "Uncertainty + low fitness",
          "uncertainty_distance": "Uncertainty + distance"}
COLORS = {"random": "#4D4D4D", "uncertainty": "#0072B2", "distance": "#E69F00",
          "fitness_high": "#009E73", "fitness_low": "#D55E00",
          "uncertainty_fitness_high": "#56B4E9", "uncertainty_fitness_low": "#CC79A7",
          "uncertainty_distance": "#6A3D9A"}
MODEL_LABELS = {"CNN_Jannis_OHE": "CNN on OHE", "mlp_small": "MLP ensemble"}
METRICS = {"spearman": "Spearman ↑", "pearson": "Pearson ↑", "r2": "R² ↑",
           "rmse": "RMSE ↓", "mae": "MAE ↓", "kl_true_to_pred_bits": "KL divergence (bits) ↓",
           "wasserstein": "Wasserstein distance ↓"}


def number(value):
    return np.nan if value in (None, "", "None") else float(value)


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def grouped(rows, keys, value):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[key] for key in keys)].append(number(row[value]))
    return {key: (float(np.nanmean(values)), float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.)
            for key, values in groups.items()}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics = []; diagnostics = []; queries = []; stratified = []
    missing = []
    for model in MODELS:
        for seed in SEEDS:
            for strategy in STRATEGIES:
                run = RAW / model / f"seed{seed}" / strategy
                if not (run / "complete.json").exists(): missing.append(str(run)); continue
                metrics += read_csv(run / "metrics.csv")
                diagnostics += read_csv(run / "acquisition_correlations.csv")
                queries += read_csv(run / "queries.csv.gz")
                stratified += read_csv(run / "stratified_metrics.csv")
    if missing: raise RuntimeError(f"Incomplete trajectories ({len(missing)}): {missing[:3]}")
    write_csv(RESULTS / "all_seeded_metrics.csv", metrics)
    write_csv(RESULTS / "all_acquisition_correlations.csv", diagnostics)
    write_csv(RESULTS / "all_query_diagnostics.csv.gz", queries)
    write_csv(RESULTS / "all_stratified_metrics.csv", stratified)

    figure_root = RESULTS / "figures"
    for metric, ylabel in METRICS.items():
        summary = grouped(metrics, ("model", "strategy", "round"), metric)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, constrained_layout=True)
        for axis, model in zip(axes, MODELS):
            for strategy in STRATEGIES:
                rounds = np.arange(11); values = np.asarray([summary[(model, strategy, str(r))] for r in rounds])
                axis.plot(rounds, values[:, 0], color=COLORS[strategy], lw=2, label=LABELS[strategy])
                axis.fill_between(rounds, values[:, 0] - values[:, 1], values[:, 0] + values[:, 1],
                                  color=COLORS[strategy], alpha=.14, linewidth=0)
            axis.set(title=MODEL_LABELS[model], xlabel="Acquisition round", ylabel=ylabel)
            axis.set_xticks(range(11)); axis.grid(alpha=.2)
            if metric in ("spearman", "pearson", "r2"): axis.set_ylim(-1.05, 1.05)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.08))
        fig.suptitle("amacGFP active-learning acquisition benchmark", fontsize=18, y=1.16)
        save(fig, figure_root / "learning_curves" / metric)

    final_predictions = {}
    for strategy in STRATEGIES:
        fig_scatter, ax_scatter = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        fig_density, ax_density = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
        for axis_s, axis_d, model in zip(ax_scatter, ax_density, MODELS):
            records = []
            for seed in SEEDS:
                records += read_csv(RAW / model / f"seed{seed}" / strategy / "test_predictions_round_10.csv.gz")
            truth = np.asarray([number(r["y_true"]) for r in records]); pred = np.asarray([number(r["y_pred"]) for r in records])
            final_predictions[(model, strategy)] = (truth, pred)
            lower, upper = min(truth.min(), pred.min()), max(truth.max(), pred.max())
            axis_s.scatter(truth, pred, s=5, alpha=.12, color=COLORS[strategy], rasterized=True)
            axis_s.plot([lower, upper], [lower, upper], color="black", ls="--", lw=1.5)
            axis_s.set(title=MODEL_LABELS[model], xlabel="True log10 fitness", ylabel="Predicted log10 fitness",
                       xlim=(lower, upper), ylim=(lower, upper))
            bins = np.linspace(lower, upper, 55)
            axis_d.hist(truth, bins=bins, density=True, histtype="step", lw=2.2, color="#222222", label="True")
            axis_d.hist(pred, bins=bins, density=True, histtype="step", lw=2.2, color=COLORS[strategy], label="Predicted")
            axis_d.set(title=MODEL_LABELS[model], xlabel="log10 fitness", ylabel="Density")
            axis_d.legend(frameon=False)
        fig_scatter.suptitle(f"Final test predictions: {LABELS[strategy]}", fontsize=17)
        fig_density.suptitle(f"Final test fitness distributions: {LABELS[strategy]}", fontsize=17)
        save(fig_scatter, figure_root / "true_vs_predicted" / strategy)
        save(fig_density, figure_root / "distributions" / strategy)

    for relation, xfield, xlabel, folder in (("uncertainty_abs_error", "uncertainty", "Predictive uncertainty", "error_vs_uncertainty"),
                                               ("distance_abs_error", "distance", "Distance to labelled set", "error_vs_distance")):
        for strategy in STRATEGIES:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
            for axis, model in zip(axes, MODELS):
                records = []
                for seed in SEEDS:
                    records += read_csv(RAW / model / f"seed{seed}" / strategy / "pool_scores_round_09.csv.gz")
                x = np.asarray([number(r[xfield]) for r in records]); error = np.asarray([number(r["absolute_error_retrospective"]) for r in records])
                axis.scatter(x, error, s=5, alpha=.12, color=COLORS[strategy], rasterized=True)
                quantiles = np.quantile(x, np.linspace(0, 1, 11)); centers=[]; means=[]
                for left, right in zip(quantiles[:-1], quantiles[1:]):
                    mask = (x >= left) & (x <= right)
                    if mask.any(): centers.append(np.mean(x[mask])); means.append(np.mean(error[mask]))
                axis.plot(centers, means, color="black", marker="o", lw=2, label="Binned mean error")
                axis.text(.03, .97, f"Pearson = {number(next(r['pearson'] for r in diagnostics if r['model']==model and r['strategy']==strategy and r['round']=='9' and r['relationship']==relation)):.2f}\n"
                                      f"Spearman = {number(next(r['spearman'] for r in diagnostics if r['model']==model and r['strategy']==strategy and r['round']=='9' and r['relationship']==relation)):.2f}",
                          transform=axis.transAxes, va="top")
                axis.set(title=MODEL_LABELS[model], xlabel=xlabel, ylabel="Absolute prediction error")
                axis.legend(frameon=False)
            fig.suptitle(f"Round 9 candidate pool: {LABELS[strategy]}", fontsize=17)
            save(fig, figure_root / folder / strategy)

    relationships = ["uncertainty_abs_error", "distance_abs_error", "uncertainty_distance",
                     "predicted_fitness_uncertainty", "predicted_fitness_distance"]
    rel_labels = ["U–error", "D–error", "U–D", "Fitness–U", "Fitness–D"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True)
    for axis, model in zip(axes, MODELS):
        matrix = np.asarray([[np.nanmean([number(r["spearman"]) for r in diagnostics
                    if r["model"] == model and r["strategy"] == strategy and r["relationship"] == relationship])
                    for relationship in relationships] for strategy in STRATEGIES])
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axis.set(title=MODEL_LABELS[model], xticks=range(len(rel_labels)), xticklabels=rel_labels,
                 yticks=range(len(STRATEGIES)), yticklabels=[LABELS[s] for s in STRATEGIES])
        axis.tick_params(axis="x", rotation=35)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]): axis.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes, label="Mean Spearman correlation", shrink=.8)
    fig.suptitle("Relationships among acquisition components and retrospective error", fontsize=17)
    save(fig, figure_root / "correlation_heatmaps" / "acquisition_components")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for axis, model in zip(axes, MODELS):
        all_values = [number(r["y_true_retrospective"]) for r in queries if r["model"] == model]
        bins = np.linspace(min(all_values), max(all_values), 50)
        for strategy in STRATEGIES:
            values = [number(r["y_true_retrospective"]) for r in queries if r["model"] == model and r["strategy"] == strategy]
            hist, edges = np.histogram(values, bins=bins, density=True)
            axis.plot((edges[:-1] + edges[1:]) / 2, hist, lw=2, color=COLORS[strategy], label=LABELS[strategy])
        axis.set(title=MODEL_LABELS[model], xlabel="True acquired log10 fitness", ylabel="Density")
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.08))
    fig.suptitle("True fitness distributions of acquired amacGFP sequences", fontsize=17, y=1.16)
    save(fig, figure_root / "acquired_batches" / "true_fitness_distributions")

    batch_groups = defaultdict(list)
    for row in queries: batch_groups[(row["model"], row["strategy"], row["seed"], row["round"])].append(row)
    batch_rows = []
    for key, values in batch_groups.items():
        truth = np.asarray([number(v["y_true_retrospective"]) for v in values]); prediction = np.asarray([number(v["y_pred_at_selection"]) for v in values])
        batch_rows.append({"model": key[0], "strategy": key[1], "seed": key[2], "round": key[3],
                           "n": len(values), "true_mean": truth.mean(), "true_sd": truth.std(ddof=1),
                           "predicted_mean": prediction.mean(), "predicted_sd": prediction.std(ddof=1),
                           "operational_bright_fraction": np.mean([v["operational_bright"] == "True" for v in values]),
                           "mean_mutations_from_WT": np.mean([number(v["n_mutations"]) for v in values]),
                           "mean_pairwise_hamming": number(values[0]["batch_mean_pairwise_hamming"])})
    write_csv(RESULTS / "acquired_batch_summary.csv", batch_rows)

    for stratifier, labels in (("true_fitness_quartile", ["Q1", "Q2", "Q3", "Q4"]),
                               ("mutations_from_WT", ["0-1", "2", "3", "4", "5+"])):
        for metric in ("spearman", "r2", "rmse"):
            subset = [row for row in stratified if row["stratifier"] == stratifier and row["round"] == "10"]
            summary = grouped(subset, ("model", "strategy", "stratum"), metric)
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
            for axis, model in zip(axes, MODELS):
                for strategy in STRATEGIES:
                    available = [label for label in labels if (model, strategy, label) in summary]
                    values = np.asarray([summary[(model, strategy, label)] for label in available])
                    axis.plot(available, values[:, 0], marker="o", lw=1.8, color=COLORS[strategy], label=LABELS[strategy])
                    axis.fill_between(available, values[:, 0]-values[:, 1], values[:, 0]+values[:, 1], color=COLORS[strategy], alpha=.12)
                axis.set(title=MODEL_LABELS[model], xlabel=stratifier.replace("_", " "), ylabel=METRICS[metric]); axis.grid(alpha=.2)
                if metric in ("spearman", "r2"): axis.set_ylim(-1.05, 1.05)
            handles, legend_labels = axes[0].get_legend_handles_labels(); fig.legend(handles, legend_labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.08))
            fig.suptitle("Final held-out test performance by " + stratifier.replace("_", " "), fontsize=17, y=1.16)
            save(fig, figure_root / "stratified_performance" / f"{stratifier}_{metric}")

    report = ["# Preliminary active-learning benchmark on amacGFP", "",
              "This benchmark compares CNN-on-OHE MC-dropout and a ten-member MLP ensemble across eight acquisition strategies. It uses three seeds, a fixed 96-sequence plate, ten acquisition rounds, matched splits within each seed, and a frozen held-out test.", "",
              "## Main outputs", "",
              "- [Learning curves](figures/learning_curves/) for Spearman, Pearson, R², RMSE, MAE, KL divergence and Wasserstein distance.",
              "- [True versus predicted plots](figures/true_vs_predicted/) and [fitness distributions](figures/distributions/).",
              "- [Error versus uncertainty](figures/error_vs_uncertainty/) and [error versus distance](figures/error_vs_distance/).",
              "- [Correlation heatmaps](figures/correlation_heatmaps/) and [acquired-sequence distributions](figures/acquired_batches/).",
              "- [Performance by true fitness and mutation count](figures/stratified_performance/).", "",
              "The raw per-seed pool scores, predictions, selected sequences, configurations and resumable checkpoints are stored outside Git at `" + str(RAW) + "`. True pool fitness appears there only as a retrospective diagnostic; acquisition scores are computed exclusively from model predictions, uncertainty and/or hidden-space distance.", "",
              "[Frozen protocol](protocol.json) · [all seeded metrics](all_seeded_metrics.csv) · [acquisition correlations](all_acquisition_correlations.csv) · [batch summary](acquired_batch_summary.csv)", ""]
    (RESULTS / "REPORT.md").write_text("\n".join(report))
    write_json(RESULTS / "finalization.json", {"trajectories": len(MODELS)*len(SEEDS)*len(STRATEGIES), "complete": True})


if __name__ == "__main__": main()
