"""Validate pool-size runs and create six publication-ready plots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.pool_mixing.prepare import MODELS, PLATE_ROUNDS, PLATE_SIZES, RAW, REQUIRED_SIZES, SEEDS, TARGETS
from scripts.regression_metrics import metrics
from scripts.transfer_benchmark.common import write_csv


OUTPUT = Path("results/cross_protein_pool_size_seed42_46")
ACTIVE_RAW = RAW
LABELS = {"aubin_1_10_1": "Aubin 1–10–1", "CNN_Jannis_OHE": "CNN on OHE", "mlp_small": "MLP"}
COLORS = {"cgreGFP": "#0072B2", "ppluGFP": "#D55E00", "amacGFP": "#D55E00"}
LINESTYLES = {"aubin_1_10_1": ":", "CNN_Jannis_OHE": "-", "mlp_small": "--"}
MARKERS = {"aubin_1_10_1": "^", "CNN_Jannis_OHE": "o", "mlp_small": "s"}


def read(path):
    with path.open() as handle: return list(csv.DictReader(handle))


def kl_divergence(y_true, y_pred, edges):
    epsilon = np.finfo(float).eps
    true = np.clip(y_true, edges[0] + epsilon, edges[-1] - epsilon)
    pred = np.clip(y_pred, edges[0] + epsilon, edges[-1] - epsilon)
    p = np.histogram(true, bins=edges)[0].astype(float) + 0.5
    q = np.histogram(pred, bins=edges)[0].astype(float) + 0.5
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def distribution_values(target, n_target, model):
    """Return frozen-test truth and predictions pooled across the five seeds."""
    by_gene = {"cgreGFP": {"truth": None, "predicted": []},
               target: {"truth": None, "predicted": []}}
    for seed in SEEDS:
        path = ACTIVE_RAW / "runs" / target / f"n{n_target:04d}" / f"seed{seed}" / model / "predictions.csv"
        groups = {gene: ([], []) for gene in by_gene}
        for row in read(path):
            truth, predicted = groups[row["evaluation_gene"]]
            truth.append(float(row["y_true"]))
            predicted.append(float(row["y_pred"]))
        for gene, (truth, predicted) in groups.items():
            truth = np.asarray(truth)
            if by_gene[gene]["truth"] is None:
                by_gene[gene]["truth"] = truth
            else:
                np.testing.assert_allclose(by_gene[gene]["truth"], truth)
            by_gene[gene]["predicted"].append(np.asarray(predicted))
    for values in by_gene.values():
        values["predicted"] = np.concatenate(values["predicted"])
    return by_gene


def plot_distributions(target, model):
    """Plot true and predicted frozen-test distributions before/after target addition."""
    budgets = (0, 1920)
    data = {budget: distribution_values(target, budget, model) for budget in budgets}
    all_values = [values[k] for budget in budgets for values in data[budget].values()
                  for k in ("truth", "predicted")]
    lower = min(float(values.min()) for values in all_values)
    upper = max(float(values.max()) for values in all_values)
    bins = np.linspace(lower, upper, 51)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), sharex=True, sharey=True,
                             constrained_layout=True)
    for row_index, gene in enumerate(("cgreGFP", target)):
        for column_index, budget in enumerate(budgets):
            axis = axes[row_index, column_index]
            values = data[budget][gene]
            axis.hist(values["truth"], bins=bins, density=True, histtype="step",
                      linewidth=2.5, color="#0072B2", label="True")
            axis.hist(values["predicted"], bins=bins, density=True, histtype="step",
                      linewidth=2.5, color="#D55E00", label="Predicted")
            axis.set_title("No target sequences" if budget == 0 else "1,920 target sequences added")
            axis.text(.03, .94, f"Frozen {gene} test", transform=axis.transAxes,
                      ha="left", va="top", fontsize=12, fontweight="bold")
            axis.grid(axis="y", alpha=.22, linestyle=":")
            axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Density")
    for axis in axes[-1, :]:
        axis.set_xlabel("log10 fluorescence")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(.5, -.02))
    fig.suptitle(f"True and predicted distributions: cgreGFP + {target} ({LABELS[model]})", fontsize=17)
    short = target.replace("GFP", "").lower()
    model_short = LABELS[model].lower().replace(" ", "_")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / "figures" / f"cgre_plus_{short}_{model_short}_true_vs_pred_distributions.{suffix}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--reuse-metrics", action="store_true",
                        help="Replot from an existing all_metrics.csv")
    parser.add_argument("--append-missing-metrics", action="store_true",
                        help="Keep existing model metrics and calculate only selected models that are absent")
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main():
    global OUTPUT, ACTIVE_RAW
    args = parse_args()
    OUTPUT = args.output
    ACTIVE_RAW = args.raw
    selected_models = args.models
    OUTPUT.mkdir(parents=True, exist_ok=True); (OUTPUT / "figures").mkdir(exist_ok=True)
    protocol = json.loads((ACTIVE_RAW / "protocol.json").read_text())
    # Process one prediction file at a time.  Keeping every seeded prediction
    # row in memory would require tens of millions of Python dictionaries.
    metrics_path = OUTPUT / "all_metrics.csv"
    if args.reuse_metrics or (args.append_missing_metrics and metrics_path.exists()):
        records = read(OUTPUT / "all_metrics.csv")
        records = [row for row in records if row["model"] in selected_models]
        for row in records:
            row["n_target_train"] = int(row["n_target_train"])
            row["seed"] = int(row["seed"])
            for name in ("r2", "pearson", "spearman", "kendall_tau", "kl_true_to_pred_bits"):
                row[name] = float(row[name])
        models_to_read = ([] if args.reuse_metrics else
                          [model for model in selected_models
                           if model not in {row["model"] for row in records}])
    else:
        records = []
        models_to_read = selected_models
    if models_to_read:
        edges = {}
        for target in TARGETS:
            for n_target in REQUIRED_SIZES:
                for seed in SEEDS:
                    for model in models_to_read:
                        directory = ACTIVE_RAW / "runs" / target / f"n{n_target:04d}" / f"seed{seed}" / model
                        if not (directory / "complete.json").exists(): raise FileNotFoundError(directory)
                        groups = {}
                        for row in read(directory / "predictions.csv"):
                            group = groups.setdefault(row["evaluation_gene"], ([], []))
                            group[0].append(float(row["y_true"]))
                            group[1].append(float(row["y_pred"]))
                        for gene, (truth, predicted) in groups.items():
                            y = np.asarray(truth); prediction = np.asarray(predicted)
                            if gene not in edges:
                                edges[gene] = np.linspace(float(y.min()), float(y.max()), 51)
                            records.append({"target_pool": target, "n_target_train": n_target,
                                            "seed": seed, "model": model, "evaluation_gene": gene,
                                            **metrics(y, prediction),
                                            "kl_true_to_pred_bits": kl_divergence(y, prediction, edges[gene])})
    summary = []
    for target in TARGETS:
        for plate_size in PLATE_SIZES:
            for round_number in range(0, PLATE_ROUNDS[plate_size] + 1):
                n_target = plate_size * round_number
                for model in selected_models:
                    for gene in ("cgreGFP", target):
                        subset = [row for row in records if row["target_pool"] == target and
                                  row["n_target_train"] == n_target and
                                  row["model"] == model and row["evaluation_gene"] == gene]
                        row = {"target_pool": target, "plate_size": plate_size, "round": round_number,
                               "n_target_train": plate_size * round_number, "model": model,
                               "evaluation_gene": gene, "n_seeds": len(subset)}
                        for metric in ("r2", "spearman", "kl_true_to_pred_bits"):
                            values = np.asarray([item[metric] for item in subset], dtype=float)
                            row[f"{metric}_mean"] = values.mean(); row[f"{metric}_sd"] = values.std(ddof=1)
                        summary.append(row)
    write_csv(OUTPUT / "all_metrics.csv", records); write_csv(OUTPUT / "summary.csv", summary)

    metric_info = {
        "r2": ("R²", "r2"),
        "spearman": ("Spearman correlation", "spearman"),
        "kl_divergence": ("KL divergence, true || predicted (bits) ↓", "kl_true_to_pred_bits"),
    }
    for target in TARGETS:
        short = target.replace("GFP", "").lower()
        for filename_metric, (ylabel, metric) in metric_info.items():
            fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=False, sharey=True, constrained_layout=True)
            handles, labels = None, None
            for axis, plate_size in zip(axes.flat, PLATE_SIZES):
                rounds = list(range(0, PLATE_ROUNDS[plate_size] + 1))
                for model in selected_models:
                    for gene in ("cgreGFP", target):
                        selected = [next(row for row in summary if row["target_pool"] == target and
                                         row["plate_size"] == plate_size and row["round"] == round_number and
                                         row["model"] == model and row["evaluation_gene"] == gene) for round_number in rounds]
                        mean = np.asarray([row[f"{metric}_mean"] for row in selected]); sd = np.asarray([row[f"{metric}_sd"] for row in selected])
                        label = f"{LABELS[model]} — test on {gene}"
                        axis.plot(rounds, mean, color=COLORS[gene], linestyle=LINESTYLES[model],
                                  marker=MARKERS[model], linewidth=2.1, markersize=5, label=label)
                        axis.fill_between(rounds, mean - sd, mean + sd, color=COLORS[gene], alpha=.12)
                axis.set_title(f"Plate size: {plate_size} sequences/round")
                axis.grid(alpha=.22)
                if metric == "r2":
                    axis.set_ylim(-1, 1)
                    axis.axhline(0, color="#555555", lw=.8)
                handles, labels = axis.get_legend_handles_labels()
            for axis in axes.flat[len(PLATE_SIZES):]:
                axis.axis("off")
            axes.flat[-1].legend(handles, labels, loc="center", frameon=False,
                                 ncol=2, fontsize=11)
            fig.supxlabel("Round"); fig.supylabel(ylabel)
            fig.suptitle(f"Adding {target} data to a fixed cgreGFP training set "
                         f"(n = {protocol['base_train_n']:,})", fontsize=16)
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(OUTPUT / "figures" / f"cgre_plus_{short}_{filename_metric}.{suffix}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        for model in selected_models:
            plot_distributions(target, model)

    lines = ["# Cross-protein labelled-pool size experiment", "", "## Question", "",
             f"When a model is trained on a fixed {protocol['base_train_n']:,}-record cgreGFP source pool, how does gradually adding a labelled pool from ppluGFP or amacGFP change prediction on the new protein and on the original cgreGFP protein? This is a supervised pool-size experiment, not active learning: target records are added by a frozen random order, without an acquisition score.", "", "## Design", "",
             f"- Fixed cgreGFP training/validation: {protocol['base_train_n']:,} / {protocol['base_validation_n']:,} records.",
             f"- Equal target training/validation pools: {protocol['target_train_pool_n']:,} / {protocol['target_validation_pool_n']:,} records.",
             "- Plate sizes: 9, 48, 96, 192 and 384 new target training records per round.",
             "- Schedules end near the same cumulative budget: 213 rounds × 9 = 1,917; 40 × 48 = 1,920; 20 × 96 = 1,920; 10 × 192 = 1,920; 5 × 384 = 1,920.",
             "- The cumulative target training size is `plate size × round`; all prefixes are nested within each seed.",
             "- Target validation grows proportionally in the original 60/20 train/validation ratio and is used only for early stopping.",
             f"- Models included in this completed result: {', '.join(LABELS[model] for model in selected_models)}; seeds 42–46.",
             "- Every round is evaluated on the same frozen test: 4,904 cgreGFP records and either 6,281 ppluGFP or 6,703 amacGFP records. Test records never enter training, validation, early stopping or sampling.",
             "- R² is displayed on an ordinary linear scale limited to [−1, 1] for readability. More negative source-only target scores are clipped only in the figure; their complete values remain in the CSV tables.",
             "- KL is D_KL(true || predicted) in bits, using 50 fixed bins over each protein's frozen-test range and a 0.5 pseudocount per bin; lower is better.", "",
             "| Protein | Available training pool | Available validation pool | Frozen test |",
             "|---|---:|---:|---:|",
             "| cgreGFP | 14,709 | 4,903 | 4,904 |",
             "| ppluGFP | 14,709 | 4,903 | 6,281 |",
             "| amacGFP | 14,709 | 4,903 | 6,703 |", "",
             "The full cgreGFP training and validation sets are used throughout. For the target protein, each round uses only the cumulative prefix specified by its plate schedule; the remaining target-pool records are unused, while the complete frozen test remains unchanged.", "", "## Figures", ""]
    for target in TARGETS:
        short = target.replace("GFP", "").lower()
        lines += [f"### cgreGFP + {target}", "",
                  f"Each figure contains five panels, one for each plate size; the x-axis is round and ends when the cumulative target set reaches approximately 1,920 sequences.", "",
                  f"![R²](figures/cgre_plus_{short}_r2.png)", "",
                  f"![Spearman](figures/cgre_plus_{short}_spearman.png)", "",
                  f"![KL divergence](figures/cgre_plus_{short}_kl_divergence.png)", ""]
        lines += ["True-versus-predicted frozen-test distributions before target addition and after 1,920 cumulative target training sequences; predictions are pooled across the five seeds.", ""]
        for model in selected_models:
            model_short = "aubin" if model == "aubin_1_10_1" else LABELS[model].lower().replace(" ", "_")
            lines += [f"#### {LABELS[model]}", "",
                      f"![{LABELS[model]} true and predicted distributions](figures/cgre_plus_{short}_{model_short}_true_vs_pred_distributions.png)", ""]
    lines += ["Machine-readable results: [summary](summary.csv) and [all seeded metrics](all_metrics.csv).", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
