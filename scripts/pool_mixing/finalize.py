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
LABELS = {"CNN_Jannis_OHE": "CNN on OHE", "mlp_small": "MLP"}
COLORS = {"cgreGFP": "#0072B2", "ppluGFP": "#D55E00", "amacGFP": "#D55E00"}
LINESTYLES = {"CNN_Jannis_OHE": "-", "mlp_small": "--"}
MARKERS = {"CNN_Jannis_OHE": "o", "mlp_small": "s"}


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--reuse-metrics", action="store_true",
                        help="Replot from an existing all_metrics.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    selected_models = args.models
    OUTPUT.mkdir(parents=True, exist_ok=True); (OUTPUT / "figures").mkdir(exist_ok=True)
    protocol = json.loads((RAW / "protocol.json").read_text())
    # Process one prediction file at a time.  Keeping every seeded prediction
    # row in memory would require tens of millions of Python dictionaries.
    if args.reuse_metrics:
        records = read(OUTPUT / "all_metrics.csv")
        records = [row for row in records if row["model"] in selected_models]
        for row in records:
            row["n_target_train"] = int(row["n_target_train"])
            row["seed"] = int(row["seed"])
            for name in ("r2", "pearson", "spearman", "kendall_tau", "kl_true_to_pred_bits"):
                row[name] = float(row[name])
    else:
        records = []
        edges = {}
        for target in TARGETS:
            for n_target in REQUIRED_SIZES:
                for seed in SEEDS:
                    for model in selected_models:
                        directory = RAW / "runs" / target / f"n{n_target:04d}" / f"seed{seed}" / model
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
        "r2": ("R² (symmetric-log scale)", "r2"),
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
                    axis.set_yscale("symlog", linthresh=.25, linscale=1)
                    axis.set_yticks([-50, -10, -1, -.25, 0, .25, .5, 1])
                    axis.axhline(0, color="#555555", lw=.8)
                handles, labels = axis.get_legend_handles_labels()
            for axis in axes.flat[len(PLATE_SIZES):]:
                axis.set_visible(False)
            fig.supxlabel("Round"); fig.supylabel(ylabel)
            fig.suptitle(f"Adding {target} data to a fixed cgreGFP training set", fontsize=16)
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.015), frameon=False, ncol=4)
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(OUTPUT / "figures" / f"cgre_plus_{short}_{filename_metric}.{suffix}", dpi=300, bbox_inches="tight")
            plt.close(fig)

    lines = ["# Cross-protein labelled-pool size experiment", "", "## Question", "",
             "When a model is trained on the complete cgreGFP source pool, how does gradually adding an equally sized labelled pool from ppluGFP or amacGFP change prediction on the new protein and on the original cgreGFP protein? This is a supervised pool-size experiment, not active learning: target records are added by a frozen random order, without an acquisition score.", "", "## Design", "",
             f"- Fixed cgreGFP training/validation: {protocol['base_train_n']:,} / {protocol['base_validation_n']:,} records.",
             f"- Equal target training/validation pools: {protocol['target_train_pool_n']:,} / {protocol['target_validation_pool_n']:,} records.",
             "- Plate sizes: 9, 48, 96, 192 and 384 new target training records per round.",
             "- Schedules end near the same cumulative budget: 213 rounds × 9 = 1,917; 40 × 48 = 1,920; 20 × 96 = 1,920; 10 × 192 = 1,920; 5 × 384 = 1,920.",
             "- The cumulative target training size is `plate size × round`; all prefixes are nested within each seed.",
             "- Target validation grows proportionally in the original 60/20 train/validation ratio and is used only for early stopping.",
             f"- Models included in this completed result: {', '.join(LABELS[model] for model in selected_models)}; seeds 42–46.",
             "- Both protein-specific test sets remain frozen and never enter training, validation, early stopping or sampling.",
             "- R² uses a symmetric-log display so the strongly negative source-only target score and the later 0–1 range remain visible on the same panel; the underlying values are unchanged.",
             "- KL is D_KL(true || predicted) in bits, using 50 fixed bins over each protein's frozen-test range and a 0.5 pseudocount per bin; lower is better.", "", "## Figures", ""]
    for target in TARGETS:
        short = target.replace("GFP", "").lower()
        lines += [f"### cgreGFP + {target}", "",
                  f"Each figure contains five panels, one for each plate size; the x-axis is round and ends when the cumulative target set reaches approximately 1,920 sequences.", "",
                  f"![R²](figures/cgre_plus_{short}_r2.png)", "",
                  f"![Spearman](figures/cgre_plus_{short}_spearman.png)", "",
                  f"![KL divergence](figures/cgre_plus_{short}_kl_divergence.png)", ""]
    lines += ["Machine-readable results: [summary](summary.csv) and [all seeded metrics](all_metrics.csv).", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
