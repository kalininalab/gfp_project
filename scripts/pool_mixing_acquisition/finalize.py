"""Overlay matched random and acquisition-sampled pool-size trajectories."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.pool_mixing.prepare import PLATE_ROUNDS, PLATE_SIZES, SEEDS, TARGETS
from scripts.regression_metrics import metrics
from scripts.transfer_benchmark.common import write_csv

MODELS = ("aubin_1_10_1", "mlp_small", "CNN_Jannis_OHE")
LABELS = {"aubin_1_10_1": "Aubin 1–10–1", "mlp_small": "MLP", "CNN_Jannis_OHE": "CNN on OHE"}
LINESTYLES = {"aubin_1_10_1": ":", "mlp_small": "--", "CNN_Jannis_OHE": "-"}
MARKERS = {"aubin_1_10_1": "^", "mlp_small": "s", "CNN_Jannis_OHE": "o"}
COLORS = {("random", "cgreGFP"): "#0072B2", ("random", "target"): "#D55E00",
          ("acquisition", "cgreGFP"): "#CC79A7", ("acquisition", "target"): "#6A51A3"}


def models_for_plate(plate_size):
    return MODELS[:-1] if plate_size == 9 else MODELS


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def acquisition_records(raw):
    records = []
    for target in TARGETS:
        for plate_size in PLATE_SIZES:
            for seed in SEEDS:
                for model in models_for_plate(plate_size):
                    directory = raw / "runs" / target / f"plate{plate_size:03d}" / f"seed{seed}" / model
                    if not (directory / "complete.json").exists():
                        raise FileNotFoundError(f"Incomplete acquisition trajectory: {directory}")
                    for round_number in range(PLATE_ROUNDS[plate_size] + 1):
                        records.extend(read(directory / f"metrics_round_{round_number:03d}.csv"))
    return records


def summarize_acquisition(records):
    summary = []
    for target in TARGETS:
        for plate_size in PLATE_SIZES:
            for round_number in range(PLATE_ROUNDS[plate_size] + 1):
                for model in models_for_plate(plate_size):
                    for gene in ("cgreGFP", target):
                        subset = [row for row in records if row["target_pool"] == target and
                                  int(row["plate_size"]) == plate_size and int(row["round"]) == round_number and
                                  row["model"] == model and row["evaluation_gene"] == gene]
                        if len(subset) != len(SEEDS):
                            raise ValueError(f"Expected five seeds for {target}, {plate_size}, {round_number}, {model}, {gene}")
                        row = {"method": "acquisition", "target_pool": target,
                               "plate_size": plate_size, "round": round_number,
                               "n_target_train": plate_size * round_number, "model": model,
                               "evaluation_gene": gene, "n_seeds": len(subset)}
                        for metric in ("r2", "spearman", "kl_true_to_pred_bits"):
                            values = np.asarray([float(item[metric]) for item in subset])
                            row[f"{metric}_mean"] = values.mean()
                            row[f"{metric}_sd"] = values.std(ddof=1)
                        summary.append(row)
    return summary


def normalize_random(rows):
    result = []
    for row in rows:
        normalized = {**row, "method": "random"}
        for field in ("plate_size", "round", "n_target_train", "n_seeds"):
            normalized[field] = int(normalized[field])
        for field in ("r2_mean", "r2_sd", "spearman_mean", "spearman_sd",
                      "kl_true_to_pred_bits_mean", "kl_true_to_pred_bits_sd"):
            normalized[field] = float(normalized[field])
        result.append(normalized)
    return result


def kl_divergence(y_true, y_predicted):
    edges = np.linspace(float(np.min(y_true)), float(np.max(y_true)), 51)
    epsilon = np.finfo(float).eps
    truth = np.clip(y_true, edges[0] + epsilon, edges[-1] - epsilon)
    predicted = np.clip(y_predicted, edges[0] + epsilon, edges[-1] - epsilon)
    p = np.histogram(truth, bins=edges)[0].astype(float) + 0.5
    q = np.histogram(predicted, bins=edges)[0].astype(float) + 0.5
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def random_cnn_seeded(reference_raw):
    """Read the matched CNN random fits at each unique cumulative size."""
    records = []
    sizes = list(range(0, 1921, 48))
    for target in TARGETS:
        for n_target in sizes:
            for seed in SEEDS:
                directory = reference_raw / "runs" / target / f"n{n_target:04d}" / f"seed{seed}" / "CNN_Jannis_OHE"
                if not (directory / "complete.json").exists():
                    raise FileNotFoundError(f"Incomplete random CNN fit: {directory}")
                groups = {}
                for row in read(directory / "predictions.csv"):
                    truth, predicted = groups.setdefault(row["evaluation_gene"], ([], []))
                    truth.append(float(row["y_true"])); predicted.append(float(row["y_pred"]))
                for gene, (truth, predicted) in groups.items():
                    y = np.asarray(truth); prediction = np.asarray(predicted)
                    records.append({"method": "random", "target_pool": target,
                                    "n_target_train": n_target, "seed": seed,
                                    "model": "CNN_Jannis_OHE", "evaluation_gene": gene,
                                    **metrics(y, prediction),
                                    "kl_true_to_pred_bits": kl_divergence(y, prediction)})
    return records


def expand_random_cnn(records):
    result = []
    for target in TARGETS:
        for plate_size in PLATE_SIZES:
            if plate_size == 9:
                continue
            for round_number in range(PLATE_ROUNDS[plate_size] + 1):
                n_target = plate_size * round_number
                for gene in ("cgreGFP", target):
                    subset = [row for row in records if row["target_pool"] == target and
                              row["n_target_train"] == n_target and row["evaluation_gene"] == gene]
                    row = {"method": "random", "target_pool": target, "plate_size": plate_size,
                           "round": round_number, "n_target_train": n_target,
                           "model": "CNN_Jannis_OHE", "evaluation_gene": gene, "n_seeds": len(subset)}
                    for metric in ("r2", "spearman", "kl_true_to_pred_bits"):
                        values = np.asarray([float(item[metric]) for item in subset])
                        row[f"{metric}_mean"] = values.mean(); row[f"{metric}_sd"] = values.std(ddof=1)
                    result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--acquisition-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.acquisition_raw / "protocol.json").read_text())
    random_summary = [row for row in normalize_random(read(args.output / "summary.csv"))
                      if row["model"] != "CNN_Jannis_OHE"]
    random_cnn = random_cnn_seeded(args.reference_raw)
    random_summary.extend(expand_random_cnn(random_cnn))
    acquisition = acquisition_records(args.acquisition_raw)
    acquisition_summary = summarize_acquisition(acquisition)
    combined = random_summary + acquisition_summary
    write_csv(args.output / "acquisition_all_seeded_metrics.csv", acquisition)
    write_csv(args.output / "random_cnn_all_seeded_metrics.csv", random_cnn)
    write_csv(args.output / "summary.csv", random_summary)
    write_csv(args.output / "summary_random_vs_acquisition.csv", combined)

    metric_info = {"r2": ("R²", "r2"), "spearman": ("Spearman correlation", "spearman"),
                   "kl_divergence": ("KL divergence, true || predicted (bits) ↓", "kl_true_to_pred_bits")}
    for target in TARGETS:
        short = target.replace("GFP", "").lower()
        for filename_metric, (ylabel, metric) in metric_info.items():
            fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True, constrained_layout=True)
            handles = labels = None
            for axis, plate_size in zip(axes.flat, PLATE_SIZES):
                rounds = list(range(PLATE_ROUNDS[plate_size] + 1))
                for model in models_for_plate(plate_size):
                    for method in ("random", "acquisition"):
                        for gene in ("cgreGFP", target):
                            selected = [next(row for row in combined if row["method"] == method and
                                             row["target_pool"] == target and row["plate_size"] == plate_size and
                                             row["round"] == round_number and row["model"] == model and
                                             row["evaluation_gene"] == gene) for round_number in rounds]
                            mean = np.asarray([row[f"{metric}_mean"] for row in selected], dtype=float)
                            sd = np.asarray([row[f"{metric}_sd"] for row in selected], dtype=float)
                            color = COLORS[(method, "cgreGFP" if gene == "cgreGFP" else "target")]
                            label = f"{LABELS[model]} — {method} — test on {gene}"
                            axis.plot(rounds, mean, color=color, linestyle=LINESTYLES[model],
                                      marker=MARKERS[model], linewidth=2.0, markersize=4.5, label=label)
                            axis.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=.10)
                axis.set_title(f"Plate size: {plate_size} sequences/round")
                axis.grid(alpha=.22)
                if metric == "r2":
                    axis.set_ylim(-1, 1); axis.axhline(0, color="#555555", lw=.8)
                handles, labels = axis.get_legend_handles_labels()
            axes.flat[-1].axis("off")
            axes.flat[-1].legend(handles, labels, loc="center", frameon=False, ncol=2, fontsize=8.5)
            fig.supxlabel("Round"); fig.supylabel(ylabel)
            fig.suptitle(f"Random versus acquisition sampling: cgreGFP + {target} "
                         f"(cgreGFP train n = {protocol['base_train_n']:,})", fontsize=16)
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(args.output / "figures" / f"cgre_plus_{short}_{filename_metric}.{suffix}",
                            dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    report_path = args.output / "REPORT.md"
    report = report_path.read_text()
    marker = "## Random versus acquisition sampling"
    if marker in report:
        report = report.split(marker)[0].rstrip() + "\n\n"
    report += (f"{marker}\n\nThe figures above now overlay the original frozen-random trajectories with "
               "acquisition-sampled trajectories. Blue/orange show random sampling on the cgreGFP/target "
               "tests; pink/violet show acquisition sampling on the same tests. At each round the acquisition "
               "score is `0.62 × scaled hidden-space distance + 0.38 × scaled predictive variance`. Aubin and "
               "MLP use five-member ensembles for variance; CNN on OHE uses 25 MC-dropout predictions. All "
               "three primary models are reinitialized and retrained on the complete cumulative labelled set "
               "at every round. CNN on OHE is evaluated for plate sizes 48, 96, 192 and 384; plate size 9 is "
               "retained for Aubin and MLP only. Validation and frozen tests are identical to the random arm, and test labels "
               "never enter acquisition.\n\nMachine-readable comparison: "
               "[summary_random_vs_acquisition.csv](summary_random_vs_acquisition.csv) and "
               "[acquisition_all_seeded_metrics.csv](acquisition_all_seeded_metrics.csv).\n")
    report_path.write_text(report)


if __name__ == "__main__":
    main()
