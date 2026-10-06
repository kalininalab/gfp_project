"""Plot AL model selection and matched random-versus-fancy transfer results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.transfer_benchmark.common import MIXES, digest, read_csv, write_csv
from scripts.transfer_benchmark.plot import COLORS, LABELS, MIX_ORDER, DIRECTIONS, style_axis

MODEL_ORDER = ["aubin_1_10_1", "aubin_linear", "mlp_small", "CNN_Jannis_OHE"]
METRICS = [("pearson", "Pearson r"), ("spearman", "Spearman ρ"), ("r2", "R²")]


def save(fig, directory, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def completed_rows(directory, models):
    protocol = json.loads((directory / "protocol.json").read_text())
    rows = []
    for mix in MIXES:
        for seed in protocol["seeds"]:
            for model in models:
                folder = directory / "fits" / mix / f"seed{seed}" / model
                complete = json.loads((folder / "complete.json").read_text())
                if complete["protocol_sha256"] != digest(directory / "protocol.json"):
                    raise ValueError(f"Protocol mismatch: {folder}")
                if complete["metrics_sha256"] != digest(folder / "metrics.csv"):
                    raise ValueError(f"Metrics mismatch: {folder}")
                rows.extend(read_csv(folder / "metrics.csv"))
    return protocol, rows


def plot_al_model_selection(fancy_directory):
    protocol, rows = completed_rows(fancy_directory, MODEL_ORDER)
    final_round = protocol["rounds"]
    chosen = [
        row for row in rows
        if row["mix"] == "cgre"
        and row["target"] == "natural"
        and int(row["round"]) == final_round
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    for axis, (metric, title) in zip(axes, METRICS):
        groups = [[float(row[metric]) for row in chosen if row["model"] == model] for model in MODEL_ORDER]
        axis.bar(x, [np.mean(values) for values in groups],
                 yerr=[np.std(values, ddof=1) for values in groups],
                 color=[COLORS[model] for model in MODEL_ORDER], capsize=4)
        axis.set_xticks(x, [LABELS[model] for model in MODEL_ORDER], rotation=25, ha="right")
        axis.set_title(title, loc="center", fontsize=17)
        axis.set_ylim(0.75, 1.01)
        style_axis(axis)
    axes[0].set_ylabel("Frozen cgreGFP test score")
    fig.suptitle("Base-model comparison with active learning", fontsize=22, y=1.01)
    fig.tight_layout(w_pad=1.2)
    save(fig, fancy_directory, "model_comparison_al")


def aggregate_final(rows, protocol):
    grouped = defaultdict(list)
    for row in rows:
        if int(row["round"]) != protocol["rounds"]:
            continue
        mix, target = row["mix"], row["target"]
        if mix != "artificial" and target == "natural":
            key = ("ortholog_mixtures", mix)
        elif mix in ("cgre", "artificial"):
            key = ("peak_transfer", f"{mix}_to_{target}")
        else:
            continue
        grouped[key].append(float(row["spearman"]))
    return {key: (float(np.mean(values)), float(np.std(values, ddof=1))) for key, values in grouped.items()}


def sampling_plot(random_scores, fancy_scores, experiment, directory):
    conditions = MIX_ORDER if experiment == "ortholog_mixtures" else [f"{a}_to_{b}" for a, b in DIRECTIONS]
    labels = ([name.replace("_", " + ") for name in conditions] if experiment == "ortholog_mixtures"
              else ["Artificial\n→ Artificial", "Artificial\n→ Natural", "Natural\n→ Artificial", "Natural\n→ Natural"])
    x = np.arange(len(conditions)); width = 0.36
    fig, axis = plt.subplots(figsize=(12.8, 5.8))
    r_mean = [random_scores[(experiment, condition)][0] for condition in conditions]
    r_sd = [random_scores[(experiment, condition)][1] for condition in conditions]
    f_mean = [fancy_scores[(experiment, condition)][0] for condition in conditions]
    f_sd = [fancy_scores[(experiment, condition)][1] for condition in conditions]
    axis.bar(x - width / 2, r_mean, width, yerr=r_sd, capsize=4, color="#56B4E9", label="Random")
    axis.bar(x + width / 2, f_mean, width, yerr=f_sd, capsize=4, color="#0072B2", label='Acquisition score ("Fancy")')
    axis.set_xticks(x, labels, rotation=25 if experiment == "ortholog_mixtures" else 0, ha="right" if experiment == "ortholog_mixtures" else "center")
    axis.set_ylabel("Test Spearman ρ")
    axis.set_title("Random versus acquisition-score sampling", loc="center", fontsize=21)
    axis.legend(frameon=False, ncol=2, fontsize=13)
    style_axis(axis)
    fig.tight_layout()
    stem = "sampling_ortholog_transfer" if experiment == "ortholog_mixtures" else "sampling_peak_transfer"
    save(fig, directory, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fancy", type=Path, default=Path("results/transfer_al_cgreGFP_seed42_46"))
    parser.add_argument("--random", type=Path, default=Path("results/transfer_random_cgreGFP_seed42_46"))
    args = parser.parse_args()
    plot_al_model_selection(args.fancy)
    random_protocol, random_rows = completed_rows(args.random, ["CNN_Jannis_OHE"])
    fancy_protocol, fancy_rows = completed_rows(args.fancy, ["CNN_Jannis_OHE"])
    if random_protocol["test_records_sha256"] != fancy_protocol["test_records_sha256"]:
        raise ValueError("Random/fancy test manifests differ")
    random_scores = aggregate_final(random_rows, random_protocol)
    fancy_scores = aggregate_final(fancy_rows, fancy_protocol)
    sampling_plot(random_scores, fancy_scores, "ortholog_mixtures", args.random)
    sampling_plot(random_scores, fancy_scores, "peak_transfer", args.random)
    write_csv(args.random / "per_round_scores.csv", random_rows)


if __name__ == "__main__":
    main()
