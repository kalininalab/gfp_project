"""Plot CPU and CNN target-adaptation results as one four-model panel."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.target_adaptation.finalize import ARMS, ARM_COLORS, ARM_LABELS
from scripts.transfer_benchmark.common import read_csv
from scripts.transfer_benchmark.plot import COLORS, LABELS, style_axis

MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small", "CNN_Jannis_OHE"]
OUTPUT = Path("results/target_adaptation_combined_seed42_46")


def completed_final_rows(root):
    protocol = json.loads((root / "protocol.json").read_text())
    grouped = defaultdict(list)
    counts = defaultdict(set)
    for pair in protocol["pairs"]:
        for seed in protocol["seeds"]:
            for model in protocol["models"]:
                folder = root / "fits" / pair / f"seed{seed}" / model
                if not (folder / "complete.json").exists():
                    continue
                counts[(pair, model)].add(seed)
                for row in read_csv(folder / "metrics.csv"):
                    final = row["arm"].startswith("oneshot") or (
                        row["arm"].startswith("iterative") and int(row["round"]) == protocol["rounds"]
                    )
                    if final:
                        grouped[(pair, model, row["arm"])].append(float(row["spearman"]))
    return protocol, grouped, counts


def plot(cpu_root, cnn_root, stem, title):
    cpu_protocol, cpu, cpu_counts = completed_final_rows(cpu_root)
    cnn_protocol, cnn, cnn_counts = completed_final_rows(cnn_root)
    if cpu_protocol["pairs"] != cnn_protocol["pairs"]:
        raise ValueError("CPU and CNN pair order differs")
    pairs = cpu_protocol["pairs"]
    fig, axes = plt.subplots(4, 1, figsize=(15.5, 17.5), sharex=True, sharey=True)
    x = np.arange(len(pairs)); width = .19
    for axis, model in zip(axes, MODELS):
        values = cnn if model == "CNN_Jannis_OHE" else cpu
        counts = cnn_counts if model == "CNN_Jannis_OHE" else cpu_counts
        for position, (arm, label, color) in enumerate(zip(ARMS, ARM_LABELS, ARM_COLORS)):
            means, errors = [], []
            for pair in pairs:
                runs = values[(pair, model, arm)]
                means.append(np.nan if not runs else np.mean(runs))
                errors.append(0 if len(runs) < 2 else np.std(runs, ddof=1))
            axis.bar(x + (position - 1.5) * width, means, width, yerr=errors,
                     capsize=3, color=color, label=label)
        model_counts = [len(counts[(pair, model)]) for pair in pairs]
        incomplete = any(n < 5 for n in model_counts)
        suffix = " — PROVISIONAL: " + ", ".join(f"{n}/5" for n in model_counts) if incomplete else ""
        axis.set_title(LABELS[model] + suffix, loc="center", color=COLORS[model],
                       fontweight="bold", fontsize=17)
        axis.set_ylabel("Target-test Spearman ρ")
        style_axis(axis)
    labels = [pair.replace("_to_", " → ").replace("GFP", "") for pair in pairs]
    axes[-1].set_xticks(x, labels, rotation=22, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, .966))
    fig.suptitle(title, fontsize=22, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=1.3)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def main():
    plot(Path("results/target_adaptation_seed42_46"),
         Path("results/target_adaptation_cnn_seed42_46"),
         "target_adaptation_all_models",
         "Target adaptation between GFP proteins")
    plot(Path("results/target_adaptation_peaks_seed42_46"),
         Path("results/target_adaptation_peaks_cnn_seed42_46"),
         "target_adaptation_peaks_all_models",
         "Target adaptation between natural cgreGFP and artificial peaks")


if __name__ == "__main__":
    main()
