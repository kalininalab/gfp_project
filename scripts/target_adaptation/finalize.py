"""Validate and plot the completed target-adaptation experiment."""

import json
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.transfer_benchmark.common import digest, read_csv, write_csv
from scripts.transfer_benchmark.plot import COLORS, LABELS, style_axis

ROOT = Path("results/target_adaptation_seed42_46")
ARMS = ["oneshot_random", "iterative_random", "oneshot_fancy", "iterative_fancy"]
ARM_LABELS = ["One-shot random", "Iterative random", "One-shot fancy", "Iterative fancy"]
ARM_COLORS = ["#9AD5F2", "#0072B2", "#F3C46B", "#D55E00"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.directory
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    for pair in protocol["pairs"]:
        for seed in protocol["seeds"]:
            for model in protocol["models"]:
                folder = root / "fits" / pair / f"seed{seed}" / model
                complete = json.loads((folder / "complete.json").read_text())
                if complete["protocol_sha256"] != digest(root / "protocol.json") or complete["metrics_sha256"] != digest(folder / "metrics.csv"):
                    raise ValueError(f"Integrity failure: {folder}")
                for row in read_csv(folder / "metrics.csv"):
                    row.update(pair=pair, seed=seed, model=model)
                    rows.append(row)
    write_csv(root / "per_round_scores.csv", rows)
    final = [row for row in rows if row["arm"].startswith("oneshot") or (row["arm"].startswith("iterative") and int(row["round"]) == protocol["rounds"])]
    grouped = defaultdict(list)
    for row in final:
        grouped[(row["pair"], row["model"], row["arm"])].append(float(row["spearman"]))
    summary = []
    for (pair, model, arm), values in grouped.items():
        if len(values) != len(protocol["seeds"]):
            raise ValueError(f"Missing seed: {pair}/{model}/{arm}")
        summary.append({"pair": pair, "model": model, "arm": arm, "n_runs": len(values),
                        "spearman_mean": float(np.mean(values)), "spearman_sd": float(np.std(values, ddof=1)),
                        "runs_json": json.dumps(values)})
    write_csv(root / "summary.csv", summary)
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 13), sharex=True, sharey=True)
    x = np.arange(len(protocol["pairs"])); width = 0.19
    for axis, model in zip(axes, protocol["models"]):
        lookup = {(row["pair"], row["arm"]): row for row in summary if row["model"] == model}
        for position, (arm, label, color) in enumerate(zip(ARMS, ARM_LABELS, ARM_COLORS)):
            means = [float(lookup[(pair, arm)]["spearman_mean"]) for pair in protocol["pairs"]]
            errors = [float(lookup[(pair, arm)]["spearman_sd"]) for pair in protocol["pairs"]]
            axis.bar(x + (position - 1.5) * width, means, width, yerr=errors, capsize=3, color=color, label=label)
        axis.set_title(LABELS[model], loc="center", color=COLORS[model], fontweight="bold", fontsize=18)
        axis.set_ylabel("Target-test Spearman ρ")
        style_axis(axis)
    labels = [pair.replace("_to_", " → ").replace("GFP", "") for pair in protocol["pairs"]]
    axes[-1].set_xticks(x, labels, rotation=25, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Target adaptation: 960 labelled target sequences", fontsize=23, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.3)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / f"target_adaptation_comparison.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)
    (root / "RESULTS.md").write_text(
        "# Target-protein adaptation\n\n"
        "Five seeds and three CPU models. Iterative arms acquire 96 target sequences in each of ten rounds; one-shot arms add 960 target sequences once. All arms share the same frozen target test.\n\n"
        "![Target adaptation](target_adaptation_comparison.png)\n"
    )


if __name__ == "__main__":
    main()
