"""Plot an explicitly provisional target-adaptation result from completed jobs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.target_adaptation.finalize import ARMS, ARM_COLORS, ARM_LABELS
from scripts.transfer_benchmark.common import read_csv, write_csv
from scripts.transfer_benchmark.plot import style_axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    completed = 0
    for pair in protocol["pairs"]:
        for seed in protocol["seeds"]:
            for model in protocol["models"]:
                folder = root / "fits" / pair / f"seed{seed}" / model
                if not (folder / "complete.json").exists():
                    continue
                completed += 1
                for row in read_csv(folder / "metrics.csv"):
                    if row["arm"].startswith("oneshot") or (
                        row["arm"].startswith("iterative") and int(row["round"]) == protocol["rounds"]
                    ):
                        rows.append({**row, "pair": pair, "seed": seed, "model": model})

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pair"], row["arm"])].append(float(row["spearman"]))
    summary = []
    for pair in protocol["pairs"]:
        for arm in ARMS:
            values = grouped[(pair, arm)]
            summary.append({
                "pair": pair, "arm": arm, "n_seeds": len(values),
                "spearman_mean": "" if not values else float(np.mean(values)),
                "spearman_sd": "" if len(values) < 2 else float(np.std(values, ddof=1)),
            })
    write_csv(root / "partial_summary.csv", summary)

    fig, axis = plt.subplots(figsize=(15.5, 6.5))
    x = np.arange(len(protocol["pairs"])); width = .19
    for position, (arm, label, color) in enumerate(zip(ARMS, ARM_LABELS, ARM_COLORS)):
        means, errors = [], []
        for pair in protocol["pairs"]:
            values = grouped[(pair, arm)]
            means.append(np.nan if not values else np.mean(values))
            errors.append(0 if len(values) < 2 else np.std(values, ddof=1))
        axis.bar(x + (position - 1.5) * width, means, width, yerr=errors,
                 capsize=3, color=color, label=label)
    counts = [len(grouped[(pair, ARMS[0])]) for pair in protocol["pairs"]]
    labels = [pair.replace("_to_", " → ").replace("GFP", "") + f"\n(n={count}/5 seeds)"
              for pair, count in zip(protocol["pairs"], counts)]
    axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_ylabel("Target-test Spearman ρ")
    style_axis(axis)
    axis.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.12))
    fig.suptitle(f"CNN target-protein adaptation — PROVISIONAL ({completed}/30 trajectories)",
                 fontsize=21, fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .9))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / f"target_adaptation_partial.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)
    (root / "STATUS.md").write_text(
        f"# CNN target-protein adaptation — provisional\n\n"
        f"**{completed}/30 trajectories are complete. This is not the final result.** "
        "Each x-axis label reports the number of completed seeds used for that pair. "
        "Missing pairs are intentionally blank. The finalizer will replace this with "
        "the preregistered five-seed figure after all jobs finish.\n\n"
        "![Provisional result](target_adaptation_partial.png)\n"
    )


if __name__ == "__main__":
    main()
