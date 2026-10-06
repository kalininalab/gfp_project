"""Plot matched Random/Fancy transfer for Aubin, Linear and MLP."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.transfer_active_learning.plot_followups import completed_rows, aggregate_final
from scripts.transfer_benchmark.plot import COLORS, LABELS, MIX_ORDER, DIRECTIONS, style_axis

FANCY = Path("results/transfer_al_cgreGFP_seed42_46")
OUTPUT = Path("results/transfer_random_cpu_models_cgreGFP_seed42_46")
RANDOM = {
    "aubin_1_10_1": Path("results/transfer_random_aubin_cgreGFP_seed42_46"),
    "aubin_linear": Path("results/transfer_random_linear_cgreGFP_seed42_46"),
    "mlp_small": Path("results/transfer_random_mlp_cgreGFP_seed42_46"),
}


def darker(color):
    import matplotlib.colors as colors
    return colors.to_hex(np.asarray(colors.to_rgb(color)) * .68)


def plot(experiment, fancy, output):
    conditions = MIX_ORDER if experiment == "ortholog_mixtures" else [f"{a}_to_{b}" for a, b in DIRECTIONS]
    labels = ([name.replace("_", " + ") for name in conditions] if experiment == "ortholog_mixtures"
              else ["Artificial\n→ Artificial", "Artificial\n→ Natural", "Natural\n→ Artificial", "Natural\n→ Natural"])
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 13), sharex=True, sharey=True)
    x = np.arange(len(conditions)); width = .36
    for axis, model in zip(axes, RANDOM):
        random_protocol, random_rows = completed_rows(RANDOM[model], [model])
        fancy_protocol, fancy_rows = completed_rows(fancy, [model])
        if random_protocol["test_records_sha256"] != fancy_protocol["test_records_sha256"]:
            raise ValueError(f"Test manifest mismatch for {model}")
        random = aggregate_final(random_rows, random_protocol)
        fancy = aggregate_final(fancy_rows, fancy_protocol)
        r_mean = [random[(experiment, condition)][0] for condition in conditions]
        r_sd = [random[(experiment, condition)][1] for condition in conditions]
        f_mean = [fancy[(experiment, condition)][0] for condition in conditions]
        f_sd = [fancy[(experiment, condition)][1] for condition in conditions]
        axis.bar(x-width/2, r_mean, width, yerr=r_sd, capsize=3,
                 color=COLORS[model], label="Random")
        axis.bar(x+width/2, f_mean, width, yerr=f_sd, capsize=3,
                 color=darker(COLORS[model]), label='Acquisition score ("Fancy")')
        axis.set_title(LABELS[model], loc="center", color=COLORS[model], fontweight="bold", fontsize=18)
        axis.set_ylabel("Test Spearman ρ")
        style_axis(axis)
    axes[-1].set_xticks(x, labels, rotation=25 if experiment == "ortholog_mixtures" else 0,
                        ha="right" if experiment == "ortholog_mixtures" else "center")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5,.965))
    title = ("Random versus acquisition-score transfer between GFP proteins" if experiment == "ortholog_mixtures"
             else "Random versus acquisition-score transfer between natural cgreGFP and artificial peaks")
    fig.suptitle(title, fontsize=22, y=.995)
    fig.tight_layout(rect=(0,0,1,.93), h_pad=1.3)
    stem = "sampling_ortholog_transfer_cpu_models" if experiment == "ortholog_mixtures" else "sampling_peak_transfer_cpu_models"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fancy", type=Path, default=FANCY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot("ortholog_mixtures", args.fancy, args.output)
    plot("peak_transfer", args.fancy, args.output)
    uncertainty = json.loads((args.fancy / "protocol.json").read_text()).get("uncertainty")
    method = ("Five-member ensemble variance supplies uncertainty for the acquisition score."
              if uncertainty else "The acquisition score uses projected hidden-space distance only.")
    (args.output / "RESULTS.md").write_text(
        '# Random versus acquisition-score transfer — CPU models\n\n'
        'Five matched seeds for Aubin, Linear and MLP. ' + method + '\n\n'
        '![Protein transfer](sampling_ortholog_transfer_cpu_models.png)\n\n'
        '![Peak transfer](sampling_peak_transfer_cpu_models.png)\n'
    )


if __name__ == "__main__":
    main()
