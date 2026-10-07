"""Four-model density panel after one 96-sequence target-AL round."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from scripts.transfer_benchmark.common import read_csv
from scripts.transfer_benchmark.plot import COLORS, LABELS

PAIR = "cgreGFP_to_amacGFP"
MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small", "CNN_Jannis_OHE"]
ROOTS = {
    "aubin_1_10_1": Path("results/target_adaptation_seed42_46"),
    "aubin_linear": Path("results/target_adaptation_seed42_46"),
    "mlp_small": Path("results/target_adaptation_seed42_46"),
    "CNN_Jannis_OHE": Path("results/target_adaptation_cnn_seed42_46"),
}
OUTPUT = Path("results/target_adaptation_density_panels")


def values(model, round_number):
    predictions = []
    correlations = []
    y_true = None
    for seed in range(42, 47):
        path = ROOTS[model] / "fits" / PAIR / f"seed{seed}" / model / "predictions.csv"
        rows = [row for row in read_csv(path)
                if row["arm"] == "iterative_fancy" and int(row["round"]) == round_number]
        truth = np.asarray([float(row["y_true"]) for row in rows])
        prediction = np.asarray([float(row["y_pred"]) for row in rows])
        if y_true is None:
            y_true = truth
        else:
            np.testing.assert_allclose(y_true, truth)
        predictions.append(prediction)
        correlations.append(spearmanr(truth, prediction).statistic)
    return y_true, np.concatenate(predictions), np.asarray(correlations)


def plot(round_number, target_count):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.7), sharex=True, sharey=True)
    bins = np.linspace(0.7, 4.2, 48)
    for axis, model in zip(axes.flat, MODELS):
        truth, prediction, correlations = values(model, round_number)
        axis.hist(truth, bins=bins, density=True, histtype="step", linewidth=2.4,
                  color="#0072B2", label="True")
        axis.hist(prediction, bins=bins, density=True, histtype="step", linewidth=2.4,
                  color="#E69F00", label="Predicted")
        axis.set_title(LABELS[model], color=COLORS[model], fontsize=17,
                       fontweight="bold", pad=8)
        axis.text(.97, .93,
                  f"Spearman ρ = {correlations.mean():.3f} ± {correlations.std(ddof=1):.3f}",
                  transform=axis.transAxes, ha="right", va="top", fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.22, linestyle=":")
        axis.tick_params(labelsize=11)
    for axis in axes[:, 0]:
        axis.set_ylabel("Density", fontsize=13)
    for axis in axes[-1, :]:
        axis.set_xlabel("log10 fluorescence", fontsize=13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(.5, .925), fontsize=12)
    rounds = "one active-learning round" if round_number == 1 else f"{round_number} active-learning rounds"
    fig.suptitle(f"cgreGFP → amacGFP after {rounds}",
                 fontsize=21, fontweight="bold", y=.995)
    fig.text(.5, .948,
             f"{target_count} labelled amacGFP target sequences added · frozen amacGFP test · predictions pooled across 5 seeds",
             ha="center", va="top", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .89), h_pad=2.0, w_pad=1.6)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"cgre_to_amac_round{round_number:02d}_{target_count}_target_density.{suffix}",
                    dpi=300, facecolor="white")
    plt.close(fig)


def main():
    plot(1, 96)
    plot(10, 960)


if __name__ == "__main__":
    main()
