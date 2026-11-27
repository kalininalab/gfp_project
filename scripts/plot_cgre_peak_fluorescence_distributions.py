"""Overlay fluorescence distributions for natural cgreGFP, large peaks, and Minis."""
from collections import defaultdict
from pathlib import Path
import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import read_fasta, reconstruct, Excluded

OUT = ROOT / "results/cgre_peak_fluorescence_distributions"
DATA = ROOT / "data/processed/baseline_v1/sequences.csv"
MINIS = ROOT / "data/raw/240228__ntdata_cgre12minis_c075-genotypes__err1__cgreWTgates.csv"
NATURAL_RAW = ROOT / "data/raw/amacGFP_cgreGFP_ppluGFP2__final_aminoacid_genotypes_to_brightness.csv"
LANDSCAPES = ["cgreGFP", "cgre132", "cgre1338", "cgre4111", "cgre9708"]
COLORS = {
    "cgreGFP": "#0072B2", "cgre132": "#E69F00", "cgre1338": "#CC79A7",
    "cgre4111": "#009E73", "cgre9708": "#D55E00", "Minis": "#6F4C9B",
}

def canonical_values():
    values = {name: [] for name in LANDSCAPES}
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["gene"] in values:
                values[row["gene"]].append(float(row["target_log10"]))
    return values

def mini_values():
    refs = read_fasta(ROOT / "data/raw/fasta_sequences/protein_seqs.fa")
    grouped = defaultdict(lambda: [[], []])
    with MINIS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                sequence, _, _ = reconstruct(row["aa_genotype"], refs[row["gene"]])
                value, weight = float(row["brightness"]), float(row["pseudocell_count"])
                if not math.isfinite(value) or value <= 1 or not math.isfinite(weight) or weight <= 0:
                    continue
            except (Excluded, ValueError, KeyError):
                continue
            grouped[sequence][0].append(value); grouped[sequence][1].append(weight)
    result = []
    for vals, weights in grouped.values():
        mean = math.fsum(v*w for v, w in zip(vals, weights)) / math.fsum(weights)
        result.append(math.log10(mean))
    return result

def wt_controls():
    with NATURAL_RAW.open(newline="") as handle:
        natural_wt = next(float(r["replicates_mean_brightness"]) for r in csv.DictReader(handle)
                          if r["gene"] == "cgreGFP" and r["aa_genotype_native"] == "wt")
    sources = {
        name: ROOT / f"data/raw/240228__ntdata_{name}_c075-genotypes__err1__cgreWTgates.csv"
        for name in LANDSCAPES if name != "cgreGFP"
    }
    sources["Minis"] = MINIS
    controls = {}
    for name, path in sources.items():
        with path.open(newline="") as handle:
            controls[name] = next(float(r["brightness"]) for r in csv.DictReader(handle)
                                  if r["aa_genotype"] == "wt_ctrl_cgre_wt")
    return natural_wt, controls

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    values = canonical_values(); values["Minis"] = mini_values()
    all_values = np.concatenate([values[x] for x in [*LANDSCAPES, "Minis"]])
    lo, hi = float(all_values.min()), float(all_values.max())
    padding = .06 * (hi - lo); grid = np.linspace(lo-padding, hi+padding, 600)

    fig, ax = plt.subplots(figsize=(16, 9))
    order = ["cgreGFP", "cgre132", "cgre1338", "cgre4111", "cgre9708", "Minis"]
    for name in order:
        array = np.asarray(values[name])
        density = gaussian_kde(array, bw_method="scott")(grid)
        label = f"{name}  (n={len(array):,})"
        alpha = .24 if name == "cgreGFP" else .12
        line_alpha = 1
        ax.fill_between(grid, density, color=COLORS[name], alpha=alpha, linewidth=0)
        ax.plot(grid, density, color=COLORS[name], linewidth=3.2, alpha=line_alpha, label=label)

    overall_min, overall_max = float(all_values.min()), float(all_values.max())
    for value, text in (
        (overall_min, f"Overall min\n{overall_min:.2f}"),
        (overall_max, f"Overall max\n{overall_max:.2f}"),
    ):
        ax.axvline(value, color="#30343B", linewidth=1.8,
                   linestyle="--", alpha=.8, zorder=1)
        ax.text(value, .025, text, transform=ax.get_xaxis_transform(),
                va="bottom", ha="center", fontsize=14,
                color="#30343B", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=2.5))

    ax.set_title("Fluorescence distributions across cgreGFP landscapes",
                 fontsize=29, fontweight="bold", pad=22)
    ax.set_xlabel("log₁₀ fluorescence", fontsize=23, labelpad=12)
    ax.set_ylabel("Density", fontsize=23, labelpad=12)
    ax.tick_params(axis="both", labelsize=18, width=1.3, length=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.3)
    ax.grid(axis="y", color="#D8D8D8", linewidth=1, alpha=.65)
    ax.set_axisbelow(True)
    ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=17, ncol=2, loc="upper left",
              handlelength=2.6, columnspacing=1.7)
    fig.subplots_adjust(left=.10, right=.98, bottom=.13, top=.88)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"cgre_peak_fluorescence_distributions.{suffix}", dpi=300,
                    facecolor="white")
    plt.close(fig)

    # Align each artificial file's cgreGFP WT control to natural cgreGFP WT.
    natural_wt, controls = wt_controls()
    shifts = {"cgreGFP": 0., **{name: math.log10(natural_wt / control)
                                for name, control in controls.items()}}
    scaled = {name: np.asarray(vals) + shifts[name] for name, vals in values.items()}
    scaled_all = np.concatenate([scaled[name] for name in order])
    scaled_lo, scaled_hi = float(scaled_all.min()), float(scaled_all.max())
    scaled_pad = .06 * (scaled_hi - scaled_lo)
    scaled_grid = np.linspace(scaled_lo-scaled_pad, scaled_hi+scaled_pad, 600)
    fig, ax = plt.subplots(figsize=(16, 9))
    for name in order:
        array = scaled[name]
        curve = gaussian_kde(array, bw_method="scott")(scaled_grid)
        alpha = .24 if name == "cgreGFP" else .12
        ax.fill_between(scaled_grid, curve, color=COLORS[name], alpha=alpha, linewidth=0)
        ax.plot(scaled_grid, curve, color=COLORS[name], linewidth=3.2,
                label=f"{name}  (n={len(array):,})")
    natural_wt_log = math.log10(natural_wt)
    ax.axvline(natural_wt_log, color="#30343B", linestyle="--", linewidth=2, alpha=.8)
    ax.text(natural_wt_log, .98, f"cgreGFP WT = {natural_wt_log:.2f}",
            transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right",
            fontsize=14, color="#30343B", fontweight="bold")
    ax.set_title("Fluorescence after alignment to the cgreGFP WT control",
                 fontsize=29, fontweight="bold", pad=22)
    ax.set_xlabel("WT-aligned log₁₀ fluorescence", fontsize=23, labelpad=12)
    ax.set_ylabel("Density", fontsize=23, labelpad=12)
    ax.tick_params(axis="both", labelsize=18, width=1.3, length=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.3)
    ax.grid(axis="y", color="#D8D8D8", linewidth=1, alpha=.65); ax.set_axisbelow(True)
    ax.set_xlim(scaled_grid[0], scaled_grid[-1]); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=17, ncol=2, loc="upper left",
              handlelength=2.6, columnspacing=1.7)
    fig.subplots_adjust(left=.10, right=.98, bottom=.13, top=.88)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"cgre_peak_fluorescence_wt_aligned.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)

    # WT-aligned count histogram in the compact presentation layout.
    fig, ax = plt.subplots(figsize=(12.5, 9))
    scaled_bins = np.linspace(scaled_lo, scaled_hi, 51)
    for name in order:
        array = scaled[name]
        fill_alpha = .27 if name == "cgreGFP" else .13
        ax.hist(array, bins=scaled_bins, density=False, histtype="stepfilled",
                color=COLORS[name], alpha=fill_alpha, linewidth=0)
        ax.hist(array, bins=scaled_bins, density=False, histtype="step",
                color=COLORS[name], linewidth=2.6,
                label=f"{name}  (n={len(array):,})")
    for value, text in ((scaled_lo, f"Overall min\n{scaled_lo:.2f}"),
                        (scaled_hi, f"Overall max\n{scaled_hi:.2f}")):
        ax.axvline(value, color="#30343B", linewidth=1.8, linestyle="--", alpha=.8, zorder=5)
        ax.text(value, -.105, text, transform=ax.get_xaxis_transform(), va="top", ha="center",
                fontsize=14, color="#30343B", fontweight="bold", clip_on=False)
    ax.axvline(natural_wt_log, color="#6B6F76", linewidth=1.8, linestyle=":", alpha=.9, zorder=5)
    ax.text(natural_wt_log, .97, f"cgreGFP WT\n{natural_wt_log:.2f}",
            transform=ax.get_xaxis_transform(), va="top", ha="center", fontsize=13,
            color="#30343B", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=2))
    fig.suptitle("WT-aligned fluorescence counts across cgreGFP landscapes",
                 fontsize=23, fontweight="bold", y=.97)
    ax.set_xlabel("WT-aligned log₁₀ fluorescence", fontsize=23, labelpad=92)
    ax.set_ylabel("Number of sequences per bin", fontsize=23, labelpad=12)
    ax.tick_params(axis="both", labelsize=18, width=1.3, length=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.3)
    ax.grid(axis="y", color="#D8D8D8", linewidth=1, alpha=.65); ax.set_axisbelow(True)
    ax.set_xlim(scaled_grid[0], scaled_grid[-1]); ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=17, ncol=3,
               loc="upper center", bbox_to_anchor=(.55, .90),
               handlelength=2.6, columnspacing=1.7)
    fig.subplots_adjust(left=.11, right=.98, bottom=.30, top=.76)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"cgre_peak_fluorescence_counts_wt_aligned.{suffix}",
                    dpi=300, facecolor="white")
    plt.close(fig)

    with (OUT / "wt_alignment_factors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["landscape", "file_cgre_wt_control", "natural_cgre_wt", "linear_factor", "log10_shift"])
        writer.writerow(["cgreGFP", natural_wt, natural_wt, 1., 0.])
        for name in order[1:]:
            writer.writerow([name, controls[name], natural_wt, natural_wt/controls[name], shifts[name]])

    # Count histogram: common bins preserve the actual number of sequences.
    fig, ax = plt.subplots(figsize=(12.5, 9))
    bins = np.linspace(overall_min, overall_max, 51)
    for name in order:
        array = np.asarray(values[name])
        fill_alpha = .27 if name == "cgreGFP" else .13
        ax.hist(array, bins=bins, density=False, histtype="stepfilled",
                color=COLORS[name], alpha=fill_alpha, linewidth=0)
        ax.hist(array, bins=bins, density=False, histtype="step",
                color=COLORS[name], linewidth=2.6,
                label=f"{name}  (n={len(array):,})")
    for value, text in (
        (overall_min, f"Overall min\n{overall_min:.2f}"),
        (overall_max, f"Overall max\n{overall_max:.2f}"),
    ):
        ax.axvline(value, color="#30343B", linewidth=1.8,
                   linestyle="--", alpha=.8, zorder=5)
        ax.text(value, -.105, text, transform=ax.get_xaxis_transform(),
                va="top", ha="center", fontsize=14, color="#30343B",
                fontweight="bold", clip_on=False)
    fig.suptitle("Fluorescence counts across cgreGFP landscapes",
                 fontsize=29, fontweight="bold", y=.97)
    ax.set_xlabel("log₁₀ fluorescence", fontsize=23, labelpad=92)
    ax.set_ylabel("Number of sequences per bin", fontsize=23, labelpad=12)
    ax.tick_params(axis="both", labelsize=18, width=1.3, length=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.3)
    ax.grid(axis="y", color="#D8D8D8", linewidth=1, alpha=.65)
    ax.set_axisbelow(True); ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=17, ncol=3,
               loc="upper center", bbox_to_anchor=(.55, .90),
               handlelength=2.6, columnspacing=1.7)
    fig.subplots_adjust(left=.11, right=.98, bottom=.30, top=.76)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"cgre_peak_fluorescence_counts.{suffix}", dpi=300,
                    facecolor="white")
    plt.close(fig)

    # Smooth count-equivalent curves: KDE density × sample count × bin width.
    fig, ax = plt.subplots(figsize=(16, 9))
    smooth_bin_width = (overall_max - overall_min) / 100
    for name in order:
        array = np.asarray(values[name])
        count_curve = gaussian_kde(array, bw_method="scott")(grid) * len(array) * smooth_bin_width
        fill_alpha = .24 if name == "cgreGFP" else .12
        ax.fill_between(grid, count_curve, color=COLORS[name], alpha=fill_alpha, linewidth=0)
        ax.plot(grid, count_curve, color=COLORS[name], linewidth=3,
                label=f"{name}  (n={len(array):,})")
    for value, text in ((overall_min, f"Overall min\n{overall_min:.2f}"),
                        (overall_max, f"Overall max\n{overall_max:.2f}")):
        ax.axvline(value, color="#30343B", linewidth=1.8, linestyle="--", alpha=.8)
        ax.text(value, .025, text, transform=ax.get_xaxis_transform(), va="bottom", ha="center",
                fontsize=14, color="#30343B", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=2.5))
    ax.set_title("Smoothed fluorescence counts across cgreGFP landscapes",
                 fontsize=29, fontweight="bold", pad=22)
    ax.set_xlabel("log₁₀ fluorescence", fontsize=23, labelpad=12)
    ax.set_ylabel("Smoothed sequence count (100-bin scale)", fontsize=21, labelpad=12)
    ax.tick_params(axis="both", labelsize=18, width=1.3, length=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.3)
    ax.grid(axis="y", color="#D8D8D8", linewidth=1, alpha=.65); ax.set_axisbelow(True)
    ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=17, ncol=2, loc="upper left",
              handlelength=2.6, columnspacing=1.7)
    fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.88)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"cgre_peak_fluorescence_counts_smooth.{suffix}", dpi=300,
                    facecolor="white")
    plt.close(fig)

    with (OUT / "landscape_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["landscape", "n", "mean_log10", "sd_log10", "median_log10"])
        for name in order:
            a = np.asarray(values[name]); writer.writerow([name, len(a), a.mean(), a.std(ddof=1), np.median(a)])
    (OUT / "README.md").write_text(
        "# Fluorescence distributions across cgreGFP landscapes\n\n"
        "![Distributions](cgre_peak_fluorescence_distributions.png)\n\n"
        "The curves show kernel-density estimates of model-ready log10 fluorescence for natural "
        "cgreGFP, each of the four large artificial peaks, and all smaller artificial landscapes "
        "pooled as `Minis`. Repeated Minis amino-acid sequences are aggregated with the same "
        "pseudocell-count-weighted arithmetic mean used for the large peaks before log10 transformation. "
        "Curve area is normalized separately for each landscape, so shapes are comparable despite different sample counts.\n"
    )
    print({k: len(v) for k,v in values.items()})

if __name__ == "__main__": main()
