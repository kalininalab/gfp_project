"""Pairwise reference-sequence identity for natural GFPs and cgreGFP peaks."""
from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from Bio.Align import PairwiseAligner, substitution_matrices

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import read_fasta

OUT = ROOT / "results/sequence_identity_all_gfps"
NATURAL = ["cgreGFP", "ppluGFP", "amacGFP", "avGFP"]
LARGE = ["cgre132", "cgre1338", "cgre4111", "cgre9708"]
MINIS = ["cgre13", "cgre567", "cgre575", "cgre3224", "cgre1414", "cgre626",
         "cgre2880", "cgre900x", "cgre121", "cgre83", "cgre985"]

def calculate(names, refs):
    aligner = PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                              open_gap_score=-10., extend_gap_score=-.5)
    result = np.eye(len(names)) * 100
    for i, a in enumerate(names):
        for j in range(i):
            b = names[j]
            alignment = aligner.align(refs[a], refs[b])[0]
            aa, bb = alignment[0], alignment[1]
            value = 100 * sum(x == y for x, y in zip(aa, bb)) / len(aa)
            result[i, j] = result[j, i] = value
    return result

def save_csv(path, names, matrix):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["sequence", *names])
        for name, row in zip(names, matrix): writer.writerow([name, *[f"{x:.6f}" for x in row]])

def heatmap(names, matrix, stem, title, figsize, annotation_size, tick_size):
    fig, ax = plt.subplots(figsize=figsize)
    palette = LinearSegmentedColormap.from_list(
        "pink_blue", ["#E8A0BF", "#315A9E"]
    )
    image = ax.imshow(matrix, cmap=palette, vmin=0, vmax=100, aspect="equal")
    ax.set_xticks(range(len(names)), names, rotation=42, ha="right", fontsize=tick_size)
    ax.set_yticks(range(len(names)), names, fontsize=tick_size)
    ax.set_title(title, fontsize=tick_size + 8, fontweight="bold", pad=22)
    for i in range(len(names)):
        for j in range(len(names)):
            rgb = np.asarray(palette(image.norm(matrix[i, j]))[:3])
            linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = linear_rgb @ np.array([0.2126, 0.7152, 0.0722])
            text_color = "black" if luminance > 0.179 else "white"
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                    fontsize=annotation_size, color=text_color,
                    fontweight="bold" if i == j else "normal")
    cbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.035)
    cbar.set_label("Sequence identity (%)", fontsize=tick_size)
    cbar.ax.tick_params(labelsize=tick_size - 1)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {k: v.rstrip("*") for k, v in read_fasta(ROOT / "data/raw/fasta_sequences/protein_seqs.fa").items()}
    compact = NATURAL + LARGE
    full = compact + MINIS
    cm = calculate(compact, refs); fm = calculate(full, refs)
    save_csv(OUT / "sequence_identity_natural_and_large_peaks.csv", compact, cm)
    save_csv(OUT / "sequence_identity_all_landscapes.csv", full, fm)
    heatmap(compact, cm, "sequence_identity_natural_and_large_peaks",
            "Reference-sequence identity across GFP landscapes", (13, 11), 12, 15)
    heatmap(full, fm, "sequence_identity_all_landscapes",
            "Reference-sequence identity across all GFP landscapes", (20, 18), 7, 11)
    (OUT / "README.md").write_text(
        "# Reference-sequence identity across GFP landscapes\n\n"
        "Identity is calculated from reference/WT protein sequences by global pairwise alignment "
        "with BLOSUM62, gap-open −10 and gap-extension −0.5. The percentage is the number of "
        "identical residues divided by all alignment columns, including gap columns.\n\n"
        "## Four natural proteins and four large artificial peaks\n\n"
        "![Compact identity matrix](sequence_identity_natural_and_large_peaks.png)\n\n"
        "## Full matrix with Minis expanded\n\n"
        "![Full identity matrix](sequence_identity_all_landscapes.png)\n\n"
        "`Minis` is a pooled dataset category rather than one protein reference, so it has no single "
        "sequence-identity value. The full matrix therefore expands it into its 11 constituent "
        "landscape references. Machine-readable values are provided in the corresponding CSV files.\n"
    )

if __name__ == "__main__": main()
