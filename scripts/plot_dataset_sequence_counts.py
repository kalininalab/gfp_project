"""Plot model-ready sequence counts for natural GFPs and artificial cgreGFP peaks."""
from pathlib import Path
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import read_fasta, reconstruct, Excluded

DATA = ROOT / "data/processed/baseline_v1/sequences.csv"
MINIS = ROOT / "data/raw/240228__ntdata_cgre12minis_c075-genotypes__err1__cgreWTgates.csv"
OUT = ROOT / "results/dataset_sequence_counts"
NATURAL = ["cgreGFP", "ppluGFP", "amacGFP", "avGFP"]
PEAKS = ["cgre132", "cgre1338", "cgre4111", "cgre9708"]

def prepared_counts():
    counts = {name: 0 for name in NATURAL + PEAKS}
    with DATA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["gene"] in counts:
                counts[row["gene"]] += 1
    return counts

def mini_count():
    refs = read_fasta(ROOT / "data/raw/fasta_sequences/protein_seqs.fa")
    sequences = set()
    retained_rows = 0
    with MINIS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            gene = row["gene"]
            try:
                sequence, _, _ = reconstruct(row["aa_genotype"], refs[gene])
                value = float(row["brightness"])
                weight = float(row["pseudocell_count"])
                if not np.isfinite(value) or value <= 1 or not np.isfinite(weight) or weight <= 0:
                    continue
            except (Excluded, ValueError, KeyError):
                continue
            retained_rows += 1
            sequences.add(sequence)
    return len(sequences), retained_rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = prepared_counts()
    minis, mini_rows = mini_count()
    counts["Minis"] = minis
    labels = NATURAL + PEAKS + ["Minis"]
    values = [counts[x] for x in labels]
    groups = ["Natural proteins"] * 4 + ["Large artificial peaks"] * 4 + ["Minis"]
    colors = {"Natural proteins": "#4477AA", "Large artificial peaks": "#EE6677", "Minis": "#AA3377"}

    fig, ax = plt.subplots(figsize=(11.2, 5.9))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=[colors[g] for g in groups], width=.74, edgecolor="white", linewidth=.8)
    ax.set_title("Number of unique protein sequences by landscape", fontsize=18, pad=14)
    ax.set_ylabel("Unique amino-acid sequences", fontsize=14)
    ax.set_xticks(x, labels, rotation=32, ha="right", fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=.8, alpha=.75)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=3, fontsize=10, rotation=0)
    handles = [plt.Rectangle((0,0),1,1,color=colors[g]) for g in colors]
    ax.legend(handles, list(colors), frameon=False, ncol=3, loc="upper right", fontsize=10)
    ax.margins(y=.12)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"sequence_counts.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # A spacious 16:9 version intended for presentation slides.
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar(x, values, color=[colors[g] for g in groups], width=.72,
                  edgecolor="white", linewidth=1.2)
    ax.set_title("Number of sequences per GFP landscape", fontsize=30,
                 fontweight="bold", pad=24)
    ax.set_ylabel("Unique amino-acid sequences", fontsize=23, labelpad=14)
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=20)
    ax.tick_params(axis="y", labelsize=19, width=1.4, length=6)
    ax.tick_params(axis="x", width=1.4, length=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(1.4)
    ax.grid(axis="y", color="#D6D6D6", linewidth=1.1, alpha=.7)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=7,
                 fontsize=18, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in colors]
    ax.legend(handles, list(colors), frameon=False, ncol=3,
              loc="upper right", fontsize=18, handlelength=1.4,
              columnspacing=1.5)
    ax.margins(y=.15)
    fig.subplots_adjust(left=.13, right=.98, bottom=.20, top=.87)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"sequence_counts_slide.{suffix}", dpi=300,
                    facecolor="white")
    plt.close(fig)
    with (OUT / "sequence_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["landscape", "group", "unique_amino_acid_sequences"])
        writer.writerows(zip(labels, groups, values))
    (OUT / "README.md").write_text(
        "# Sequence counts by landscape\n\n"
        "![Sequence counts](sequence_counts.png)\n\n"
        "Counts are unique, valid amino-acid sequences after the canonical project filters. "
        "The four large artificial peaks are shown separately. `Minis` pools all smaller peaks "
        "from `cgre12minis` and deduplicates identical protein sequences across them. "
        f"The minis source contains {mini_rows:,} retained source rows and {minis:,} unique protein sequences.\n\n"
        "Machine-readable counts: [sequence_counts.csv](sequence_counts.csv).\n"
    )
    print(dict(zip(labels, values)))
    print("minis retained rows", mini_rows)

if __name__ == "__main__":
    main()
