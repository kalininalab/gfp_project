"""Plot the fraction of epistatic variants at each mutation count."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/epistatic"
OUTPUT = ROOT / "results/epistasis_by_mutation_count"
GENES = ("cgreGFP", "ppluGFP", "amacGFP")
COLORS = {"cgreGFP": "#0072B2", "ppluGFP": "#D55E00", "amacGFP": "#CC79A7"}
MIN_N = 20


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for gene in GENES:
        grouped = defaultdict(lambda: {"total": 0, "epistatic": 0, "classifiable": 0})
        with (INPUT / f"{gene}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                mutations = int(row["num_mutations"])
                if mutations < 2: continue
                grouped[mutations]["total"] += 1
                grouped[mutations]["epistatic"] += row["epistatic"] == "True"
                grouped[mutations]["classifiable"] += row["epistatic"] in ("True", "False")
        for mutations, counts in sorted(grouped.items()):
            summary.append({"gene": gene, "n_mutations": mutations,
                "n_sequences": counts["total"], "n_epistatic": counts["epistatic"],
                "n_classifiable": counts["classifiable"],
                "epistatic_percent_all_sequences": 100 * counts["epistatic"] / counts["total"],
                "classification_coverage_percent": 100 * counts["classifiable"] / counts["total"],
                "included_in_main_figure": counts["total"] >= MIN_N})
    write_csv(OUTPUT / "epistasis_by_mutation_count.csv", summary)

    fig, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for gene in GENES:
        rows = [row for row in summary if row["gene"] == gene and row["included_in_main_figure"]]
        axis.plot([row["n_mutations"] for row in rows],
                  [row["epistatic_percent_all_sequences"] for row in rows],
                  marker="o", markersize=7, lw=2.4, color=COLORS[gene], label=gene)
    axis.set(xlabel="Number of amino-acid mutations",
             ylabel="Epistatic sequences (%)",
             xticks=range(2, 11), ylim=(-0.15, 5.7))
    axis.grid(alpha=.22); axis.legend(frameon=False, fontsize=12)
    axis.set_title("Fraction of epistatic variants by mutation count", fontsize=17, pad=12)
    axis.text(.99, .98, f"Points shown when n ≥ {MIN_N}", transform=axis.transAxes,
              ha="right", va="top", color="#555555", fontsize=10)
    save(fig, OUTPUT / "epistatic_fraction_by_mutation_count")

    fig, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for gene in GENES:
        rows = [row for row in summary if row["gene"] == gene and row["included_in_main_figure"]]
        axis.plot([row["n_mutations"] for row in rows],
                  [row["classification_coverage_percent"] for row in rows],
                  marker="o", markersize=7, lw=2.4, color=COLORS[gene], label=gene)
    axis.set(xlabel="Number of amino-acid mutations", ylabel="Sequences with all single-mutant measurements (%)",
             xticks=range(2, 11), ylim=(0, 103))
    axis.grid(alpha=.22); axis.legend(frameon=False, fontsize=12)
    axis.set_title("Coverage of the additive epistasis test", fontsize=17, pad=12)
    save(fig, OUTPUT / "epistasis_classification_coverage")

    report = ["# Epistatic fraction by mutation count", "",
        "The requested percentage is calculated separately for each protein and mutation count as:", "",
        "`100 × number classified epistatic / total number of sequences`.", "",
        "![Epistatic fraction](epistatic_fraction_by_mutation_count.png)", "",
        "A sequence is called epistatic by the existing historical pipeline when the absolute difference between its measured wild-type-relative brightness and the sum of its constituent single-mutant effects exceeds the summed measurement uncertainties.", "",
        "Some multi-mutants cannot be classified because one or more required single mutants were not measured. Following the requested denominator, these records remain in the total sequence count but not in the epistatic numerator. This makes the plotted percentage conservative, especially at high mutation counts. The coverage diagnostic shows this effect:", "",
        "![Classification coverage](epistasis_classification_coverage.png)", "",
        f"The main figures show mutation counts represented by at least {MIN_N} sequences. The [complete table](epistasis_by_mutation_count.csv) retains every observed count, including sparse high-mutation tails.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(report))


if __name__ == "__main__": main()
