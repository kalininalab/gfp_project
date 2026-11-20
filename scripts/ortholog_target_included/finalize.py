"""Validate, summarize, and plot the target-included four-protein benchmark."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.regression_metrics import metrics
from .common import (DATA, DEFAULT, GENES, MODELS, PUBLISHED, digest,
                     read_csv, write_csv)

COLORS = {"aubin_1_10_1": "#0072B2", "aubin_linear": "#E69F00",
          "mlp_small": "#CC79A7", "CNN_Jannis_OHE": "#56B4E9"}
LABELS = {"aubin_1_10_1": "Aubin", "aubin_linear": "Linear",
          "mlp_small": "MLP", "CNN_Jannis_OHE": "CNN on OHE"}
SHORT = {"cgreGFP": "cgre", "amacGFP": "amac", "ppluGFP": "pplu", "avGFP": "av"}
METRICS = [("spearman", "Spearman ρ"), ("pearson", "Pearson r"), ("r2", "R²")]


def collect(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    assert digest(DATA) == protocol["dataset_sha256"]
    table = {row["record_id"]: row for row in read_csv(DATA)}
    tests = read_csv(directory / "test_records.csv")
    assert digest(directory / "test_records.csv") == protocol["test_records_sha256"]
    per_run, inputs = [], {}
    for target in GENES:
        expected = {row["record_id"] for row in tests if row["target"] == target}
        for condition in protocol["conditions"][target]:
            for seed in protocol["seeds"]:
                for model in MODELS:
                    out = directory / "fits" / target / condition / f"seed{seed}" / model
                    report = json.loads((out / "metrics.json").read_text())
                    assert report["protocol_sha256"] == digest(directory / "protocol.json")
                    assert report["source_hashes"] == protocol["source_hashes"]
                    assert digest(out / "predictions.csv") == report["predictions_sha256"]
                    predictions = [row for row in read_csv(out / "predictions.csv")
                                   if row["partition"] == "test"]
                    assert {row["record_id"] for row in predictions} == expected
                    assert all(row["gene"] == target for row in predictions)
                    assert all(float(row["y_true"]) == float(table[row["record_id"]]["target_log10"])
                               for row in predictions)
                    score = metrics([float(row["y_true"]) for row in predictions],
                                    [float(row["y_pred"]) for row in predictions])
                    for name, _ in METRICS:
                        np.testing.assert_allclose(score[name], report["metrics"]["test"][name],
                                                   rtol=1e-10, atol=1e-12)
                    per_run.append(dict(target=target, condition=condition, seed=seed,
                                        model=model, n_train=protocol["n_train"],
                                        n_validation=protocol["n_validation"], **score))
                    inputs[str((out / "metrics.json").relative_to(directory))] = digest(out / "metrics.json")
    grouped = defaultdict(list)
    for row in per_run:
        grouped[row["target"], row["condition"], row["model"]].append(row)
    summary = []
    for (target, condition, model), rows in grouped.items():
        item = dict(target=target, condition=condition,
                    n_sources=condition.count("+") + 1, model=model,
                    n_runs=len(rows), n_test=rows[0]["n"],
                    n_train=rows[0]["n_train"], n_validation=rows[0]["n_validation"])
        for metric in ["spearman", "pearson", "r2", "rmse"]:
            values = [row[metric] for row in rows]
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_sd"] = float(np.std(values, ddof=1))
            item[metric + "_runs_json"] = json.dumps(values)
        summary.append(item)
    return protocol, per_run, summary, inputs


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB2BF")
    axis.yaxis.grid(True, color="#DDE2EA", linestyle=":", linewidth=.9)
    axis.set_axisbelow(True)
    axis.axhline(0, color="#737E8D", linewidth=.8)
    axis.tick_params(colors="#344052", labelsize=10)


def plot(summary, protocol, output, metric, label):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 16,
                         "axes.labelsize": 13, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(18, 12.3))
    outer = figure.add_gridspec(2, 2, left=.075, right=.99, top=.87, bottom=.09,
                                hspace=.28, wspace=.13)
    width = .19
    offsets = np.array([-1.5, -.5, .5, 1.5]) * width
    all_values = [row[metric + "_mean"] for row in summary]
    lower = -1 if metric == "r2" else max(-1, min(-.1, np.floor((min(all_values) - .08) * 10) / 10))
    for panel, target in enumerate(GENES):
        cell = outer[panel // 2, panel % 2].subgridspec(2, 1, height_ratios=[4.5, 1.55], hspace=.01)
        axis = figure.add_subplot(cell[0])
        matrix = figure.add_subplot(cell[1], sharex=axis)
        conditions = protocol["conditions"][target]
        for model, offset in zip(MODELS, offsets):
            lookup = {row["condition"]: row for row in summary
                      if row["target"] == target and row["model"] == model}
            values = [lookup[condition][metric + "_mean"] for condition in conditions]
            errors = [lookup[condition][metric + "_sd"] for condition in conditions]
            x = np.arange(len(conditions)) + offset
            axis.bar(x, values, width=width * .92, color=COLORS[model], edgecolor="white",
                     linewidth=.5, zorder=3, label=LABELS[model])
            axis.errorbar(x, values, yerr=errors, fmt="none", ecolor="#414B5A",
                          capsize=2, linewidth=.9, zorder=4)
            if metric == "r2":
                for xpos, value in zip(x, values):
                    if value < -1:
                        axis.scatter(xpos, -.965, marker="v", s=24, color="#253244",
                                     zorder=6, clip_on=False)
        axis.set_title(f"Predicted protein: {target}", color="#233044", pad=8)
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
        axis.set_ylim(lower, 1)
        style(axis)
        if panel % 2 == 0:
            axis.set_ylabel("Frozen target test " + label)
        else:
            axis.tick_params(axis="y", labelleft=False)
        for row_index, gene in enumerate(GENES):
            for column, condition in enumerate(conditions):
                included = gene in condition.split("+")
                matrix.scatter(column, row_index, s=92,
                               facecolor="#344052" if included else "white",
                               edgecolor="#344052" if included else "#C4CDD9", linewidth=1.25)
        matrix.set_yticks(range(len(GENES)), [SHORT[gene] for gene in GENES], fontsize=10)
        matrix.set_ylim(len(GENES) - .5, -.5)
        matrix.set_xlim(-.6, len(conditions) - .4)
        matrix.set_xticks([])
        matrix.tick_params(axis="y", length=0, pad=7, colors="#344052")
        for spine in matrix.spines.values():
            spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[model]) for model in MODELS]
    figure.legend(handles, [LABELS[model] for model in MODELS], loc="upper center",
                  ncol=4, frameon=False, bbox_to_anchor=(.5, .93), fontsize=12)
    figure.suptitle("Prediction from mixtures that include the target GFP protein",
                    fontsize=22, color="#182535", y=.985)
    note = ("Fixed 14,709-train / 4,903-validation total budget; target is included in fitting; "
            "target test remains frozen.")
    if metric == "r2":
        note += "  ▼ marks a mean R² below −1."
    figure.text(.5, .025, note, ha="center", fontsize=11, color="#4D596A")
    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(output / f"four_protein_target_included_{metric}.{suffix}",
                       dpi=240, facecolor="white")
    plt.close(figure)


def report(protocol, summary, output):
    lines = [
        "# Four-protein mixtures with the target included", "", "## Question", "",
        "How does target-protein prediction change when the target's labelled training records are combined with every non-empty combination of the other three GFP proteins?", "",
        "## Design", "",
        f'Every fit uses a fixed total of **{protocol["n_train"]:,} training** and **{protocol["n_validation"]:,} validation** records, divided equally among all proteins shown as filled circles. The predicted protein is always included. Its original test split remains frozen and never enters fitting or checkpoint selection.', "",
        "Five seeds (42–46) vary subsampling, initialization and training order. This is the target-included counterpart to the leave-one-out experiment; the total label budget and model definitions are matched.", "",
        "## Main figures", "",
        "![Spearman](four_protein_target_included_spearman.png)", "",
        "![Pearson](four_protein_target_included_pearson.png)", "",
        "![R2](four_protein_target_included_r2.png)", "",
        "Filled circles show the proteins included in each fixed-budget training mixture. The predicted protein is filled in every column. R² is displayed from −1 to 1; exact values remain in the CSV and table.", "",
        "## Main findings", "",
        "- Including labelled target-protein records raises the best mean Spearman correlation from 0.525 to 0.875 for cgreGFP, 0.450 to 0.868 for amacGFP, 0.195 to 0.891 for ppluGFP, and 0.630 to 0.857 for avGFP compared with the matched leave-one-out experiment.",
        "- CNN on OHE gives the best mean Spearman result for cgreGFP (0.875, cgreGFP + amacGFP), amacGFP (0.868, amacGFP + ppluGFP), and ppluGFP (0.891, amacGFP + ppluGFP). MLP is best for avGFP (0.857, cgreGFP + avGFP).",
        "- Adding more source proteins does not consistently improve performance under the fixed total label budget. Two-protein mixtures are best for all four targets, because adding proteins reduces the number of labelled target sequences available to the model.", "",
        "## Results", "",
        "| Target | Training mixture | Model | Test n | Spearman | Pearson | R² | RMSE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for target in GENES:
        for condition in protocol["conditions"][target]:
            for model in MODELS:
                row = next(item for item in summary if item["target"] == target
                           and item["condition"] == condition and item["model"] == model)
                value = lambda name: f'{row[name + "_mean"]:.3f} ± {row[name + "_sd"]:.3f}'
                lines.append(f'| {target} | {condition} | {LABELS[model]} | {row["n_test"]} | '
                             f'{value("spearman")} | {value("pearson")} | {value("r2")} | {value("rmse")} |')
    lines += ["", "Values are mean ± run SD across five seeds. `per_run_scores.csv` contains every repeat and `sample_counts.csv` preserves exact allocations."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, default=PUBLISHED)
    args = parser.parse_args()
    protocol, per_run, summary, inputs = collect(args.directory)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_run_scores.csv", per_run)
    write_csv(args.output / "summary.csv", summary)
    for metric, label in METRICS:
        plot(summary, protocol, args.output, metric, label)
    for name in ["test_records.csv", "sample_counts.csv"]:
        (args.output / name).write_bytes((args.directory / name).read_bytes())
    report(protocol, summary, args.output)
    final = dict(status="complete", fits=len(inputs),
                 protocol_sha256=digest(args.directory / "protocol.json"),
                 input_metrics=inputs,
                 outputs={path.name: digest(path) for path in args.output.iterdir()
                          if path.is_file() and path.name != "finalization.json"})
    (args.output / "finalization.json").write_text(json.dumps(final, indent=2) + "\n")
    print(f"Validated {len(inputs)} fits and wrote {args.output}")


if __name__ == "__main__":
    main()
