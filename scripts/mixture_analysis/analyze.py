"""Fit train-only fluorescence mixtures and produce the stage-1 report."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

DATA = Path("data/processed/baseline_v1/sequences.csv")
OUTPUT = Path("results/cgre_peak_mixture_analysis")
GENES = ["cgreGFP", "cgre132", "cgre1338", "cgre4111", "cgre9708"]
SEEDS = range(42, 47)
COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#E69F00"]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ordered_parameters(model):
    order = np.argsort(model.means_.ravel())
    return (
        model.weights_[order], model.means_.ravel()[order],
        np.sqrt(model.covariances_.reshape(-1)[order]),
    )


def two_component_diagnostics(model, values):
    weights, means, stds = ordered_parameters(model)
    grid = np.linspace(values.min(), values.max(), 20000)
    components = np.asarray([
        weights[index] * norm.pdf(grid, means[index], stds[index]) for index in range(2)
    ])
    difference = components[0] - components[1]
    crossings = np.flatnonzero(np.signbit(difference[:-1]) != np.signbit(difference[1:]))
    between = [index for index in crossings if means[0] <= grid[index] <= means[1]]
    threshold = grid[between[-1] if between else np.argmin(np.abs(difference))]
    posterior = model.predict_proba(values[:, None])
    posterior = posterior[:, np.argsort(model.means_.ravel())]
    return {
        "dark_weight": weights[0], "bright_weight": weights[1],
        "dark_mean": means[0], "bright_mean": means[1],
        "dark_sd": stds[0], "bright_sd": stds[1], "threshold": threshold,
        "density_overlap": np.trapezoid(np.minimum(components[0], components[1]), grid),
        "ambiguous_fraction_p20_p80": np.mean((posterior[:, 1] >= 0.2) & (posterior[:, 1] <= 0.8)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(DATA.open()))
    model_rows, two_rows, bootstrap_rows = [], [], []
    data = {}
    models = {}
    for gene in GENES:
        train = np.asarray([
            float(row["target_log10"]) for row in rows
            if row["gene"] == gene and row["split"] != "test"
        ])
        test_count = sum(row["gene"] == gene and row["split"] == "test" for row in rows)
        data[gene] = train
        floor = train.min()
        floor_count = int(np.sum(np.isclose(train, floor, atol=1e-10)))
        for components in range(1, 5):
            model = GaussianMixture(components, n_init=10, random_state=42, reg_covar=1e-6).fit(train[:, None])
            models[(gene, components)] = model
            model_rows.append({
                "gene": gene, "components": components, "n_development": len(train),
                "n_frozen_test_excluded": test_count, "aic": model.aic(train[:, None]),
                "bic": model.bic(train[:, None]),
            })
        diagnostics = two_component_diagnostics(models[(gene, 2)], train)
        two_rows.append({
            "gene": gene, "n_development": len(train), "measurement_floor": floor,
            "floor_count": floor_count, "floor_fraction": floor_count / len(train), **diagnostics,
        })
        for seed in SEEDS:
            sample = np.random.default_rng(seed).choice(train, len(train), replace=True)
            scores = {}
            for components in range(1, 5):
                model = GaussianMixture(components, n_init=5, random_state=seed, reg_covar=1e-6).fit(sample[:, None])
                scores[components] = model.bic(sample[:, None])
            model = GaussianMixture(2, n_init=10, random_state=seed, reg_covar=1e-6).fit(sample[:, None])
            bootstrap_rows.append({
                "gene": gene, "seed": seed, "bic_selected_components": min(scores, key=scores.get),
                **two_component_diagnostics(model, sample),
            })

    write_csv(OUTPUT / "model_selection.csv", model_rows)
    write_csv(OUTPUT / "two_component_summary.csv", two_rows)
    write_csv(OUTPUT / "bootstrap_stability.csv", bootstrap_rows)

    fig, axes = plt.subplots(len(GENES), 2, figsize=(11, 14), constrained_layout=True)
    for row_index, gene in enumerate(GENES):
        values = data[gene]
        axis = axes[row_index, 0]
        histogram, bins, _ = axis.hist(values, bins=70, density=True, color="#B3B3B3", alpha=0.6, edgecolor="none")
        grid = np.linspace(values.min(), values.max(), 1200)
        model = models[(gene, 2)]
        weights, means, stds = ordered_parameters(model)
        components = []
        for index in range(2):
            density = weights[index] * norm.pdf(grid, means[index], stds[index])
            components.append(density)
            axis.plot(grid, density, color=COLORS[index], lw=2, label=f"Component {index + 1}")
        axis.plot(grid, np.sum(components, axis=0), color="#000000", lw=2, label="2-GMM total")
        threshold = next(item["threshold"] for item in two_rows if item["gene"] == gene)
        summary = next(item for item in two_rows if item["gene"] == gene)
        axis.axvline(threshold, color="#000000", ls="--", lw=1.5, label="Posterior boundary")
        # A true point mass at the assay floor becomes an arbitrarily tall density
        # bar. Clip that bar so the continuous part remains readable, and report
        # its probability explicitly instead of disguising it as a Gaussian.
        floor_bin = np.searchsorted(bins, summary["measurement_floor"], side="right") - 1
        visible = np.delete(histogram, np.clip(floor_bin, 0, len(histogram) - 1))
        axis.set_ylim(0, max(visible.max() * 1.25, 0.5))
        axis.text(0.02, 0.96, f"At floor: {summary['floor_fraction']:.1%}", transform=axis.transAxes,
                  ha="left", va="top", fontsize=10)
        axis.set(title=gene, xlabel="log10 fluorescence", ylabel="Density")
        if row_index == 0:
            axis.legend(frameon=False, ncol=2, fontsize=9)

        axis = axes[row_index, 1]
        subset = [item for item in model_rows if item["gene"] == gene]
        best = min(item["bic"] for item in subset)
        axis.plot([item["components"] for item in subset], [item["bic"] - best + 1 for item in subset],
                  marker="o", color="#0072B2", lw=2)
        axis.set(xticks=[1, 2, 3, 4], xlabel="Gaussian components", ylabel="BIC − best BIC + 1",
                 yscale="log")
        axis.axhline(1, color="#555555", lw=1)
    fig.suptitle("Train-only fluorescence mixture analysis", fontsize=18)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"mixture_analysis.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    floor_fractions = [item["floor_fraction"] for item in two_rows]
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.bar(GENES, floor_fractions, color="#0072B2")
    axis.set(ylabel="Fraction exactly at measurement floor", xlabel="Landscape",
             title="Point mass at the lower fluorescence limit")
    axis.tick_params(axis="x", rotation=25)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"measurement_floor.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected = {gene: sorted({int(row["bic_selected_components"]) for row in bootstrap_rows if row["gene"] == gene}) for gene in GENES}
    report = [
        "# Fluorescence distributions in natural cgreGFP and artificial peaks", "",
        "This analysis uses development records (`train` + `validation`) only. The frozen test records are excluded from mixture selection, threshold estimation and every plotted density.", "",
        "## Main result", "",
        "A two-Gaussian description is much better than a single Gaussian, but it is not the best model for any landscape among the tested 1–4 component models. BIC and AIC prefer four components in all five full development sets; this does not establish that four is the true number of biological states. Five seeded bootstrap refits give the selected component counts shown below.", "",
        "| Landscape | Development N | Frozen test excluded | Exact floor fraction | 2-GMM boundary | Bootstrap BIC choices |", "|---|---:|---:|---:|---:|---|",
    ]
    for item in two_rows:
        test_count = next(row["n_frozen_test_excluded"] for row in model_rows if row["gene"] == item["gene"])
        report.append(f"| {item['gene']} | {item['n_development']:,} | {test_count:,} | {item['floor_fraction']:.1%} | {item['threshold']:.3f} | {', '.join(map(str, selected[item['gene']]))} |")
    report += [
        "", "![Mixture fits and BIC](mixture_analysis.png)", "",
        "![Measurement-floor fractions](measurement_floor.png)", "",
        "## What AIC and BIC mean", "",
        "A Gaussian mixture represents the fluorescence histogram with several bell-shaped curves. Adding curves always makes the training fit more flexible, so we use AIC and BIC to penalize unnecessary complexity. **Lower AIC/BIC is better.** These scores only compare the candidate models fitted to the same data; they are not prediction accuracy, and selecting four among the tested 1–4 components does not prove that four biological states exist.", "",
        "## Interpretation", "",
        "The data contain a discrete pile-up at the lower measurement limit plus structured continuous fluorescence above it. Ordinary GMMs approximate that point mass with a nearly zero-variance Gaussian and then divide the continuous part into additional components. Consequently, `dark versus bright` is a useful prediction task, but the observations do not support a literal claim that fluorescence is generated by exactly two Gaussian populations.", "",
        "For the predictor, the defensible next baseline is a hurdle model: classify `at measurement floor` versus `above floor`, then regress continuous fluorescence among above-floor variants. A soft two-Gaussian model should still be evaluated as a comparison, using the train-only boundaries in this report, but it should not define the biological ground truth.", "",
        "## Reproduce", "", "```bash", "/nethome/akolchina/miniconda3/envs/combi/bin/python -m scripts.mixture_analysis.analyze", "```", "",
        "Machine-readable outputs: [model selection](model_selection.csv), [two-component summaries](two_component_summary.csv), and [bootstrap stability](bootstrap_stability.csv).", "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(report))
    (OUTPUT / "protocol.json").write_text(json.dumps({
        "data": str(DATA), "genes": GENES, "development_splits": ["train", "validation"],
        "excluded_split": "test", "components": [1, 2, 3, 4], "seeds": list(SEEDS),
        "gmm": {"covariance_type": "full", "reg_covar": 1e-6, "full_n_init": 10, "bootstrap_n_init": 5},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
