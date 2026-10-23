"""Compare direct, floor-hurdle and two-GMM-gated linear OHE predictors."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, matthews_corrcoef, roc_auc_score
from sklearn.mixture import GaussianMixture

from scripts.aubin_model import ALPHABET
from scripts.hurdle_predictor.neural_run import load
from scripts.hurdle_predictor.run import GENES
from scripts.mixture_analysis.analyze import two_component_diagnostics
from scripts.regression_metrics import metrics


OUTPUT = Path("results/gmm_gated_linear_seed42_46")
LABELS = {"direct": "Direct linear", "floor_hurdle": "Floor hurdle", "gmm_gate": "2-GMM gate"}
COLORS = {"direct": "#56B4E9", "floor_hurdle": "#0072B2", "gmm_gate": "#D55E00"}


def positional_ohe(tokens):
    """Return sparse flattened positional OHE without fitting a vocabulary."""
    n, length = tokens.shape
    rows = np.repeat(np.arange(n), length)
    columns = np.tile(np.arange(length), n) * len(ALPHABET) + tokens.reshape(-1)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, length * len(ALPHABET)))


def classifier_scores(labels, probability):
    predicted = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc_high": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "test_high_fraction": float(labels.mean()),
    }


def fit_predictions(x, y, ids, high, threshold, seed):
    train, test = ids["train"], ids["test"]
    direct = Ridge(alpha=10.0, solver="lsqr", tol=1e-8).fit(x[train], y[train]).predict(x[test])
    gate = LogisticRegression(C=1.0, max_iter=3000, solver="liblinear", random_state=seed).fit(x[train], high[train])
    probability = gate.predict_proba(x[test])[:, 1]
    low = Ridge(alpha=10.0, solver="lsqr", tol=1e-8).fit(x[train][~high[train]], y[train][~high[train]])
    high_model = Ridge(alpha=10.0, solver="lsqr", tol=1e-8).fit(x[train][high[train]], y[train][high[train]])
    low_prediction = np.minimum(low.predict(x[test]), threshold)
    high_prediction = np.maximum(high_model.predict(x[test]), threshold)
    mixture = (1.0 - probability) * low_prediction + probability * high_prediction
    return direct, probability, low_prediction, high_prediction, mixture


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics, classifiers, thresholds = [], [], []
    for gene in GENES:
        rows, tokens, y, ids = load(gene, "cnn_ohe")
        x = positional_ohe(tokens)
        for seed in range(42, 47):
            gmm = GaussianMixture(2, n_init=10, random_state=seed, reg_covar=1e-6).fit(y[ids["train"], None])
            diagnostics = two_component_diagnostics(gmm, y[ids["train"]])
            threshold = float(diagnostics["threshold"]); gmm_high = y > threshold
            direct, probability, low, high, gmm_prediction = fit_predictions(x, y, ids, gmm_high, threshold, seed)

            floor = float(y[ids["train"]].min()); floor_high = ~np.isclose(y, floor, atol=1e-10)
            _, floor_probability, _, floor_conditional, floor_prediction = fit_predictions(
                x, y, ids, floor_high, floor, seed
            )
            test = ids["test"]
            predictions = {"direct": direct, "floor_hurdle": floor_prediction, "gmm_gate": gmm_prediction}
            for model, prediction in predictions.items():
                all_metrics.append({"gene": gene, "seed": seed, "model": model, **metrics(y[test], prediction)})
            classifiers.append({"gene": gene, "seed": seed, "gate": "gmm", **classifier_scores(gmm_high[test], probability)})
            classifiers.append({"gene": gene, "seed": seed, "gate": "floor", **classifier_scores(floor_high[test], floor_probability)})
            thresholds.append({"gene": gene, "seed": seed, "gmm_threshold": threshold, "floor": floor,
                               "train_low_fraction": float((~gmm_high[ids["train"]]).mean())})

            destination = OUTPUT / gene / f"seed{seed}"; destination.mkdir(parents=True, exist_ok=True)
            test_rows = []
            for position, index in enumerate(test):
                test_rows.append({"record_id": rows[index]["record_id"], "y_true": y[index],
                                  "is_gmm_high": int(gmm_high[index]), "p_gmm_high": probability[position],
                                  "gmm_low_expert": low[position], "gmm_high_expert": high[position],
                                  "direct": direct[position], "floor_hurdle": floor_prediction[position],
                                  "gmm_gate": gmm_prediction[position]})
            write_csv(destination / "predictions.csv", test_rows)
            (destination / "complete.json").write_text(json.dumps({
                "gene": gene, "seed": seed, "encoding": "sparse positional one-hot",
                "linear_regression": "Ridge(alpha=10, solver=lsqr)",
                "gate": "LogisticRegression(C=1, solver=liblinear)",
                "threshold": threshold, "threshold_source": "two-GMM fitted to training labels only",
                "frozen_test_used_for_fitting": False,
            }, indent=2) + "\n")

    summary = []
    for gene in GENES:
        for model in LABELS:
            subset = [row for row in all_metrics if row["gene"] == gene and row["model"] == model]
            record = {"gene": gene, "model": model}
            for metric in ("r2", "pearson", "spearman", "rmse", "wasserstein", "js_divergence"):
                values = np.asarray([row[metric] for row in subset])
                record[f"{metric}_mean"] = values.mean(); record[f"{metric}_sd"] = values.std(ddof=1)
            summary.append(record)
    write_csv(OUTPUT / "all_metrics.csv", all_metrics); write_csv(OUTPUT / "summary.csv", summary)
    write_csv(OUTPUT / "classifier_metrics.csv", classifiers); write_csv(OUTPUT / "thresholds.csv", thresholds)

    positions, width = np.arange(len(GENES)), .24
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, metric, title in zip(axes, ("r2", "wasserstein"), ("R² ↑", "Wasserstein distance ↓")):
        for index, model in enumerate(LABELS):
            selected = [next(row for row in summary if row["gene"] == gene and row["model"] == model) for gene in GENES]
            axis.bar(positions + (index - 1) * width, [row[f"{metric}_mean"] for row in selected], width,
                     yerr=[row[f"{metric}_sd"] for row in selected], color=COLORS[model], capsize=3, label=LABELS[model])
        axis.set(xticks=positions, xticklabels=GENES, ylabel=title); axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=.2); axis.axhline(0, color="#555555", lw=.8)
    axes[0].legend(frameon=False)
    fig.suptitle("Does a train-only two-Gaussian split improve linear OHE prediction?", fontsize=16)
    for suffix in ("png", "pdf", "svg"): fig.savefig(OUTPUT / f"comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    lines = ["# Two-GMM-gated linear predictor", "", "A positional one-hot sequence is passed to linear models only: logistic regression estimates `p_high`, and separate Ridge regressors predict the low and high fluorescence values. The soft output is `(1 - p_high) × low_prediction + p_high × high_prediction`. The GMM boundary and every predictor are fitted on training data only; all methods share the frozen test.", "", "![Comparison](comparison.png)", "", "## Results", "", "| Landscape | Direct linear R² | Floor-hurdle R² | 2-GMM-gated R² | 2-GMM-gated Pearson |", "|---|---:|---:|---:|---:|"]
    for gene in GENES:
        get = lambda model: next(row for row in summary if row["gene"] == gene and row["model"] == model)
        direct, floor, gate = get("direct"), get("floor_hurdle"), get("gmm_gate")
        lines.append(f"| {gene} | {direct['r2_mean']:.3f} ± {direct['r2_sd']:.3f} | {floor['r2_mean']:.3f} ± {floor['r2_sd']:.3f} | {gate['r2_mean']:.3f} ± {gate['r2_sd']:.3f} | {gate['pearson_mean']:.3f} ± {gate['pearson_sd']:.3f} |")
    lines += ["", "Five seeds repeat the GMM initialization. The downstream linear fits are otherwise deterministic, so identical repeated values must not be interpreted as five independent model-training replicates.", "", "Machine-readable outputs: [summary](summary.csv), [all seeded metrics](all_metrics.csv), [thresholds](thresholds.csv), and [classifier metrics](classifier_metrics.csv).", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
