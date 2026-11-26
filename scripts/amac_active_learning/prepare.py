"""Freeze the reproducible protocol and create the HTCondor DAG."""

import json
from datetime import datetime, timezone

from .common import DATA, MODELS, RAW, RESULTS, SEEDS, STRATEGIES, digest, read_csv, write_json


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv(DATA) if row["gene"] == "amacGFP"]
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preliminary exploratory benchmark",
        "data": str(DATA.resolve()), "data_sha256": digest(DATA), "gene": "amacGFP",
        "n_records": len(rows), "seeds": list(SEEDS), "models": list(MODELS),
        "strategies": list(STRATEGIES), "rounds": 10, "batch_size": 96,
        "total_acquisition_budget": 960,
        "split": "seeded paper-style 20% fixed test; 1% initial labels, 10% fixed validation, 10% candidate pool; remainder unused",
        "fairness": "within each seed every model and strategy shares exact record IDs for initial labels, validation, pool, test and unused data",
        "cnn": {"uncertainty": "SD of 25 MC-dropout predictions", "prediction": "mean of the same 25 predictions"},
        "mlp": {"members": 10, "uncertainty": "SD across independently initialized members", "prediction": "ensemble mean"},
        "distance": "Euclidean distance in the primary model's penultimate representation to the nearest currently labelled record; the primary MLP ensemble member defines this coherent hidden coordinate system",
        "score_normalization": "within-round percentile ranks; equal 0.5/0.5 weights for combined strategies; low fitness is one minus high-fitness percentile",
        "selection": "global descending score; stable record-ID tie break; no true candidate labels",
        "training": "fresh initialization each round; complete cumulative labelled set; existing Adam/AdamW, MSE, validation early stopping and epoch limits",
        "model_settings": {"mlp_small": {"epochs": 30, "patience": 10, "batch_size": 32, "learning_rate": 0.001},
                           "CNN_Jannis_OHE": {"epochs": 60, "patience": 10, "batch_size": 64, "learning_rate": 0.0003, "weight_decay": 0.0001}},
        "distribution_metrics": {"bins": 50, "edges": "fixed global amacGFP min/max", "pseudocount": 0.5, "kl_units": "bits"},
        "bright_fraction": "retrospective operational dark/bright boundary from a two-component GMM fitted to initial labels only; not asserted as biological ground truth",
        "raw_directory": str(RAW), "summary_directory": str(RESULTS.resolve()),
        "leakage_policy": "true pool fitness is written only after selection for retrospective diagnostics and never enters scores",
    }
    existing = RESULTS / "protocol.json"
    if existing.exists():
        old = json.loads(existing.read_text()); protocol["created_utc"] = old["created_utc"]
        if old != protocol: raise ValueError("Frozen protocol differs; use a new result directory")
    else: write_json(existing, protocol)

    dag = []
    for model in MODELS:
        submit = "amac_al_cnn.sub" if model.startswith("CNN") else "amac_al_mlp.sub"
        for seed in SEEDS:
            for strategy in STRATEGIES:
                name = f"RUN_{model.replace('_','')}_{seed}_{strategy}"
                dag += [f"JOB {name} condor/{submit}",
                        f'VARS {name} model="{model}" seed="{seed}" strategy="{strategy}"']
    dag.append("JOB FINALIZE condor/amac_al_finalize.sub")
    dag.append("PARENT " + " ".join(line.split()[1] for line in dag if line.startswith("JOB RUN_")) + " CHILD FINALIZE")
    (RESULTS / "benchmark.dag").write_text("\n".join(dag) + "\n")
    print(f"Frozen {len(MODELS)*len(SEEDS)*len(STRATEGIES)} trajectories in {RESULTS}")


if __name__ == "__main__": main()
