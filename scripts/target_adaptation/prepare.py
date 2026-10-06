"""Freeze pairwise natural-GFP target-adaptation inputs and parameters."""

import datetime
import json
from pathlib import Path

from scripts.transfer_benchmark.common import DATA, NATURAL, digest, read_csv, write_csv

OUTPUT = Path("results/target_adaptation_seed42_46")
REFERENCE = Path("results/transfer_cgreGFP_seed42_51")
MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small"]


def main():
    if (OUTPUT / "protocol.json").exists():
        raise FileExistsError("Target-adaptation protocol is already frozen")
    rows = read_csv(DATA)
    pairs = [f"{source}_to_{target}" for source in NATURAL for target in NATURAL if source != target]
    tests = [
        {"record_id": row["record_id"], "gene": row["gene"], "target_group": row["gene"]}
        for row in rows if row["gene"] in NATURAL and row["split"] == "test"
    ]
    test_counts = {gene: sum(row["gene"] == gene for row in tests) for gene in NATURAL}
    pool_counts = {gene: sum(row["gene"] == gene and row["split"] != "test" for row in rows) for gene in NATURAL}
    if min(test_counts.values()) < 4904 or min(pool_counts.values()) < 19612:
        raise ValueError(f"Natural-GFP partitions are unexpectedly small: {test_counts}, {pool_counts}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "test_records.csv", tests)
    reference = json.loads((REFERENCE / "protocol.json").read_text())
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_sha256": digest(DATA),
        "alignment": str(REFERENCE / "alignment.json"),
        "alignment_sha256": digest(REFERENCE / "alignment.json"),
        "test_records_sha256": digest(OUTPUT / "test_records.csv"),
        "seeds": list(range(42, 47)),
        "models": MODELS,
        "pairs": pairs,
        "source_train_count": 14709,
        "source_validation_count": 4903,
        "target_test_counts": test_counts,
        "target_pool_counts": pool_counts,
        "target_sequences_per_round": 96,
        "rounds": 10,
        "one_shot_target_count": 960,
        "arms": ["iterative_fancy", "iterative_random", "oneshot_fancy", "oneshot_random"],
        "fancy_score": "0.62 * scaled projected-penultimate distance + 0.38 * scaled MC-dropout variance",
        "selection_note": "CPU models have no dropout, so their uncertainty term is zero and fancy ranking is representation-distance based",
        "fancy_selection": "global descending score; dense SpectralClustering/3-mer stage omitted",
        "evaluation_policy": "same frozen target test for every arm, model and seed; excluded from source training, target pool, acquisition and checkpoint selection",
        "validation_policy": "fixed source-protein validation records only",
        "onehot_settings": reference["onehot_settings"],
        "projection_dimensions": 32,
        "distance_anchors": 256,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_csv(OUTPUT / "jobs.csv", [
        {"pair": pair, "seed": seed, "model": model}
        for pair in pairs for seed in protocol["seeds"] for model in MODELS
    ])
    print(f"Frozen {len(pairs) * len(protocol['seeds']) * len(MODELS)} CPU jobs")


if __name__ == "__main__":
    main()
