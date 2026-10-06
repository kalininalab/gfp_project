"""Freeze natural-cgreGFP/artificial-peak target adaptation."""

import datetime
import json
from pathlib import Path

from scripts.transfer_benchmark.common import DATA, digest, read_csv, write_csv

OUTPUT = Path("results/target_adaptation_peaks_seed42_46")
REFERENCE = Path("results/transfer_cgreGFP_seed42_51")
MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small"]
PEAKS = {"cgre132", "cgre1338", "cgre4111", "cgre9708"}


def main():
    if (OUTPUT / "protocol.json").exists():
        raise FileExistsError("Peak-adaptation protocol is already frozen")
    rows = read_csv(DATA)
    domains = {"cgreGFP": {"cgreGFP"}, "artificial": PEAKS}
    test_counts = {name: sum(row["gene"] in genes and row["split"] == "test" for row in rows) for name, genes in domains.items()}
    pool_counts = {name: sum(row["gene"] in genes and row["split"] != "test" for row in rows) for name, genes in domains.items()}
    tests = [{"record_id": row["record_id"], "gene": row["gene"], "target_group": name}
             for name, genes in domains.items() for row in rows if row["gene"] in genes and row["split"] == "test"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "test_records.csv", tests)
    reference = json.loads((REFERENCE / "protocol.json").read_text())
    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_sha256": digest(DATA), "alignment": str(REFERENCE / "alignment.json"),
        "alignment_sha256": digest(REFERENCE / "alignment.json"),
        "test_records_sha256": digest(OUTPUT / "test_records.csv"),
        "seeds": list(range(42, 47)), "models": MODELS,
        "pairs": ["cgreGFP_to_artificial", "artificial_to_cgreGFP"],
        "source_train_count": 14709, "source_validation_count": 4903,
        "target_test_counts": test_counts, "target_pool_counts": pool_counts,
        "target_sequences_per_round": 96, "rounds": 10, "one_shot_target_count": 960,
        "arms": ["iterative_fancy", "iterative_random", "oneshot_fancy", "oneshot_random"],
        "fancy_score": "0.62 * scaled projected-penultimate distance + 0.38 * scaled MC-dropout variance",
        "selection_note": "CPU models have no dropout, so fancy ranking is representation-distance based",
        "fancy_selection": "global descending score; dense SpectralClustering/3-mer stage omitted",
        "evaluation_policy": "same frozen target test for every arm, model and seed; unavailable to selection",
        "validation_policy": "fixed source-domain validation records only",
        "onehot_settings": reference["onehot_settings"], "projection_dimensions": 32, "distance_anchors": 256,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_csv(OUTPUT / "jobs.csv", [{"pair": pair, "seed": seed, "model": model}
                                     for pair in protocol["pairs"] for seed in protocol["seeds"] for model in MODELS])
    print("Frozen 30 natural/artificial CPU jobs", test_counts, pool_counts)


if __name__ == "__main__":
    main()
