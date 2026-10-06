"""Freeze the five-member ensemble-uncertainty transfer-AL protocol."""

import json
from pathlib import Path

from scripts.transfer_benchmark.common import digest, read_csv, write_csv

SOURCE = Path("results/transfer_al_cgreGFP_seed42_46")
OUTPUT = Path("results/transfer_al_ensemble_cpu_seed42_46")
MODELS = ["aubin_1_10_1", "aubin_linear", "mlp_small"]


def main():
    if (OUTPUT / "protocol.json").exists():
        raise FileExistsError("Ensemble-uncertainty protocol is already frozen")
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    protocol["models"] = MODELS
    protocol["matched_distance_only_protocol_sha256"] = digest(SOURCE / "protocol.json")
    protocol["ensemble_members"] = 5
    protocol["acquisition"] = "0.62 * scaled projected-penultimate distance + 0.38 * scaled five-member ensemble prediction variance"
    protocol["uncertainty"] = {
        "aubin_1_10_1": "primary fit plus four bootstrap-resampled auxiliary fits",
        "aubin_linear": "primary fit plus four bootstrap-resampled auxiliary fits",
        "mlp_small": "five independently initialized fits on the same labelled records",
    }
    protocol["prediction_policy"] = "primary fit evaluates the frozen test; auxiliary fits are used only for acquisition uncertainty"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "test_records.csv", read_csv(SOURCE / "test_records.csv"))
    protocol["test_records_sha256"] = digest(OUTPUT / "test_records.csv")
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_csv(OUTPUT / "jobs.csv", [{"mix": mix, "seed": seed, "model": model}
                                     for mix in protocol["mixes"] for seed in protocol["seeds"] for model in MODELS])


if __name__ == "__main__":
    main()
