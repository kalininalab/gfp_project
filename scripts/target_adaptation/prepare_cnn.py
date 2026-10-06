"""Freeze CNN target-adaptation protocols using the existing test manifests."""

import argparse
import json
from pathlib import Path

from scripts.transfer_benchmark.common import digest, read_csv, write_csv


def freeze(source: Path, output: Path):
    if (output / "protocol.json").exists():
        raise FileExistsError(f"Protocol already frozen: {output}")
    protocol = json.loads((source / "protocol.json").read_text())
    transfer = json.loads(Path("results/transfer_al_cgreGFP_seed42_46/protocol.json").read_text())
    protocol["models"] = ["CNN_Jannis_OHE"]
    protocol["cnn_settings"] = transfer["cnn_settings"]
    protocol["uncertainty_samples"] = transfer["uncertainty_samples"]
    protocol["fancy_score"] = "0.62 * scaled projected-penultimate distance + 0.38 * scaled MC-dropout variance"
    protocol["selection_note"] = "CNN uncertainty is prediction variance from 25 stochastic MC-dropout passes"
    protocol["matched_cpu_protocol_sha256"] = digest(source / "protocol.json")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "test_records.csv", read_csv(source / "test_records.csv"))
    protocol["test_records_sha256"] = digest(output / "test_records.csv")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.source, args.output)


if __name__ == "__main__":
    main()
