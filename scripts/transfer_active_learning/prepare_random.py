"""Freeze a random-acquisition arm matched to the transfer-AL protocol."""

import argparse
import json
from pathlib import Path

from scripts.transfer_benchmark.common import digest, read_csv, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fancy",
        type=Path,
        default=Path("results/transfer_al_cgreGFP_seed42_46"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/transfer_random_cgreGFP_seed42_46"),
    )
    parser.add_argument("--model", default="CNN_Jannis_OHE")
    args = parser.parse_args()
    if (args.output / "protocol.json").exists():
        raise FileExistsError("Random transfer protocol is already frozen")

    protocol = json.loads((args.fancy / "protocol.json").read_text())
    protocol["matched_fancy_protocol_sha256"] = digest(args.fancy / "protocol.json")
    if args.model not in protocol["models"]:
        raise ValueError(f"Model {args.model} is absent from the matched fancy protocol")
    protocol["models"] = [args.model]
    protocol["acquisition"] = "nested seeded random sampling without replacement"
    protocol["selection"] = "seeded random permutation; identical initial set and round budgets to fancy acquisition"
    protocol["comparison"] = f"{args.model} random versus fancy acquisition at matched label budgets"

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "test_records.csv", read_csv(args.fancy / "test_records.csv"))
    protocol["test_records_sha256"] = digest(args.output / "test_records.csv")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
