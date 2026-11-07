"""Register and generate a DAG for an additional pool-mixing base model."""

import argparse
import json
from datetime import date
from pathlib import Path

from scripts.pool_mixing.prepare import REQUIRED_SIZES, SEEDS, TARGETS
from scripts.transfer_benchmark.common import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--model", choices=["aubin_1_10_1"], required=True)
    parser.add_argument("--dag", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, default=25)
    args = parser.parse_args()
    protocol_path = args.raw / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["required_cumulative_sizes"] != REQUIRED_SIZES or protocol["seeds"] != SEEDS:
        raise ValueError("Existing protocol does not match the pool-mixing schedule")
    addition = {
        "model": args.model,
        "registered": date.today().isoformat(),
        "base_protocol": str(protocol_path),
        "base_protocol_sha256": digest(protocol_path),
        "dataset_sha256": protocol["dataset_sha256"],
        "training_settings": protocol["mlp_settings"],
        "note": "Uses the frozen source subset, target order, validation schedule and tests from the base protocol.",
    }
    addition_path = args.raw / "model_additions" / f"{args.model}.json"
    addition_path.parent.mkdir(parents=True, exist_ok=True)
    addition_path.write_text(json.dumps(addition, indent=2) + "\n")
    lines = []
    for target in TARGETS:
        for n_target in REQUIRED_SIZES:
            for seed in SEEDS:
                node = f"AUBIN_{target}_{n_target}_{seed}"
                lines += [f"JOB {node} condor/pool_mixing_cpu_model.sub",
                          f'VARS {node} target="{target}" n_target="{n_target}" seed="{seed}" model="{args.model}" raw="{args.raw}"',
                          f"CATEGORY {node} CPU", f"RETRY {node} 1"]
    lines.append(f"MAXJOBS CPU {args.max_jobs}")
    args.dag.write_text("\n".join(lines) + "\n")
    print(json.dumps({**addition, "jobs": len(TARGETS) * len(REQUIRED_SIZES) * len(SEEDS),
                      "max_jobs": args.max_jobs, "dag": str(args.dag)}, indent=2))


if __name__ == "__main__":
    main()
