"""Freeze and write the dependent HTCondor DAG for acquisition pool mixing."""

import argparse
import json
from pathlib import Path

from scripts.pool_mixing.prepare import PLATE_ROUNDS, PLATE_SIZES, REQUIRED_SIZES, SEEDS, TARGETS
from scripts.transfer_benchmark.common import ROOT, digest

MODELS = ("aubin_1_10_1", "mlp_small", "CNN_Jannis_OHE")


def models_for_plate(plate_size):
    """CNN plate 9 is excluded: its 214 dependent GPU fits dominate runtime."""
    return MODELS[:-1] if plate_size == 9 else MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dag", type=Path, required=True)
    args = parser.parse_args()
    if (args.raw / "protocol.json").exists():
        raise FileExistsError(args.raw / "protocol.json")
    reference = json.loads((args.reference_raw / "protocol.json").read_text())
    args.raw.mkdir(parents=True); (args.raw / "condor").mkdir()
    protocol = {
        "question": "How does acquisition sampling change the cross-protein labelled-pool curves relative to the matched random order?",
        "reference_raw": str(args.reference_raw),
        "reference_protocol_sha256": digest(args.reference_raw / "protocol.json"),
        "base_train_n": reference["base_train_n"], "targets": TARGETS,
        "plate_sizes": PLATE_SIZES, "plate_rounds": PLATE_ROUNDS,
        "seeds": SEEDS, "models": list(MODELS),
        "acquisition_formula": "0.62 * min-max-scaled hidden-space distance + 0.38 * min-max-scaled predictive variance",
        "distance_weight": 0.62, "uncertainty_weight": 0.38,
        "distance_anchors": 256, "projection_dimensions": 32,
        "ensemble_members": 5, "mc_dropout_samples": 25,
        "non_cnn_uncertainty": "Five-member prediction variance. Aubin auxiliary fits bootstrap the labelled training records; MLP auxiliary fits use independent initialization and batch order.",
        "cnn_uncertainty": "Variance of 25 stochastic MC-dropout predictions from the current fitted CNN.",
        "selection": "Global descending acquisition score within the frozen target-training pool.",
        "retraining": "At every round, the primary model and acquisition uncertainty estimator are fitted from new initialization on the complete cumulative labelled set.",
        "validation_and_test": "Exactly the reference experiment's growing target validation prefix and frozen protein-specific tests; test labels never enter fitting or acquisition.",
    }
    (args.raw / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    lines, final_nodes = [], []
    # The published random arm contains Aubin and MLP.  Fit its missing CNN arm
    # on the same frozen prefixes so every acquisition curve has a matched
    # random comparator.
    cnn_random_sizes = list(range(0, 1921, 48))
    for target in TARGETS:
        for n_target in cnn_random_sizes:
            for seed in SEEDS:
                node = f"RANDOM_GPU_{target}_{n_target}_{seed}"
                lines += [f"JOB {node} condor/pool_mixing_gpu_custom.sub",
                          f'VARS {node} target="{target}" n_target="{n_target}" seed="{seed}" raw="{args.reference_raw}"',
                          f"CATEGORY {node} GPU", f"RETRY {node} 2"]
                final_nodes.append(node)
    for target in TARGETS:
        for plate_size in PLATE_SIZES:
            last_round = PLATE_ROUNDS[plate_size]
            for seed in SEEDS:
                for model in models_for_plate(plate_size):
                    previous = None
                    category = "GPU" if model == "CNN_Jannis_OHE" else "CPU"
                    submit = "condor/pool_mixing_acquisition_gpu.sub" if category == "GPU" else "condor/pool_mixing_acquisition_cpu.sub"
                    for round_number in range(last_round + 1):
                        node = f"{category}_{target}_{plate_size}_{seed}_{model}_{round_number}"
                        lines += [f"JOB {node} {submit}",
                                  f'VARS {node} target="{target}" plate="{plate_size}" round="{round_number}" seed="{seed}" model="{model}" reference_raw="{args.reference_raw}" raw="{args.raw}"',
                                  f"CATEGORY {node} {category}", f"RETRY {node} 2"]
                        if previous:
                            lines.append(f"PARENT {previous} CHILD {node}")
                        previous = node
                    final_nodes.append(previous)
    finish = "FINALIZE"
    lines += [f"JOB {finish} condor/pool_mixing_acquisition_finalize.sub",
              f'VARS {finish} reference_raw="{args.reference_raw}" acquisition_raw="{args.raw}" output="{args.output}"',
              f"PARENT {' '.join(final_nodes)} CHILD {finish}",
              "MAXJOBS CPU 50", "MAXJOBS GPU 50"]
    args.dag.write_text("\n".join(lines) + "\n")
    print(json.dumps({"dag": str(args.dag), "nodes": sum(1 for line in lines if line.startswith("JOB ")),
                      "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
