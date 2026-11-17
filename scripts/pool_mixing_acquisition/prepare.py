"""Freeze and write the dependent HTCondor DAG for acquisition pool mixing."""

import argparse
import json
from pathlib import Path

from scripts.pool_mixing.prepare import PLATE_ROUNDS, PLATE_SIZES, REQUIRED_SIZES, SEEDS, TARGETS
from scripts.transfer_benchmark.common import ROOT, digest

MODELS = ("aubin_1_10_1", "mlp_small", "CNN_Jannis_OHE")


def models_for_plate(plate_size, cnn_plate_sizes):
    """CPU models retain every schedule; CNN uses the prespecified subset."""
    return MODELS if plate_size in cnn_plate_sizes else MODELS[:-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dag", type=Path, required=True)
    parser.add_argument("--rerun-random-all", action="store_true",
                        help="Schedule fresh random Aubin, MLP and CNN comparators under the reference protocol")
    parser.add_argument("--max-cpu-jobs", type=int, default=50)
    parser.add_argument("--max-gpu-jobs", type=int, default=50)
    parser.add_argument("--resume-existing", action="store_true",
                        help="Rewrite an existing frozen acquisition DAG without deleting completed artifacts")
    args = parser.parse_args()
    protocol_path = args.raw / "protocol.json"
    if protocol_path.exists() and not args.resume_existing:
        raise FileExistsError(protocol_path)
    reference = json.loads((args.reference_raw / "protocol.json").read_text())
    cnn_plate_sizes = reference.get("cnn_plate_sizes", [48, 96, 192, 384])
    cnn_random_sizes = reference.get("cnn_random_sizes", list(range(0, 1921, 48)))
    if args.resume_existing:
        protocol = json.loads(protocol_path.read_text())
        protocol["cnn_plate_sizes"] = cnn_plate_sizes
        protocol["cnn_random_sizes"] = cnn_random_sizes
        protocol["reference_protocol_sha256"] = digest(args.reference_raw / "protocol.json")
    else:
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
            "laboratory_budget": reference.get("laboratory_budget", False),
            "validation_and_test": (
                "No target validation labels are used. The fixed cgreGFP validation split is diagnostic only; "
                "fixed epoch counts determine checkpoints. Frozen tests never enter fitting or acquisition."
                if reference.get("laboratory_budget", False) else
                "Exactly the reference experiment's growing target validation prefix and frozen protein-specific tests; test labels never enter fitting or acquisition."),
            "label_budget_policy": reference.get("label_budget_policy"),
            "checkpoint_policy": reference.get("checkpoint_policy"),
            "cnn_plate_sizes": cnn_plate_sizes, "cnn_random_sizes": cnn_random_sizes,
        }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    lines, final_nodes = [], []
    if args.rerun_random_all:
        random_jobs = [
            ("aubin_1_10_1", REQUIRED_SIZES, "condor/pool_mixing_cpu_model.sub", "CPU"),
            ("mlp_small", REQUIRED_SIZES, "condor/pool_mixing_cpu_model.sub", "CPU"),
            ("CNN_Jannis_OHE", cnn_random_sizes, "condor/pool_mixing_gpu_custom.sub", "GPU")]
        for model, sizes, submit, category in random_jobs:
            for target in TARGETS:
                for n_target in sizes:
                    for seed in SEEDS:
                        node = f"RANDOM_{category}_{model}_{target}_{n_target}_{seed}"
                        variables = (f'target="{target}" n_target="{n_target}" seed="{seed}" '
                                     f'model="{model}" raw="{args.reference_raw}"')
                        lines += [f"JOB {node} {submit}", f"VARS {node} {variables}",
                                  f"CATEGORY {node} {category}", f"RETRY {node} 2"]
                        final_nodes.append(node)
    else:
        # Legacy extension: CPU comparators already exist; add only missing CNN.
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
                for model in models_for_plate(plate_size, cnn_plate_sizes):
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
              f"MAXJOBS CPU {args.max_cpu_jobs}", f"MAXJOBS GPU {args.max_gpu_jobs}"]
    args.dag.write_text("\n".join(lines) + "\n")
    print(json.dumps({"dag": str(args.dag), "nodes": sum(1 for line in lines if line.startswith("JOB ")),
                      "protocol": protocol}, indent=2))


if __name__ == "__main__":
    main()
