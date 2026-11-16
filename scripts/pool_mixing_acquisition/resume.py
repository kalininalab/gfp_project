"""Build a reduced DAG containing only unfinished pool-mixing work."""

import argparse
import json
from pathlib import Path

from scripts.pool_mixing.prepare import PLATE_ROUNDS, PLATE_SIZES, SEEDS, TARGETS
from scripts.pool_mixing_acquisition.prepare import MODELS, models_for_plate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dag", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.raw / "protocol.json").read_text())
    lines, final_nodes, counts = [], [], {"random_cnn": 0, "acquisition": 0}

    # With CNN plate 9 removed, every remaining random-CNN schedule is a
    # prefix of the 48-sequence schedule (0, 48, ..., 1,920).
    for target in TARGETS:
        for n_target in range(0, 1921, 48):
            for seed in SEEDS:
                complete = (args.reference_raw / "runs" / target / f"n{n_target:04d}" /
                            f"seed{seed}" / "CNN_Jannis_OHE" / "complete.json")
                if complete.exists():
                    continue
                node = f"RANDOM_GPU_{target}_{n_target}_{seed}"
                lines += [f"JOB {node} condor/pool_mixing_gpu_custom.sub",
                          f'VARS {node} target="{target}" n_target="{n_target}" seed="{seed}" raw="{args.reference_raw}"',
                          f"CATEGORY {node} GPU", f"RETRY {node} 2"]
                final_nodes.append(node); counts["random_cnn"] += 1

    for target in TARGETS:
        for plate_size in PLATE_SIZES:
            last_round = PLATE_ROUNDS[plate_size]
            for seed in SEEDS:
                for model in models_for_plate(plate_size):
                    trajectory = (args.raw / "runs" / target / f"plate{plate_size:03d}" /
                                  f"seed{seed}" / model)
                    state_path = trajectory / "state.json"
                    next_round = (json.loads(state_path.read_text())["next_round"]
                                  if state_path.exists() else 0)
                    if next_round > last_round:
                        continue
                    category = "GPU" if model == "CNN_Jannis_OHE" else "CPU"
                    submit = ("condor/pool_mixing_acquisition_gpu.sub" if category == "GPU"
                              else "condor/pool_mixing_acquisition_cpu.sub")
                    previous = None
                    for round_number in range(next_round, last_round + 1):
                        node = f"{category}_{target}_{plate_size}_{seed}_{model}_{round_number}"
                        lines += [f"JOB {node} {submit}",
                                  f'VARS {node} target="{target}" plate="{plate_size}" round="{round_number}" seed="{seed}" model="{model}" reference_raw="{args.reference_raw}" raw="{args.raw}"',
                                  f"CATEGORY {node} {category}", f"RETRY {node} 2"]
                        if previous:
                            lines.append(f"PARENT {previous} CHILD {node}")
                        previous = node; counts["acquisition"] += 1
                    final_nodes.append(previous)

    finish = "FINALIZE"
    lines += [f"JOB {finish} condor/pool_mixing_acquisition_finalize.sub",
              f'VARS {finish} reference_raw="{args.reference_raw}" acquisition_raw="{args.raw}" output="{args.output}"']
    if final_nodes:
        lines.append(f"PARENT {' '.join(final_nodes)} CHILD {finish}")
    lines += ["MAXJOBS CPU 50", "MAXJOBS GPU 50"]
    args.dag.write_text("\n".join(lines) + "\n")
    amendment = {
        "base_train_n": protocol["base_train_n"],
        "change": "CNN on OHE plate size 9 removed; Aubin and MLP plate size 9 retained.",
        "cnn_plate_sizes": [48, 96, 192, 384],
        "cnn_random_cumulative_sizes": list(range(0, 1921, 48)),
        "unfinished_jobs_at_resume": counts,
    }
    (args.raw / "plate9_cnn_exclusion.json").write_text(json.dumps(amendment, indent=2) + "\n")
    print(json.dumps({"dag": str(args.dag), **counts,
                      "total_before_finalizer": sum(counts.values())}, indent=2))


if __name__ == "__main__":
    main()
