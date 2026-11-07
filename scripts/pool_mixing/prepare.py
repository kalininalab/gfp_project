"""Freeze equal-sized cgreGFP/target pools for the pool-size experiment."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prepare_data import read_fasta
from scripts.transfer_benchmark.common import DATA, GENES, ROOT, digest, read_csv, write_csv
from scripts.transfer_benchmark.prepare import star_alignment


RAW = Path("/data/users/akolchina/gfp_project/cross_protein_pool_size_seed42_46_runs")
TARGETS = ["ppluGFP", "amacGFP"]
PLATE_SIZES = [9, 48, 96, 192, 384]
PLATE_ROUNDS = {9: 213, 48: 40, 96: 20, 192: 10, 384: 5}
REQUIRED_SIZES = sorted({0} | {plate_size * round_number for plate_size, last_round in PLATE_ROUNDS.items()
                              for round_number in range(1, last_round + 1)})
SEEDS = list(range(42, 47))
MODELS = ["aubin_1_10_1", "mlp_small", "CNN_Jannis_OHE"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--base-train-n", type=int, default=14709)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--dag", type=Path, default=ROOT / "condor/pool_mixing.dag")
    return parser.parse_args()


def main():
    args = parse_args()
    raw = args.raw
    if (raw / "protocol.json").exists():
        raise FileExistsError(f"Frozen protocol already exists: {raw / 'protocol.json'}")
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "condor").mkdir(exist_ok=True)
    rows = read_csv(DATA)
    pools = {(gene, split): [row for row in rows if row["gene"] == gene and row["split"] == split]
             for gene in ["cgreGFP"] + TARGETS for split in ("train", "validation", "test")}
    source_train_n, n_validation = len(pools["cgreGFP", "train"]), len(pools["cgreGFP", "validation"])
    if (source_train_n, n_validation) != (14709, 4903):
        raise ValueError("Unexpected cgreGFP split sizes")
    if not 1 <= args.base_train_n <= source_train_n:
        raise ValueError(f"base-train-n must be between 1 and {source_train_n}")
    references = {gene: sequence.rstrip("*") for gene, sequence in
                  read_fasta(ROOT / "data/raw/fasta_sequences/protein_seqs.fa").items() if gene in GENES}
    alignment = star_alignment(references)
    (raw / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")

    manifests, hashes = [], {}
    base_hashes = {}
    for seed in SEEDS:
        available = pools["cgreGFP", "train"]
        rng = np.random.default_rng(np.random.SeedSequence([seed, args.base_train_n, 2026]))
        order = rng.permutation(len(available))[:args.base_train_n]
        selected = [{"record_id": available[index]["record_id"], "gene": "cgreGFP",
                     "split": "train", "pool_rank": rank}
                    for rank, index in enumerate(order)]
        path = raw / "base_pools" / f"cgreGFP_seed{seed}.csv"
        write_csv(path, selected)
        base_hashes[str(path.relative_to(raw))] = digest(path)
    for target_index, target in enumerate(TARGETS):
        for seed in SEEDS:
            selected = []
            for split, budget in (("train", source_train_n), ("validation", n_validation)):
                available = pools[target, split]
                if len(available) < budget:
                    raise ValueError(f"Insufficient {target} {split} records")
                rng = np.random.default_rng(np.random.SeedSequence([seed, target_index, int(split == "validation")]))
                order = rng.permutation(len(available))[:budget]
                selected.extend({"record_id": available[index]["record_id"], "gene": target,
                                 "split": split, "pool_rank": rank}
                                for rank, index in enumerate(order))
            path = raw / "target_pools" / f"{target}_seed{seed}.csv"
            write_csv(path, selected)
            hashes[str(path.relative_to(raw))] = digest(path)
            manifests.append({"target": target, "seed": seed, "path": str(path)})

    protocol = {
        "question": "How does adding a labelled second-protein pool affect prediction on the new protein and the fixed cgreGFP base protein?",
        "dataset": str(DATA), "dataset_sha256": digest(DATA),
        "raw_output": str(raw), "targets": TARGETS, "plate_sizes": PLATE_SIZES,
        "plate_rounds": PLATE_ROUNDS, "required_cumulative_sizes": REQUIRED_SIZES,
        "seeds": SEEDS, "models": args.models,
        "base_gene": "cgreGFP", "base_train_n": args.base_train_n, "base_validation_n": n_validation,
        "base_source_train_n": source_train_n,
        "base_pool_hashes": base_hashes,
        "target_train_pool_n": source_train_n, "target_validation_pool_n": n_validation,
        "pool_definition": "A seeded cgreGFP training subset is fixed per trajectory. For plate sizes 9, 48, 96, 192 and 384, each round adds one new plate to all previous plates; cumulative target size is plate_size times round. Every schedule ends near 1,920 target records.",
        "validation_definition": "Target validation grows proportionally in the original 60/20 train/validation ratio and is used only for early stopping.",
        "test_definition": {gene: len(pools[gene, "test"]) for gene in ["cgreGFP"] + TARGETS},
        "test_policy": "Both protein-specific tests are frozen and never enter training, validation, checkpoint selection, or pool sampling.",
        "alignment_sha256": digest(raw / "alignment.json"), "target_pool_hashes": hashes,
        "mlp_settings": {"epochs": 30, "patience": 10, "batch_size": 32, "learning_rate": 0.001,
                         "optimizer": "Adam", "loss": "MSE"},
        "cnn_settings": {"epochs": 60, "patience": 10, "batch_size": 64, "learning_rate": 0.0003,
                         "weight_decay": 0.0001, "optimizer": "AdamW", "loss": "MSE",
                         "target_scaling": "combined training mean/std"},
    }
    (raw / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_csv(raw / "pool_manifests.csv", manifests)
    dag_lines, cpu_nodes, gpu_nodes = [], [], []
    for target in TARGETS:
        for n_target in REQUIRED_SIZES:
            for seed in SEEDS:
                jobs = (("aubin_1_10_1", "condor/pool_mixing_cpu_model.sub", "CPU"),
                        ("mlp_small", "condor/pool_mixing_cpu_model.sub", "CPU"),
                        ("CNN_Jannis_OHE", "condor/pool_mixing_gpu_custom.sub", "GPU"))
                for model, submit, category in jobs:
                    if model not in args.models:
                        continue
                    node = f"{category}_{target}_{n_target}_{seed}"
                    dag_lines += [f"JOB {node} {submit}",
                                  f'VARS {node} target="{target}" n_target="{n_target}" seed="{seed}" model="{model}" raw="{raw}"',
                                  f"CATEGORY {node} {category}", f"RETRY {node} 1"]
                    (cpu_nodes if category == "CPU" else gpu_nodes).append(node)
    dag_lines += ["MAXJOBS CPU 50", "MAXJOBS GPU 50"]
    args.dag.write_text("\n".join(dag_lines) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
