"""Freeze equal-sized cgreGFP/target pools for the pool-size experiment."""

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
MODELS = ["mlp_small", "CNN_Jannis_OHE"]


def main():
    if (RAW / "protocol.json").exists():
        raise FileExistsError(f"Frozen protocol already exists: {RAW / 'protocol.json'}")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "condor").mkdir(exist_ok=True)
    rows = read_csv(DATA)
    pools = {(gene, split): [row for row in rows if row["gene"] == gene and row["split"] == split]
             for gene in ["cgreGFP"] + TARGETS for split in ("train", "validation", "test")}
    n_train, n_validation = len(pools["cgreGFP", "train"]), len(pools["cgreGFP", "validation"])
    if (n_train, n_validation) != (14709, 4903):
        raise ValueError("Unexpected cgreGFP split sizes")
    references = {gene: sequence.rstrip("*") for gene, sequence in
                  read_fasta(ROOT / "data/raw/fasta_sequences/protein_seqs.fa").items() if gene in GENES}
    alignment = star_alignment(references)
    (RAW / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")

    manifests, hashes = [], {}
    for target_index, target in enumerate(TARGETS):
        for seed in SEEDS:
            selected = []
            for split, budget in (("train", n_train), ("validation", n_validation)):
                available = pools[target, split]
                if len(available) < budget:
                    raise ValueError(f"Insufficient {target} {split} records")
                rng = np.random.default_rng(np.random.SeedSequence([seed, target_index, int(split == "validation")]))
                order = rng.permutation(len(available))[:budget]
                selected.extend({"record_id": available[index]["record_id"], "gene": target,
                                 "split": split, "pool_rank": rank}
                                for rank, index in enumerate(order))
            path = RAW / "target_pools" / f"{target}_seed{seed}.csv"
            write_csv(path, selected)
            hashes[str(path.relative_to(RAW))] = digest(path)
            manifests.append({"target": target, "seed": seed, "path": str(path)})

    protocol = {
        "question": "How does adding a labelled second-protein pool affect prediction on the new protein and the fixed cgreGFP base protein?",
        "dataset": str(DATA), "dataset_sha256": digest(DATA),
        "raw_output": str(RAW), "targets": TARGETS, "plate_sizes": PLATE_SIZES,
        "plate_rounds": PLATE_ROUNDS, "required_cumulative_sizes": REQUIRED_SIZES,
        "seeds": SEEDS, "models": MODELS,
        "base_gene": "cgreGFP", "base_train_n": n_train, "base_validation_n": n_validation,
        "target_train_pool_n": n_train, "target_validation_pool_n": n_validation,
        "pool_definition": "All cgreGFP source records are fixed. For plate sizes 9, 48, 96, 192 and 384, each round adds one new plate to all previous plates; cumulative target size is plate_size times round. Every schedule ends near 1,920 target records.",
        "validation_definition": "Target validation grows proportionally in the original 60/20 train/validation ratio and is used only for early stopping.",
        "test_definition": {gene: len(pools[gene, "test"]) for gene in ["cgreGFP"] + TARGETS},
        "test_policy": "Both protein-specific tests are frozen and never enter training, validation, checkpoint selection, or pool sampling.",
        "alignment_sha256": digest(RAW / "alignment.json"), "target_pool_hashes": hashes,
        "mlp_settings": {"epochs": 30, "patience": 10, "batch_size": 32, "learning_rate": 0.001,
                         "optimizer": "Adam", "loss": "MSE"},
        "cnn_settings": {"epochs": 60, "patience": 10, "batch_size": 64, "learning_rate": 0.0003,
                         "weight_decay": 0.0001, "optimizer": "AdamW", "loss": "MSE",
                         "target_scaling": "combined training mean/std"},
    }
    (RAW / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_csv(RAW / "pool_manifests.csv", manifests)
    dag_lines, cpu_nodes, gpu_nodes = [], [], []
    for target in TARGETS:
        for n_target in REQUIRED_SIZES:
            for seed in SEEDS:
                for model, submit, category in (("mlp_small", "condor/pool_mixing_cpu.sub", "CPU"),
                                                ("CNN_Jannis_OHE", "condor/pool_mixing_gpu.sub", "GPU")):
                    node = f"{category}_{target}_{n_target}_{seed}"
                    dag_lines += [f"JOB {node} {submit}",
                                  f'VARS {node} target="{target}" n_target="{n_target}" seed="{seed}"',
                                  f"CATEGORY {node} {category}", f"RETRY {node} 1"]
                    (cpu_nodes if category == "CPU" else gpu_nodes).append(node)
    dag_lines += ["JOB FINAL condor/pool_mixing_finalize.sub"]
    dag_lines += [f"PARENT {node} CHILD FINAL" for node in cpu_nodes + gpu_nodes]
    dag_lines += ["MAXJOBS CPU 50", "MAXJOBS GPU 50"]
    (ROOT / "condor/pool_mixing.dag").write_text("\n".join(dag_lines) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
