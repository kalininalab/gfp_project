"""Freeze nested training subsets and create the saturation DAG."""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from scripts.ortholog_holdout.prepare import star_alignment
from scripts.prepare_data import read_fasta
from .common import (BASE_SIZES, DATA, GENES, MODELS, PUBLISHED, RAW, ROOT, SEEDS,
                     digest, read_csv, scientific_sources, write_csv)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=type(RAW), default=RAW)
    parser.add_argument("--dag", type=Path, default=ROOT / "condor/protein_saturation.dag")
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists(): raise FileExistsError("Protocol already frozen")
    rows = [row for row in read_csv(DATA) if row["gene"] in GENES]
    references = {gene: sequence.rstrip("*") for gene, sequence in
                  read_fasta(ROOT / "data/raw/fasta_sequences/protein_seqs.fa").items() if gene in GENES}
    alignment = star_alignment(references); (out / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    counts = {}; sizes = {}; subset_hashes = {}
    for gene in GENES:
        counts[gene] = {split: sum(row["gene"] == gene and row["split"] == split for row in rows)
                        for split in ("train", "validation", "test")}
        sizes[gene] = [*BASE_SIZES, counts[gene]["train"]]
        train = [row for row in rows if row["gene"] == gene and row["split"] == "train"]
        for seed in SEEDS:
            order = np.random.default_rng(np.random.SeedSequence([seed, GENES.index(gene), 2026])).permutation(len(train))
            manifest = [{"record_id": train[i]["record_id"], "pool_rank": rank}
                        for rank, i in enumerate(order)]
            path = out / "subsets" / gene / f"seed{seed}.csv"; write_csv(path, manifest)
            subset_hashes[str(path.relative_to(out))] = digest(path)
    protocol = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": "Within each GFP, where does held-out predictive performance saturate as same-protein training labels increase?",
        "dataset_sha256": digest(DATA), "genes": list(GENES), "models": list(MODELS), "seeds": list(SEEDS),
        "sizes": sizes, "counts": counts, "nested_subsets": True,
        "validation_policy": "complete original protein-specific validation split fixed at every training size",
        "test_policy": "complete original protein-specific test split fixed and unavailable to fitting or checkpoint selection",
        "onehot_settings": {"epochs": 30, "patience": 10, "batch_size": 32, "lr": .001},
        "cnn_settings": {"epochs": 60, "patience": 10, "batch_size": 64, "lr": .0003, "weight_decay": .0001},
        "distribution_metrics": {"bins": 50, "pseudocount": .5, "kl_units": "bits"},
        "alignment_sha256": digest(out / "alignment.json"), "subset_hashes": subset_hashes,
        "source_hashes": scientific_sources(), "published_directory": str(PUBLISHED)}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n"); (out / "condor").mkdir(exist_ok=True)
    lines=[]; parents=[]; index=0
    for model in MODELS:
        submit = "condor/protein_saturation_gpu.sub" if model.startswith("CNN") else "condor/protein_saturation_cpu.sub"
        for gene in GENES:
            for seed in SEEDS:
                node=f"FIT_{index:03d}"; index += 1; parents.append(node)
                lines += [f"JOB {node} {submit}", f'VARS {node} directory="{out}" gene="{gene}" seed="{seed}" model="{model}"']
    lines += ["JOB FINAL condor/protein_saturation_finalize.sub",
              f'VARS FINAL directory="{out}" published="{PUBLISHED}"',
              "PARENT " + " ".join(parents) + " CHILD FINAL", "MAXJOBS GPU 20", "MAXJOBS CPU 40"]
    args.dag.write_text("\n".join(lines) + "\n")
    print(json.dumps({"trajectory_jobs": len(parents), "fits": sum(len(sizes[g]) for g in GENES)*len(SEEDS)*len(MODELS), "dag": str(args.dag)}, indent=2))


if __name__ == "__main__": main()
