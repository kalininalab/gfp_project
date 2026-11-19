"""Freeze target-included mixtures for all four GFP proteins."""
import argparse
import datetime
import json

import numpy as np

from scripts.ortholog_holdout.prepare import star_alignment
from scripts.prepare_data import read_fasta
from .common import (DATA, DEFAULT, GENES, MODELS, ROOT, condition_id, digest,
                     included_sources, other_combinations, read_csv,
                     scientific_sources, write_csv)

N_TRAIN, N_VALIDATION = 14709, 4903


def allocate(total, n):
    counts = [total // n] * n
    for index in range(total % n):
        counts[index] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=type(DEFAULT), default=DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    parser.add_argument("--dag", type=str, default="condor/ortholog_target_included.dag")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError("Choose a new output directory")

    rows = [row for row in read_csv(DATA) if row["gene"] in GENES]
    assert len({row["record_id"] for row in rows}) == len(rows)
    owner = {}
    for gene in GENES:
        for row in rows:
            if row["gene"] == gene:
                owner.setdefault(row["sequence"], gene)
    pools = {(gene, split): [row for row in rows if row["gene"] == gene and row["split"] == split
                             and owner[row["sequence"]] == gene]
             for gene in GENES for split in ["train", "validation", "test"]}
    references = {gene: sequence.rstrip("*") for gene, sequence in
                  read_fasta(ROOT / "data/raw/fasta_sequences/protein_seqs.fa").items()
                  if gene in GENES}
    alignment = star_alignment(references)
    (out / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    tests = [dict(target=gene, record_id=row["record_id"], gene=gene)
             for gene in GENES for row in rows
             if row["gene"] == gene and row["split"] == "test"]
    write_csv(out / "test_records.csv", tests)

    subset_hashes, counts, jobs, condition_map = {}, [], [], {}
    for target in GENES:
        target_test_sequences = {row["sequence"] for row in rows
                                 if row["gene"] == target and row["split"] == "test"}
        condition_map[target] = []
        for other_genes in other_combinations(target):
            sources = included_sources(target, other_genes)
            condition = condition_id(sources)
            condition_map[target].append(condition)
            for seed in args.seeds:
                selected = []
                for split, total in [("train", N_TRAIN), ("validation", N_VALIDATION)]:
                    for gene, budget in zip(sources, allocate(total, len(sources))):
                        pool = [row for row in pools[gene, split]
                                if row["sequence"] not in target_test_sequences]
                        assert budget <= len(pool), (target, condition, gene, split, budget, len(pool))
                        rng = np.random.default_rng(np.random.SeedSequence(
                            [seed, GENES.index(gene), int(split == "validation")]))
                        chosen = sorted(rng.permutation(len(pool))[:budget].tolist())
                        selected += [dict(record_id=pool[i]["record_id"], gene=gene, split=split)
                                     for i in chosen]
                        counts.append(dict(target=target, condition=condition, seed=seed,
                                           split=split, source=gene, n=budget,
                                           available=len(pool)))
                assert len({row["record_id"] for row in selected}) == len(selected)
                path = out / "subsets" / target / f"{condition}_seed{seed}.csv"
                write_csv(path, selected)
                subset_hashes[str(path.relative_to(out))] = digest(path)
                jobs += [dict(target=target, condition=condition, seed=seed, model=model)
                         for model in MODELS]

    write_csv(out / "sample_counts.csv", counts)
    write_csv(out / "jobs.csv", jobs)
    (out / "condor").mkdir(exist_ok=True)
    protocol = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        dataset_sha256=digest(DATA), genes=GENES, seeds=args.seeds, models=MODELS,
        conditions=condition_map, n_train=N_TRAIN, n_validation=N_VALIDATION,
        design=("For each predicted protein, its training and validation records are included in "
                "every fit together with each of the seven nonempty combinations of the other "
                "three proteins."),
        source_balance=("Fixed total budget; all included proteins contribute equally within each "
                        "combination, rounded by at most one record."),
        target_test_policy=("The original target 20% test split is frozen and excluded from training, "
                            "validation, checkpoint selection, and subsampling."),
        model_selection="mixed-source validation MSE; frozen target test labels never select checkpoints",
        test_counts={gene: sum(row["gene"] == gene and row["split"] == "test" for row in rows)
                     for gene in GENES},
        alignment_sha256=digest(out / "alignment.json"),
        test_records_sha256=digest(out / "test_records.csv"),
        subset_hashes=subset_hashes, source_hashes=scientific_sources(),
        onehot_settings=dict(epochs=30, patience=10, batch_size=32, lr=.001),
        cnn_settings=dict(epochs=60, patience=10, batch_size=64, lr=.0003,
                          weight_decay=.0001))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    lines, fit_nodes = [], []
    for index, job in enumerate(jobs):
        node = f"FIT_{index:04d}"
        gpu = job["model"] == "CNN_Jannis_OHE"
        submit = "condor/ortholog_target_included_gpu.sub" if gpu else "condor/ortholog_target_included_cpu.sub"
        lines += [f"JOB {node} {submit}",
                  (f'VARS {node} directory="{out}" target="{job["target"]}" '
                   f'condition="{job["condition"]}" seed="{job["seed"]}" model="{job["model"]}"'),
                  f"CATEGORY {node} {'GPU' if gpu else 'CPU'}"]
        fit_nodes.append(node)
    lines += ["JOB FINAL condor/ortholog_target_included_finalize.sub",
              f'VARS FINAL directory="{out}" output="{ROOT / "results/ortholog_target_included_seed42_46"}"',
              "PARENT " + " ".join(fit_nodes) + " CHILD FINAL",
              "MAXJOBS CPU 50", "MAXJOBS GPU 40"]
    dag = ROOT / args.dag
    dag.write_text("\n".join(lines) + "\n")
    print(json.dumps({"fits": len(jobs), "cpu": sum(j["model"] != "CNN_Jannis_OHE" for j in jobs),
                      "gpu": sum(j["model"] == "CNN_Jannis_OHE" for j in jobs),
                      "dag": str(dag), "test_counts": protocol["test_counts"]}, indent=2))


if __name__ == "__main__":
    main()
