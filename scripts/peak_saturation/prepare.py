"""Freeze nested training subsets and create the artificial-peak saturation DAG."""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from .common import (BASE_SIZES, DATA, MODELS, PEAKS, PUBLISHED, RAW, ROOT, SEEDS,
                     digest, read_csv, scientific_sources, write_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RAW)
    parser.add_argument("--dag", type=Path, default=ROOT / "condor/peak_saturation.dag")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen")

    all_rows = read_csv(DATA)
    rows = [row for row in all_rows if row["gene"] in PEAKS]
    cgre_validation = [row for row in all_rows if row["gene"] == "cgreGFP" and row["split"] == "validation"]
    cgre_test = [row for row in all_rows if row["gene"] == "cgreGFP" and row["split"] == "test"]
    if len({row["sequence"] for row in rows}) != len(rows):
        raise ValueError("Duplicate sequences can leak across artificial peaks")
    if set(row["sequence"] for row in rows) & set(row["sequence"] for row in cgre_validation + cgre_test):
        raise ValueError("Source peaks overlap natural cgreGFP validation/test sequences")
    alignment = {"length": 235, "mapping": {name: list(range(235)) for name in (*PEAKS, "cgreGFP")},
                 "method": "native cgreGFP coordinates; source peaks and natural cgreGFP have length 235"}
    (out / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")

    counts, sizes, subset_hashes = {}, {}, {}
    for peak_index, peak in enumerate(PEAKS):
        source = [row for row in rows if row["gene"] == peak]
        counts[peak] = {"source_total": len(source),
                        **{split: sum(row["split"] == split for row in source)
                           for split in ("train", "validation", "test")}}
        n_complete = counts[peak]["source_total"]
        sizes[peak] = [size for size in BASE_SIZES if size < n_complete] + [n_complete]
        for seed in SEEDS:
            order = np.random.default_rng(
                np.random.SeedSequence([seed, peak_index, 2026, 1127])).permutation(len(source))
            manifest = [{"record_id": source[i]["record_id"], "pool_rank": rank}
                        for rank, i in enumerate(order)]
            path = out / "subsets" / peak / f"seed{seed}.csv"
            write_csv(path, manifest)
            subset_hashes[str(path.relative_to(out))] = digest(path)

    protocol = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "question": "How does prediction on natural cgreGFP change as training data from one artificial peak increase?",
        "dataset_sha256": digest(DATA), "peaks": list(PEAKS), "models": list(MODELS),
        "seeds": list(SEEDS), "sizes": sizes, "counts": counts, "nested_subsets": True,
        "source_policy": "all sequences from each artificial peak form its nested source-training space",
        "validation_policy": "complete original natural-cgreGFP validation split fixed at every training size and used for checkpoint selection",
        "test_policy": "complete original natural-cgreGFP test split fixed and unavailable to fitting or checkpoint selection",
        "target_counts": {"validation": len(cgre_validation), "test": len(cgre_test)},
        "onehot_settings": {"epochs": 30, "patience": 10, "batch_size": 32, "lr": .001},
        "cnn_settings": {"epochs": 60, "patience": 10, "batch_size": 64, "lr": .0003,
                         "weight_decay": .0001},
        "distribution_metrics": {"bins": 50, "pseudocount": .5, "kl_units": "bits"},
        "alignment_sha256": digest(out / "alignment.json"), "subset_hashes": subset_hashes,
        "source_hashes": scientific_sources(), "published_directory": str(PUBLISHED)
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "condor").mkdir(exist_ok=True)

    lines, parents = [], []
    for index, (model, peak, seed) in enumerate(
            (item for model in MODELS for peak in PEAKS for seed in SEEDS
             for item in [(model, peak, seed)])):
        node = f"FIT_{index:03d}"
        parents.append(node)
        submit = "condor/peak_saturation_gpu.sub" if model.startswith("CNN") else "condor/peak_saturation_cpu.sub"
        lines += [f"JOB {node} {submit}",
                  f'VARS {node} directory="{out}" peak="{peak}" seed="{seed}" model="{model}"']
    lines += ["JOB FINAL condor/peak_saturation_finalize.sub",
              f'VARS FINAL directory="{out}" published="{PUBLISHED}"',
              "PARENT " + " ".join(parents) + " CHILD FINAL", "MAXJOBS GPU 20", "MAXJOBS CPU 40"]
    args.dag.write_text("\n".join(lines) + "\n")
    print(json.dumps({"trajectory_jobs": len(parents),
                      "fits": sum(len(sizes[p]) for p in PEAKS) * len(SEEDS) * len(MODELS),
                      "sizes": sizes, "dag": str(args.dag)}, indent=2))


if __name__ == "__main__":
    main()
