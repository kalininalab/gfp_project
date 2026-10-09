"""Create mean-pooled ESM-2 650M features for one frozen dataset domain."""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/baseline_v1/sequences.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/users/akolchina/gfp_project_kermut/features/esm2_t33_650M"))
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--revision", default="08e4846e537177426273712802403f7ba8261b6c")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = [row for row in csv.DictReader(args.data.open()) if row["gene"] == args.gene]
    if not rows:
        raise ValueError(f"No records for {args.gene}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("record_id must be unique")
    lengths = {len(row["sequence"]) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"Expected one sequence length, found {sorted(lengths)}")

    output = args.output_root / args.gene
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        print(f"Completed output already exists: {output}")
        return

    torch.manual_seed(42)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, local_files_only=True, add_pooling_layer=False
    ).eval().to(args.device)
    if model.config.num_hidden_layers != 33 or model.config.hidden_size != 1280:
        raise ValueError("Checkpoint is not ESM-2 t33 650M")

    start = time.perf_counter()
    embeddings = np.lib.format.open_memmap(
        output / "mean_embeddings.npy", mode="w+", dtype="float32", shape=(len(rows), 1280)
    )
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            tokens = tokenizer(
                [row["sequence"] for row in batch], return_tensors="pt", padding=True,
                return_special_tokens_mask=True,
            )
            special = tokens.pop("special_tokens_mask").bool()
            valid = tokens["attention_mask"].bool() & ~special
            hidden = model(**{key: value.to(args.device) for key, value in tokens.items()}).last_hidden_state.cpu()
            for index in range(len(batch)):
                embeddings[offset + index] = hidden[index, valid[index]].float().mean(0).numpy()
            if offset % (args.batch_size * 25) == 0:
                print(f"{args.gene}: {offset}/{len(rows)}", flush=True)
    embeddings.flush()

    with (output / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "gene", "sequence"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    manifest = {
        "model": args.model,
        "revision": args.revision,
        "layer": 33,
        "shape": [len(rows), 1280],
        "dtype": "float32",
        "pooling": "arithmetic mean over residues; special tokens excluded",
        "gene": args.gene,
        "dataset_sha256": digest(args.data),
        "records_sha256": digest(output / "records.csv"),
        "embeddings_sha256": digest(output / "mean_embeddings.npy"),
        "elapsed_seconds": time.perf_counter() - start,
        "host": platform.node(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
