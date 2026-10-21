"""Extract frozen ProteinBERT final global representations for GFP sequences."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.hurdle_predictor.run import DATA, GENES

DEFAULT_CHECKPOINT = Path("/data/users/akolchina/gfp_project_proteinbert/models/full_go_epoch_92400_sample_23500000.pkl")
DEFAULT_OUTPUT = Path("/data/users/akolchina/gfp_project_artifacts/proteinbert_embeddings")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    from tensorflow import keras
    from proteinbert import load_pretrained_model

    generator, encoder = load_pretrained_model(
        local_model_dump_dir=str(args.checkpoint.parent),
        local_model_dump_file_name=args.checkpoint.name,
        download_model_dump_if_not_exists=False,
    )
    pretrained = generator.create_model(args.sequence_length, compile=False)
    # The final 512-dimensional global hidden state has integrated sequence
    # information through all six global-attention blocks. We deliberately do
    # not include the 8,943-dimensional GO output probabilities.
    embedding_model = keras.Model(
        pretrained.inputs,
        pretrained.get_layer("global-merge2-norm-block6").output,
        name="proteinbert_final_global_embedding",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256(args.checkpoint)
    for gene in GENES:
        destination = args.output / gene
        complete = destination / "complete.json"
        if complete.exists():
            continue
        rows = [row for row in csv.DictReader(DATA.open()) if row["gene"] == gene]
        encoded = encoder.encode_X([row["sequence"] for row in rows], args.sequence_length)
        representations = embedding_model.predict(encoded, batch_size=args.batch_size, verbose=1).astype(np.float32)
        if representations.shape != (len(rows), 512) or not np.isfinite(representations).all():
            raise ValueError(f"Unexpected ProteinBERT output for {gene}: {representations.shape}")
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "global_embeddings.npy", representations)
        with (destination / "records.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["record_id", "split"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in ("record_id", "split")} for row in rows)
        complete.write_text(json.dumps({
            "gene": gene, "n": len(rows), "dimensions": 512,
            "representation": "ProteinBERT final global hidden state after block 6",
            "sequence_length": args.sequence_length, "checkpoint_sha256": checkpoint_hash,
            "labels_used": False,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
