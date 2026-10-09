"""Compute the Kermut ESM-2 650M masked-marginal mean feature."""

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.kermut.preprocess_embeddings import digest

DATA = Path("data/processed/baseline_v1/sequences.csv")
FASTA = Path("data/raw/fasta_sequences/protein_seqs.fa")
OUTPUT = Path("/data/users/akolchina/gfp_project_kermut/features/zero_shot")
MODEL = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"
GENES = {"cgreGFP", "cgre132", "cgre1338", "cgre4111", "cgre9708"}


def read_reference() -> str:
    lines = FASTA.read_text().splitlines()
    for index, line in enumerate(lines):
        if line == ">cgreGFP":
            return lines[index + 1].rstrip("*")
    raise ValueError("cgreGFP reference is absent from FASTA")


def main() -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "manifest.json").exists():
        print("Zero-shot features already complete")
        return
    reference = read_reference()
    rows = [row for row in csv.DictReader(DATA.open()) if row["gene"] in GENES]
    if any(len(row["sequence"]) != len(reference) for row in rows):
        raise ValueError("All variants must use the cgreGFP coordinate system")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval().cuda()
    encoded = tokenizer(reference, return_tensors="pt")
    base = encoded["input_ids"]
    attention = encoded["attention_mask"]
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    token_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids}
    if len(set(token_ids.values())) != 20:
        raise ValueError("Tokenizer amino-acid mapping is invalid")

    log_odds = np.zeros((len(reference), 20), dtype=np.float32)
    start = time.perf_counter()
    batch_size = 16
    with torch.inference_mode():
        for offset in range(0, len(reference), batch_size):
            positions = list(range(offset, min(offset + batch_size, len(reference))))
            tokens = base.repeat(len(positions), 1)
            masks = attention.repeat(len(positions), 1)
            for row_index, position in enumerate(positions):
                tokens[row_index, position + 1] = tokenizer.mask_token_id
            logits = model(input_ids=tokens.cuda(), attention_mask=masks.cuda()).logits.cpu()
            for row_index, position in enumerate(positions):
                values = torch.log_softmax(logits[row_index, position + 1], dim=-1)
                wt_value = values[token_ids[reference[position]]]
                log_odds[position] = np.asarray(
                    [(values[token_ids[aa]] - wt_value).item() for aa in amino_acids], dtype=np.float32
                )
            print(f"masked positions: {positions[-1] + 1}/{len(reference)}", flush=True)

    aa_index = {aa: index for index, aa in enumerate(amino_acids)}
    scores = np.zeros(len(rows), dtype=np.float32)
    for row_index, row in enumerate(rows):
        scores[row_index] = sum(
            log_odds[position, aa_index[mutant]]
            for position, (wild_type, mutant) in enumerate(zip(reference, row["sequence"]))
            if wild_type != mutant
        )
    np.save(OUTPUT / "scores.npy", scores)
    np.save(OUTPUT / "reference_log_odds.npy", log_odds)
    with (OUTPUT / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "gene", "sequence"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    manifest = {
        "model": MODEL, "revision": REVISION, "method": "wild-type masked marginals",
        "reference": reference, "genes": sorted(GENES), "records": len(rows),
        "dataset_sha256": digest(DATA), "records_sha256": digest(OUTPUT / "records.csv"),
        "scores_sha256": digest(OUTPUT / "scores.npy"), "elapsed_seconds": time.perf_counter() - start,
        "host": platform.node(), "gpu": torch.cuda.get_device_name(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
