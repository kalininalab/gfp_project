"""Validate and freeze cgreGFP ProteinMPNN probabilities and CA coordinates."""

import json
from pathlib import Path

import numpy as np

from scripts.kermut.preprocess_embeddings import digest
from scripts.kermut.preprocess_zero_shot import read_reference

ROOT = Path("/data/users/akolchina/gfp_project_kermut")
PDB = ROOT / "inputs/AF-D7PM05-F1-model_v6.pdb"
RAW = ROOT / "features/proteinmpnn_alphafold/conditional_probs_only/AF-D7PM05-F1-model_v6.npz"
OUTPUT = ROOT / "features/structure"
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def main() -> None:
    reference = read_reference()
    raw = np.load(RAW)
    sequence = "".join(ALPHABET[index] for index in raw["S"])
    if sequence != reference:
        raise ValueError("ProteinMPNN structure sequence differs from the project reference")
    if raw["log_p"].shape != (10, len(reference), 21):
        raise ValueError(f"Unexpected ProteinMPNN shape: {raw['log_p'].shape}")
    probabilities = np.exp(raw["log_p"].mean(axis=0))[:, :20].astype(np.float32)

    residues = []
    seen = set()
    for line in PDB.read_text().splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[21].strip() not in {"", "A"}:
            continue
        key = (line[21], line[22:26], line[26])
        if key in seen:
            continue
        seen.add(key)
        residues.append((THREE_TO_ONE[line[17:20].strip()], [float(line[30:38]), float(line[38:46]), float(line[46:54])]))
    pdb_sequence = "".join(amino_acid for amino_acid, _ in residues)
    if pdb_sequence != reference:
        raise ValueError("AlphaFold CA sequence differs from the project reference")
    coords = np.asarray([xyz for _, xyz in residues], dtype=np.float32)
    if coords.shape != (len(reference), 3) or not np.isfinite(coords).all():
        raise ValueError(f"Invalid coordinates: {coords.shape}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "conditional_probs.npy", probabilities)
    np.save(OUTPUT / "coords.npy", coords)
    manifest = {
        "reference": reference,
        "length": len(reference),
        "proteinmpnn_replicates": 10,
        "proteinmpnn_alphabet": ALPHABET,
        "structure": "AF-D7PM05-F1-model_v6",
        "structure_source": "AlphaFold DB API entry D7PM05",
        "raw_proteinmpnn_sha256": digest(RAW),
        "pdb_sha256": digest(PDB),
        "conditional_probs_sha256": digest(OUTPUT / "conditional_probs.npy"),
        "coords_sha256": digest(OUTPUT / "coords.npy"),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
