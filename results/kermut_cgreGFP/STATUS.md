# Kermut cgreGFP status

The Kermut experiment is restricted to the shared cgreGFP sequence coordinate system:

- natural cgreGFP on its frozen test;
- natural cgreGFP to the four artificial-peak landscapes;
- the four artificial-peak landscapes to natural cgreGFP;
- matched non-AL, random and acquisition-score AL comparisons.

The cgreGFP/amacGFP/ppluGFP ortholog-transfer scenarios are excluded from the published Kermut comparison because they do not share one wild type and one structure.

## Running preprocessing

| Cluster | Jobs | Purpose |
|---|---:|---|
| `65836` | 5 GPU jobs | Mean-pooled ESM-2 t33 650M embeddings for natural cgreGFP and four peaks |
| `65838` | 1 GPU job | ESM-2 t33 650M wild-type masked-marginal scores |
| `65839` | 1 GPU job | ProteinMPNN conditional probabilities on the full 235-aa AlphaFold D7PM05 structure |

The official Kermut implementation is pinned to commit `7e9e2e62a59773f6cc8291d85e6d6006a41a6862`. Large inputs and features live under `/data/users/akolchina/gfp_project_kermut`; they will not be committed.

No model metric is reported yet. Training starts only after the feature manifests pass record-order, sequence, reference-coordinate and hash checks.

## Structural inputs

- The experimental cgreGFP structure is PDB 2HPW (1.55 Å).
- The ProteinMPNN input is AlphaFold DB model `AF-D7PM05-F1-model_v6`, whose sequence exactly matches the full 235-aa project reference. This avoids the missing terminal atoms and mature chromophore representation in 2HPW.

See [the integration audit](../../docs/KERMUT_INTEGRATION.md) for the model definition and scaling constraints.
