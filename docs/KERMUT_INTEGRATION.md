# Adding Kermut

## Decision

[Kermut](https://github.com/petergroth/kermut) is relevant because it is a Gaussian-process model that returns both a prediction and posterior uncertainty. The latter can replace ensemble or MC-dropout uncertainty in the acquisition score. However, the published model cannot be inserted unchanged into every experiment in this repository.

We will distinguish two models in figures and tables:

1. **Kermut** means the published composite model with its structure kernel, ESM-2 650M sequence kernel and ESM-2 zero-shot mean.
2. **ESM-GP (Kermut-inspired)** means a scalable GP approximation over ESM embeddings. It can be evaluated across proteins, but it is an adaptation and must never be labelled as the published Kermut model.

## What the published model needs

The audit is based on upstream commit `7e9e2e62a59773f6cc8291d85e6d6006a41a6862` and the NeurIPS 2024 paper. Published Kermut uses:

- mean-pooled `esm2_t33_650M_UR50D` embeddings with 1,280 features;
- ESM-2 650M zero-shot scores as a learned mean function;
- ProteinMPNN conditional amino-acid distributions for the reference protein;
- reference-protein 3D coordinates;
- an exact Gaussian process fitted by exact marginal likelihood.

Our existing caches use `esm2_t30_150M_UR50D` and contain 640-feature mean embeddings. They are reusable for an adapted ESM-GP baseline, but they are not the embeddings used by published Kermut. The structure and zero-shot inputs have not yet been generated for our assays.

## Compatibility with our experiments

| Experiment | Published Kermut | Reason |
|---|---:|---|
| Frozen natural cgreGFP model selection | Conceptually valid | One sequence coordinate system and one reference protein. New 650M, zero-shot, ProteinMPNN and structure inputs are required. |
| Natural cgreGFP and artificial peaks | Potentially valid | All sequences use the cgreGFP coordinate system; the common reference and structure must be declared explicitly. |
| Transfer among cgreGFP, amacGFP and ppluGFP | Not unchanged | The published structure kernel assumes equal-length variants of one reference protein and one structure. Mixing orthologs violates that assumption. |
| Target-protein adaptation among orthologs | Not unchanged | The refitted source-plus-target set contains multiple reference proteins and structures. |
| Active learning within one protein | Valid in principle | GP posterior variance provides epistemic uncertainty directly. Exact-GP scaling remains a problem. |

Kermut supports multi-mutants. The problem for ortholog transfer is combining variants defined relative to different wild types and structures in one published structure kernel.

## Scaling

The official implementation subclasses GPyTorch `ExactGP`. The paper identifies cubic scaling with the number of training examples as a limitation. Our standard source training set has 14,709 records, and the natural cgreGFP 80/20 fit has 19,612 training records. Repeating exact factorizations across five seeds, ten AL rounds and every transfer direction is therefore not a practical all-scenario baseline.

The repository's current `combi` environment also does not contain `gpytorch`. Kermut must get an isolated, pinned environment rather than silently changing the environment used for completed experiments.

## Experiment plan

### Published-model pilot

Run the official composite Kermut model on natural cgreGFP first, using the existing frozen record split. Generate the required 650M embeddings, zero-shot scores, reference structure and ProteinMPNN probabilities under `/data/users/akolchina`; keep only manifests, metrics, predictions and figures in Git. Before launching the five-seed suite, benchmark one fit at increasing training sizes and record wall time and peak memory.

If the full frozen split is tractable, compare non-AL and AL using the same test records, initial labelled records, acquisition pool, budgets and seeds as the existing models. Acquisition uses scaled distance plus scaled GP posterior variance, with the weights frozen in the protocol.

### All-scenario model

Implement a sparse variational or random-feature GP on ESM embeddings and call it **ESM-GP (Kermut-inspired)**. It can use the existing 150M features for a fast first benchmark or newly generated 650M features for a closer sequence component. Its GP posterior variance supplies the uncertainty term during AL.

Evaluate it on every existing frozen scenario without changing tests, seeds, round budgets or sampling-arm definitions. This produces a fair comparison to Aubin, Linear, MLP and CNN on OHE, while keeping the model identity explicit.

A multi-reference structural kernel for ortholog transfer would be a separate methodological extension. It requires a defined cross-protein covariance and one structure/ProteinMPNN context per reference protein; it should be developed and ablated as a new model rather than presented as stock Kermut.

## Provenance

Preprocessing began on 9 October 2026. The full-length structure input is
AlphaFold DB `AF-D7PM05-F1-model_v6` (235 aa); experimental PDB 2HPW is retained
as structural provenance but is not passed to ProteinMPNN because its atom
records omit terminal residues and represent the mature chromophore as a
modified residue. Current cluster IDs and validation state are recorded in the
[live status](../results/kermut_cgreGFP/STATUS.md).

- Official implementation: <https://github.com/petergroth/kermut>
- Paper: <https://proceedings.neurips.cc/paper_files/paper/2024/file/34547650b2ca69d91f3b3c3ae8b21962-Paper-Conference.pdf>
- Upstream license: MIT
