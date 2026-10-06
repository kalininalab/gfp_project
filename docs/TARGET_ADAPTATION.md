# Target-protein adaptation

This experiment tests whether a small labelled sample from the target protein
improves transfer from another GFP protein. It covers all six directed pairs
among cgreGFP, amacGFP and ppluGFP, using Aubin, Linear and MLP over seeds 42–46.

For a direction such as cgreGFP → ppluGFP, the model starts with 14,709 cgreGFP
training records and a fixed 4,903-record cgreGFP validation set. The ppluGFP
non-test records form the acquisition pool. The full frozen ppluGFP 20% test set
is unavailable to fitting, validation, acquisition and checkpoint selection.

Four arms use the same source split, target pool and target test:

- **Iterative fancy:** acquire 96 target sequences, refit, and repeat for ten rounds.
- **Iterative random:** the same schedule with seeded random selection.
- **One-shot fancy:** select all 960 target sequences from the source-only model and refit once.
- **One-shot random:** add 960 seeded-random target sequences and refit once.

For these models the acquisition uncertainty term is zero because they contain
no dropout. Therefore “fancy” means ranking by scaled distance in the projected
penultimate representation. Selection is global; the student's dense
SpectralClustering/3-mer stage is not used.

Every iterative round saves metrics, test predictions, a true/predicted
distribution plot and a true-versus-predicted scatter plot. Heavy per-fit files
live under `/data/users/akolchina/gfp_project_artifacts`.

```bash
python -m scripts.target_adaptation.prepare
condor_submit condor/target_adaptation_cpu.sub
condor_submit condor/target_adaptation_finish.sub
```

The finalizer requires all 90 jobs and validates their protocol and metrics
hashes before creating the aggregate comparison.
