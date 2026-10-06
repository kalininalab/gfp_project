# Reproducing the cgreGFP architecture comparison

## What the supplied manuscript actually specifies

Local source files supplied for this audit were `GFP Shotgun.pdf` and
`GFP Shotgun-2.docx`, both ten pages. They are manuscript drafts and are not
distributed in the Git repository. The relevant “Choice of the model” section
states that cgreGFP is split randomly into **80% training and 20% held-out test**.
Its CNN is trained for **50 epochs with Adam**. The table reports CNN Pearson
0.943, Spearman 0.878 and R² 0.889.

The text describes one ordinary convolution, two dilated convolutions, adaptive
max pooling, three fully connected layers and GELU. It says final implementation
details are in an appendix, but no appendix is present in either supplied file.
The files do not give a random seed, batch size, learning rate, channel widths,
kernel sizes, dilation values, loss, validation policy or the exact retained
record list. Therefore an exact numerical reproduction of 0.943 is not specified.

There are material conflicts with the available code:

- `CNN/model_legacy.py` has two convolutional and two fully connected layers,
  ReLU, 100 epochs and early stopping directly on the test set.
- `master_thesis_jaca00001/src/model.py` is a later residual CNN with configurable
  depth, mean+max pooling and dropout. Its AL defaults use five convolutions.
- The manuscript's AL section defines `alpha × distance + (1-alpha) × variance`,
  while the submitted code computes `alpha × variance + (1-alpha) × distance`.
- `master_thesis_jaca00001/main.py` has training commented out, `use_al=False`,
  and explicit 1%/10%/10% data fractions. Its dataclass defaults instead say
  90%/10%/0%. Neither is a saved configuration for the manuscript table.

These conflicts are recorded rather than silently resolved in favor of the
number that looks closest to the paper.

## Reproduction tiers

The repository already contains a stricter reproducible benchmark with fixed
train/validation/test IDs, ten-fold CV, input hashes and independent final test
evaluation. Its results are in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).
The requested models already obtain high CV Pearson there: Aubin 0.953, small
MLP 0.938, deep MLP 0.952, Jannis OHE 0.956 and Jannis ESM 0.946. That validates
that the implementations can reproduce the manuscript's accuracy scale without
using test labels for model selection.

The new manuscript-protocol comparison isolates the stated 80/20 split. All
five requested models use identical record IDs, seed 42, 50 fixed epochs, Adam,
MSE, and deterministic evaluation. Because the manuscript omits the seed,
learning rate and batch size, we freeze seed 42 and take batch 32 / lr 1e-3 from
the legacy code. The test set is evaluated only after training and is never used
for early stopping. The architectures are the requested existing base models;
they are not relabeled as the incompletely specified manuscript CNN.

```bash
python -m unittest tests.test_paper_reproduction -v
python -m scripts.paper_reproduction.run --model aubin_1_10_1
python -m scripts.paper_reproduction.run --model mlp_small
python -m scripts.paper_reproduction.run --model mlp_deep
python -m scripts.paper_reproduction.run --model CNN_Jannis_OHE --device cuda
python -m scripts.paper_reproduction.run --model CNN_Jannis_ESM --device cuda
python -m scripts.paper_reproduction.finalize
python -m scripts.paper_reproduction.plot
```

Cluster wrappers are `condor/paper_reproduction_cpu.sub` and
`condor/paper_reproduction_gpu.sub`; `condor/paper_reproduction_finish.sub`
waits for all five fits and validates/assembles the final report. Outputs include
`protocol.json`, the exact
`split.csv`, per-epoch training loss, individual test predictions, model weights,
metrics and checksums under `results/paper_reproduction_cgre_80_20/`.
The plotting entry point exposes `plot_metrics(rows, ax=None)` so its panel can
be reused directly in a manuscript figure without copying training code.

## Relationship to active learning

The 80%-training result is a fully supervised upper reference, not an AL result.
An AL experiment must begin with far fewer labels, so its initial Pearson should
be lower. For a valid comparison, the 20% test IDs above must remain fixed and
unavailable to acquisition; AL operates only on the other 80%. A high score is
not obtained by moving test examples into training or evaluating on a changing,
model-specific pool. The earlier `results/active_learning_cgre/` run preserves
the student's changing-pool metric for audit purposes and must not be presented
as the manuscript reproduction.
