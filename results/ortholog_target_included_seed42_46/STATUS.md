# Four-protein mixtures with the target included

## Complete — 20 November 2026

All **560/560 fits** completed successfully. The strict finalizer validated
all frozen-test predictions and generated the publication figures, report and
machine-readable tables: **[open the completed results](RESULTS.md)**.

The original DAG's 560 fit nodes succeeded; its finalizer node initially
failed because an HTCondor macro name collided with the report-output variable.
The macro was renamed and the same 560 completed fits were finalized without
rerunning any model.

This is the target-included counterpart to the four-protein leave-one-out figure.
For each predicted protein, every training mixture contains that protein plus one
of the seven non-empty combinations of the other three GFP proteins.

- Models: Aubin, Linear, MLP and CNN on OHE.
- Seeds: 42–46.
- Fixed total budget: 14,709 training and 4,903 validation sequences per fit,
  balanced equally among the included proteins.
- Evaluation: the same frozen protein-specific test splits used by the
  leave-one-out experiment. Test records never enter training, validation,
  subsampling or checkpoint selection.
- Planned fits: 560 total (420 CPU and 140 GPU).
- Submitted 19 November 2026 as HTCondor DAG **120055** (561 nodes including
  the strict finalizer). The DAG limits concurrent fits to 50 CPU and 40 GPU
  jobs. One validated CPU smoke fit is already present and will be reused.
- Final outputs: Spearman, Pearson and R² panels in the same visual format as
  the leave-one-out figure, plus per-run and summarized CSV tables.

Raw output: `/data/users/akolchina/gfp_project/ortholog_target_included_seed42_46_runs`.

The final figures and report will replace this status page only after all fits
finish and the finalizer validates every frozen test prediction.
