# Cross-protein plate-size experiment with 7,350 cgreGFP training records

## Corrected laboratory-budget rerun — 17 November 2026

The earlier Random/Acquisition extension was stopped because it used additional
target-validation labels outside the stated plate budget. Its figures must not
be interpreted as the final laboratory simulation.

The corrected protocol uses **no target validation sequences**. Every target
label produced by a plate enters training, so a plate of 96 means exactly 96
new measurements. Aubin and MLP train for 30 fixed epochs and CNN on OHE for 60
fixed epochs. The fixed 4,903-record cgreGFP validation split is diagnostic
only and never selects a checkpoint. Frozen tests remain unchanged.

The fresh matched rerun contains Random and acquisition-score sampling for
Aubin, MLP and CNN on OHE over seeds 42–46. CNN retains plate sizes 48–384;
plate size 9 remains excluded. The new DAG **95871** contains 11,881 nodes and
replaces the stopped DAG 92330. Its GPU cap is 25, paired with the full-source
DAG for a total limit of 50 GPU jobs.

Raw output:

- Random fits: `/data/users/akolchina/gfp_project/cross_protein_pool_size_cgre7350_lab_seed42_46_runs`
- Acquisition fits: `/data/users/akolchina/gfp_project/cross_protein_pool_size_cgre7350_lab_acquisition_seed42_46_runs`

## Superseded historical workflow

The material below documents the older runs retained for provenance. It does
not describe the corrected laboratory-budget rerun above.

This experiment repeats the completed MLP plate-size study while reducing the cgreGFP training set from 14,709 to 7,350 records.

## Controlled comparison

- Model: MLP on aligned positional OHE.
- Seeds: 42–46.
- A reproducible random subset of 7,350 cgreGFP training records is frozen separately for each seed.
- The complete 4,903-record cgreGFP validation set remains available for early stopping.
- Target sequence order and target-validation schedule are identical to the full-training experiment; their manifest hashes match exactly.
- Frozen tests are unchanged: 4,904 cgreGFP, 6,281 ppluGFP and 6,703 amacGFP records.
- Target plate sizes are 9, 48, 96, 192 and 384 records per round; cumulative plates end at 1,917 or 1,920 target training records.
- R², Spearman correlation and KL divergence will be reported for both the cgreGFP and target-protein frozen tests.
- True-versus-predicted distributions will compare no target addition with 1,920 added target records.

## Workflow

- Submitted 7 November 2026 as CPU-only DAG **74194**; it completed **2,410/2,410 MLP fits with zero failures**.
- Unique fits: **2 targets × 241 cumulative sizes × 5 seeds = 2,410 MLP fits**.
- Concurrency limit: 50 CPU jobs; no GPU jobs are requested.
- The pre-submission smoke fit completed successfully with exactly 7,350 cgreGFP training records.
- Raw manifests, checkpoints, predictions and logs are stored under `/data/users/akolchina/gfp_project/cross_protein_pool_size_cgre7350_seed42_46_runs`.

## Aubin addition

- Aubin 1-10-1 was registered as an additional base model on 7 November 2026 using the same frozen 7,350-record source subset for each seed.
- CPU-only DAG **74546** completed **2,410/2,410 Aubin fits with zero failures**.
- A smoke fit completed successfully before submission.
- Final MLP-versus-Aubin report: [`REPORT.md`](REPORT.md).

## Random versus acquisition extension

- Submitted 15 November 2026 as DAG **82313**. On 16 November that DAG was
  replaced by reduced resume DAG **92330**, preserving all completed results.
- Repeats the matched acquisition comparison with 7,350 fixed cgreGFP
  training records for Aubin, MLP and CNN on OHE and seeds 42–46. Aubin and
  MLP retain all five plate sizes. CNN uses 48, 96, 192 and 384; its plate-9
  trajectories were removed on 16 November.
- The DAG also computes the missing random CNN comparator using this report's
  frozen target order, validation policy and tests.
- Acquisition score: `0.62 × scaled hidden-space distance + 0.38 × scaled
  predictive variance`; Aubin/MLP use five-member ensembles and CNN uses 25
  MC-dropout predictions.
- Primary models are reinitialized and trained on all cumulative labelled
  records at each round. Test data never enter fitting or selection.
- Final overlays use pink for acquisition results on cgreGFP and violet for
  acquisition results on the target protein.
- Raw output: `/data/users/akolchina/gfp_project/cross_protein_pool_size_cgre7350_acquisition_seed42_46_runs`.
- The resume snapshot contained 1,759 unfinished jobs before finalization:
  1,379 acquisition rounds and 380 matched-random CNN fits.
