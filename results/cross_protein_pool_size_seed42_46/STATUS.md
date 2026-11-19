# Cross-protein labelled-pool size experiment

**18 November 2026: [preliminary corrected laboratory-budget figures](preliminary/README.md).** Available random/acquisition rounds for Aubin, MLP and CNN; incomplete seed counts are marked. The experiment is still running.

## Corrected laboratory-budget rerun — 17 November 2026

The earlier Random/Acquisition extension was stopped because it used additional
target-validation labels outside the stated plate budget. Its figures must not
be interpreted as the final laboratory simulation.

The corrected protocol uses **no target validation sequences**. Every target
label produced by a plate enters training, so a plate of 96 means exactly 96
new measurements. Aubin trains for 60 fixed epochs, MLP for 40 and CNN on OHE
for 60 fixed epochs. The fixed 4,903-record cgreGFP validation split is recorded only
as a diagnostic and never selects a checkpoint. Frozen protein-specific tests
remain unchanged.

The fresh matched rerun contains Random and acquisition-score sampling for
Aubin, MLP and CNN on OHE over seeds 42–46. Following a runtime audit, CNN is
restricted to plate sizes 96 and 384; Aubin and MLP retain all five schedules.
Random CNN fits use the matching 96-sequence budget grid. DAG **99903** contains
11,161 nodes and replaces exhaustive DAG 98972. Completed compatible artifacts
are retained and reused. Its GPU cap is 25, paired with the second
source-size DAG for a total limit of 50 GPU jobs.

Raw output:

- Random fits: `/data/users/akolchina/gfp_project/cross_protein_pool_size_lab_seed42_46_runs`
- Acquisition fits: `/data/users/akolchina/gfp_project/cross_protein_pool_size_lab_acquisition_seed42_46_runs`

## Superseded historical workflow

The material below documents the older runs retained for provenance. It does
not describe the corrected laboratory-budget rerun above.

This experiment keeps the full cgreGFP training pool fixed and simulates laboratory plates of **9, 48, 96, 192 or 384 new target sequences per round**. Cumulative target data equal `plate size × round`, and every schedule ends near 1,920 target sequences: 213, 40, 20, 10 or 5 rounds, respectively. Each round retains all sequences from previous plates. MLP is evaluated on frozen tests of both participating proteins across seeds 42–46.

This is supervised random pool-size addition, not active learning. The final folder will contain six main figures: R², Spearman and KL divergence for cgreGFP+ppluGFP and cgreGFP+amacGFP. Each figure contains five panels, one per plate size, with round on the x-axis.

Final figures: [`figures/`](figures/)
Final report: [`REPORT.md`](REPORT.md)

## Completed workflow

- The original mixed MLP/CNN DAG **69490** was cancelled on 6 November 2026 at the user’s request because CNN throughput was too slow. Its incomplete CNN trajectories remain as partial raw data and are excluded from the balanced result.
- The CPU-only MLP DAG **71741** completed successfully: **2,409/2,409 DAG nodes**, zero failures, plus the independently completed smoke-test point, for **2,410 MLP fits** total.
- Identical cumulative target sizes shared by different plate schedules are fitted once and reused.
- Seeds: 42–46; targets: ppluGFP and amacGFP; frozen tests also include the original cgreGFP base protein.
- The completed result is MLP-only. No CNN jobs from this workflow remain in the queue.
- Raw checkpoints, predictions, manifests and Condor logs are stored under `/data/users/akolchina/gfp_project/cross_protein_pool_size_seed42_46_runs`.

## Aubin addition

- Aubin 1-10-1 was registered as an additional base model on 7 November 2026 without changing the frozen source pool, target order, validation schedule or tests.
- CPU-only DAG **74545** completed **2,410/2,410 Aubin fits with zero failures**.
- A smoke fit completed successfully before submission.
- The final report combines MLP and Aubin across all five seeds: [`REPORT.md`](REPORT.md).

## Random versus acquisition extension

- Submitted 15 November 2026 as DAG **82312**. On 16 November that DAG was
  replaced by reduced resume DAG **92329**, preserving all completed results.
- Adds matched acquisition trajectories across both targets and seeds 42–46.
  Aubin and MLP retain all five plate sizes. CNN on OHE uses 48, 96, 192 and
  384; plate 9 was removed because its 214 sequential GPU fits dominated the
  completion time.
- Because the earlier CNN grid was cancelled, the same DAG also completes the
  missing random CNN comparator on the frozen random prefixes.
- Acquisition score: `0.62 × scaled hidden-space distance + 0.38 × scaled
  predictive variance`; Aubin/MLP use five-member ensembles and CNN uses 25
  MC-dropout predictions.
- Every primary fit is reinitialized and trained on the complete cumulative
  labelled set. Frozen validation and test records are unchanged.
- When the DAG completes, the existing R², Spearman and KL figures will be
  replaced by overlays: random cgreGFP/target in blue/orange and acquisition
  cgreGFP/target in pink/violet.
- Raw output: `/data/users/akolchina/gfp_project/cross_protein_pool_size_acquisition_seed42_46_runs`.
- The resume snapshot contained 3,017 unfinished jobs before finalization:
  2,642 acquisition rounds and 375 matched-random CNN fits.
