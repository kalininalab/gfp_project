# Cross-protein plate-size experiment with 7,350 cgreGFP training records

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
