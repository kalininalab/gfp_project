# Cross-protein labelled-pool size experiment

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
- CPU-only DAG **74545** contains 2,410 Aubin fits and has a 25-job concurrency limit.
- A smoke fit completed successfully before submission.
