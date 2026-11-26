# Preliminary amacGFP active-learning benchmark

The benchmark implementation and frozen protocol are ready. It contains **48 resumable trajectories**:

- two models: CNN on OHE with 25-pass MC-dropout uncertainty and a ten-member MLP ensemble;
- eight acquisition strategies: Random, Uncertainty, Distance, High fitness, Low fitness, Uncertainty + high fitness, Uncertainty + low fitness, and Uncertainty + distance;
- three seeds (42–44), ten acquisition rounds and 96 new labels per round.

Within a seed, every architecture and strategy receives identical initial labels, validation records, candidate pool, unused records and held-out test. True candidate fitness is inaccessible to score calculation and is attached only after selection for retrospective diagnostics.

## Execution status

The 24 CNN GPU trajectories, 24 MLP CPU trajectories and dependent finalizer were submitted to HTCondor on 25 November 2026 as **DAG cluster 120946**. All 48 trajectory nodes were materialized successfully; the finalizer will start only after every trajectory completes.

Monitor the DAG with:

```bash
ssh lsv-submit 'condor_q -constraint "DAGManJobId == 120946 || ClusterId == 120946"'
```

Large per-seed predictions, pool scores, selected sequences, uncertainties, configurations and resumable checkpoints are written to `/data/users/akolchina/gfp_project/amac_active_learning_seed42_44`. Final tables, the report and publication figures are written to this Git-facing directory.

The MLP path passed a two-round end-to-end smoke run with ten independently initialized members. All eight score functions, percentile normalization, output schemas and Python syntax were also validated. The CNN path compiled successfully; its local CPU smoke run was stopped because MC-dropout inference is intentionally assigned to GPU production jobs.

[Frozen protocol](protocol.json)
