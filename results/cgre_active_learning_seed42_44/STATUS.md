# Preliminary cgreGFP acquisition benchmark

This is the direct cgreGFP counterpart to the completed amacGFP acquisition benchmark.

## Design

- Models: CNN on OHE with 25-pass MC-dropout uncertainty; ten-member independently initialized MLP ensemble.
- Strategies: Random, Uncertainty, Distance, High fitness, Low fitness, Uncertainty + high fitness, Uncertainty + low fitness, and Uncertainty + distance.
- Three seeds (42–44), ten acquisition rounds and 96 labels per round.
- Per seed: 245 initial labels, 2,451 fixed validation records, 2,452 candidate records and 4,904 fixed held-out test records.
- All strategies and architectures share record IDs within a seed. True candidate fitness is attached only after selection for retrospective diagnostics.

## Execution

Submitted to HTCondor on 26 November 2026 as **DAG cluster 121082**:

- 24 MLP CPU trajectories;
- 24 CNN GPU trajectories;
- one automatic finalizer after all 48 trajectories complete.

Large per-seed predictions, models, pool scores and resumable checkpoints are written to `/data/users/akolchina/gfp_project/cgre_active_learning_seed42_44`. Final figures, tables and the report will be written here.

```bash
ssh lsv-submit 'condor_q -constraint "DAGManJobId == 121082 || ClusterId == 121082"'
```

[Frozen protocol](protocol.json)
