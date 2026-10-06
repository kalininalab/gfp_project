# Jannis OHE follow-up experiments

Experiment 1 compares five requested models on manuscript-style random 80/20
cgreGFP splits. Across seeds 42–46, Aubin has the highest mean Pearson and R²;
Jannis OHE has the highest mean Spearman. The test is never used for fitting or
selection. The acquisition comparison retains Jannis OHE because it was the
pre-registered one-seed choice and supplies MC-dropout uncertainty.

Experiment 2 compares acquisition against random sampling. Its random arm uses
nested subsets containing 10%, 20%, 40%, 60% and 80% of all cgreGFP records.
The matched fancy arm starts from the same 10% records and acquires examples by
`0.62 × scaled hidden-space distance + 0.38 × scaled MC-dropout variance` until
it reaches the same target sizes. At this full-dataset scale, global score
ranking replaces the student's dense spectral clustering, whose affinity matrix
has quadratic memory cost. This tests the acquisition formula without an
infeasible clustering step and records every selected record and score.

This weighting matches the student's executable default (`alpha=0.38` applied
to uncertainty). The draft manuscript swaps the two terms; that discrepancy is
recorded rather than silently conflated with the implemented method.

Every fit saves test predictions plus two figure families in PNG, PDF and SVG:
an overlaid true/predicted density histogram and a true-versus-predicted point
plot. Aggregate figures show mean ± sample SD across seeds.

Transfer experiments before and after AL must use the exact record IDs in
`results/transfer_cgreGFP_seed42_46/test_records.csv`. Acquisition, validation
and training must exclude every record in this manifest. Its SHA-256 is stored
in the transfer finalization manifest so the before/after comparison can reject
test-set drift.

```bash
# The training-size runner supplies the matched random-sampling arm.
condor_submit condor/jannis_training_size.sub
condor_submit condor/jannis_fancy_sampling.sub
python -m scripts.jannis_experiments.finalize
```
