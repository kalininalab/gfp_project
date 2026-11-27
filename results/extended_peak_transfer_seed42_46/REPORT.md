# Extended artificial-peak transferability

Status: submitted to Condor as DAG **121206**; results pending. This is separate from peak-to-cgreGFP saturation. The DAG caps concurrent GPU trajectories at 20 and CPU trajectories at 40.

Each of cgre1338, cgre4111, cgre132 and cgre9708 rotates as central. WT means the natural cgreGFP landscape, not a single reference sequence. Other means the remaining three artificial peaks.

Each budget produces four columns (central peaks), with two rows:

- Upper: WT → central, central → central, other → central.
- Lower: central → WT, central → central, central → other.

MLP (64 hidden neurons) and Jannis CNN on OHE are compared over seeds 42–46. Separate PNG/PDF/SVG panels will report Pearson, Spearman and R², mean ± seed SD. R² axes are limited to [−1, 1]; tables retain unclipped values.

## Data and validation

Every setting uses 1,000, 2,000 or 2,508 training records. The maximum common budget is set by cgre9708's original training partition. Training subsets are nested within each seed and shared by both architectures.

Validation has 836 source-only records, fixed across budgets. Cross-domain target landscapes are excluded from training and validation. In central → central, validation comes from a disjoint partition of that landscape. Source-validation MSE selects checkpoints.

Other-peak training contributions are 334/333/333, 667/667/666 or 836/836/836; validation contributions are 279/279/278. Single-landscape sources provide the whole budget.

Original 20% tests are frozen across seeds and models: cgreGFP 4,904; cgre1338 2,049; cgre4111 1,643; cgre132 854; cgre9708 836. Pooled other-peak metrics retain all test records and therefore weight peaks by test size.

Original processed fluorescence is used without exploratory WT-control rescaling. No active learning is used.

## Computation and reproduction

Nine distinct sources × two models × five seeds × three budgets = **270 fits**, grouped into 90 resumable trajectory jobs. WT fits are reused across target panels; central fits supply all their evaluation directions. Repeated central → central bars are the same results.

The existing transfer training recipes are retained: MLP at most 30 epochs, CNN at most 60; early-stopping patience 10. Saved artifacts include selected IDs, frozen tests, configuration/code hashes, training histories, checkpoints and predictions.

```bash
python -m scripts.extended_peak_transfer prepare
condor_submit_dag condor/extended_peak_transfer.dag
# Finalization runs as a dependent Condor job, or manually:
python -m scripts.extended_peak_transfer finalize
```

Preparation refuses to overwrite a frozen protocol. Raw outputs: `/data/users/akolchina/gfp_project/extended_peak_transfer_seed42_46`. Figures and tables will be written beside this report.
