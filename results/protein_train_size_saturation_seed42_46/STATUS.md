# Four-protein training-size saturation

**Complete: 280/280 fits. [Open the results and figures](REPORT.md).** Status updated 27 November 2026. Submission details below are historical.

This experiment tests whether prediction saturates as more labelled sequences from the **same protein** are added. It runs separately for cgreGFP, amacGFP, ppluGFP and avGFP with MLP and CNN on OHE.

## Design

- Nested training sizes: **500, 1,000, 2,000, 4,000, 8,000, 12,000, and the complete available training set**.
- Five seeds (42–46).
- The original protein-specific validation and held-out test sets remain fixed at every size.
- Each larger subset contains the complete preceding subset within a seed.
- Models, optimizer settings, losses, early stopping and encodings match the four-protein experiments.

Complete training-set sizes are 14,709 cgreGFP, 20,106 amacGFP, 18,841 ppluGFP and 31,029 avGFP records.

## Execution

Submitted to HTCondor on 26 November 2026 as **DAG cluster 121041**:

- 20 MLP CPU trajectories;
- 20 CNN GPU trajectories;
- seven fits per trajectory, giving **280 fits**;
- one automatic finalizer after all trajectories finish.

The final report will contain Spearman, Pearson, R², RMSE, MAE, KL-divergence and Wasserstein-distance curves with mean ± SD across seeds. Large models and per-fit predictions are stored under `/data/users/akolchina/gfp_project/protein_train_size_saturation_seed42_46`.

```bash
ssh lsv-submit 'condor_q -constraint "DAGManJobId == 121041 || ClusterId == 121041"'
```
