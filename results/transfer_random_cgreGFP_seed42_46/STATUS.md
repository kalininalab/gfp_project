# Random versus acquisition-score transfer

**Status: running.** This five-seed experiment compares seeded random sampling
with the student acquisition score for CNN on OHE at identical label budgets.
Both arms use the same initial labelled records, validation records, round
schedule, and frozen natural/artificial test records.

The acquisition-score arm uses:

```text
0.62 × scaled projected hidden-space distance
+ 0.38 × scaled MC-dropout variance
```

It selects globally by descending score. The student's additional dense
SpectralClustering and 3-mer-diversity stage is not run at this dataset scale,
because its pool-by-pool affinity matrix has quadratic memory cost. Accordingly,
the figures use the precise label **Acquisition score ("Fancy")**, rather than
claiming that the full clustering pipeline was reproduced.

After all 40 random trajectories finish and pass validation, this directory will
contain:

- `sampling_ortholog_transfer.png` — transfer among GFP proteins;
- `sampling_peak_transfer.png` — natural cgreGFP/artificial-peak transfer;
- matching PDF and SVG files;
- `per_round_scores.csv` with all rounds and seeds.
