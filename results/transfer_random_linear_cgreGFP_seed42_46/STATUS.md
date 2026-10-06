# Linear random versus acquisition-score transfer

**Status: running.** The 40 acquisition-score trajectories for Linear are
already complete in the main transfer-AL experiment. This matched CPU run adds
40 seeded-random trajectories: eight source settings across seeds 42–46.

Random and acquisition-score arms use the same initial labelled records, round
budgets, validation records and frozen target tests. Linear has no dropout, so
the acquisition-score arm ranks by projected hidden-space distance; its
uncertainty contribution is zero.

The finalizer will create:

- `sampling_ortholog_transfer.png`;
- `sampling_peak_transfer.png`;
- matching PDF and SVG files.
