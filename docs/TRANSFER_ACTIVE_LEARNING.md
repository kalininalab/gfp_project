# Transferability with active learning

This experiment repeats the transfer benchmark with active selection of labelled
source sequences. It uses the same five seeds (42–46), four models, source
mixtures, training budget, validation size, and frozen target test records as the
non-active-learning benchmark. Test records never enter training, validation,
acquisition scoring, or checkpoint selection.

## Protocol

Each trajectory starts with 10% of the final 14,709-record label budget and adds
labels over ten acquisition rounds, ending at exactly 14,709 records. For a
multi-protein source, the initial labelled set is allocated proportionally with
the same deterministic allocation used by the transfer benchmark.

The acquisition score reproduces the executable student implementation:

```text
0.62 × scaled distance + 0.38 × scaled uncertainty
```

Distance is the minimum Euclidean distance from a pool sequence to 256
deterministically sampled labelled anchors in a 32-dimensional projection of the
model's penultimate representation. Uncertainty is MC-dropout prediction
variance from 25 passes for CNN on OHE. In the original completed protocol,
models without dropout have zero uncertainty, so their ranking is
distance-based. Selection is global and stable by descending score.

### Ensemble-uncertainty extension for CPU models

The versioned extension in
`results/transfer_al_ensemble_cpu_seed42_46` supplies uncertainty for Aubin,
Linear and MLP with five-member ensembles. Aubin and Linear use the primary fit
plus four bootstrap-resampled fits of the labelled training set. MLP uses five
independently initialized fits on the same labelled records. For every pool
sequence, uncertainty is the sample variance of the five predictions. The
acquisition score remains exactly:

```text
0.62 × scaled projected-hidden-space distance
+ 0.38 × scaled ensemble prediction variance
```

Auxiliary ensemble members affect acquisition only. The primary fit produces
all frozen-test predictions and metrics. This keeps the reported base model
unchanged while adding model-specific epistemic uncertainty to selection. The
extension reuses the same seeds, initial labelled records, budgets, validation
records and frozen test manifest as the completed distance-only protocol.

Every round refits from scratch with the model seed and the same architecture,
optimizer, validation checkpointing, and epoch limit as the matched non-AL run.
The fixed seed makes the final-budget AL/non-AL comparison differ through the
selected training sequences rather than model initialization.

## Outputs

Each of the 160 trajectories saves metrics, selected record IDs and scores,
predictions, the fitted checkpoint, and two diagnostic plots for every round:

- `round_NN_TARGET_distribution.png`: true and predicted test distributions;
- `round_NN_TARGET_scatter.png`: true versus predicted test fluorescence.

There are 11 evaluated fits per trajectory: the initial fit (`round_00`) and ten
acquisition updates (`round_01` through `round_10`). Large per-run artifacts live
under `/data/users/akolchina/gfp_project_artifacts`; the repository contains the
frozen protocol, final tables, and publication figures.

## Reproduction

Prepare the frozen protocol once:

```bash
python -m scripts.transfer_active_learning.prepare
```

Submit 120 CPU trajectories, 40 GPU trajectories, and the finalizer:

```bash
condor_submit condor/transfer_al_cpu.sub
condor_submit condor/transfer_al_gpu.sub
condor_submit condor/transfer_al_finish.sub
```

The finalizer validates all trajectory hashes and all expected round/target
rows before creating AL-only figures and paired non-AL/AL figures. To regenerate
them after successful fits:

```bash
python -m scripts.transfer_active_learning.plot
```

The exact experiment parameters and test-manifest SHA-256 are recorded in
`results/transfer_al_cgreGFP_seed42_46/protocol.json`.

To reproduce the ensemble-uncertainty extension and its matched Random/Fancy
figures:

```bash
python -m scripts.transfer_active_learning.prepare_ensemble_cpu
condor_submit condor/transfer_al_ensemble_cpu.sub
condor_submit condor/transfer_al_ensemble_finish.sub
```

The final cgre-to-natural round also supplies the four-model comparison with
active learning. This comparison uses the transfer protocol's 14,709 labelled
training records, 4,903 validation records, and frozen 4,904-record test—not the
manuscript reproduction's validation-free 80/20 fitting protocol.

For the acquisition ablation, `CNN on OHE` is rerun with nested seeded random
selection. Random and fancy arms share the initial labelled set, round budgets,
five seeds, validation sets, and frozen target tests. Run it with:

```bash
python -m scripts.transfer_active_learning.prepare_random
condor_submit condor/transfer_random_gpu.sub
condor_submit condor/transfer_random_finish.sub
```

The finalizer produces adjacent Random/Fancy bars for the seven protein-source
mixtures and the four natural/artificial transfer directions.
