# Experiment timeline and figure finder

Use this page when the question is **“what did we run or plot most recently?”**
It is ordered newest first. For the scientific story organized by question,
use the [experiment catalog](EXPERIMENT_CATALOG.md).

## Latest activity

### 15 November 2026 — acquisition-sampled plate-size experiment submitted

The newest work is still running, so it has **no final figures yet**. It repeats
the cross-protein plate-size experiment with acquisition-selected target
sequences and adds CNN on OHE to both the random and acquisition arms. CNN uses
plates 48–384; the 214-round plate-9 CNN trajectories were removed on 16
November. Aubin and MLP retain plate 9.

- [14,709-cgreGFP status](../results/cross_protein_pool_size_seed42_46/STATUS.md)
- [7,350-cgreGFP status](../results/cross_protein_pool_size_cgre7350_seed42_46/STATUS.md)
- [Method and reproduction guide](POOL_SIZE_EXPERIMENT.md)

When complete, the finalizer will update the existing R², Spearman and KL
figures in the two pool-size result folders. Random cgreGFP/target curves remain
blue/orange; acquisition cgreGFP/target curves are added in pink/violet.

## Most recently completed figures

### 10 November 2026 — four-protein leave-one-out transfer

These are the most recently published main performance figures. The panels
predict each of cgreGFP, amacGFP, ppluGFP and avGFP from every combination of
the other three proteins. The target protein is named in each panel title.

- [Spearman figure](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_spearman.png)
- [Pearson figure](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_pearson.png)
- [R² figure](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_r2.png)
- [Report](../results/ortholog_leave_one_out_seed42_46/RESULTS.md)
- [Archived older layout](../results/ortholog_leave_one_out_seed42_46/archive_text_labels/)

### 16 November 2026 — rotating-central artificial-peak transfer completed

The 10 November interpretation was incorrect and is being replaced. Natural
`cgreGFP` is excluded. Each of `cgre1338`, `cgre4111`, `cgre132` and `cgre9708`
becomes the pseudo-natural central peak in turn, while the other three form its
artificial domain. Four directions are evaluated per panel on frozen tests.
All 160 fits completed successfully in Condor DAG `92517`; the old result is
archived.

- [Spearman figure](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_spearman.png)
- [Pearson figure](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_pearson.png)
- [R² figure](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_r2.png)
- [Report](../results/artificial_peak_holdout_seed42_46/RESULTS.md)
- [Superseded interpretation](../results/artificial_peak_holdout_seed42_46/archive_previous_interpretation/)

### 8 November 2026 — random cross-protein plate-size curves

These panels add nested random plates from ppluGFP or amacGFP to a fixed
cgreGFP source set. Each folder contains R², Spearman, KL and true/predicted
distribution figures for Aubin and MLP.

- [14,709-cgreGFP report](../results/cross_protein_pool_size_seed42_46/REPORT.md) · [all figures](../results/cross_protein_pool_size_seed42_46/figures/)
- [7,350-cgreGFP report](../results/cross_protein_pool_size_cgre7350_seed42_46/REPORT.md) · [all figures](../results/cross_protein_pool_size_cgre7350_seed42_46/figures/)

## Chronological experiment history

| Date | Experiment | What changed | Report and figures |
|---|---|---|---|
| 2026-11-15 | Acquisition-sampled plate sizes | Added matched random/acquisition arms and CNN for both cgreGFP source sizes; running | [Full-source status](../results/cross_protein_pool_size_seed42_46/STATUS.md) · [Half-source status](../results/cross_protein_pool_size_cgre7350_seed42_46/STATUS.md) |
| 2026-11-10 | Four-protein leave-one-out | Added avGFP; predict each protein from all combinations of the other three; improved panel layout | [Report](../results/ortholog_leave_one_out_seed42_46/RESULTS.md) · [figures](../results/ortholog_leave_one_out_seed42_46/) |
| 2026-11-16 | Rotating-central artificial peaks | Exclude natural cgreGFP; rotate four artificial peaks through the central role and test all four within/transfer directions; 160/160 fits complete | [Report](../results/artificial_peak_holdout_seed42_46/RESULTS.md) · [archived prior interpretation](../results/artificial_peak_holdout_seed42_46/archive_previous_interpretation/) |
| 2026-11-08 | Plate-size experiment, two source sizes | Completed random nested target additions for Aubin and MLP with plate sizes 9–384 | [14,709 source](../results/cross_protein_pool_size_seed42_46/REPORT.md) · [7,350 source](../results/cross_protein_pool_size_cgre7350_seed42_46/REPORT.md) |
| 2026-10-23 | Two-GMM-gated predictors | Compared probabilistic low/high gates with direct and hurdle prediction for CNN and linear OHE models | [CNN](../results/gmm_gated_cnn_seed42_46/REPORT.md) · [Linear](../results/gmm_gated_linear_seed42_46/REPORT.md) |
| 2026-10-21 | Distribution-aware predictor study | GMM/AIC/BIC analysis, hurdle models, ProteinBERT, ESM-2, XGBoost, MLP and CNN comparisons | [Combined report](../results/cgre_predictor_study/REPORT.md) · [mixture analysis](../results/cgre_peak_mixture_analysis/REPORT.md) · [neural hurdle](../results/hurdle_neural_seed42_46/REPORT.md) |
| 2026-10-20 | Initial hurdle pilot | Direct versus classification–regression prediction using ESM/Ridge | [Report](../results/hurdle_esm_ridge_seed42_46/REPORT.md) |
| 2026-10-09 | Kermut feasibility | Recorded Kermut integration and preprocessing status | [Status](../results/kermut_cgreGFP/STATUS.md) |
| 2026-10-08 | Complete target-adaptation presentation | Unified all target-adaptation scenarios, bar plots, 56 density panels and 56 dotplots | [Report](../results/target_adaptation_complete_report/README.md) · [diagnostic atlas](../results/target_adaptation_diagnostic_atlas/README.md) |
| 2026-10-07 | Four-model target adaptation | Combined Aubin, Linear, MLP and CNN; added 96-sequence, 960-sequence and one-shot density panels | [Combined figures](../results/target_adaptation_combined_seed42_46/) · [density panels](../results/target_adaptation_density_panels/) |
| 2026-10-06 | AL, uncertainty and target adaptation | Source-pool AL, random versus acquisition, ensemble uncertainty, protein transfer and cgreGFP/artificial adaptation | [AL report](../results/transfer_al_cgreGFP_seed42_46/RESULTS.md) · [uncertainty report](../results/transfer_ensemble_sampling_cpu_seed42_46/RESULTS.md) · [target adaptation](../results/target_adaptation_seed42_46/RESULTS.md) · [peak adaptation](../results/target_adaptation_peaks_seed42_46/RESULTS.md) |
| 2026-10-05 | First cgreGFP AL prototypes | Compared remaining-pool and fixed-test Pearson; retained as historical prototypes | [Original](../results/active_learning_cgre/RESULTS.md) · [fixed-test version](../results/active_learning_cgre_fixed_test/RESULTS.md) |
| 2026-10-05/06 | Manuscript benchmark and early ablations | Reproduced the 80/20 cgreGFP comparison; tested training-size and sampling variants | [80/20 reproduction](../results/paper_reproduction_cgre_80_20/RESULTS.md) · [training-size archive](../results/experiment_2_training_size/RESULTS.md) · [sampling archive](../results/experiment_2_sampling/RESULTS.md) |
| 2026-09-29 | Initial benchmark suite | ESM/OHE baselines, CV learning curves, three-seed transfer and prediction panels | [ESM benchmark](../results/esm_cgreGFP/RESULTS.md) · [learning curves](../results/learning_curves_cgre_cv10/RESULTS.md) · [three-seed transfer](../results/transfer_cgreGFP_seed42_44/RESULTS.md) |

## Complete result-folder inventory

This inventory includes publication reports, intermediate artifacts and
superseded runs. **Current** means it is a result we would normally show;
**supporting** means it feeds another report; **archive** means it is retained
for provenance and should not be mistaken for the latest result.

| Result folder | Role | Contents / where to start |
|---|---|---|
| [`active_learning_cgre`](../results/active_learning_cgre/) | Archive | First AL prototype; [results](../results/active_learning_cgre/RESULTS.md) |
| [`active_learning_cgre_fixed_test`](../results/active_learning_cgre_fixed_test/) | Archive | Fixed-test correction of the first prototype; [results](../results/active_learning_cgre_fixed_test/RESULTS.md) |
| [`artificial_peak_holdout_seed42_46`](../results/artificial_peak_holdout_seed42_46/) | Current | Four artificial peaks rotate through the central role; natural cgreGFP excluded; 160/160 fits complete; [report](../results/artificial_peak_holdout_seed42_46/RESULTS.md) |
| [`aubin_cgreGFP_seed42`](../results/aubin_cgreGFP_seed42/) | Supporting | Single-seed Aubin fit artifact |
| [`baselines_cgreGFP_seed42`](../results/baselines_cgreGFP_seed42/) | Supporting | Single-seed baseline artifacts |
| [`cgre_peak_mixture_analysis`](../results/cgre_peak_mixture_analysis/) | Current | GMM, AIC/BIC and measurement-floor analysis; [report](../results/cgre_peak_mixture_analysis/REPORT.md) |
| [`cgre_predictor_study`](../results/cgre_predictor_study/) | Current | Combined distribution-aware predictor study; [report](../results/cgre_predictor_study/REPORT.md) |
| [`cross_protein_pool_size_cgre7350_seed42_46`](../results/cross_protein_pool_size_cgre7350_seed42_46/) | Current | Plate-size experiment with 7,350 cgreGFP source records; [report](../results/cross_protein_pool_size_cgre7350_seed42_46/REPORT.md) · [figures](../results/cross_protein_pool_size_cgre7350_seed42_46/figures/) |
| [`cross_protein_pool_size_seed42_46`](../results/cross_protein_pool_size_seed42_46/) | Current | Plate-size experiment with 14,709 cgreGFP source records; [report](../results/cross_protein_pool_size_seed42_46/REPORT.md) · [figures](../results/cross_protein_pool_size_seed42_46/figures/) |
| [`cv10_cgreGFP_seed42`](../results/cv10_cgreGFP_seed42/) | Supporting | Single-seed CV artifacts used by benchmark reports |
| [`datasail_audit_cgreGFP`](../results/datasail_audit_cgreGFP/) | Supporting | Dataset-splitting audit |
| [`esm_cgreGFP`](../results/esm_cgreGFP/) | Current | ESM/CNN benchmark and prediction plots; [results](../results/esm_cgreGFP/RESULTS.md) |
| [`evaluation_cgreGFP_seed42`](../results/evaluation_cgreGFP_seed42/) | Supporting | Base-model evaluation plots |
| [`experiment_2_sampling`](../results/experiment_2_sampling/) | Archive | Early sampling comparison; [results](../results/experiment_2_sampling/RESULTS.md) |
| [`experiment_2_training_size`](../results/experiment_2_training_size/) | Archive | Student-style training-size ablation; [results](../results/experiment_2_training_size/RESULTS.md) |
| [`experiment_3_sampling`](../results/experiment_3_sampling/) | Archive | Earlier random/Fancy sampling experiment; [results](../results/experiment_3_sampling/RESULTS.md) |
| [`gmm_gated_cnn_seed42_46`](../results/gmm_gated_cnn_seed42_46/) | Current | Two-GMM-gated CNN; [report](../results/gmm_gated_cnn_seed42_46/REPORT.md) |
| [`gmm_gated_linear_seed42_46`](../results/gmm_gated_linear_seed42_46/) | Current | Two-GMM-gated linear predictor; [report](../results/gmm_gated_linear_seed42_46/REPORT.md) |
| [`hurdle_esm_ridge_seed42_46`](../results/hurdle_esm_ridge_seed42_46/) | Supporting | First hurdle pilot; [report](../results/hurdle_esm_ridge_seed42_46/REPORT.md) |
| [`hurdle_neural_seed42_46`](../results/hurdle_neural_seed42_46/) | Current | MLP/CNN direct-versus-hurdle comparison; [report](../results/hurdle_neural_seed42_46/REPORT.md) |
| [`hurdle_xgboost_seed42_46`](../results/hurdle_xgboost_seed42_46/) | Supporting | XGBoost hurdle raw summaries, incorporated into predictor study |
| [`jannis_ohe_cgreGFP`](../results/jannis_ohe_cgreGFP/) | Supporting | Jannis OHE benchmark artifacts |
| [`kermut_cgreGFP`](../results/kermut_cgreGFP/) | Incomplete | Kermut feasibility/preprocessing; [status](../results/kermut_cgreGFP/STATUS.md) |
| [`knn_cgreGFP_seed42`](../results/knn_cgreGFP_seed42/) | Supporting | KNN baseline artifact |
| [`learning_curves_cgre_cv10`](../results/learning_curves_cgre_cv10/) | Current | Ten-fold learning curves for cgreGFP and peaks; [results](../results/learning_curves_cgre_cv10/RESULTS.md) |
| [`ortholog_leave_one_out_seed42_46`](../results/ortholog_leave_one_out_seed42_46/) | Current | Four-protein leave-one-out; [report](../results/ortholog_leave_one_out_seed42_46/RESULTS.md) |
| [`paper_reproduction_cgre_80_20`](../results/paper_reproduction_cgre_80_20/) | Current | Manuscript 80/20 base-model comparison; [results](../results/paper_reproduction_cgre_80_20/RESULTS.md) |
| [`proteinbert_models_seed42_46`](../results/proteinbert_models_seed42_46/) | Supporting | ProteinBERT model outputs incorporated into predictor study |
| [`target_adaptation_cnn_seed42_46`](../results/target_adaptation_cnn_seed42_46/) | Current | CNN target-protein adaptation; [results](../results/target_adaptation_cnn_seed42_46/RESULTS.md) |
| [`target_adaptation_combined_seed42_46`](../results/target_adaptation_combined_seed42_46/) | Current | Combined four-model target-adaptation figures |
| [`target_adaptation_complete_report`](../results/target_adaptation_complete_report/) | Current | Presentation-level index of all target-adaptation results; [report](../results/target_adaptation_complete_report/README.md) |
| [`target_adaptation_density_panels`](../results/target_adaptation_density_panels/) | Current | Selected 96/960/one-shot density panels |
| [`target_adaptation_diagnostic_atlas`](../results/target_adaptation_diagnostic_atlas/) | Current | All 56 density and 56 dotplot panels; [index](../results/target_adaptation_diagnostic_atlas/README.md) |
| [`target_adaptation_peaks_cnn_seed42_46`](../results/target_adaptation_peaks_cnn_seed42_46/) | Current | CNN natural/artificial target adaptation; [results](../results/target_adaptation_peaks_cnn_seed42_46/RESULTS.md) |
| [`target_adaptation_peaks_seed42_46`](../results/target_adaptation_peaks_seed42_46/) | Current | CPU-model natural/artificial target adaptation; [results](../results/target_adaptation_peaks_seed42_46/RESULTS.md) |
| [`target_adaptation_seed42_46`](../results/target_adaptation_seed42_46/) | Current | CPU-model target-protein adaptation; [results](../results/target_adaptation_seed42_46/RESULTS.md) |
| [`transfer_al_cgreGFP_seed42_46`](../results/transfer_al_cgreGFP_seed42_46/) | Current | Source-pool AL and non-AL comparison; [results](../results/transfer_al_cgreGFP_seed42_46/RESULTS.md) |
| [`transfer_al_ensemble_cpu_seed42_46`](../results/transfer_al_ensemble_cpu_seed42_46/) | Supporting | Frozen protocols and raw tables for ensemble-uncertainty transfer |
| [`transfer_cgreGFP_seed42_44`](../results/transfer_cgreGFP_seed42_44/) | Archive | Original three-seed transfer and individual prediction plots; [results](../results/transfer_cgreGFP_seed42_44/RESULTS.md) |
| [`transfer_cgreGFP_seed42_46`](../results/transfer_cgreGFP_seed42_46/) | Current | Five-seed transfer before AL; [results](../results/transfer_cgreGFP_seed42_46/RESULTS.md) |
| [`transfer_cgreGFP_seed42_51`](../results/transfer_cgreGFP_seed42_51/) | Archive | Incomplete/older ten-seed transfer artifacts |
| [`transfer_ensemble_sampling_cpu_seed42_46`](../results/transfer_ensemble_sampling_cpu_seed42_46/) | Current | Aubin/Linear/MLP random versus acquisition with ensemble uncertainty; [results](../results/transfer_ensemble_sampling_cpu_seed42_46/RESULTS.md) |
| [`transfer_random_aubin_cgreGFP_seed42_46`](../results/transfer_random_aubin_cgreGFP_seed42_46/) | Supporting | Aubin matched-random protocol artifacts |
| [`transfer_random_cgreGFP_seed42_46`](../results/transfer_random_cgreGFP_seed42_46/) | Incomplete | CNN random-versus-acquisition status; [status](../results/transfer_random_cgreGFP_seed42_46/STATUS.md) |
| [`transfer_random_cpu_models_cgreGFP_seed42_46`](../results/transfer_random_cpu_models_cgreGFP_seed42_46/) | Archive | Earlier distance-only CPU comparison; [results](../results/transfer_random_cpu_models_cgreGFP_seed42_46/RESULTS.md) |
| [`transfer_random_cpu_models_condor`](../results/transfer_random_cpu_models_condor/) | Supporting | Condor-local bookkeeping, not a scientific result |
| [`transfer_random_linear_cgreGFP_seed42_46`](../results/transfer_random_linear_cgreGFP_seed42_46/) | Archive | Linear-only random/Fancy comparison; [status and figures](../results/transfer_random_linear_cgreGFP_seed42_46/STATUS.md) |
| [`transfer_random_mlp_cgreGFP_seed42_46`](../results/transfer_random_mlp_cgreGFP_seed42_46/) | Supporting | MLP matched-random protocol artifacts |

The repository check `python scripts/check_experiment_index.py` fails if a new
top-level result folder is added without being linked from this page.
