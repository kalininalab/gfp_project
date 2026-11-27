# Experiment catalog by scientific question

This page organizes the scientific story by question, so it is not
chronological. To find the newest experiment, remember what was plotted last,
or inspect every result folder including archives, open the
**[experiment timeline and figure finder](EXPERIMENT_TIMELINE.md)**. For the
project overview, dataset description and first reproduction steps, start with
the [main README](../README.md).

## Experimental story

Proposed follow-up (not launched): [Library-level active learning for GFP landscape mapping](LIBRARY_LEVEL_ACTIVE_LEARNING.md) — adaptively choose parental peaks and error-prone PCR conditions for the next measured batch.

The experiments answer four questions in sequence. Throughout, the **target
test set is frozen**, is shared by all compared methods, and is never available
for training or acquisition.

### Results navigation

- [cgreGFP MLP classification and regression](../results/valley_mlp_cgre_seed42_46/REPORT.md): hard/soft routing with separate MLPs, KDE valley boundary, direct comparator and true-class routing diagnostic; five seeds, results pending.
- [Extended artificial-peak transferability](../results/extended_peak_transfer_seed42_46/REPORT.md): WT/central/other directions, source-only validation, MLP/CNN, five seeds and three matched training budgets; results pending.
- **[Train-only fluorescence distribution analysis for cgreGFP and artificial peaks](../results/cgre_peak_mixture_analysis/REPORT.md):** 1–4 component GMM comparison, measurement-floor audit, bootstrap stability and implications for a binary-first predictor.
- **[Classification–regression predictor pilot](../results/hurdle_esm_ridge_seed42_46/REPORT.md):** matched direct versus soft/hard hurdle prediction on all five frozen tests.
- **[Neural classification–regression experiment](../results/hurdle_neural_seed42_46/REPORT.md):** five-seed MLP-on-ESM and CNN-on-OHE matched direct-versus-hurdle comparisons.
- **[cgreGFP distribution and predictor study](../results/cgre_predictor_study/REPORT.md):** two-part report covering Gaussian-mixture evidence and matched ProteinBERT/ESM-2/OHE predictor comparisons with R² and distribution metrics.
- **[Two-GMM-gated CNN experiment](../results/gmm_gated_cnn_seed42_46/REPORT.md):** completed 25/25 train-only low/high Gaussian-boundary fits with two regression experts, compared against direct and measurement-floor-hurdle CNNs on identical frozen tests.
- **[Two-GMM-gated linear experiment](../results/gmm_gated_linear_seed42_46/REPORT.md):** the same train-only split with positional-OHE logistic/Ridge components; gating helps three of five landscapes but remains below the direct CNN.
- **[Complete target-adaptation report](../results/target_adaptation_complete_report/README.md):** R² and other metrics, bar plots and links to prediction diagnostics.
- [Scenario-by-scenario diagnostic atlas](../results/target_adaptation_diagnostic_atlas/README.md)
- [All 56 true/predicted distribution panels](../results/target_adaptation_diagnostic_atlas/distributions/)
- [All 56 true-vs-predicted dotplot panels](../results/target_adaptation_diagnostic_atlas/dotplots/)

The atlas covers all 8 transfer directions, 7 main training/acquisition
scenarios, 4 models and 5 seeds. Filenames state the source, target, selection
method, round and number of target sequences added.

| Stage | What enters training after the initial fit? | Target sequences added? |
|---|---|---|
| Transfer without AL | Nothing | **No** |
| Source-pool AL transfer | Sequences selected from the original **source-protein training pool** | **No** |
| Target adaptation | Sequences selected from a separate labelled pool of the **target protein/domain** | **Yes: 96 per round for 10 rounds** |
| One-shot target-adaptation control | 960 target-pool sequences added once | **Yes: 960 once, no AL rounds** |

### 1. Establish the base models without active learning

We first compare Aubin, Linear, MLP and CNN on OHE without iterative
acquisition. These fits establish predictive accuracy on natural cgreGFP and
the initial transferability between proteins and between natural cgreGFP and
artificial peaks.

Every fit uses the same fixed budget: **14,709 source-training sequences** and
**4,903 source-validation sequences**. The frozen evaluation sets contain
**4,904 natural cgreGFP sequences** and, where applicable, **5,382 sequences
from the four artificial peaks combined**. No target-test sequence enters
training or validation. Source composition is:

| Source training condition | Training composition | Validation composition |
|---|---:|---:|
| cgreGFP, amacGFP or ppluGFP alone | 14,709 from that protein | 4,903 from that protein |
| Any two natural proteins | 7,355 + 7,354 | 2,452 + 2,451 |
| cgreGFP + amacGFP + ppluGFP | 4,903 each | 1,635 cgreGFP + 1,634 amacGFP + 1,634 ppluGFP |
| Four artificial peaks | 2,333 cgre132 + 5,599 cgre1338 + 4,491 cgre4111 + 2,286 cgre9708 | 777 + 1,867 + 1,497 + 762, respectively |

The exact per-run sizes are retained in the
[summary CSV](../results/transfer_cgreGFP_seed42_46/summary.csv), and the full
results and sampling explanation are in the
[transfer report](../results/transfer_cgreGFP_seed42_46/RESULTS.md).

![Transfer without AL between GFP proteins](../results/transfer_cgreGFP_seed42_46/ortholog_mixtures_spearman.png)

#### Predict each of four GFP proteins from the other three

The four-protein follow-up adds `avGFP` and applies a strict
leave-one-protein-out design. Each of `cgreGFP`, `amacGFP`, `ppluGFP` and
`avGFP` becomes the frozen target in turn, while all seven non-empty
combinations of the other three proteins are tested as sources. The target is
absent from training and validation; every fit keeps the same 14,709-training
and 4,903-validation source budget.

**[Open the complete four-protein transfer report](../results/ortholog_leave_one_out_seed42_46/RESULTS.md)**

![Four-protein leave-one-out Spearman](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_spearman.png)

![Four-protein leave-one-out Pearson](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_pearson.png)

![Four-protein leave-one-out R2](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_r2.png)

![Transfer without AL between natural cgreGFP and artificial peaks](../results/transfer_cgreGFP_seed42_46/artificial_peak_transfer_spearman.png)

#### Rotate the central role among four artificial peaks

Natural `cgreGFP` is excluded. Each of the four largest artificial landscapes
(`cgre1338`, `cgre4111`, `cgre132`, `cgre9708`) becomes the pseudo-natural
central peak in turn; the remaining three form its pooled other-peaks domain.
Each panel tests central→central, central→other peaks, other peaks→other peaks
and other peaks→central using equal source budgets and frozen tests. The
corrected 160-fit benchmark completed as Condor DAG `92517`; the earlier
interpretation is archived inside the result folder.

**[Open the rotating-central artificial-peak report](../results/artificial_peak_holdout_seed42_46/RESULTS.md)**

![Held-out artificial peaks: Spearman](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_spearman.png)

![Held-out artificial peaks: Pearson](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_pearson.png)

![Held-out artificial peaks: R2](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_r2.png)

The supervised random-pool plate-size experiments are reported for
[14,709 cgreGFP training records](../results/cross_protein_pool_size_seed42_46/REPORT.md)
and [7,350 cgreGFP training records](../results/cross_protein_pool_size_cgre7350_seed42_46/REPORT.md).
Target-pool order, validation schedules and frozen tests are matched between
the two reports.

### 2. Ask whether active selection improves source-only transfer

Here AL selects additional records from the **source training pool**. It does
**not add any sequence from the target protein or target domain**. Each
trajectory starts at 10% of the 14,709-record source budget and reaches 14,709
source records over ten rounds. The target test remains untouched.

![Non-AL versus source-pool AL between GFP proteins](../results/transfer_al_cgreGFP_seed42_46/ortholog_mixtures_spearman_non_al_vs_al.png)

![Non-AL versus source-pool AL between natural and artificial cgreGFP](../results/transfer_al_cgreGFP_seed42_46/artificial_peak_transfer_spearman_non_al_vs_al.png)

### 3. Test the acquisition rule and its uncertainty term

Random and Fancy arms use identical initial source records, round budgets and
frozen tests. Fancy ranks the source pool by
`0.62 × scaled distance + 0.38 × scaled uncertainty`. For Aubin and Linear,
uncertainty is variance across a five-member bootstrap ensemble; for MLP it is
variance across five independently initialized fits. All **120/120**
ensemble-uncertainty trajectories are complete. This experiment still adds
**zero target sequences**.

![Ensemble uncertainty between GFP proteins](../results/transfer_ensemble_sampling_cpu_seed42_46/sampling_ortholog_transfer_cpu_models.png)

![Ensemble uncertainty between natural cgreGFP and artificial peaks](../results/transfer_ensemble_sampling_cpu_seed42_46/sampling_peak_transfer_cpu_models.png)

### 4. Add target sequences: target adaptation

This is the experiment in which target data actually enter training. Starting
from a source-trained model, the iterative arms add **96 selected target-pool
sequences after every round for 10 rounds**, for a total of 960. The matched
one-shot controls add **960 target-pool sequences once** and have no acquisition
rounds. Random and Fancy use the same initial fit, total target-label budget,
validation records and frozen target test.

All four models are shown together below. Aubin, Linear and MLP are complete.
CNN protein-to-protein adaptation is also complete; every row uses all five
seeds.

![Target adaptation for all four models between GFP proteins](../results/target_adaptation_combined_seed42_46/target_adaptation_all_models.png)

All four models are complete for natural cgreGFP ↔ artificial peaks:

![Target adaptation for all four models between natural and artificial cgreGFP](../results/target_adaptation_combined_seed42_46/target_adaptation_peaks_all_models.png)

The density panels below show the same `cgreGFP → amacGFP` iterative-Fancy
trajectory after adding the first 96 target sequences and after all ten rounds
(960 target sequences). Curves use the frozen amacGFP test; predicted densities
pool five model seeds, while the true test density is shown once.

![True and predicted densities after adding 96 target sequences](../results/target_adaptation_density_panels/cgre_to_amac_round01_96_target_density.png)

![True and predicted densities after adding 960 target sequences](../results/target_adaptation_density_panels/cgre_to_amac_round10_960_target_density.png)

For the matched non-AL control, 960 target sequences are sampled randomly and
added in a single fit. There is no Fancy score and there are no iterative
refitting/acquisition rounds:

![True and predicted densities after one-shot random addition of 960 target sequences](../results/target_adaptation_density_panels/cgre_to_amac_oneshot_random_960_target_density.png)

The same one-shot random control for transfer between natural cgreGFP and its
artificial peaks:

![Natural cgreGFP to artificial peaks after one-shot random addition](../results/target_adaptation_density_panels/cgre_to_artificial_oneshot_random_960_target_density.png)

![Artificial peaks to natural cgreGFP after one-shot random addition](../results/target_adaptation_density_panels/artificial_to_cgre_oneshot_random_960_target_density.png)

The complete diagnostic atlas contains every direction and all seven main
conditions, with densities and dotplots in separate folders:

- [Browse all 56 distribution panels](../results/target_adaptation_diagnostic_atlas/distributions/)
- [Browse all 56 true-vs-predicted dotplot panels](../results/target_adaptation_diagnostic_atlas/dotplots/)
- [Scenario-by-scenario index](../results/target_adaptation_diagnostic_atlas/README.md)

## Complete result index

| Experiment | What was done | Tables | Main figures |
|---|---|---|---|
| cgreGFP distribution and predictor study | Gaussian-mixture analysis followed by matched ProteinBERT, ESM-2 and OHE comparisons using MLP, XGBoost and CNN; direct and soft-hurdle prediction; 5 seeds | [Report](../results/cgre_predictor_study/REPORT.md) · [Summary CSV](../results/cgre_predictor_study/summary.csv) · [All seeded metrics](../results/cgre_predictor_study/all_metrics.csv) | [Model comparison](../results/cgre_predictor_study/model_comparison.png) · [Distributions and dot plots](../results/cgre_predictor_study/best_model_diagnostics.png) |
| Epistatic fraction by mutation count | Existing additive single-mutant epistasis calls for cgreGFP, ppluGFP and amacGFP; epistatic count divided by all sequences at each mutation count | [Report](../results/epistasis_by_mutation_count/REPORT.md) · [Complete table](../results/epistasis_by_mutation_count/epistasis_by_mutation_count.csv) | [Epistatic percentage](../results/epistasis_by_mutation_count/epistatic_fraction_by_mutation_count.png) · [Classification coverage](../results/epistasis_by_mutation_count/epistasis_classification_coverage.png) |
| Two-GMM-gated CNN | Train-only GMM low/high boundary, probabilistic gate and separate low/high regression experts; direct and exact-floor-hurdle controls; 5 landscapes × 5 seeds | [Report](../results/gmm_gated_cnn_seed42_46/REPORT.md) · [Summary CSV](../results/gmm_gated_cnn_seed42_46/summary.csv) · [All seeded metrics](../results/gmm_gated_cnn_seed42_46/all_metrics.csv) | [CNN comparison](../results/gmm_gated_cnn_seed42_46/comparison.png) |
| Two-GMM-gated linear predictor | Positional OHE with logistic low/high gate and separate Ridge experts; direct and exact-floor-hurdle controls; 5 landscapes × 5 GMM seeds | [Report](../results/gmm_gated_linear_seed42_46/REPORT.md) · [Summary CSV](../results/gmm_gated_linear_seed42_46/summary.csv) · [All seeded metrics](../results/gmm_gated_linear_seed42_46/all_metrics.csv) | [Linear comparison](../results/gmm_gated_linear_seed42_46/comparison.png) |
| cgreGFP benchmark | Fixed 60/20/20 holdout and 10-fold CV; 24,516 sequences; 121 ESM fits + 11 Jannis OHE fits + sequence baselines | [Report](BENCHMARK_RESULTS.md) · [CSV](../results/esm_cgreGFP/comparison.csv) | [CNN predictions](../results/esm_cgreGFP/cv_cnn_true_vs_predicted.png) · [Mean-ESM predictions](../results/esm_cgreGFP/cv_mean_true_vs_predicted.png) · [Accuracy/runtime](../results/esm_cgreGFP/accuracy_vs_runtime.png) |
| Transfer before AL | Aubin, Linear, MLP and CNN on OHE; seven ortholog mixtures and four natural/artificial directions; 5 seeds | [Report](../results/transfer_cgreGFP_seed42_46/RESULTS.md) · [CSV](../results/transfer_cgreGFP_seed42_46/summary.csv) | [GFP proteins](../results/transfer_cgreGFP_seed42_46/ortholog_mixtures_spearman.png) · [Artificial peaks](../results/transfer_cgreGFP_seed42_46/artificial_peak_transfer_spearman.png) |
| Four-protein leave-one-out transfer | cgreGFP, amacGFP, ppluGFP and avGFP each become the frozen target; all 7 combinations of the other 3 proteins; 4 models × 5 seeds | [Report](../results/ortholog_leave_one_out_seed42_46/RESULTS.md) · [Summary CSV](../results/ortholog_leave_one_out_seed42_46/summary.csv) · [All seeded metrics](../results/ortholog_leave_one_out_seed42_46/per_run_scores.csv) | [Spearman](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_spearman.png) · [Pearson](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_pearson.png) · [R²](../results/ortholog_leave_one_out_seed42_46/four_protein_holdout_r2.png) |
| Rotating-central artificial-peak transfer | Natural cgreGFP excluded; each of four artificial peaks becomes central; four within/transfer directions; 4 models × 5 seeds; 160/160 fits complete | [Report](../results/artificial_peak_holdout_seed42_46/RESULTS.md) · [Summary CSV](../results/artificial_peak_holdout_seed42_46/summary.csv) · [All seeded metrics](../results/artificial_peak_holdout_seed42_46/per_run_scores.csv) · [Archived prior result](../results/artificial_peak_holdout_seed42_46/archive_previous_interpretation/) | [Spearman](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_spearman.png) · [Pearson](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_pearson.png) · [R²](../results/artificial_peak_holdout_seed42_46/artificial_peak_holdout_r2.png) |
| Transfer with AL | Same models, seeds, budgets and frozen tests; 10 acquisition rounds; 160/160 trajectories | [Results](../results/transfer_al_cgreGFP_seed42_46/RESULTS.md) · [CSV](../results/transfer_al_cgreGFP_seed42_46/summary.csv) · [Protocol](TRANSFER_ACTIVE_LEARNING.md) | [GFP proteins](../results/transfer_al_cgreGFP_seed42_46/ortholog_mixtures_spearman.png) · [Non-AL vs AL](../results/transfer_al_cgreGFP_seed42_46/ortholog_mixtures_spearman_non_al_vs_al.png) · [Artificial peaks](../results/transfer_al_cgreGFP_seed42_46/artificial_peak_transfer_spearman.png) |
| Preliminary amacGFP acquisition benchmark | CNN MC dropout and 10-member MLP ensemble; 8 acquisition strategies; 3 seeds; 10 × 96 labels; matched initial sets, pools and tests | [Status](../results/amac_active_learning_seed42_44/STATUS.md) · [Frozen protocol](../results/amac_active_learning_seed42_44/protocol.json) | Generated automatically after all trajectories complete |
| Preliminary cgreGFP acquisition benchmark | Matched amacGFP design on cgreGFP: CNN MC dropout and 10-member MLP ensemble; 8 strategies; 3 seeds; 10 × 96 labels | [Live status](../results/cgre_active_learning_seed42_44/STATUS.md) · [Frozen protocol](../results/cgre_active_learning_seed42_44/protocol.json) | Generated automatically after all trajectories complete |
| Four-protein training-size saturation | Within-protein MLP and CNN fits at 500–12,000 plus the complete training set; nested subsets; fixed validation/test; 5 seeds | [Live status](../results/protein_train_size_saturation_seed42_46/STATUS.md) | Generated automatically after all 280 fits complete |
| Random vs acquisition score ("Fancy") transfer | CNN on OHE; matched initial sets, budgets, seeds and frozen tests | [Live status and exact method](../results/transfer_random_cgreGFP_seed42_46/STATUS.md) · [Frozen protocol](../results/transfer_random_cgreGFP_seed42_46/protocol.json) | Final CNN figures are not published; use the status page |
| Random vs acquisition score transfer — Linear | 40 acquisition-score and 40 matched random CPU trajectories; 5 seeds | [Status](../results/transfer_random_linear_cgreGFP_seed42_46/STATUS.md) · [Frozen protocol](../results/transfer_random_linear_cgreGFP_seed42_46/protocol.json) | [GFP proteins](../results/transfer_random_linear_cgreGFP_seed42_46/sampling_ortholog_transfer.png) · [Natural/artificial peaks](../results/transfer_random_linear_cgreGFP_seed42_46/sampling_peak_transfer.png) |
| Ensemble uncertainty for non-CNN models | Aubin, Linear and MLP; 5-member ensembles; matched Random/Fancy comparison; 120/120 trajectories | [Results and method](../results/transfer_ensemble_sampling_cpu_seed42_46/RESULTS.md) · [Protocol](../results/transfer_al_ensemble_cpu_seed42_46/protocol.json) | [GFP proteins](../results/transfer_ensemble_sampling_cpu_seed42_46/sampling_ortholog_transfer_cpu_models.png) · [Natural/artificial peaks](../results/transfer_ensemble_sampling_cpu_seed42_46/sampling_peak_transfer_cpu_models.png) |
| Target-protein adaptation | 96 target sequences × 10 or 960 once; fancy/random; 6 directed protein pairs; 3 CPU models; 5 seeds; 90/90 jobs | [Results](../results/target_adaptation_seed42_46/RESULTS.md) · [CSV](../results/target_adaptation_seed42_46/summary.csv) · [Design](TARGET_ADAPTATION.md) | [Four-arm comparison](../results/target_adaptation_seed42_46/target_adaptation_comparison.png) |
| Natural cgreGFP/artificial-peak adaptation | Same 96 × 10 versus 960-once design in both directions; 3 CPU models; 5 seeds; 30/30 jobs | [Results](../results/target_adaptation_peaks_seed42_46/RESULTS.md) · [CSV](../results/target_adaptation_peaks_seed42_46/summary.csv) · [Design](TARGET_ADAPTATION.md) | [Four-arm comparison](../results/target_adaptation_peaks_seed42_46/target_adaptation_comparison.png) |
| CNN target-protein adaptation | CNN on OHE; 96 × 10 versus 960 once; fancy/random; 6 directed protein pairs; 5 seeds; 30/30 complete | [Results](../results/target_adaptation_cnn_seed42_46/RESULTS.md) · [CSV](../results/target_adaptation_cnn_seed42_46/summary.csv) | [Four-arm comparison](../results/target_adaptation_cnn_seed42_46/target_adaptation_comparison.png) |
| CNN natural/artificial adaptation | CNN on OHE; same four arms in both directions; 5 seeds; 10/10 complete | [Results](../results/target_adaptation_peaks_cnn_seed42_46/RESULTS.md) · [CSV](../results/target_adaptation_peaks_cnn_seed42_46/summary.csv) | [Four-arm comparison](../results/target_adaptation_peaks_cnn_seed42_46/target_adaptation_comparison.png) |

<details>
<summary>Detailed result notes and archived intermediate figures</summary>

[Selected models: cgre → cgre scatter panel](../results/transfer_cgreGFP_seed42_44/cgre_true_vs_predicted/cgre_test_all_seeds.png)
shows individual saved predictions from all three cgre-only control fits.
[PDF](../results/transfer_cgreGFP_seed42_44/cgre_true_vs_predicted/cgre_test_seed42.pdf) ·
[All seeds and formats](../results/transfer_cgreGFP_seed42_44/cgre_true_vs_predicted/) ·
[Metrics](../results/transfer_cgreGFP_seed42_44/cgre_true_vs_predicted/metrics.csv).

[amacGFP + ppluGFP → cgreGFP scatter panel](../results/transfer_cgreGFP_seed42_44/amac_pplu_to_cgre_true_vs_predicted/amac_pplu_to_cgre_all_seeds.png)
shows the four selected models across all three seeds on the same 4,904 held-out
cgreGFP sequences ([metrics](../results/transfer_cgreGFP_seed42_44/amac_pplu_to_cgre_true_vs_predicted/metrics.csv)).

**Best measured accuracy: Jannis CNN on OHE** (CV RMSE 0.227, R² 0.912,
Spearman 0.900). **Efficient default: Aubin 1–10–1 on OHE** (RMSE 0.234,
R² 0.908): its holdout fit takes 65 seconds on CPU versus 20.7 minutes on GPU
for Jannis OHE. Deep MLP has nearly the same ranking score (0.899) at lower cost.
Small differences are descriptive, not claims of statistical significance.

The transfer experiment compares Aubin, Linear, MLP and CNN on OHE over five
seeds. All models use aligned one-hot inputs and identical source budgets.

![Transferability of models between GFP proteins](../results/transfer_cgreGFP_seed42_46/ortholog_mixtures_spearman.png)

![Transferability between natural cgreGFP and artificial peaks](../results/transfer_cgreGFP_seed42_46/artificial_peak_transfer_spearman.png)

Transfer uses **14,709 train / 4,903 validation** records per fit, with fixed
**20% target holdouts**: 4,904 natural or 5,382 artificial sequences. Sources
contribute equally in ortholog mixtures. Error bars show SD across five seeds;
validation uses training sources only. Artificial → artificial tests within the
four represented peaks, not an unseen peak. The natural test was used in earlier
comparisons, so these transfer results are exploratory.

### Active-learning transfer

All **160 five-seed trajectories** for Aubin, Linear, MLP and CNN on OHE are
complete. Each trajectory has an initial fit plus ten acquisition rounds and ends
at the same 14,709-label budget as the non-AL transfer benchmark. AL and non-AL
use the exact same frozen test records. Bars show the mean and error bars show
sample SD across seeds 42–46.

**Matched model selection without and with AL on frozen natural cgreGFP test:**

![Non-AL versus AL model comparison](../results/transfer_al_cgreGFP_seed42_46/model_comparison_al.png)

**Transfer between GFP proteins after AL:**

![AL transfer between GFP proteins](../results/transfer_al_cgreGFP_seed42_46/ortholog_mixtures_spearman.png)

**Matched non-AL and AL transfer between GFP proteins:**

![Non-AL versus AL transfer between GFP proteins](../results/transfer_al_cgreGFP_seed42_46/ortholog_mixtures_spearman_non_al_vs_al.png)

**Transfer between natural cgreGFP and artificial peaks after AL:**

![AL transfer between natural cgreGFP and artificial peaks](../results/transfer_al_cgreGFP_seed42_46/artificial_peak_transfer_spearman.png)

**Matched non-AL and AL transfer between natural cgreGFP and artificial peaks:**

![Non-AL versus AL peak transfer](../results/transfer_al_cgreGFP_seed42_46/artificial_peak_transfer_spearman_non_al_vs_al.png)

### Target-protein adaptation

The next experiment adds 96 labelled target sequences over ten iterative rounds
or adds 960 sequences once. It compares random and acquisition-score selection
for all six directed natural-protein pairs and both natural/artificial directions.

![Natural-protein target adaptation](../results/target_adaptation_seed42_46/target_adaptation_comparison.png)

![Natural/artificial target adaptation](../results/target_adaptation_peaks_seed42_46/target_adaptation_comparison.png)

### Random versus acquisition score ("Fancy") transfer

This is a separate five-seed CNN-on-OHE experiment for both GFP-protein transfer
and natural cgreGFP/artificial-peak transfer. Random and acquisition-score arms
share the initial labelled records, label budgets, validation sets and frozen
tests. [Current status and exact method](../results/transfer_random_cgreGFP_seed42_46/STATUS.md).

The acquisition arm uses `0.62 × scaled hidden-space distance + 0.38 × scaled
MC-dropout variance`, followed by global descending-score selection. It does not
run the student's dense SpectralClustering/3-mer stage, whose pool affinity
matrix is quadratic at this scale. The labels and documentation state this
explicitly.

Final CNN figures are not published. The status page above records the frozen
protocol and completion state; completed CPU-model comparisons follow below.

### Random versus acquisition score transfer — CPU models

This five-seed CPU experiment repeats the same comparison for **Aubin, Linear
and MLP** for GFP-protein transfer and natural cgreGFP/artificial-peak transfer.
The acquisition-score and matched random trajectories are complete for all
three models. Every arm shares initial labelled records, label budgets,
validation sets and frozen tests.

Aubin, Linear and MLP contain no dropout, so their uncertainty term is zero.
Their acquisition score is therefore the scaled distance from the labelled
training set in each model's projected hidden representation. As in the CNN
experiment, selection uses global descending scores and does not run the dense
SpectralClustering/3-mer stage.

We retain those completed distance-only trajectories as the reproducible first
version. A versioned five-member ensemble experiment recalculates the
three CPU models with uncertainty included. Aubin and Linear use bootstrap
ensembles; MLP uses independent initializations. In each case the pool score is
`0.62 × scaled distance + 0.38 × scaled ensemble prediction variance`, while
the primary model alone evaluates the unchanged frozen test. See the
[exact method and reproduction commands](TRANSFER_ACTIVE_LEARNING.md#ensemble-uncertainty-extension-for-cpu-models).

Final ensemble-uncertainty comparisons:

![Aubin, Linear and MLP between GFP proteins](../results/transfer_ensemble_sampling_cpu_seed42_46/sampling_ortholog_transfer_cpu_models.png)

![Aubin, Linear and MLP between natural cgreGFP and artificial peaks](../results/transfer_ensemble_sampling_cpu_seed42_46/sampling_peak_transfer_cpu_models.png)

The earlier Linear-only figures remain available for provenance:

![Linear Random versus acquisition score between GFP proteins](../results/transfer_random_linear_cgreGFP_seed42_46/sampling_ortholog_transfer.png)

**Random versus acquisition-score sampling between natural cgreGFP and artificial peaks:**

![Linear Random versus acquisition score between natural cgreGFP and artificial peaks](../results/transfer_random_linear_cgreGFP_seed42_46/sampling_peak_transfer.png)

Distance-only three-model figure paths:

- [Aubin, Linear and MLP between GFP proteins](../results/transfer_random_cpu_models_cgreGFP_seed42_46/sampling_ortholog_transfer_cpu_models.png)
- [Aubin, Linear and MLP between natural cgreGFP and artificial peaks](../results/transfer_random_cpu_models_cgreGFP_seed42_46/sampling_peak_transfer_cpu_models.png)

R² is reported in the result tables alongside correlations: 1 is perfect, 0
matches the test-mean predictor, and negative values indicate worse squared error.

</details>
