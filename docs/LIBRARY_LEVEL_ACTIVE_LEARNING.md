# Library-level active learning for GFP landscape mapping

Status: proposed follow-up, recorded 26 November 2026. No jobs submitted and no changes to existing experiments.

## Scientific question

Can adaptive selection of parental sequences and error-prone PCR conditions improve prediction across a predefined cgreGFP landscape with fewer measured variants than non-adaptive library generation?

Suggested project name: **Active learning-guided library design for GFP fitness-landscape mapping**. Short name: **library-level active learning**. More broadly, this is sequential experimental design. These are descriptive names, not a claim of a new established algorithm.

## What is selected

Each action is a parent sequence, a feasible mutagenesis protocol/rate, and an allocation of clones to assay. Existing natural cgreGFP and artificial peak reference sequences provide starting parents. A parent is a library centre; it need not be a verified local fitness maximum. New measured variants could become eligible parents in later rounds.

Error-prone PCR generates a distribution of offspring, not an exact selected sequence. Model the nucleotide mutation spectrum and translate offspring to protein sequences where possible. Account for synonymous changes, stops, duplicate clones and variation in mutation count; an average mutation rate does not guarantee a particular amino-acid distance.

## Proposed loop

1. Fit the fitness predictor and uncertainty estimator on measured sequence–fluorescence pairs.
2. Simulate possible batches for each feasible parent/protocol choice using an empirically calibrated mutagenesis distribution.
3. Score the expected usefulness of a batch for prediction in the intended landscape. Uncertainty plus diversity is a practical initial proxy; it is not automatically equivalent to expected information gain or error reduction.
4. Select a library or allocate a plate across libraries. A preliminary budget is 96 measured clones per round, with controls and assay failures explicitly accounted for.
5. Generate the library, sample clones, obtain their actual sequences and fluorescence measurements, and retain both bright and dark observations.
6. Add the measurements to training, retrain, and choose the next experiment.

The acquisition decision must use predictions and currently available measurements only. Unknown offspring labels and held-out test labels cannot inform library selection.

## Objective and evaluation

The objective is better landscape prediction per experimental cost, including low- and high-fluorescence regions. Selecting the brightest parent alone is not the objective. Define the target region and its weighting before comparison; uniform coverage of all possible protein sequences is infeasible.

Compare adaptive parent/protocol choice against random choices and a fixed balanced allocation across parents, using identical initial data, feasible actions and measurement budgets. Use frozen tests that cover intended regions, plus distance-stratified metrics; assess errors against distance to the nearest training sequence as well as WT distance. Select hyperparameters using validation, not final test scores.

Existing peak-labelled datasets can support a restricted retrospective pilot: select a landscape and randomly reveal a batch from its available training pool. This tests adaptive allocation among measured landscapes, but does not reproduce arbitrary error-prone PCR libraries. Simulated new sequences have no experimental labels and cannot be treated as measured training data. Prospective experiments are needed to validate the full workflow.

Before pooling natural and artificial data, audit calibration and measurement floors. WT-control alignment is a preliminary multiplicative correction, not proof that the full assay scales are interchangeable.

## Related work

[Active learning-assisted directed evolution](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739421/) provides related experimental precedent for iterative model-guided protein engineering. Its optimization objective and acquisition procedure should not be presented as identical to this landscape-mapping proposal.

[Learning to Design Protein and DNA Libraries](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/Archive/EECS-2023-73.pdf) discusses designing distributions of sequences through library construction choices, including error-prone PCR.
