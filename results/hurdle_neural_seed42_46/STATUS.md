# Neural classification–regression status

This experiment asks whether a soft measurement-floor classifier plus a conditional regressor improves final fluorescence prediction.

## Design

- Five landscapes: natural cgreGFP, cgre132, cgre1338, cgre4111 and cgre9708.
- Two architectures: MLP on mean ESM-2 embeddings and CNN on native one-hot sequence.
- Two matched models: direct regression and shared-trunk soft hurdle.
- Five independent initializations: seeds 42–46.
- Fixed 60% train, 20% validation and 20% test records.
- The frozen test is not used for fitting or checkpoint selection.

The workflow contains 50 GPU jobs followed by one automatic finalizer. It was submitted on 21 October 2026 as DAG `66478`; worker clusters are `66479`–`66528`. At submission all workers were queued without holds. Final results will appear in this directory after every run passes validation.
