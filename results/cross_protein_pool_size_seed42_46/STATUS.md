# Cross-protein labelled-pool size experiment

This experiment keeps the full cgreGFP training pool fixed and simulates laboratory plates of **9, 48, 96, 192 or 384 new target sequences per round**. Cumulative target data equal `plate size × round`, and every schedule ends near 1,920 target sequences: 213, 40, 20, 10 or 5 rounds, respectively. Each round retains all sequences from previous plates. MLP and CNN-on-OHE models are evaluated on frozen tests of both participating proteins across seeds 42–46.

This is supervised random pool-size addition, not active learning. The final folder will contain six main figures: R², Spearman and KL divergence for cgreGFP+ppluGFP and cgreGFP+amacGFP. Each figure contains five panels, one per plate size, with round on the x-axis.

Final figures: [`figures/`](figures/)
Final report after completion: [`REPORT.md`](REPORT.md)
