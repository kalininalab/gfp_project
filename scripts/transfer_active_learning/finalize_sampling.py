"""Validate and plot matched random/fancy transfer for one model."""

import argparse
from pathlib import Path

from scripts.transfer_active_learning.plot_followups import completed_rows, aggregate_final, sampling_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fancy", type=Path, default=Path("results/transfer_al_cgreGFP_seed42_46"))
    parser.add_argument("--random", type=Path, required=True)
    args = parser.parse_args()
    random_protocol, random_rows = completed_rows(args.random, [args.model])
    fancy_protocol, fancy_rows = completed_rows(args.fancy, [args.model])
    if random_protocol["test_records_sha256"] != fancy_protocol["test_records_sha256"]:
        raise ValueError("Random/fancy test manifests differ")
    random_scores = aggregate_final(random_rows, random_protocol)
    fancy_scores = aggregate_final(fancy_rows, fancy_protocol)
    sampling_plot(random_scores, fancy_scores, "ortholog_mixtures", args.random)
    sampling_plot(random_scores, fancy_scores, "peak_transfer", args.random)


if __name__ == "__main__":
    main()
