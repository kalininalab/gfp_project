"""Fit all nested training sizes for one artificial-peak/model/seed trajectory."""
import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits

from scripts.peak_holdout.run import predict_cnn, train_cnn
from scripts.regression_metrics import metrics
from scripts.train_aubin import predict_encoded, train
from scripts.transfer_benchmark.models import AlignedModel, AlignedOHECNN, encode_aligned
from .common import DATA, MODELS, PEAKS, RAW, SEEDS, digest, read_csv, scientific_sources, write_csv


def kl_divergence(y, prediction, edges):
    eps = np.finfo(float).eps
    y = np.clip(y, edges[0] + eps, edges[-1] - eps)
    prediction = np.clip(prediction, edges[0] + eps, edges[-1] - eps)
    p = np.histogram(y, edges)[0].astype(float) + .5
    q = np.histogram(prediction, edges)[0].astype(float) + .5
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=RAW)
    parser.add_argument("--peak", choices=PEAKS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    if digest(DATA) != protocol["dataset_sha256"] or scientific_sources() != protocol["source_hashes"]:
        raise ValueError("Frozen source changed")
    subset_path = args.directory / "subsets" / args.peak / f"seed{args.seed}.csv"
    relative = str(subset_path.relative_to(args.directory))
    if digest(subset_path) != protocol["subset_hashes"][relative]:
        raise ValueError("Nested subset changed")
    rank = {row["record_id"]: int(row["pool_rank"]) for row in read_csv(subset_path)}
    dataset = read_csv(DATA)
    source = [row for row in dataset if row["gene"] == args.peak]
    validation = [row for row in dataset if row["gene"] == "cgreGFP" and row["split"] == "validation"]
    test = [row for row in dataset if row["gene"] == "cgreGFP" and row["split"] == "test"]
    alignment = json.loads((args.directory / "alignment.json").read_text())
    values = [float(row["target_log10"]) for row in validation + test]
    edges = np.linspace(min(values), max(values), 51)
    trajectory = args.directory / "fits" / args.peak / f"seed{args.seed}" / args.model
    trajectory.mkdir(parents=True, exist_ok=True)

    for n_train in protocol["sizes"][args.peak]:
        out = trajectory / f"n{n_train:05d}"
        out.mkdir(exist_ok=True)
        if (out / "complete.json").exists():
            continue
        training = sorted((row for row in source if rank[row["record_id"]] < n_train),
                          key=lambda row: rank[row["record_id"]])
        if len(training) != n_train:
            raise ValueError("Wrong nested training size")
        rows = training + validation + test
        ids = {"train": np.arange(len(training)),
               "validation": np.arange(len(training), len(training) + len(validation)),
               "test": np.arange(len(training) + len(validation), len(rows))}
        y = np.asarray([float(row["target_log10"]) for row in rows])
        x = encode_aligned(rows, alignment)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed); torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            if args.model.startswith("CNN"):
                if args.device != "cuda":
                    raise ValueError("CNN requires CUDA")
                model = AlignedOHECNN(alignment["length"]).to(args.device)
                history, best, center, scale = train_cnn(
                    model, x, y, ids, args.seed, args.device, protocol["cnn_settings"])
                prediction = predict_cnn(model, x, ids["test"], args.device, center, scale)
            else:
                model = AlignedModel(alignment["length"], args.model)
                settings = protocol["onehot_settings"]
                history, best = train(
                    model, x[ids["train"]], torch.tensor(y[ids["train"]], dtype=torch.float32),
                    x[ids["validation"]], torch.tensor(y[ids["validation"]], dtype=torch.float32),
                    args.seed, settings["epochs"], settings["patience"], settings["batch_size"], settings["lr"])
                prediction = predict_encoded(model, x[ids["test"]]).numpy()
                center, scale = 0., 1.
        score = metrics(y[ids["test"]], prediction)
        score.update(kl_true_to_pred_bits=kl_divergence(y[ids["test"]], prediction, edges),
                     wasserstein=float(wasserstein_distance(y[ids["test"]], prediction)))
        write_csv(out / "history.csv", history)
        write_csv(out / "predictions.csv", (
            {"record_id": row["record_id"], "y_true": float(true), "y_pred": float(pred)}
            for row, true, pred in zip(test, y[ids["test"]], prediction)))
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "center": center, "scale": scale, "best_epoch": best}, out / "model.pt")
        report = {"peak": args.peak, "model": args.model, "seed": args.seed,
                  "n_train": n_train, "n_validation": len(validation), "n_test": len(test),
                  "metrics": score, "best_epoch": best, "fit_seconds": time.perf_counter() - started,
                  "protocol_sha256": digest(args.directory / "protocol.json")}
        (out / "complete.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(args.peak, args.model, args.seed, n_train, score["pearson"], flush=True)


if __name__ == "__main__":
    main()
