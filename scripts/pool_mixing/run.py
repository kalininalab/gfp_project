"""Fit one MLP/CNN trajectory point for the cross-protein pool-size study."""

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from scripts.regression_metrics import metrics
from scripts.train_aubin import predict_encoded, train
from scripts.transfer_benchmark.common import DATA, digest, read_csv, write_csv
from scripts.transfer_benchmark.models import AlignedModel, AlignedOHECNN, encode_aligned
from scripts.pool_mixing.prepare import MODELS, RAW, REQUIRED_SIZES, SEEDS, TARGETS


@torch.inference_mode()
def predict_cnn(model, x, indexes, device, center, scale, batch_size=128):
    model.eval()
    return np.concatenate([(model(x[indexes[start:start + batch_size]].to(device)).cpu().numpy() * scale + center)
                           for start in range(0, len(indexes), batch_size)])


def fit_cnn(x, y, ids, seed, device, settings):
    model = AlignedOHECNN(x.shape[1]).to(device)
    center, scale = float(y[ids["train"]].mean()), max(float(y[ids["train"]].std()), 1e-8)
    normalized = (y - center) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(seed); best = float("inf"); best_state = None; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train(); order = rng.permutation(ids["train"]); total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            indexes = order[start:start + settings["batch_size"]]
            xb = x[indexes].to(device)
            target = torch.tensor(normalized[indexes], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite CNN loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()) * len(indexes)
        validation = predict_cnn(model, x, ids["validation"], device, center, scale)
        validation_mse = float(np.mean((validation - y[ids["validation"]]) ** 2))
        history.append({"epoch": epoch, "train_mse": total / len(order) * scale ** 2,
                        "validation_mse": validation_mse})
        print(f"CNN epoch={epoch} validation_mse={validation_mse:.6f}", flush=True)
        if validation_mse < best:
            best, best_epoch, stale = validation_mse, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    return model, center, scale, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--n-target", type=int, choices=REQUIRED_SIZES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    destination = args.raw / "runs" / args.target / f"n{args.n_target:04d}" / f"seed{args.seed}" / args.model
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists():
        print(f"Verified completed run: {destination}"); return
    protocol = json.loads((args.raw / "protocol.json").read_text())
    if digest(DATA) != protocol["dataset_sha256"] or digest(args.raw / "alignment.json") != protocol["alignment_sha256"]:
        raise ValueError("Frozen data or alignment changed")
    pool_path = args.raw / "target_pools" / f"{args.target}_seed{args.seed}.csv"
    if digest(pool_path) != protocol["target_pool_hashes"][str(pool_path.relative_to(args.raw))]:
        raise ValueError("Target pool changed")
    all_rows = read_csv(DATA); by_id = {row["record_id"]: row for row in all_rows}
    target_pool = read_csv(pool_path)
    n_target_train = args.n_target
    n_target_validation = round(protocol["target_validation_pool_n"] / protocol["target_train_pool_n"] * n_target_train)
    chosen_ids = {split: [row["record_id"] for row in target_pool if row["split"] == split and
                          int(row["pool_rank"]) < (n_target_train if split == "train" else n_target_validation)]
                  for split in ("train", "validation")}
    partitions = {
        "train": [row for row in all_rows if row["gene"] == "cgreGFP" and row["split"] == "train"] +
                 [by_id[record_id] for record_id in chosen_ids["train"]],
        "validation": [row for row in all_rows if row["gene"] == "cgreGFP" and row["split"] == "validation"] +
                      [by_id[record_id] for record_id in chosen_ids["validation"]],
        "test_cgreGFP": [row for row in all_rows if row["gene"] == "cgreGFP" and row["split"] == "test"],
        f"test_{args.target}": [row for row in all_rows if row["gene"] == args.target and row["split"] == "test"],
    }
    sequence_ids = [row["record_id"] for records in partitions.values() for row in records]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("Sequence identity leakage between partitions")
    rows, ids = [], {}
    for name, records in partitions.items():
        ids[name] = np.arange(len(rows), len(rows) + len(records)); rows.extend(records)
    y = np.asarray([float(row["target_log10"]) for row in rows], dtype=np.float32)
    alignment = json.loads((args.raw / "alignment.json").read_text())
    x = encode_aligned(rows, alignment)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if args.model == "mlp_small":
        if args.device != "cpu": raise ValueError("MLP must run on CPU")
        model = AlignedModel(alignment["length"], "mlp_small")
        settings = protocol["mlp_settings"]
        history, best_epoch = train(model, x[ids["train"]], torch.tensor(y[ids["train"]]),
                                    x[ids["validation"]], torch.tensor(y[ids["validation"]]),
                                    args.seed, settings["epochs"], settings["patience"],
                                    settings["batch_size"], settings["learning_rate"])
        prediction = lambda indexes: predict_encoded(model, x[indexes]).numpy()
        center, scale = 0.0, 1.0
    else:
        if args.device != "cuda": raise ValueError("CNN must run on GPU")
        model, center, scale, history, best_epoch = fit_cnn(x, y, ids, args.seed, args.device,
                                                            protocol["cnn_settings"])
        prediction = lambda indexes: predict_cnn(model, x, indexes, args.device, center, scale)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model": args.model, "target": args.target, "n_target_train": args.n_target,
                "seed": args.seed, "alignment_sha256": protocol["alignment_sha256"],
                "best_epoch": best_epoch, "center": center, "scale": scale}, destination / "model.pt")
    write_csv(destination / "history.csv", history)
    prediction_rows, score_rows = [], []
    for evaluation_gene in ("cgreGFP", args.target):
        indexes = ids[f"test_{evaluation_gene}"]; values = prediction(indexes)
        score_rows.append({"target_pool": args.target, "n_target_train": n_target_train, "seed": args.seed,
                           "model": args.model, "evaluation_gene": evaluation_gene,
                           **metrics(y[indexes], values)})
        prediction_rows.extend({"record_id": rows[index]["record_id"], "evaluation_gene": evaluation_gene,
                                "y_true": float(y[index]), "y_pred": float(value)}
                               for index, value in zip(indexes, values))
    write_csv(destination / "predictions.csv", prediction_rows)
    write_csv(destination / "metrics.csv", score_rows)
    (destination / "complete.json").write_text(json.dumps({
        "target_pool": args.target, "n_target_train": args.n_target, "seed": args.seed, "model": args.model,
        "n_cgre_train": protocol["base_train_n"], "n_target_train": n_target_train,
        "n_cgre_validation": protocol["base_validation_n"], "n_target_validation": n_target_validation,
        "best_epoch": best_epoch, "frozen_tests_used_for_fitting": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
