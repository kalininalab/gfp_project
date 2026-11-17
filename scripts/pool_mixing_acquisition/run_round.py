"""Run one dependent round of a cross-protein acquisition trajectory.

The frozen source, validation and test partitions come from the corresponding
random pool-size experiment.  Only the order in which target-training records
are labelled changes: the next plate maximizes a distance/uncertainty score.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from scripts.active_learning.acquisition import acquisition_scores
from scripts.pool_mixing.run import fit_cnn, predict_cnn, train_fixed
from scripts.regression_metrics import metrics
from scripts.train_aubin import predict_encoded, train
from scripts.transfer_benchmark.common import DATA, digest, read_csv, write_csv
from scripts.transfer_benchmark.models import AlignedModel, AlignedOHECNN, encode_aligned

MODELS = ("aubin_1_10_1", "mlp_small", "CNN_Jannis_OHE")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def fit_model(model_name, x, y, train_ids, validation_ids, seed, device, protocol):
    seed_everything(seed)
    ids = {"train": np.asarray(train_ids), "validation": np.asarray(validation_ids)}
    if model_name == "CNN_Jannis_OHE":
        model, center, scale, history, best_epoch = fit_cnn(
            x, y, ids, seed, device, protocol["cnn_settings"])
        prediction = lambda indexes: predict_cnn(model, x, np.asarray(indexes), device, center, scale)
    else:
        model = AlignedModel(x.shape[1], model_name)
        settings = protocol.get("model_settings", {}).get(model_name, protocol.get("mlp_settings"))
        if settings is None:
            raise ValueError(f"No training settings for {model_name}")
        if settings.get("checkpoint_policy", "early_stopping") == "last_epoch":
            history, best_epoch = train_fixed(
                model, x[ids["train"]], torch.tensor(y[ids["train"]]),
                x[ids["validation"]], torch.tensor(y[ids["validation"]]), seed,
                settings["epochs"], settings["batch_size"], settings["learning_rate"])
        else:
            history, best_epoch = train(
                model, x[ids["train"]], torch.tensor(y[ids["train"]]),
                x[ids["validation"]], torch.tensor(y[ids["validation"]]), seed,
                settings["epochs"], settings["patience"], settings["batch_size"],
                settings["learning_rate"])
        prediction = lambda indexes: predict_encoded(model, x[np.asarray(indexes)]).numpy()
        center, scale = 0.0, 1.0
    return model, prediction, history, best_epoch, center, scale


@torch.inference_mode()
def representations(model, x, indexes, device, batch_size=256):
    """Return the input to the fitted model's final regression layer."""
    captured = []
    layer = model.base.regressor if isinstance(model, AlignedOHECNN) else model.network[-1]
    hook = layer.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().cpu()))
    model.eval()
    indexes = np.asarray(indexes)
    for start in range(0, len(indexes), batch_size):
        model(x[indexes[start:start + batch_size]].to(device))
    hook.remove()
    return torch.cat(captured).numpy()


@torch.inference_mode()
def mc_dropout_variance(model, x, indexes, device, scale, samples, batch_size=128):
    values = []
    indexes = np.asarray(indexes)
    model.train()
    for start in range(0, len(indexes), batch_size):
        xb = x[indexes[start:start + batch_size]].to(device)
        draws = torch.stack([model(xb) * scale for _ in range(samples)])
        values.append(draws.var(0, unbiased=True).cpu().numpy())
    model.eval()
    return np.concatenate(values).astype(np.float32)


def ensemble_variance(model_name, x, y, train_ids, validation_ids, seed, protocol,
                      pool_ids, primary_prediction, members):
    predictions = [np.asarray(primary_prediction(pool_ids), dtype=np.float32)]
    for member in range(1, members):
        member_seed = seed + 100003 * member
        member_train = np.asarray(train_ids)
        if model_name == "aubin_1_10_1":
            rng = np.random.default_rng(np.random.SeedSequence([seed, member, 7919]))
            member_train = rng.choice(member_train, len(member_train), replace=True)
        _, prediction, _, _, _, _ = fit_model(
            model_name, x, y, member_train, validation_ids, member_seed, "cpu", protocol)
        predictions.append(np.asarray(prediction(pool_ids), dtype=np.float32))
    return np.var(np.stack(predictions), axis=0, ddof=1).astype(np.float32)


def projected(values, seed, dimensions):
    if values.shape[1] <= dimensions:
        return values.astype(np.float32)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(values.shape[1], dimensions)).astype(np.float32) / np.sqrt(dimensions)
    return values.astype(np.float32) @ matrix


def minimum_distances(pool, labelled, chunk_size=1024):
    pool_tensor = torch.as_tensor(pool, dtype=torch.float32)
    labelled_tensor = torch.as_tensor(labelled, dtype=torch.float32)
    return torch.cat([torch.cdist(chunk, labelled_tensor).min(dim=1).values
                      for chunk in pool_tensor.split(chunk_size)]).numpy()


def kl_divergence(y_true, y_predicted):
    edges = np.linspace(float(np.min(y_true)), float(np.max(y_true)), 51)
    epsilon = np.finfo(float).eps
    truth = np.clip(y_true, edges[0] + epsilon, edges[-1] - epsilon)
    predicted = np.clip(y_predicted, edges[0] + epsilon, edges[-1] - epsilon)
    p = np.histogram(truth, bins=edges)[0].astype(float) + 0.5
    q = np.histogram(predicted, bins=edges)[0].astype(float) + 0.5
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log2(p / q)))


def scaled(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.min()) / (np.ptp(values) + 1e-8)


def read_state(path):
    return json.loads(path.read_text()) if path.exists() else {"next_round": 0, "selected_record_ids": []}


def append_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("ppluGFP", "amacGFP"), required=True)
    parser.add_argument("--plate-size", type=int, choices=(9, 48, 96, 192, 384), required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--reference-raw", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    protocol = json.loads((args.raw / "protocol.json").read_text())
    reference_protocol = json.loads((args.reference_raw / "protocol.json").read_text())
    last_round = int(protocol["plate_rounds"][str(args.plate_size)])
    if not 0 <= args.round <= last_round:
        raise ValueError("Round is outside the frozen plate schedule")
    if digest(DATA) != reference_protocol["dataset_sha256"]:
        raise ValueError("Processed dataset changed")
    trajectory = args.raw / "runs" / args.target / f"plate{args.plate_size:03d}" / f"seed{args.seed}" / args.model
    trajectory.mkdir(parents=True, exist_ok=True)
    state_path = trajectory / "state.json"
    state = read_state(state_path)
    if state["next_round"] > args.round:
        print(f"Round {args.round} already complete"); return
    if state["next_round"] != args.round:
        raise RuntimeError(f"Expected round {state['next_round']}, received {args.round}")

    all_rows = read_csv(DATA); by_id = {row["record_id"]: row for row in all_rows}
    target_path = args.reference_raw / "target_pools" / f"{args.target}_seed{args.seed}.csv"
    target_relative = str(target_path.relative_to(args.reference_raw))
    if digest(target_path) != reference_protocol["target_pool_hashes"][target_relative]:
        raise ValueError(f"Frozen pool changed: {target_path}")
    if "base_pool_hashes" in reference_protocol:
        base_path = args.reference_raw / "base_pools" / f"cgreGFP_seed{args.seed}.csv"
        base_relative = str(base_path.relative_to(args.reference_raw))
        if digest(base_path) != reference_protocol["base_pool_hashes"][base_relative]:
            raise ValueError(f"Frozen pool changed: {base_path}")
        base_ids = [row["record_id"] for row in read_csv(base_path)]
    else:
        base_ids = [row["record_id"] for row in all_rows
                    if row["gene"] == "cgreGFP" and row["split"] == "train"]
    target_manifest = read_csv(target_path)
    target_train_ids = [row["record_id"] for row in target_manifest if row["split"] == "train"]
    target_validation_ids = [row["record_id"] for row in target_manifest if row["split"] == "validation"]
    selected_record_ids = list(state["selected_record_ids"])
    expected = args.plate_size * args.round
    if len(selected_record_ids) != expected or not set(selected_record_ids) <= set(target_train_ids):
        raise ValueError("Acquisition state does not match this round")
    n_target_validation = round(reference_protocol["target_validation_pool_n"] /
                                reference_protocol["target_train_pool_n"] * expected)
    validation_record_ids = ([row["record_id"] for row in all_rows
                              if row["gene"] == "cgreGFP" and row["split"] == "validation"] +
                             target_validation_ids[:n_target_validation])
    test_ids = {gene: [row["record_id"] for row in all_rows
                       if row["gene"] == gene and row["split"] == "test"]
                for gene in ("cgreGFP", args.target)}
    remaining_ids = [record_id for record_id in target_train_ids if record_id not in set(selected_record_ids)]
    ordered_ids = base_ids + selected_record_ids + validation_record_ids + remaining_ids + sum(test_ids.values(), [])
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("Record leakage between training, validation, pool and test")
    rows = [by_id[record_id] for record_id in ordered_ids]
    index = {record_id: i for i, record_id in enumerate(ordered_ids)}
    x = encode_aligned(rows, json.loads((args.reference_raw / "alignment.json").read_text()))
    y = np.asarray([float(row["target_log10"]) for row in rows], dtype=np.float32)
    train_indexes = np.asarray([index[record_id] for record_id in base_ids + selected_record_ids])
    validation_indexes = np.asarray([index[record_id] for record_id in validation_record_ids])
    pool_indexes = np.asarray([index[record_id] for record_id in remaining_ids])

    model, prediction, history, best_epoch, center, scale = fit_model(
        args.model, x, y, train_indexes, validation_indexes, args.seed, args.device, reference_protocol)
    history_rows = [{"round": args.round, **row} for row in history]
    write_csv(trajectory / f"history_round_{args.round:03d}.csv", history_rows)
    score_rows, prediction_rows = [], []
    for gene, record_ids in test_ids.items():
        indexes = np.asarray([index[record_id] for record_id in record_ids])
        values = prediction(indexes)
        score_rows.append({"method": "acquisition", "target_pool": args.target,
                           "plate_size": args.plate_size, "round": args.round,
                           "n_target_train": expected, "seed": args.seed, "model": args.model,
                           "evaluation_gene": gene, **metrics(y[indexes], values),
                           "kl_true_to_pred_bits": kl_divergence(y[indexes], values)})
        if args.round in (0, last_round):
            prediction_rows.extend({"record_id": record_id, "evaluation_gene": gene,
                                    "y_true": float(y[i]), "y_pred": float(value)}
                                   for record_id, i, value in zip(record_ids, indexes, values))
    write_csv(trajectory / f"metrics_round_{args.round:03d}.csv", score_rows)
    if prediction_rows:
        write_csv(trajectory / f"predictions_round_{args.round:03d}.csv", prediction_rows)

    if args.round < last_round:
        hidden_train = representations(model, x, train_indexes, args.device)
        hidden_pool = representations(model, x, pool_indexes, args.device)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.plate_size, args.round, 2026]))
        anchors = rng.choice(len(hidden_train), min(protocol["distance_anchors"], len(hidden_train)), replace=False)
        projection_seed = args.seed * 100000 + args.plate_size * 100 + args.round
        labelled = projected(hidden_train[anchors], projection_seed, protocol["projection_dimensions"])
        pool = projected(hidden_pool, projection_seed, protocol["projection_dimensions"])
        if args.model == "CNN_Jannis_OHE":
            variance = mc_dropout_variance(model, x, pool_indexes, args.device, scale,
                                           protocol["mc_dropout_samples"])
        else:
            variance = ensemble_variance(args.model, x, y, train_indexes, validation_indexes,
                                         args.seed, reference_protocol, pool_indexes, prediction,
                                         protocol["ensemble_members"])
        combined = acquisition_scores(labelled, pool, variance, alpha=protocol["distance_weight"])
        chosen_local = np.argsort(-combined, kind="stable")[:args.plate_size]
        chosen_ids = [remaining_ids[i] for i in chosen_local]
        distances = minimum_distances(pool, labelled)
        query_rows = [{"selected_after_round": args.round, "first_fit_round": args.round + 1,
                       "record_id": remaining_ids[i], "acquisition_score": float(combined[i]),
                       "scaled_distance": float(scaled(distances)[i]),
                       "scaled_uncertainty": float(scaled(variance)[i]),
                       "predictive_variance": float(variance[i])} for i in chosen_local]
        append_csv(trajectory / "queries.csv", query_rows)
        selected_record_ids.extend(chosen_ids)
    else:
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "model": args.model, "target": args.target, "plate_size": args.plate_size,
                    "seed": args.seed, "round": args.round, "best_epoch": best_epoch,
                    "center": center, "scale": scale}, trajectory / "final_model.pt")
    atomic_json(state_path, {"next_round": args.round + 1,
                             "selected_record_ids": selected_record_ids})
    if args.round == last_round:
        atomic_json(trajectory / "complete.json", {
            "method": "acquisition", "rounds_completed": last_round + 1,
            "final_target_train_n": expected, "frozen_tests_used_for_fitting_or_sampling": False,
            "protocol_sha256": digest(args.raw / "protocol.json")})


if __name__ == "__main__":
    main()
