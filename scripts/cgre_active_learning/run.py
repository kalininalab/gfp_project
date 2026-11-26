"""Run one resumable cgreGFP active-learning trajectory."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture

from scripts.active_learning.run import Predictor, infer, input_batch, split_indices
from scripts.train_aubin import train
from .common import (DATA, MODELS, RAW, STRATEGIES, digest, percentile, read_csv,
                     strategy_scores, write_csv, write_json)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def safe_corr(a, b, kind):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0: return None
    return float((pearsonr if kind == "pearson" else spearmanr)(a, b).statistic)


def distribution_metrics(y, prediction, edges):
    y, prediction = np.asarray(y), np.asarray(prediction)
    eps = np.finfo(float).eps
    yt = np.clip(y, edges[0] + eps, edges[-1] - eps)
    yp = np.clip(prediction, edges[0] + eps, edges[-1] - eps)
    p = np.histogram(yt, edges)[0].astype(float) + 0.5
    q = np.histogram(yp, edges)[0].astype(float) + 0.5
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log2(p / q))), float(wasserstein_distance(y, prediction))


def scores(y, prediction, edges):
    y, prediction = np.asarray(y), np.asarray(prediction)
    mse = float(mean_squared_error(y, prediction)); kl, wd = distribution_metrics(y, prediction, edges)
    return {"n": len(y), "spearman": safe_corr(y, prediction, "spearman"),
            "pearson": safe_corr(y, prediction, "pearson"), "r2": float(r2_score(y, prediction)),
            "rmse": math.sqrt(mse), "mae": float(mean_absolute_error(y, prediction)),
            "kl_true_to_pred_bits": kl, "wasserstein": wd}


def minimum_distance(pool, labelled, chunk=512):
    labelled = torch.as_tensor(labelled, dtype=torch.float32)
    return torch.cat([torch.cdist(torch.as_tensor(part, dtype=torch.float32), labelled).min(1).values
                      for part in np.array_split(pool, max(1, math.ceil(len(pool) / chunk))) if len(part)]).numpy()


def mean_pairwise_hamming(sequences):
    total = pairs = 0
    for i, left in enumerate(sequences):
        for right in sequences[i + 1:]:
            total += sum(a != b for a, b in zip(left, right)); pairs += 1
    return float(total / pairs) if pairs else 0.0


def predict_members(models, x, indexes, device, batch_size, centers, scales, cnn, mc_samples=25):
    if cnn:
        mean, variance = infer(models[0], x, indexes, device, batch_size, samples=mc_samples)
        prediction = mean * scales[0] + centers[0]
        uncertainty = np.sqrt(np.maximum(variance, 0)) * scales[0]
        return prediction.astype(np.float32), uncertainty.astype(np.float32)
    values = []
    for model, center, scale in zip(models, centers, scales):
        prediction, _ = infer(model, x, indexes, device, batch_size)
        values.append(prediction * scale + center)
    values = np.stack(values)
    return values.mean(0).astype(np.float32), values.std(0, ddof=1).astype(np.float32)


def fit_fresh(model_name, x, y, train_ids, validation_ids, seed, device, epochs_override=None):
    """Reuse the established baseline optimizer, loss and early-stopping recipes."""
    seed_all(seed)
    model = Predictor(model_name, x.shape[1], 20).to(device)
    model.architecture = model_name
    if model_name == "mlp_small":
        epochs = epochs_override or 30
        history, _ = train(model, torch.as_tensor(x[train_ids]), torch.as_tensor(y[train_ids]),
                           torch.as_tensor(x[validation_ids]), torch.as_tensor(y[validation_ids]),
                           seed, epochs, 10, 32, 0.001)
        return model, 0.0, 1.0, history
    epochs = epochs_override or 60
    center = float(y[train_ids].mean()); scale = max(float(y[train_ids].std()), 1e-8)
    normalized = (y - center) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    rng = np.random.default_rng(seed); best = float("inf"); best_state = None; stale = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(train_ids); total = 0.0
        for start in range(0, len(order), 64):
            indexes = order[start:start + 64]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(input_batch(x, indexes, device))
            target = torch.as_tensor(normalized[indexes], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss): raise ValueError("Non-finite CNN loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()) * len(indexes)
        validation_prediction, _ = infer(model, x, validation_ids, device, 64)
        validation_mse = float(np.mean((validation_prediction * scale + center - y[validation_ids]) ** 2))
        history.append({"epoch": epoch, "train_mse": total / len(order) * scale ** 2,
                        "validation_mse": validation_mse})
        if validation_mse < best:
            best, stale = validation_mse, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10: break
    model.load_state_dict(best_state)
    return model, center, scale, history


def primary_embeddings(models, x, indexes, device, batch_size):
    """Use one coherent hidden coordinate system for nearest-neighbour distance."""
    return infer(models[0], x, indexes, device, batch_size, embeddings=True).astype(np.float32)


def stratified_rows(base, y, prediction, mutation_count, fitness_edges, metric_edges):
    output = []
    fitness_groups = np.digitize(y, fitness_edges[1:-1], right=False)
    for group in range(4):
        ix = np.flatnonzero(fitness_groups == group)
        if len(ix) >= 2:
            output.append({**base, "stratifier": "true_fitness_quartile", "stratum": f"Q{group+1}",
                           **scores(y[ix], prediction[ix], metric_edges)})
    mutation_groups = [("0-1", mutation_count <= 1), ("2", mutation_count == 2),
                       ("3", mutation_count == 3), ("4", mutation_count == 4),
                       ("5+", mutation_count >= 5)]
    for label, mask in mutation_groups:
        ix = np.flatnonzero(mask)
        if len(ix) >= 2:
            output.append({**base, "stratifier": "mutations_from_WT", "stratum": label,
                           **scores(y[ix], prediction[ix], metric_edges)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--epochs", type=int, help="Smoke-test override")
    parser.add_argument("--rounds", type=int, default=10)
    args = parser.parse_args()
    if args.model.startswith("CNN") != (args.device == "cuda") and args.epochs is None:
        raise ValueError("Production CNN runs use CUDA; MLP ensemble runs use CPU")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rows = [row for row in read_csv(DATA) if row["gene"] == "cgreGFP"]
    if digest(DATA) != json.loads(Path("results/cgre_active_learning_seed42_44/protocol.json").read_text())["data_sha256"]:
        raise ValueError("Processed dataset changed after protocol freeze")
    sequences = [row["sequence"] for row in rows]
    if len(set(sequences)) != len(sequences): raise ValueError("Duplicate cgreGFP sequences")
    alphabet = "ACDEFGHIKLMNPQRSTVWY"; lookup = {aa: i for i, aa in enumerate(alphabet)}
    tokens = np.asarray([[lookup[aa] for aa in sequence] for sequence in sequences], dtype=np.int64)
    x = np.eye(20, dtype=np.float32)[tokens] if args.model.startswith("CNN") else tokens
    y = np.asarray([float(row["target_log10"]) for row in rows], dtype=np.float32)
    mutations = np.asarray([int(row["n_mutations"]) for row in rows])
    partitions = split_indices(len(rows), args.seed)
    initial_train = partitions["train"].copy(); validation = partitions["validation"].copy()
    test = partitions["test"].copy(); train = initial_train.copy(); pool = partitions["pool"].copy()
    metric_edges = np.linspace(float(y.min()), float(y.max()), 51)
    fitness_edges = np.quantile(y[initial_train], [0, .25, .5, .75, 1])
    gmm = GaussianMixture(2, n_init=10, random_state=args.seed, reg_covar=1e-6).fit(y[initial_train, None])
    means = gmm.means_.ravel(); order = np.argsort(means)
    grid = np.linspace(float(y[initial_train].min()), float(y[initial_train].max()), 10000)
    densities = np.exp(gmm._estimate_weighted_log_prob(grid[:, None]))[:, order]
    between = (grid >= means[order[0]]) & (grid <= means[order[1]])
    bright_threshold = float(grid[np.flatnonzero(between)[np.argmin(np.abs(densities[between, 0] - densities[between, 1]))]])

    destination = args.raw / args.model / f"seed{args.seed}" / args.strategy
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "complete.json").exists():
        print(f"Already complete: {destination}"); return
    split_rows = [{"record_id": rows[i]["record_id"], "partition": part}
                  for part, indexes in partitions.items() for i in indexes]
    if not (destination / "splits.csv.gz").exists(): write_csv(destination / "splits.csv.gz", split_rows)
    run_config = {"model": args.model, "strategy": args.strategy, "seed": args.seed,
                  "rounds": args.rounds, "batch_size": 96, "ensemble_members": 10 if args.model == "mlp_small" else 1,
                  "mc_dropout_samples": 25, "bright_threshold_initial_train_gmm": bright_threshold,
                  "initial_counts": {key: len(value) for key, value in partitions.items()},
                  "test_record_ids_sha256": digest(destination / "splits.csv.gz")}
    write_json(destination / "config.json", run_config)

    cnn = args.model.startswith("CNN"); members = 1 if cnn else 10
    inference_batch_size = 64 if cnn else 32
    start_round = 0; all_metrics = []; all_diagnostics = []; all_queries = []; all_stratified = []
    checkpoint = destination / "resume.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
        start_round, train, pool = saved["next_round"], saved["train"], saved["pool"]
        all_metrics, all_diagnostics = saved["metrics"], saved["diagnostics"]
        all_queries, all_stratified = saved["queries"], saved["stratified"]

    for round_number in range(start_round, args.rounds + 1):
        started = time.perf_counter(); models = []; centers, scales, histories = [], [], []
        for member in range(members):
            member_seed = args.seed + 100003 * member
            model, center, scale, history = fit_fresh(
                args.model, x, y, train, validation, member_seed, args.device, args.epochs)
            models.append(model)
            centers.append(center); scales.append(scale)
            histories.extend({"member": member, **item} for item in history)
        write_csv(destination / f"history_round_{round_number:02d}.csv.gz", histories)

        test_prediction, test_uncertainty = predict_members(models, x, test, args.device,
                                                             inference_batch_size, centers, scales, cnn)
        base = {"model": args.model, "strategy": args.strategy, "seed": args.seed,
                "round": round_number, "n_train": len(train), "n_pool": len(pool)}
        all_metrics.append({**base, **scores(y[test], test_prediction, metric_edges),
                            "fit_seconds": time.perf_counter() - started})
        all_stratified.extend(stratified_rows(base, y[test], test_prediction, mutations[test],
                                              fitness_edges, metric_edges))
        write_csv(destination / f"test_predictions_round_{round_number:02d}.csv.gz",
                  ({"record_id": rows[i]["record_id"], "sequence": sequences[i],
                    "n_mutations": mutations[i], "y_true": float(y[i]), "y_pred": float(prediction),
                    "uncertainty": float(uncertainty)}
                   for i, prediction, uncertainty in zip(test, test_prediction, test_uncertainty)))

        if round_number < args.rounds:
            pool_prediction, pool_uncertainty = predict_members(models, x, pool, args.device,
                                                                 inference_batch_size, centers, scales, cnn)
            labelled_embedding = primary_embeddings(models, x, train, args.device, inference_batch_size)
            pool_embedding = primary_embeddings(models, x, pool, args.device, inference_batch_size)
            distance = minimum_distance(pool_embedding, labelled_embedding)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, round_number, STRATEGIES.index(args.strategy), 2026]))
            acquisition, u_rank, d_rank, high_rank, low_rank = strategy_scores(
                args.strategy, pool_uncertainty, distance, pool_prediction, rng)
            absolute_error = np.abs(pool_prediction - y[pool])
            correlation_pairs = (("uncertainty_abs_error", pool_uncertainty, absolute_error),
                                 ("distance_abs_error", distance, absolute_error),
                                 ("uncertainty_distance", pool_uncertainty, distance),
                                 ("predicted_fitness_uncertainty", pool_prediction, pool_uncertainty),
                                 ("predicted_fitness_distance", pool_prediction, distance))
            for name, left, right in correlation_pairs:
                all_diagnostics.append({**base, "relationship": name,
                                        "pearson": safe_corr(left, right, "pearson"),
                                        "spearman": safe_corr(left, right, "spearman")})
            order = np.lexsort((np.asarray([rows[i]["record_id"] for i in pool]), -acquisition))
            selected_local = order[:96]; selected = pool[selected_local]
            selected_set = set(selected.tolist())
            batch_hamming = mean_pairwise_hamming([sequences[i] for i in selected])
            for rank, local in enumerate(selected_local, 1):
                i = int(pool[local])
                all_queries.append({**base, "selection_rank": rank, "record_id": rows[i]["record_id"],
                    "sequence": sequences[i], "n_mutations": int(mutations[i]),
                    "y_true_retrospective": float(y[i]), "y_pred_at_selection": float(pool_prediction[local]),
                    "absolute_error_retrospective": float(absolute_error[local]),
                    "uncertainty": float(pool_uncertainty[local]), "distance": float(distance[local]),
                    "uncertainty_percentile": float(u_rank[local]), "distance_percentile": float(d_rank[local]),
                    "high_fitness_percentile": float(high_rank[local]), "low_fitness_percentile": float(low_rank[local]),
                    "acquisition_score": float(acquisition[local]),
                    "operational_bright": bool(y[i] >= bright_threshold),
                    "batch_mean_pairwise_hamming": batch_hamming})
            pool_rows = []
            for local, i in enumerate(pool):
                pool_rows.append({"record_id": rows[i]["record_id"], "selected": int(int(i) in selected_set),
                    "y_true_retrospective": float(y[i]), "y_pred": float(pool_prediction[local]),
                    "absolute_error_retrospective": float(absolute_error[local]),
                    "uncertainty": float(pool_uncertainty[local]), "distance": float(distance[local]),
                    "uncertainty_percentile": float(u_rank[local]), "distance_percentile": float(d_rank[local]),
                    "high_fitness_percentile": float(high_rank[local]), "low_fitness_percentile": float(low_rank[local]),
                    "acquisition_score": float(acquisition[local]), "n_mutations": int(mutations[i])})
            write_csv(destination / f"pool_scores_round_{round_number:02d}.csv.gz", pool_rows)
            train = np.concatenate([train, selected])
            pool = np.asarray([i for i in pool if int(i) not in selected_set], dtype=int)

        write_csv(destination / "metrics.csv", all_metrics)
        write_csv(destination / "acquisition_correlations.csv", all_diagnostics)
        write_csv(destination / "queries.csv.gz", all_queries)
        write_csv(destination / "stratified_metrics.csv", all_stratified)
        artifact = {"next_round": round_number + 1, "train": train, "pool": pool,
                    "metrics": all_metrics, "diagnostics": all_diagnostics, "queries": all_queries,
                    "stratified": all_stratified}
        torch.save(artifact, destination / "resume.pt.tmp"); (destination / "resume.pt.tmp").replace(checkpoint)
        print(f"{args.model} seed={args.seed} {args.strategy} round={round_number}/{args.rounds} "
              f"test Spearman={all_metrics[-1]['spearman']:.4f}", flush=True)

    write_json(destination / "complete.json", {"rounds_completed": args.rounds + 1,
               "final_n_train": len(train), "final_n_pool": len(pool),
               "true_pool_fitness_used_for_acquisition": False,
               "metrics_sha256": digest(destination / "metrics.csv")})


if __name__ == "__main__": main()
