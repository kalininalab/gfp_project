"""Run one four-arm target-protein adaptation job."""

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from scripts.active_learning.acquisition import acquisition_scores
from scripts.regression_metrics import metrics
from scripts.transfer_active_learning.run import fit, projected, representations, save_plots
from scripts.transfer_benchmark.common import DATA, digest, read_csv, write_csv
from scripts.transfer_benchmark.models import encode_aligned


def evaluate(predict, y, test, title, stem):
    prediction = predict(test)
    score = metrics(y[test], prediction)
    save_plots(y[test], prediction, title, stem)
    return prediction, score


def select_fancy(model, x, train, pool, seed, round_number, device, protocol, count):
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_number]))
    hidden_train = representations(model, x, train, device)
    hidden_pool = representations(model, x, pool, device)
    anchors = rng.choice(len(hidden_train), min(protocol["distance_anchors"], len(hidden_train)), replace=False)
    projection_seed = seed * 1000 + round_number
    labeled = projected(hidden_train[anchors], projection_seed, protocol["projection_dimensions"])
    candidates = projected(hidden_pool, projection_seed, protocol["projection_dimensions"])
    variance = np.zeros(len(pool), dtype=np.float32)
    scores = acquisition_scores(labeled, candidates, variance, alpha=0.62)
    chosen = np.argsort(-scores, kind="stable")[:count]
    return chosen, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directory", type=Path, default=Path("results/target_adaptation_seed42_46"))
    args = parser.parse_args()
    protocol = json.loads((args.directory / "protocol.json").read_text())
    if args.pair not in protocol["pairs"] or args.seed not in protocol["seeds"] or args.model not in protocol["models"]:
        raise ValueError("Job is absent from frozen protocol")
    source, target = args.pair.split("_to_")
    out = args.directory / "fits" / args.pair / f"seed{args.seed}" / args.model
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "complete.json").exists():
            return
        all_rows = read_csv(DATA); table = {row["record_id"]: row for row in all_rows}
        source_alias = {"cgreGFP": "cgre", "amacGFP": "amac", "ppluGFP": "pplu"}[source]
        subset = read_csv(Path(protocol["alignment"]).parent / "subsets" / f"{source_alias}_seed{args.seed}.csv")
        source_train_rows = [table[row["record_id"]] for row in subset if row["split"] == "train"]
        source_validation_rows = [table[row["record_id"]] for row in subset if row["split"] == "validation"]
        target_pool_rows = [row for row in all_rows if row["gene"] == target and row["split"] != "test"]
        target_test_rows = [row for row in all_rows if row["gene"] == target and row["split"] == "test"]
        counts = tuple(map(len, (source_train_rows, source_validation_rows, target_pool_rows, target_test_rows)))
        expected = (protocol["source_train_count"], protocol["source_validation_count"], protocol["target_pool_counts"][target], protocol["target_test_counts"][target])
        if counts != expected:
            raise ValueError(f"Partition sizes {counts} do not match {expected}")
        rows = source_train_rows + source_validation_rows + target_pool_rows + target_test_rows
        alignment = json.loads(Path(protocol["alignment"]).read_text())
        x = encode_aligned(rows, alignment)
        y = np.asarray([float(row["target_log10"]) for row in rows])
        n_source = len(source_train_rows); n_validation = len(source_validation_rows); n_pool = len(target_pool_rows)
        source_train = np.arange(n_source)
        validation = np.arange(n_source, n_source + n_validation)
        pool_base = np.arange(n_source + n_validation, n_source + n_validation + n_pool)
        test = np.arange(n_source + n_validation + n_pool, len(rows))
        if set(source_train) & set(test) or set(pool_base) & set(test):
            raise ValueError("Target-test leakage")

        result_rows = []; prediction_rows = []; query_rows = []
        settings = {"onehot_settings": protocol["onehot_settings"]}
        ids = {"train": source_train, "validation": validation}
        baseline_model, _, _, _, _, baseline_predict = fit(args.model, x, y, ids, args.seed, "cpu", settings)
        baseline_prediction, baseline_score = evaluate(
            baseline_predict, y, test, f"{args.pair} · source only",
            out / "source_only_round_00",
        )
        result_rows.append({"arm": "source_only", "round": 0, "n_target": 0, **baseline_score})
        prediction_rows.extend({"arm": "source_only", "round": 0, "record_id": rows[i]["record_id"], "y_true": float(y[i]), "y_pred": float(value)} for i, value in zip(test, baseline_prediction))

        first_fancy, first_scores = select_fancy(baseline_model, x, source_train, pool_base, args.seed, 0, "cpu", protocol, protocol["one_shot_target_count"])
        random_order = np.random.default_rng(args.seed).permutation(len(pool_base))
        one_shot = {"oneshot_fancy": first_fancy, "oneshot_random": random_order[:protocol["one_shot_target_count"]]}
        for arm, local in one_shot.items():
            selected = pool_base[local]
            model, _, _, _, _, predict = fit(args.model, x, y, {"train": np.concatenate([source_train, selected]), "validation": validation}, args.seed, "cpu", settings)
            prediction, score = evaluate(predict, y, test, f"{args.pair} · {arm}", out / arm)
            result_rows.append({"arm": arm, "round": 1, "n_target": len(selected), **score})
            prediction_rows.extend({"arm": arm, "round": 1, "record_id": rows[i]["record_id"], "y_true": float(y[i]), "y_pred": float(value)} for i, value in zip(test, prediction))
            query_rows.extend({"arm": arm, "selected_after_round": 0, "record_id": rows[i]["record_id"], "score": "" if arm.endswith("random") else float(first_scores[j])} for j, i in zip(local, selected))

        for arm in ("iterative_fancy", "iterative_random"):
            train = source_train.copy(); pool = pool_base.copy()
            for round_number in range(1, protocol["rounds"] + 1):
                if arm == "iterative_fancy":
                    if round_number == 1:
                        chosen = first_fancy[:protocol["target_sequences_per_round"]]
                        scores = first_scores
                    else:
                        chosen, scores = select_fancy(model, x, train, pool, args.seed, round_number - 1, "cpu", protocol, protocol["target_sequences_per_round"])
                else:
                    rng = np.random.default_rng(np.random.SeedSequence([args.seed, round_number - 1, 1]))
                    chosen = rng.choice(len(pool), protocol["target_sequences_per_round"], replace=False)
                    scores = None
                selected = pool[chosen]
                query_rows.extend({"arm": arm, "selected_after_round": round_number - 1, "record_id": rows[i]["record_id"], "score": "" if scores is None else float(scores[j])} for j, i in zip(chosen, selected))
                train = np.concatenate([train, selected]); pool = np.delete(pool, chosen)
                model, _, _, _, _, predict = fit(args.model, x, y, {"train": train, "validation": validation}, args.seed, "cpu", settings)
                prediction, score = evaluate(predict, y, test, f"{args.pair} · {arm} · round {round_number}", out / f"{arm}_round_{round_number:02d}")
                result_rows.append({"arm": arm, "round": round_number, "n_target": round_number * protocol["target_sequences_per_round"], **score})
                prediction_rows.extend({"arm": arm, "round": round_number, "record_id": rows[i]["record_id"], "y_true": float(y[i]), "y_pred": float(value)} for i, value in zip(test, prediction))
        write_csv(out / "metrics.csv", result_rows)
        write_csv(out / "predictions.csv", prediction_rows)
        write_csv(out / "queries.csv", query_rows)
        (out / "complete.json").write_text(json.dumps({
            "protocol_sha256": digest(args.directory / "protocol.json"),
            "metrics_sha256": digest(out / "metrics.csv"),
            "test_records_sha256": protocol["test_records_sha256"],
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
