"""Shared constants and small I/O helpers for the amacGFP AL benchmark."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

DATA = Path("data/processed/baseline_v1/sequences.csv")
RESULTS = Path("results/amac_active_learning_seed42_44")
RAW = Path("/data/users/akolchina/gfp_project/amac_active_learning_seed42_44")
SEEDS = (42, 43, 44)
MODELS = ("CNN_Jannis_OHE", "mlp_small")
STRATEGIES = (
    "random", "uncertainty", "distance", "fitness_high", "fitness_low",
    "uncertainty_fitness_high", "uncertainty_fitness_low", "uncertainty_distance",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(temporary, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def percentile(values):
    """Return deterministic percentile ranks in [0, 1], averaging ties."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float32)
    return ((rankdata(values, method="average") - 1) / (len(values) - 1)).astype(np.float32)


def strategy_scores(strategy, uncertainty, distance, prediction, rng):
    """Score a pool without accepting true candidate labels."""
    u, d, high = map(percentile, (uncertainty, distance, prediction))
    low = 1.0 - high
    if strategy == "random":
        score = rng.random(len(high))
    elif strategy == "uncertainty": score = u
    elif strategy == "distance": score = d
    elif strategy == "fitness_high": score = high
    elif strategy == "fitness_low": score = low
    elif strategy == "uncertainty_fitness_high": score = 0.5 * (u + high)
    elif strategy == "uncertainty_fitness_low": score = 0.5 * (u + low)
    elif strategy == "uncertainty_distance": score = 0.5 * (u + d)
    else: raise ValueError(f"Unknown strategy: {strategy}")
    return np.asarray(score, dtype=np.float32), u, d, high, low
