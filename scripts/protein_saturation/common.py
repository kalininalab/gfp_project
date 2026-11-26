import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/baseline_v1/sequences.csv"
RAW = Path("/data/users/akolchina/gfp_project/protein_train_size_saturation_seed42_46")
PUBLISHED = ROOT / "results/protein_train_size_saturation_seed42_46"
GENES = ("cgreGFP", "amacGFP", "ppluGFP", "avGFP")
MODELS = ("mlp_small", "CNN_Jannis_OHE")
SEEDS = tuple(range(42, 47))
BASE_SIZES = (500, 1000, 2000, 4000, 8000, 12000)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream: return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def scientific_sources():
    paths = [Path(__file__).with_name(name) for name in ("common.py", "prepare.py", "run.py")]
    paths += [ROOT / "scripts" / name for name in (
        "ortholog_holdout/prepare.py", "peak_holdout/run.py", "transfer_benchmark/models.py",
        "train_aubin.py", "regression_metrics.py", "baseline_models.py", "esm_benchmark/models.py")]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}
