"""Shared constants and provenance for the learning-curve experiment."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/'results/learning_curves_cgre_cv10'
DATA = ROOT/'data/processed/baseline_v1/sequences.csv'
PEAKS = ['cgre132', 'cgre1338', 'cgre4111', 'cgre9708']
SOURCES = ['natural', 'artificial']
MODELS = ['aubin_1_10_1', 'mlp_small', 'mlp_deep', 'CNN_Jannis_OHE']
SIZES = [500, 1000, 2500, 5000, 10000, 15000, 17500]
COLORS = ['#4267AC', '#D6813A', '#865BA6', '#BE577C']
LABELS = ['Aubin 1–10–1', 'Small MLP', 'Deep MLP', 'Jannis OHE']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scientific_sources():
    files = [Path(__file__).with_name(name) for name in ['common.py', 'prepare.py', 'run.py']]
    files += [ROOT/'scripts'/name for name in ['aubin_model.py', 'baseline_models.py',
        'train_aubin.py', 'regression_metrics.py', 'esm_benchmark/run.py',
        'esm_benchmark/models.py', 'esm_benchmark/embed.py']]
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def fit_directory(directory, source, fold, size, model):
    return Path(directory)/'fits'/source/f'fold_{fold:02d}'/f'n_{size:05d}'/model
