import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/processed/baseline_v1/sequences.csv'
DEFAULT = Path('/data/users/akolchina/gfp_project/artificial_peak_holdout_seed42_46_runs')
PUBLISHED = ROOT / 'results/artificial_peak_holdout_seed42_46'
PEAKS = ['cgre132', 'cgre1338', 'cgre4111', 'cgre9708']
TARGETS = ['cgre1338', 'cgre4111', 'cgre132']
MODES = ['artificial_only', 'with_natural']
MODELS = ['aubin_1_10_1', 'aubin_linear', 'mlp_small', 'CNN_Jannis_OHE']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
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
    paths = [Path(__file__).with_name(name) for name in ['common.py', 'prepare.py', 'run.py']]
    paths += [ROOT / 'scripts' / name for name in [
        'transfer_benchmark/models.py', 'aubin_model.py', 'baseline_models.py',
        'train_aubin.py', 'regression_metrics.py', 'esm_benchmark/models.py']]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}
