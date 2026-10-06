import csv
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'results/transfer_cgreGFP_seed42_46_runs'
DATA=ROOT/'data/processed/baseline_v1/sequences.csv'
PEAKS=['cgre132','cgre1338','cgre4111','cgre9708']
NATURAL=['cgreGFP','amacGFP','ppluGFP']
GENES=NATURAL+PEAKS
MODELS=['aubin_1_10_1','aubin_linear','mlp_small','CNN_Jannis_OHE']
MIXES={
    'cgre':['cgreGFP'], 'amac':['amacGFP'], 'pplu':['ppluGFP'],
    'cgre_amac':['cgreGFP','amacGFP'], 'cgre_pplu':['cgreGFP','ppluGFP'],
    'amac_pplu':['amacGFP','ppluGFP'], 'cgre_amac_pplu':NATURAL,
    'artificial':PEAKS,
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def scientific_sources():
    files=[Path(__file__).with_name(n) for n in ['common.py','prepare.py','models.py','run.py','audit_features.py']]
    files += [ROOT/'scripts'/n for n in ['aubin_model.py','baseline_models.py','train_aubin.py','regression_metrics.py','esm_benchmark/models.py']]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}
