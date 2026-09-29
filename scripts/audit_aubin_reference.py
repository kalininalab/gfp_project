"""Recompute reference metrics from Aubin's archived prediction CSVs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]


def audit(gene):
    base = ROOT/'Orthologous_GFP_Fitness_Peaks/analysis/ML/_08_Figures/for_paper'
    result = {}
    for path in sorted(base.glob(f'{gene}_*.csv')):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {'Observed fluorescence', 'Predicted fluorescence'} <= rows[0].keys():
            continue
        y = [float(r['Observed fluorescence']) for r in rows]
        pred = [float(r['Predicted fluorescence']) for r in rows]
        result[path.name] = dict(n=len(y), r2=float(r2_score(y, pred)),
                                 spearman=float(spearmanr(y, pred).statistic),
                                 source=str(path.relative_to(ROOT)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not result:
        raise ValueError(f'No archived prediction tables for {gene}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gene', default='cgreGFP')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(audit(args.gene), stream, indent=2)
        stream.write('\n')
