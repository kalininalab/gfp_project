"""Validate completed folds, aggregate OOF predictions, and draw CV figures."""

import argparse
import json
from pathlib import Path

try:
    from .cross_validate import MODELS, summarize, sha256
    from .plot_cross_validation import plot
except ImportError:
    from cross_validate import MODELS, summarize, sha256
    from plot_cross_validation import plot


def finalize(directory):
    config = json.loads((directory/'config.json').read_text())
    for fold in range(config['n_splits']):
        path = directory/f'fold_{fold:02d}'/'metrics.json'
        if not path.exists() or set(json.loads(path.read_text())) != set(MODELS):
            raise ValueError(f'Fold {fold} is not complete')
    if sha256(Path(config['data'])) != config['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch')
    results = summarize(config)
    report = dict(config=config, status='complete', results=results)
    handoff = directory/'scheduler_handoff.json'
    if handoff.exists():
        report['scheduler_handoff'] = json.loads(handoff.read_text())
    report['finalization_scripts'] = {p.name:sha256(p) for p in [Path(__file__), Path(__file__).with_name('plot_cross_validation.py')]}
    (directory/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(directory)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    finalize(parser.parse_args().directory)
