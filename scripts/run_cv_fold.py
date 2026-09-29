"""Run/resume one frozen CV fold on a batch worker without refitting finished models."""

import argparse
import fcntl
import json
from pathlib import Path

try:
    from . import cross_validate as cv
except ImportError:
    import cross_validate as cv


def run(config_path, fold):
    config = json.loads(config_path.read_text())
    if not 0 <= fold < config['n_splits']:
        raise ValueError('Invalid fold index')
    for name, expected in config['scripts'].items():
        if cv.sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Scientific code changed since folds were frozen: {name}')
    if cv.sha256(Path(config['data'])) != config['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch')
    output = Path(config['output'])/f'fold_{fold:02d}'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        all_models = list(cv.MODELS)
        completed = {}
        for name in all_models:
            destination = output/name
            metrics_path = destination/'metrics.json'
            if metrics_path.exists() and (destination/'predictions.csv').exists() and (
                    (destination/'model.pt').exists() or (destination/'model.pkl').exists()):
                try:
                    report = json.loads(metrics_path.read_text())
                    if {'validation','test'} <= report['metrics'].keys():
                        completed[name] = report
                except (json.JSONDecodeError, KeyError):
                    pass  # Interrupted writes are rerun, never counted complete.
        missing = [name for name in all_models if name not in completed]
        print(f'Fold {fold}: reuse {list(completed)}; fit {missing}', flush=True)
        if missing:
            try:
                cv.MODELS = missing
                cv.one_fold(fold, config)
            finally:
                cv.MODELS = all_models
        combined = {name:json.loads((output/name/'metrics.json').read_text()) for name in all_models}
        temporary = output/'metrics.json.tmp'
        temporary.write_text(json.dumps(combined, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'metrics.json')
        print(f'Fold {fold}: all eight models complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.fold)
