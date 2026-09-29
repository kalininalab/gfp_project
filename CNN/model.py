"""Corrected CNN entry point. See README.md for the frozen-split benchmark.

The historical implementation is preserved verbatim as model_legacy.py.
Architecture definitions live in scripts.esm_benchmark.models; the shared trainer
keeps labels aligned, separates validation/test, and restores copied checkpoints.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.esm_benchmark.models import CNN_old, CNN_new, CNN_Jannis

if __name__ == '__main__':
    from scripts.esm_benchmark.run import main
    if '--model' not in sys.argv:
        sys.argv.extend(['--model', 'CNN_old'])
    main()
