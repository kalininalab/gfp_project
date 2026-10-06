"""Wait for the five cluster fits, then validate and assemble their report."""
import time
from pathlib import Path
from .finalize import main as finalize


def main():
    root=Path('results/paper_reproduction_cgre_80_20/seed_42')
    models=('aubin_1_10_1','mlp_small','mlp_deep','CNN_Jannis_OHE','CNN_Jannis_ESM')
    deadline=time.monotonic()+86400
    while time.monotonic()<deadline:
        complete=[(root/model/'complete.json').exists() for model in models]
        print(f'Completed {sum(complete)}/{len(models)} models',flush=True)
        if all(complete):
            finalize();return
        time.sleep(60)
    raise TimeoutError('80/20 reproduction incomplete after 24 hours; inspect model logs')


if __name__=='__main__':main()
