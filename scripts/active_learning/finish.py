"""Refresh AL result tables during cluster runs and validate their completion."""
import time
from pathlib import Path

from .plot import main as plot_main


def main():
    """Monitor the default pilot without keeping an interactive session alive.

    Run as a separate lightweight cluster job. Model jobs own their output
    folders; this process only regenerates top-level tables and plots. Fail after
    24 hours if trajectories remain incomplete, rather than report success.
    """
    directory = Path('results/active_learning_cgre_fixed_test')
    deadline = time.monotonic() + 86400
    signature = None
    while time.monotonic() < deadline:
        paths = sorted(directory.glob('seed_*/*/metrics.csv'))
        completed = sorted(directory.glob('seed_*/*/complete.json'))
        current = [(str(p),p.stat().st_mtime_ns) for p in paths+completed]
        if paths and current != signature:
            # main() uses its own CLI parser; this monitor has no CLI arguments.
            plot_main()
            signature = current
            from .plot import collect_results
            _, issues = collect_results(directory)
            if not issues:
                print('All five pilot trajectories complete; final report and figures validated.', flush=True)
                return
            print(f'Report refreshed; {len(issues)} incomplete-run notices.', flush=True)
        time.sleep(60)
    raise TimeoutError('Active learning remains incomplete after 24 hours; inspect model job logs')


if __name__ == '__main__':
    main()
