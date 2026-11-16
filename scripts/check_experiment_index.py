"""Fail when a top-level results folder is missing from the figure timeline."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "EXPERIMENT_TIMELINE.md"


def main():
    folders = {path.name for path in (ROOT / "results").iterdir() if path.is_dir()}
    linked = set(re.findall(r"\.\./results/([^/)#]+)", INDEX.read_text()))
    missing = sorted(folders - linked)
    stale = sorted(linked - folders)
    if missing or stale:
        details = []
        if missing:
            details.append("missing from timeline: " + ", ".join(missing))
        if stale:
            details.append("timeline links absent folders: " + ", ".join(stale))
        raise SystemExit("; ".join(details))
    print(f"Experiment timeline covers all {len(folders)} top-level result folders.")


if __name__ == "__main__":
    main()
