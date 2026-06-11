#!/usr/bin/env python3
"""run_eval.py — backward-compatible shim for scripts/fit_models.py.

This file is kept for backward compatibility with existing notebooks and docs.
New usage should call scripts/fit_models.py directly:

    uv run python scripts/fit_models.py --name my_run
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_models import main  # noqa: E402

if __name__ == "__main__":
    main()
