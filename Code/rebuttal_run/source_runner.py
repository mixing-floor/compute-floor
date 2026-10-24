"""Load the submitted experiment runner as the implementation source of truth."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import run_experiment as core  # noqa: E402,F401


SOURCE_RUNNER_PATH = CODE_DIR / "run_experiment.py"
