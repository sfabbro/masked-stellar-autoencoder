"""Pytest configuration: repo root on sys.path for `data/` and `src/` imports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Tests exercise the installed dependency, matching production. Do not silently
# substitute an unrelated sibling checkout for the pinned environment.
for path in (str(SRC), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
