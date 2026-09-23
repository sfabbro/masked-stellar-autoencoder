"""Pytest configuration: repo root on sys.path for `data/` and `src/` imports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# ~/src/astroai/torchregress when this repo lives at ~/src/sfabbro/masked-stellar-autoencoder
TORCHREGRESS = ROOT.parents[1] / "astroai" / "torchregress" / "src"
for path in (str(SRC), str(ROOT), str(TORCHREGRESS)):
    if Path(path).is_dir() and path not in sys.path:
        sys.path.insert(0, path)
