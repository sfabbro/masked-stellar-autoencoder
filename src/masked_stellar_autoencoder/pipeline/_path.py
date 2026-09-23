"""Put the local torchregress checkout on sys.path when it is not installed."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_torchregress() -> None:
    if importlib.util.find_spec("torchregress") is not None:
        return
    path = Path(__file__).resolve()
    # repo/src/masked_stellar_autoencoder/pipeline/_path.py -> ~/src/astroai/torchregress
    if len(path.parents) <= 5:
        raise ImportError("torchregress is not installed")
    candidate = path.parents[5] / "astroai" / "torchregress" / "src"
    if not candidate.is_dir():
        raise ImportError(
            "torchregress is not installed and was not found at "
            f"{candidate}. Set PYTHONPATH to its src directory."
        )
    sys.path.insert(0, str(candidate))


ensure_torchregress()
