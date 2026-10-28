"""Strict CPU replay gates preserve numerical failures without stopping other arms."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = (
    Path(__file__).resolve().parents[1]
    / "experiments/xp/monitor/verify_frozen_screens.py"
)
spec = importlib.util.spec_from_file_location("frozen_verify", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_original_nll_discrepancy_remains_strict_failure():
    maximum, rows = module.prediction_errors(
        np.array([-0.8167616129]), np.array([-0.8169975281])
    )
    assert maximum > 0.0002 and rows.tolist() == [0]
    _, rows = module.prediction_errors(np.array([1.0 + 6e-5]), np.array([1.0]))
    assert rows.tolist() == []  # declared rtol+atol, not an absolute-only rule


def test_nonfinite_replay_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        module.prediction_errors(np.array([np.nan]), np.array([0.0]))
