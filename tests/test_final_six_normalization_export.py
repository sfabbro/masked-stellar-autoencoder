"""Synthetic metadata only; no checkpoint or native model loading."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_final_six_normalization import extract_normalization


def test_six_metadata_values_and_coordinate_order():
    value = dict(
        label_center=np.arange(6),
        label_scale=np.ones(6),
        target_names=["teff", "logg", "fe_h", "alpha", "age", "distance"],
    )
    result = extract_normalization(value)
    assert result["center"] == list(range(6))
    assert result["coordinates"][-2:] == ["ln(Gyr)", "ln(kpc)"]
    with pytest.raises(ValueError):
        extract_normalization(dict(value, label_scale=np.zeros(6)))
    with pytest.raises(ValueError):
        extract_normalization(
            dict(value, target_names=list(reversed(value["target_names"])))
        )
