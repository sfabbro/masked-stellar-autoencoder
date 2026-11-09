import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "andrae_census",
    Path(__file__).parents[1] / "scripts/run_andrae_full_footprint_census.py",
)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def test_exact_integer_match_above_float_precision_and_boundary():
    ids = np.array([2**60, 2**60 + 1, 2**60 + 3], dtype="i8")
    rows, positions = census.match(
        ids, np.array([2**60 + 1, 0, 2**60 + 4, 2**60], dtype="i8")
    )
    np.testing.assert_array_equal(rows, [0, 3])
    np.testing.assert_array_equal(positions, [1, 0])
    with pytest.raises(ValueError):
        census.match(ids, ids.astype(float))
