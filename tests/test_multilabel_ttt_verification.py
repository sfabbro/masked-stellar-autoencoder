"""Invariant tests for the independent reconstruction-TTT reference."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/xp/monitor"))
reference = importlib.import_module("verify_multilabel_ttt")


def test_manual_adam_and_per_object_exact_reset():
    reference.self_test()


def test_hash_selection_is_input_order_independent_and_holdout_observed_only():
    ids = np.array([19, 77, 3, 5, 120], dtype="i8")
    permutation = np.array([4, 2, 3, 0, 1])
    np.testing.assert_array_equal(
        ids[reference.choose(ids, 3, 1042)],
        ids[permutation][reference.choose(ids[permutation], 3, 1042)],
    )
    visible = np.array([True, False, True, True])
    columns = reference.heldout(visible, [0, 1, 2, 3], 77)
    assert 0 < len(columns) < 3
    assert set(columns).issubset({0, 2, 3})
    with pytest.raises(ValueError, match="Two genuine"):
        reference.heldout(np.array([True, False]), [0, 1], 77)
