"""Paired bootstrap and exact identity staging contracts."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "catalogue_validation",
    Path(__file__).parents[1] / "scripts/run_catalogue_comparison_validation.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_bootstrap_uses_paired_indices():
    a = np.array([1.0, 5.0, 12.0, 31.0])
    np.testing.assert_array_equal(
        module.paired_bootstrap(a, a, repetitions=20), np.zeros((2, 3))
    )
    b = a + 2
    ci = module.paired_bootstrap(a, b, repetitions=20)
    np.testing.assert_allclose(ci[:, 0:2], -2)
    np.testing.assert_allclose(ci[:, 2], 0)


def test_stage_exact_int64_and_heldout(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "source_id": [2050237616959273729, 2050237616959273731],
            "feh_true": [-3.0, 0.0],
            "ours": [-2.0, 0.1],
            "catalogue": [-1.0, 0.2],
            "andrae_training_sample": [True, False],
        }
    ).to_csv(source, index=False)
    dest = tmp_path / "input.npz"
    module.stage(source, dest, [("FeH", "feh_true", "ours", "catalogue")])
    with np.load(dest) as f:
        assert f["source_id"][0] == 2050237616959273729
        np.testing.assert_array_equal(f["heldout"], [False, True])


def test_duplicate_identity_refused(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "source_id": [1, 1],
            "feh_true": [0.0, 0.0],
            "ours": [0.0, 0.0],
            "catalogue": [0.0, 0.0],
        }
    ).to_csv(source, index=False)
    with pytest.raises(ValueError, match="unique"):
        module.stage(
            source, tmp_path / "input.npz", [("FeH", "feh_true", "ours", "catalogue")]
        )


def test_bootstrap_requires_finite_same_rows():
    with pytest.raises(ValueError):
        module.paired_bootstrap(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
