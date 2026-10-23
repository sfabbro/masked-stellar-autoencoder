"""Frozen-array role integrity and honest calibration controls."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "conformal_validation",
    Path(__file__).parents[1] / "scripts/run_conformal_validation.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_arrays():
    data = {
        "source_id": np.arange(8),
        "fit_rows": np.array([0, 1]),
        "tune_rows": np.array([2, 3]),
        "calibration_rows": np.array([4, 5]),
        "test_rows": np.array([6, 7]),
    }
    pred = {}
    for role in ("test", "calibration"):
        pred[role + "_rows"] = data[role + "_rows"]
        pred[role + "_source_id"] = data["source_id"][data[role + "_rows"]]
        pred[role + "_mean"] = np.zeros((2, 3))
        pred[role + "_variance"] = np.ones((2, 3))
    return data, pred


def test_valid_roles_and_source_binding():
    module.validate_roles(*fixture_arrays())


@pytest.mark.parametrize(
    "failure", ["overlap", "source_shuffle", "negative_variance", "nan_mean"]
)
def test_role_integrity_fail_closed(failure):
    data, pred = fixture_arrays()
    if failure == "overlap":
        data["source_id"][7] = 1
    elif failure == "source_shuffle":
        pred["test_source_id"] = pred["test_source_id"][::-1]
    elif failure == "negative_variance":
        pred["test_variance"][0, 0] = -1
    else:
        pred["test_mean"][0, 0] = np.nan
    with pytest.raises(ValueError):
        module.validate_roles(data, pred)


def test_metalpoor_only_posthoc_evaluation():
    y = np.array([[5000.0, 3.0, -3.0], [6000.0, 4.0, 0.0]])
    result = module.evaluate(
        y,
        y - 1,
        y + 1,
        group="FeH_below_minus2_posthoc",
        method="global/fixed",
        target=2,
    )
    assert result["n"] == 1
    assert result["coverage"] == 1.0


def test_saved_standardized_moments_are_converted_before_native_calibration():
    data, pred = fixture_arrays()
    data["label_center"] = np.array([5000.0, 3.0, -0.2])
    data["label_scale"] = np.array([700.0, 1.1, 0.3])
    pred["test_mean"] = np.array([[1.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
    native = module.native_moments(data, pred)
    np.testing.assert_allclose(
        native["test_mean"], [[5700.0, 1.9, 0.4], [5000.0, 3.0, -0.2]]
    )
    np.testing.assert_allclose(
        native["test_variance"], np.tile([490000.0, 1.21, 0.09], (2, 1))
    )
    # Mixed-coordinate residuals would be enormous; the affine-native conversion is exact.
    y = native["test_mean"].copy()
    np.testing.assert_allclose(y - native["test_mean"], 0)
    assert abs(y[0, 0] - pred["test_mean"][0, 0]) > 5000
    np.testing.assert_array_equal(pred["test_mean"][0], [1.0, -1.0, 2.0])
    data["label_scale"][2] = -1
    with pytest.raises(ValueError, match="positive scale"):
        module.native_moments(data, pred)
