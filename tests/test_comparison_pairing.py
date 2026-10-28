"""Comparisons must reject plausible-looking metrics from mispaired arrays."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_uncertainties import coverage, split_conformal
from comparison_pairing import bound_predictions


def fixture():
    cohort = dict(
        source_id=np.array([101, 102, 103]),
        test_rows=np.array([2, 0]),
        calibration_rows=np.array([1]),
        value=np.ones((3, 3)),
        label_center=np.array([4000.0, 2.0, -1.0]),
        label_scale=np.array([500.0, 1.0, 0.5]),
    )
    pred = dict(
        test_source_id=np.array([103, 101]),
        test_mean=np.ones((2, 3)),
        test_variance=np.ones((2, 3)),
    )
    return cohort, pred


def test_native_units_and_variance():
    c, p = fixture()
    mean, std = bound_predictions(c, p, "test", variance=True)
    np.testing.assert_array_equal(mean, [[4500, 3, -0.5]] * 2)
    np.testing.assert_array_equal(std, [[500, 1, 0.5]] * 2)


@pytest.mark.parametrize(
    "fault", ["reorder", "missing", "shape", "nan", "overlap", "duplicates", "variance"]
)
def test_pairing_fails_closed(fault):
    c, p = fixture()
    if fault == "reorder":
        p["test_source_id"] = p["test_source_id"][::-1]
    elif fault == "missing":
        del p["test_source_id"]
    elif fault == "shape":
        p["test_mean"] = np.ones((2, 1))
    elif fault == "nan":
        p["test_mean"][0, 0] = np.nan
    elif fault == "overlap":
        c["calibration_rows"] = np.array([2])
    elif fault == "duplicates":
        c["source_id"][0] = c["source_id"][1]
    elif fault == "variance":
        p["test_variance"][0, 0] = -1
    with pytest.raises(ValueError):
        bound_predictions(c, p, "test", variance=True)


def test_unattainable_conformal_rank_produces_unbounded_interval():
    lo, hi, q = split_conformal(
        np.zeros(3), np.ones(3), np.arange(3), np.array([1.0]), np.array([1.0]), 0.1
    )
    assert q == np.inf and lo[0] == -np.inf and hi[0] == np.inf
    import pandas as pd

    result = coverage(
        pd.DataFrame(dict(lo=lo, hi=hi, truth=[1000.0])), "lo", "hi", "truth"
    )
    assert result["coverage"] == 1 and result["mean_width"] == np.inf


def test_finite_conformal_rank_and_invalid_alpha():
    _, _, q = split_conformal(
        np.zeros(3), np.ones(3), np.arange(3), np.zeros(1), np.ones(1), 0.5
    )
    assert q == 1
    with pytest.raises(ValueError):
        split_conformal(
            np.zeros(3), np.ones(3), np.arange(3), np.zeros(1), np.ones(1), 0
        )


def test_andrae_reports_both_systems_on_identical_heldout_rows(tmp_path, monkeypatch):
    import json

    import compare_released_andrae as comparison
    import pandas as pd

    c, p = fixture()
    np.savez(tmp_path / "cohort.npz", **c)
    np.savez(tmp_path / "pred.npz", **p)
    monkeypatch.setattr(
        comparison,
        "scan_fits",
        lambda *args: pd.DataFrame(
            {
                "source_id": [103, 101],
                "teff_andrae": [4100.0, 4300.0],
                "logg_andrae": [2.0, 3.0],
                "mh_andrae": [-1.0, -0.5],
                "andrae_training_sample": [True, False],
            }
        ),
    )
    comparison.main(
        [
            "--cohort",
            str(tmp_path / "cohort.npz"),
            "--predictions",
            str(tmp_path / "pred.npz"),
            "--fits",
            "unused",
            "--output",
            str(tmp_path / "result.png"),
        ]
    )
    summary = json.loads((tmp_path / "result.json").read_text())
    for key in ("teff", "logg", "feh"):
        assert summary["ours_heldout"]["all"][key]["n"] == 1
        assert summary["andrae_heldout"]["all"][key]["n"] == 1
    assert summary["ours_heldout"]["all"]["teff"]["mae"] == 4499.0
