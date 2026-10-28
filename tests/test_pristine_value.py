import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_pristine_value import adjuncts, compare_heads, replay, ridge_fit_predict


def test_ridge_does_not_fit_test_outliers_and_reloads_exact():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 4))
    y = x[:, 0] * 2
    fit = np.arange(60)
    test = np.arange(60, 100)
    pred, state = ridge_fit_predict(x, y, fit, test)
    changed = y.copy()
    changed[test] = 1e6
    pred2, state2 = ridge_fit_predict(x, changed, fit, test)
    np.testing.assert_array_equal(pred, pred2)
    np.testing.assert_array_equal(state["center"], x[fit].mean(0))
    np.testing.assert_array_equal(pred, replay(x[test], state2))


def test_six_nested_heads_share_tests_and_exclude_benchmark_validation(tmp_path):
    rng = np.random.default_rng(1)
    z = rng.normal(size=(120, 4))
    extra = rng.normal(size=(120, 3))
    split = np.r_[np.zeros(80, dtype="u1"), np.full(40, 2, dtype="u1")]
    membership = np.r_[
        np.zeros(30, dtype="u1"), np.ones(10, dtype="u1"), np.full(80, 255, dtype="u1")
    ]
    y = z[:, 0] + extra[:, 2]
    test, arrays, initial, metrics = compare_heads(
        z, extra, y, split, membership, tmp_path
    )
    assert len(test) == 40 and len(metrics) == 6
    assert not np.isin(
        arrays["broad_excluding_benchmark_validation__fit_row"], np.arange(30, 40)
    ).any()
    assert metrics["strict_benchmark_train__encoder"]["fit_rows"] == 30
    assert metrics["broad_excluding_benchmark_validation__encoder"]["fit_rows"] == 70
    for cohort in ["strict_benchmark_train", "broad_excluding_benchmark_validation"]:
        np.testing.assert_array_equal(
            initial[cohort + "__encoder"], initial[cohort + "__encoder_gaia_cahk"]
        )
    phot = {
        "Bpmag0": np.array([3.0]),
        "Rpmag0": np.array([1.0]),
        "Gmag0": np.array([2.0]),
        "CaHKmag0": np.array([4.0]),
    }
    np.testing.assert_array_equal(adjuncts(phot), [[2.0, 1.0, 2.0]])


def test_nonfinite_encoder_features_cannot_publish_success():
    import pytest

    x = np.ones((50, 3))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        ridge_fit_predict(x, np.ones(50), np.arange(25), np.arange(25, 50))
