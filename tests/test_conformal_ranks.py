"""Finite-sample conformal ranks, joint rank-score regions and Mondrian strata.

The central claim under test is the split-conformal guarantee:

    exact order statistic  q_hat = s_( ceil((n+1)(1-alpha)) )
    => P( s_test <= q_hat ) >= 1 - alpha      for exchangeable scores,

and that the plug-in ``numpy.quantile`` version does **not** have it.  Every
Monte-Carlo assertion states its tolerance: with ``REPS = 2000`` the standard
error of a 0.9 coverage estimate is ``sqrt(0.9*0.1/2000) = 0.0067``, so the
checks use three standard errors (and one scenario is deliberately chosen with
a discrete floor well above 1 - alpha so that "empirical >= nominal" is not a
coin flip).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from masked_stellar_autoencoder.reporting.calibration import (
    InsufficientCalibrationError,
    bonferroni_box_region,
    box_region_volume,
    ellipsoid_region_volume,
    exact_quantile,
    finite_sample_coverage_floor,
    finite_sample_rank,
    joint_rank_score_region,
    levelset_method_for,
    mahalanobis_scores,
    mondrian_conformal,
    mondrian_membership,
    neg_log_p_scores,
    region_volume,
    split_conformal_interval,
)
from masked_stellar_autoencoder.training.conformal import (
    apply_cqr_offsets_inplace,
    calibrate_cqr_offsets,
    conformity_scores_two_sided,
    exact_order_stat_index,
    split_conformal_quantile,
)

REPS = 2000
ALPHA = 0.1


def _standard_error(p: float, reps: int = REPS) -> float:
    return math.sqrt(p * (1.0 - p) / reps)


def _coverage_exact_and_plugin(
    rng: np.random.Generator, n_cal: int, alpha: float, reps: int = REPS
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Exchangeable Gaussian scores; returns (coverage_exact, coverage_plugin)."""
    calibration = rng.normal(size=(reps, n_cal))
    test = rng.normal(size=reps)
    rank = finite_sample_rank(n_cal, alpha)
    ordered = np.sort(calibration, axis=1)
    exact_threshold = ordered[:, rank - 1]
    plugin_threshold = np.quantile(calibration, 1.0 - alpha, axis=1)
    cover_exact = test <= exact_threshold
    cover_plugin = test <= plugin_threshold
    return (
        float(cover_exact.mean()),
        float(cover_plugin.mean()),
        exact_threshold,
        plugin_threshold,
    )


# ---------------------------------------------------------------------------
# exact rank
# ---------------------------------------------------------------------------
def test_finite_sample_rank_formula_and_implementations_agree():
    for n in (1, 2, 5, 19, 24, 49, 141, 199, 2046):
        for alpha in (0.01, 0.05, 0.1, 0.2, 0.5):
            rank = finite_sample_rank(n, alpha)
            assert rank == math.ceil((n + 1) * (1.0 - alpha))
            assert rank == exact_order_stat_index(n, alpha)
            floor = finite_sample_coverage_floor(n, alpha)
            assert floor == rank / (n + 1)
            # The discrete achievable coverage always meets the nominal level.
            assert floor >= 1.0 - alpha - 1e-12


def test_exact_rank_covers_at_least_nominal_over_2000_replications():
    # n_cal = 24, alpha = 0.1 gives rank 23 of 25 => achievable coverage 0.92,
    # i.e. a discrete floor 3.3 standard errors above the 0.90 nominal level.
    n_cal = 24
    rng = np.random.default_rng(20261022)
    cover_exact, cover_plugin, _, _ = _coverage_exact_and_plugin(rng, n_cal, ALPHA)
    floor = finite_sample_coverage_floor(n_cal, ALPHA)
    assert floor == pytest.approx(0.92)
    assert cover_exact >= 1.0 - ALPHA, (
        f"exact rank under-covered: {cover_exact:.4f} < {1 - ALPHA}"
    )
    assert cover_exact == pytest.approx(floor, abs=3.0 * _standard_error(floor))
    # The plug-in quantile is visibly worse at small n: it interpolates below
    # the 23rd of 25 order statistics, so its coverage is ~0.868.
    assert cover_plugin < cover_exact - 0.02
    assert cover_plugin < 1.0 - ALPHA - 0.01


def test_training_quantile_helper_matches_reporting_exact_quantile():
    rng = np.random.default_rng(13)
    scores = rng.normal(size=137)
    from masked_stellar_autoencoder.reporting.calibration import exact_quantile as exact

    for alpha in (0.01, 0.05, 0.1, 0.2):
        assert split_conformal_quantile(scores, alpha) == exact(scores, alpha)


def test_plugin_quantile_is_below_the_exact_order_statistic_at_scale():
    rng = np.random.default_rng(7)
    for n_cal in (24, 49, 199, 2046):
        scores = np.sort(rng.normal(size=n_cal))
        rank = finite_sample_rank(n_cal, ALPHA)
        plugin = float(np.quantile(scores, 1.0 - ALPHA))
        exact = float(scores[rank - 1])
        assert plugin <= exact + 1e-15
        if n_cal > 49:
            assert plugin < exact, "plug-in interpolation must not reach the exact rank"


def test_large_n_coverage_matches_the_achievable_floor():
    # Realistic n: n_cal = 199, rank = 180, achievable coverage = 180/200.
    n_cal = 199
    rng = np.random.default_rng(11)
    cover_exact, cover_plugin, exact_thr, plugin_thr = _coverage_exact_and_plugin(
        rng, n_cal, ALPHA
    )
    floor = finite_sample_coverage_floor(n_cal, ALPHA)
    assert floor == pytest.approx(0.9)
    assert cover_exact >= 1.0 - ALPHA - 3.0 * _standard_error(floor)
    assert cover_exact == pytest.approx(floor, abs=3.0 * _standard_error(floor))
    # Every replication has plugin_threshold strictly below the exact one.
    assert np.all(plugin_thr < exact_thr)
    assert cover_plugin <= cover_exact


# ---------------------------------------------------------------------------
# rank too large: raise, never a zero offset
# ---------------------------------------------------------------------------
def test_rank_exceeding_n_raises_instead_of_zero_offset():
    with pytest.raises(InsufficientCalibrationError, match="exceeds n_cal"):
        exact_quantile(np.arange(8, dtype=float), 0.1)
    # alpha=0.01 needs n >= 99 for the rank not to exceed n.
    with pytest.raises(InsufficientCalibrationError):
        exact_quantile(np.arange(98, dtype=float), 0.01)
    assert exact_quantile(np.arange(99, dtype=float), 0.01) == 98.0
    with pytest.raises(InsufficientCalibrationError, match="no conformity scores"):
        exact_quantile(np.array([]), 0.1)
    with pytest.raises(ValueError, match="must be in"):
        exact_quantile(np.arange(10, dtype=float), 0.0)


def test_cqr_calibration_raises_at_small_n():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(18, 1))
    pred = np.stack([y - 0.1, y, y + 0.1], axis=2)
    # per-tail level alpha/2 = 0.05 needs rank ceil(19*0.95) = 19 > 18.
    with pytest.raises(ValueError, match="exceeds n_cal"):
        calibrate_cqr_offsets(y, pred, alpha=0.1)
    # 19 rows is the smallest split that supports the level; no zero fallback.
    y = rng.normal(size=(19, 1))
    pred = np.stack([y - 0.1, y, y + 0.1], axis=2)
    doc = calibrate_cqr_offsets(y, pred, alpha=0.1)
    assert doc["order_stat_index_lower"] == [19]
    assert doc["finite_sample_exact"] is True


# ---------------------------------------------------------------------------
# two-sided constructions
# ---------------------------------------------------------------------------
def test_two_sided_levels_are_documented_and_cover():
    rng = np.random.default_rng(19)
    n_cal = 99
    y_cal = rng.normal(size=(REPS, n_cal))
    y_test = rng.normal(size=REPS)
    # Constant predicted box [-0.5, 0.5]: a guaranteed miss for many stars, so
    # the calibrated offsets actually have to do work.
    lower = np.full((REPS, n_cal), -0.5)
    upper = np.full((REPS, n_cal), 0.5)

    for two_sided, expected_level in (
        ("per_tail", 1.0 - ALPHA / 2.0),
        ("one_level", 1.0 - ALPHA),
    ):
        index = finite_sample_rank(
            n_cal, ALPHA / 2.0 if two_sided == "per_tail" else ALPHA
        )
        cover = np.zeros(REPS, dtype=bool)
        for rep in range(REPS):
            doc = split_conformal_interval(
                y_cal[rep], lower[rep], upper[rep], alpha=ALPHA, two_sided=two_sided
            )
            assert doc["level_lower"] == pytest.approx(expected_level)
            assert doc["order_stat_index_lower"] == index
            cover[rep] = (
                lower[rep][0] - doc["offsets_lower"]
                <= y_test[rep]
                <= upper[rep][0] + doc["offsets_upper"]
            )
        coverage = float(cover.mean())
        assert coverage >= 1.0 - ALPHA - 3.0 * _standard_error(1.0 - ALPHA), (
            f"{two_sided} coverage {coverage:.4f} below nominal"
        )


def test_reporting_and_training_calibration_agree():
    rng = np.random.default_rng(23)
    n, ell = 200, 2
    y = rng.normal(size=(n, ell))
    pred = np.stack([y - 0.4, y, y + 0.3], axis=2)
    for two_sided in ("per_tail", "one_level"):
        doc = calibrate_cqr_offsets(y, pred, alpha=0.1, two_sided=two_sided)
        assert doc["two_sided"] == two_sided
        for j in range(ell):
            reference = split_conformal_interval(
                y[:, j], pred[:, j, 0], pred[:, j, 2], alpha=0.1, two_sided=two_sided
            )
            assert doc["offsets_lower"][j] == reference["offsets_lower"]
            assert doc["offsets_upper"][j] == reference["offsets_upper"]
            assert (
                doc["order_stat_index_lower"][j] == reference["order_stat_index_lower"]
            )
            assert doc["n_cal"][j] == reference["n_cal"]
        # And the one-level score is the documented max() form.
        combined = conformity_scores_two_sided(y, pred[:, :, 0], pred[:, :, 2])
        np.testing.assert_allclose(
            combined,
            np.maximum(pred[:, :, 0] - y, y - pred[:, :, 2]),
        )


def test_exact_offsets_exceed_plugin_offsets_on_spread_scores():
    rng = np.random.default_rng(29)
    n = 199
    y = rng.normal(size=(n, 1))
    # A fixed, deliberately too-narrow predicted box.
    pred = np.stack(
        [np.full((n, 1), -0.5), np.zeros((n, 1)), np.full((n, 1), 0.5)], axis=2
    )
    doc = calibrate_cqr_offsets(y, pred, alpha=0.1, two_sided="per_tail")
    score_lower = np.maximum(0.0, pred[:, :, 0] - y)[:, 0]
    plugin = float(np.quantile(score_lower, 1.0 - 0.05))
    assert doc["offsets_lower"][0] >= plugin
    assert doc["offsets_lower"][0] == pytest.approx(
        float(np.sort(score_lower)[finite_sample_rank(n, 0.05) - 1])
    )


def test_apply_offsets_applies_exact_values():
    rng = np.random.default_rng(31)
    n = 60
    y = rng.normal(size=(n, 1))
    pred = np.stack([y - 0.5, y, y + 0.5], axis=2)
    doc = calibrate_cqr_offsets(y, pred, alpha=0.1, two_sided="one_level")
    out = apply_cqr_offsets_inplace(pred.copy(), doc)
    np.testing.assert_allclose(out[:, :, 0], pred[:, :, 0] - doc["offsets_lower"][0])
    np.testing.assert_allclose(out[:, :, 2], pred[:, :, 2] + doc["offsets_upper"][0])


# ---------------------------------------------------------------------------
# joint rank-score region
# ---------------------------------------------------------------------------
def _joint_scores(rng: np.random.Generator, reps: int, n_cal: int, cov, mean):
    d = np.asarray(cov).shape[0]
    draws = rng.standard_normal(size=(reps, n_cal + 1, d))
    chol = np.linalg.cholesky(np.asarray(cov, dtype=float))
    samples = draws @ chol.T
    flat = samples.reshape(-1, d)
    scores = mahalanobis_scores(flat, mean, cov).reshape(reps, n_cal + 1)
    return scores[:, :n_cal], scores[:, n_cal]


def test_joint_rank_score_region_has_finite_sample_coverage():
    corr = np.array(
        [
            [1.0, 0.9, 0.3],
            [0.9, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ]
    )
    mean = np.array([0.0, 0.0, 0.0])
    n_cal = 199
    rng = np.random.default_rng(41)
    cal_scores, test_scores = _joint_scores(rng, REPS, n_cal, corr, mean)

    rank = finite_sample_rank(n_cal, ALPHA)
    thresholds = np.sort(cal_scores, axis=1)[:, rank - 1]
    coverage = float(np.mean(test_scores <= thresholds))
    floor = finite_sample_coverage_floor(n_cal, ALPHA)
    assert coverage >= 1.0 - ALPHA - 3.0 * _standard_error(floor)
    assert coverage == pytest.approx(floor, abs=3.0 * _standard_error(floor))

    # The public helper reproduces the exact threshold and membership.
    region = joint_rank_score_region(
        cal_scores[0], test_scores[:50], ALPHA, score_kind="rank_score_mahalanobis"
    )
    assert region["threshold"] == pytest.approx(thresholds[0], abs=1e-12)
    assert region["rank"] == rank
    np.testing.assert_array_equal(
        region["test_membership"], test_scores[:50] <= thresholds[0]
    )


def test_joint_region_uses_rank_score_not_bonferroni_box():
    corr = np.full((3, 3), 0.9)
    np.fill_diagonal(corr, 1.0)
    mean = np.zeros(3)
    n_cal = 199
    rng = np.random.default_rng(43)
    cal_scores, test_scores = _joint_scores(rng, REPS, n_cal, corr, mean)
    rank = finite_sample_rank(n_cal, ALPHA)
    rank_score_coverage = float(
        np.mean(test_scores <= np.sort(cal_scores, axis=1)[:, rank - 1])
    )

    # Bonferroni box at alpha/3 per label, from the same exchangeable draws.
    d = 3
    draws = rng.standard_normal(size=(REPS, n_cal + 1, d))
    chol = np.linalg.cholesky(corr)
    samples = draws @ chol.T
    y_cal = samples[:, :n_cal, :]
    y_test = samples[:, n_cal, :]
    lower_cal = np.full_like(y_cal, -1.0)
    upper_cal = np.full_like(y_cal, 1.0)
    lower_test = np.full_like(y_test, -1.0)
    upper_test = np.full_like(y_test, 1.0)
    joint_cover = np.zeros(REPS, dtype=bool)
    for rep in range(REPS):
        box = bonferroni_box_region(
            y_cal[rep],
            lower_cal[rep],
            upper_cal[rep],
            y_test[rep : rep + 1],
            lower_test[rep : rep + 1],
            upper_test[rep : rep + 1],
            alpha=ALPHA,
            n_labels=d,
        )
        joint_cover[rep] = box["joint_membership"][0]
    box_coverage = float(joint_cover.mean())
    assert box_coverage >= 1.0 - ALPHA - 3.0 * _standard_error(1.0 - ALPHA)
    assert box["level"] == pytest.approx(1.0 - ALPHA / d)
    # Rank-score is the primary construction: it is not the Bonferroni box.
    assert rank_score_coverage >= 1.0 - ALPHA - 3.0 * _standard_error(1.0 - ALPHA)
    # At rho=0.9 the rank-score ellipsoid is far smaller than the Bonferroni
    # box at the same nominal level; both sizes are reported, not assumed.
    rank_threshold = float(np.sort(cal_scores[0])[rank - 1])
    ellipsoid_volume = ellipsoid_region_volume(corr, rank_threshold)
    box_volume = float(
        np.prod(
            2.0 + np.asarray(box["offsets_lower"]) + np.asarray(box["offsets_upper"])
        )
    )
    assert 0.0 < ellipsoid_volume < 0.5 * box_volume


def test_mahalanobis_scores_reject_non_pd_covariance():
    with pytest.raises(ValueError, match="not positive definite"):
        mahalanobis_scores(
            np.zeros((3, 2)), np.zeros(2), np.array([[1.0, 1.0], [1.0, 1.0]])
        )
    with pytest.raises(ValueError, match="symmetric"):
        mahalanobis_scores(
            np.zeros((3, 2)), np.zeros(2), np.array([[1.0, 1.0], [0.0, 1.0]])
        )
    scores = mahalanobis_scores(np.zeros((1, 2)), np.zeros(2), np.eye(2))
    assert scores[0] == 0.0
    with pytest.raises(ValueError, match="score_kind"):
        joint_rank_score_region(np.zeros(3), np.zeros(3), 0.1, score_kind="nope")
    log_p = np.array([-1.0, -2.0])
    np.testing.assert_allclose(neg_log_p_scores(log_p), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Mondrian
# ---------------------------------------------------------------------------
def test_mondrian_reports_exact_index_and_achievable_coverage():
    rng = np.random.default_rng(53)
    scores = rng.normal(size=141)
    strata = np.where(np.arange(141) % 2 == 0, "metal_poor", "bulk")
    doc = mondrian_conformal(scores, strata, alpha=0.1)
    for stratum, count in doc["n_cal"].items():
        rank = finite_sample_rank(count, 0.1)
        assert doc["order_stat_index"][stratum] == rank
        assert doc["achievable_coverage"][stratum] == pytest.approx(rank / (count + 1))
        assert doc["achievable_coverage"][stratum] >= 0.9
    # n = 141, alpha = 0.1 is the plan's worked example: ceil(142*0.9) = 128.
    assert finite_sample_rank(141, 0.1) == 128
    assert finite_sample_coverage_floor(141, 0.1) == pytest.approx(128 / 142.0)

    membership = mondrian_membership(
        np.array([0.0, 10.0]), np.array(["metal_poor", "bulk"]), doc["thresholds"]
    )
    np.testing.assert_array_equal(membership, [True, False])


def test_mondrian_stratum_too_small_raises():
    scores = np.arange(40, dtype=float)
    strata = np.array(["big"] * 36 + ["tiny"] * 4)
    with pytest.raises(InsufficientCalibrationError, match="tiny"):
        mondrian_conformal(scores, strata, alpha=0.05)


# ---------------------------------------------------------------------------
# region size
# ---------------------------------------------------------------------------
def test_region_volume_helpers():
    # 1-D ellipsoid: V_1 * sqrt(4) * sqrt(2.25) = 2 * 2 * 1.5 = 6.
    assert ellipsoid_region_volume(np.array([[4.0]]), 2.25) == pytest.approx(6.0)
    # 2-D unit covariance: pi * sqrt(det I) * q.
    assert ellipsoid_region_volume(np.eye(2), 2.0) == pytest.approx(2.0 * math.pi)
    np.testing.assert_allclose(
        box_region_volume(np.array([[0.0, 0.0]]), np.array([[2.0, 3.0]])), [6.0]
    )
    assert region_volume(
        kind="box", lower=[[0.0, 0.0]], upper=[[2.0, 3.0]]
    ) == pytest.approx(6.0)
    assert region_volume(
        kind="ellipsoid", cov=np.array([[4.0]]), threshold=2.25
    ) == pytest.approx(6.0)
    assert region_volume(kind="box") is None
    assert levelset_method_for("ellipsoid") == "analytic_ellipsoid"
    assert levelset_method_for("box") == "box"
    with pytest.raises(ValueError, match="positive definite"):
        ellipsoid_region_volume(np.array([[1.0, 2.0], [2.0, 1.0]]), 1.0)
    with pytest.raises(ValueError, match="below lower bounds"):
        box_region_volume(np.array([[1.0]]), np.array([[0.0]]))
