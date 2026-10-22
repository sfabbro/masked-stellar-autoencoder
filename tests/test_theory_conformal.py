"""R1: Mondrian stage/tier strata and theory-error conformal calibration.

What is asserted, and with which tolerance
------------------------------------------
* **Mondrian coverage.**  With ``REPS = 1000`` exchangeable replications the
  Monte-Carlo standard error of a 0.90 coverage estimate is
  ``sqrt(0.9 * 0.1 / 1000) = 0.0095``; per-stratum coverage must be at least
  ``1 - alpha`` within three standard errors and must match the exact discrete
  floor ``rank / (n_cal + 1)``.  Each replication draws a fresh calibration set
  *and* a fresh test score, so the check is the split-conformal guarantee, not a
  conditional-on-calibration number.
* **Theory-scaled coverage.**  A single calibration set with a huge test set
  measures conditional coverage given that calibration draw and can sit a couple
  of standard errors below nominal; the coverage test therefore averages over
  ``REPS = 200`` independent calibration/test draws and uses the standard error
  of that mean.
* **Pooling.**  A stratum below ``min_count`` is pooled with its pre-registered
  adjacent stage, the pooled group gets one exact threshold, and a group that
  still cannot support the level raises (default) or is flagged.  Pooling is
  reported, never silent.
* **No leakage.**  ``fit`` has no test argument for either calibrator; changing
  test rows changes coverage but never a threshold or ``gamma``.
* **Theory envelope.**  The envelope contains every tested model by
  construction, yet is reported separately from conformal coverage and is never
  called coverage.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from masked_stellar_autoencoder.reporting import calibration, io, schema
from masked_stellar_autoencoder.reporting.strata import (
    STAGE_BOUNDARIES,
    STAGE_DESCRIPTIONS,
    STAGE_ORDER,
    TIER_ORDER,
    MondrianCalibrator,
    compose_strata,
    stage_stratum,
    stratum_metrics,
    tier_stratum,
)
from masked_stellar_autoencoder.reporting.theory import (
    CALIBRATION_KIND,
    THEORY_MODEL_ORDER,
    TheoryScaledScore,
    theory_dispersion,
    theory_envelope,
)

ALPHA = 0.1
REPS = 1000


def _mc_se(p: float, reps: int = REPS) -> float:
    return math.sqrt(p * (1.0 - p) / reps)


def _theory_dgp(rng: np.random.Generator, n: int, true_gamma: float = 1.0):
    """Heteroscedastic exchangeable DGP: ``y = mu + N(0, sigma_exp + gamma sigma_theo)``."""
    sigma_exp = rng.uniform(0.05, 0.20, n)
    sigma_theo = rng.uniform(0.0, 0.50, n)
    median = rng.normal(0.0, 1.0, n)
    y = median + rng.normal(size=n) * (sigma_exp + true_gamma * sigma_theo)
    return y, median, sigma_exp, sigma_theo


# ---------------------------------------------------------------------------
# pre-registered strata
# ---------------------------------------------------------------------------
def test_stage_boundaries_are_pre_registered_and_exhaustive():
    # Boundary rows are chosen to pin every inclusive/exclusive edge.
    teff = np.array(
        [5800.0, 6000.0, 4800.0, 4100.0, 4300.0, 6000.0, 5500.0, 4800.0, 4600.0, 5400.0]
    )
    logg = np.array([4.5, 3.8, 3.0, 3.0, 2.0, 3.0, 4.0, 3.5, 2.5, 3.5])
    labels = stage_stratum(teff, logg)
    assert list(labels) == [
        "dwarf_ms",  # logg 4.5
        "turnoff_sgb",  # 3.5 <= logg < 4.0
        "rc",  # RC Teff box, 2.5 <= logg < 3.5
        "rgb",  # cool giant inside the logg giant band
        "agb_other",  # logg < 2.5
        "agb_other",  # hot low-gravity star
        "dwarf_ms",  # logg == 4.0 is inclusive dwarf
        "turnoff_sgb",  # logg == 3.5 is inclusive turn-off/SGB
        "rc",  # both RC bounds inclusive
        "turnoff_sgb",  # logg == 3.5 wins before the RC Teff box
    ]
    assert set(labels) == set(STAGE_ORDER)
    assert set(STAGE_BOUNDARIES) == set(STAGE_ORDER)
    assert set(STAGE_DESCRIPTIONS) == set(STAGE_ORDER)
    # Only Teff/logg enter the assignment: no label/residual argument exists.
    assert set(inspect.signature(stage_stratum).parameters) == {"teff", "logg"}

    with pytest.raises(ValueError, match="finite"):
        stage_stratum([5000.0, np.nan], [3.0, 3.0])
    with pytest.raises(ValueError, match="1-D"):
        stage_stratum(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="equal length"):
        stage_stratum([5000.0], [3.0, 3.0])
    with pytest.raises(ValueError, match="at least one"):
        stage_stratum([], [])


def test_tier_strata_from_explicit_flags_and_composition():
    xp_full = np.array([1, 0, 0, 0], dtype=bool)
    xp_partial = np.array([0, 1, 0, 0], dtype=bool)
    no_xp = np.array([0, 0, 1, 0], dtype=bool)
    gaia_only = np.array([0, 0, 0, 1], dtype=bool)
    labels = tier_stratum(
        xp_full=xp_full,
        xp_partial=xp_partial,
        no_xp=no_xp,
        gaia_only=gaia_only,
    )
    assert list(labels) == list(TIER_ORDER)

    with pytest.raises(ValueError, match="exactly one true tier flag"):
        tier_stratum(xp_full=np.array([1, 1]), no_xp=np.array([0, 1]))
    with pytest.raises(ValueError, match="exactly one true tier flag"):
        tier_stratum(
            xp_full=np.array([0, 0]),
            xp_partial=np.array([0, 1]),
            no_xp=np.array([0, 0]),
            gaia_only=np.array([0, 0]),
        )
    with pytest.raises(ValueError, match="at least one tier flag"):
        tier_stratum()
    with pytest.raises(ValueError, match="boolean"):
        tier_stratum(xp_full=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="share one length"):
        tier_stratum(xp_full=np.array([1, 0]), no_xp=np.array([0, 0, 1]))

    composed = compose_strata(
        ["rgb", "rc"], ["xp_partial", "no_xp"], ["GALAH", "APOGEE"]
    )
    assert list(composed) == [
        "rgb|xp_partial|survey:GALAH",
        "rc|no_xp|survey:APOGEE",
    ]
    np.testing.assert_array_equal(compose_strata(labels), np.asarray(labels, dtype=str))
    with pytest.raises(ValueError, match="separator"):
        compose_strata(["a|b"])
    with pytest.raises(ValueError, match="one label per stage row"):
        compose_strata(["rgb", "rc"], ["xp_full"])


# ---------------------------------------------------------------------------
# Mondrian coverage and pooling
# ---------------------------------------------------------------------------
def test_mondrian_exchangeable_coverage_per_stratum_over_1000_reps():
    labels = ("dwarf_ms", "rgb", "agb_other")
    scales = np.array([1.0, 2.0, 0.5])
    n_cal = 149  # ceil(150 * 0.9) / 150 == 0.9 exactly
    rng = np.random.default_rng(20261022)
    covers = np.zeros((REPS, len(labels)), dtype=bool)
    floors = None
    for rep in range(REPS):
        scores = np.concatenate(
            [rng.exponential(scales[index], size=n_cal) for index in range(len(labels))]
        )
        strata = np.concatenate([[label] * n_cal for label in labels])
        calibrator = MondrianCalibrator(alpha=ALPHA, min_count=50)
        document = calibrator.fit(scores, strata)
        assert document["finite_sample_exact"] is True
        assert document["pooled"] is False
        assert document["pooling_notes"] == []
        floors = document["achievable_coverage"]
        covers[rep] = calibrator.apply(rng.exponential(scales), np.asarray(labels))
    for index, label in enumerate(labels):
        coverage = float(covers[:, index].mean())
        exact_rank = calibration.finite_sample_rank(n_cal, ALPHA)
        assert floors[label] >= 1.0 - ALPHA
        assert floors[label] == pytest.approx(exact_rank / (n_cal + 1.0))
        assert coverage >= 1.0 - ALPHA - 3.0 * _mc_se(1.0 - ALPHA), (
            f"{label}: coverage {coverage:.4f} below nominal within MC error"
        )
        assert coverage == pytest.approx(floors[label], abs=3.0 * _mc_se(floors[label]))
    # Scale differences are actually resolved by stratum-specific thresholds.
    assert len(set(document["thresholds"].values())) == len(labels)


def test_mondrian_pools_adjacent_strata_below_min_count():
    counts = {
        "dwarf_ms": 120,
        "turnoff_sgb": 120,
        "rgb": 60,
        "rc": 30,
        "agb_other": 120,
    }
    labels = np.concatenate([[label] * count for label, count in counts.items()])
    rng = np.random.default_rng(7)
    scores = rng.normal(size=labels.size)
    calibrator = MondrianCalibrator(alpha=ALPHA, min_count=100)
    document = calibrator.fit(scores, labels)

    assert document["pooled"] is True
    assert document["pooling_notes"], "pooling of a small stratum must be reported"
    note = document["pooling_notes"][0]
    assert note["reason"] == "adjacent_pool_below_min_count"
    assert "rgb" in note["members"] and "rc" in note["members"]
    assert note["n"] >= 100
    for group_size in document["group_n_cal"].values():
        assert group_size >= 100
    # The two originally small strata share one exact pooled threshold.
    assert document["groups"]["rgb"] == document["groups"]["rc"]
    assert document["groups"]["rgb"] == ["rgb", "rc", "agb_other"]
    assert document["thresholds"]["rgb"] == document["thresholds"]["rc"]
    members = document["groups"]["rgb"]
    mask = np.isin(labels, members)
    assert document["n_cal"]["rgb"] == int(mask.sum())
    assert document["thresholds"]["rgb"] == pytest.approx(
        calibration.exact_quantile(scores[mask], ALPHA)
    )
    assert document["order_stat_index"]["rgb"] == calibration.finite_sample_rank(
        int(mask.sum()), ALPHA
    )
    assert document["achievable_coverage"]["rgb"] >= 1.0 - ALPHA
    # A stratum that already meets the minimum is not pooled.
    assert document["groups"]["dwarf_ms"] == ["dwarf_ms"]
    assert document["thresholds"]["dwarf_ms"] == pytest.approx(
        calibration.exact_quantile(scores[labels == "dwarf_ms"], ALPHA)
    )


def test_mondrian_raises_or_flags_impossible_calibration():
    labels = np.array(["a"] * 30 + ["b"] * 10)
    scores = np.random.default_rng(11).normal(size=40)
    with pytest.raises(
        calibration.InsufficientCalibrationError, match="cannot be calibrated"
    ):
        MondrianCalibrator(min_count=100).fit(scores, labels)

    flagged = MondrianCalibrator(min_count=100, on_insufficient="flag")
    document = flagged.fit(scores, labels)
    assert document["uncalibrated"] == ["a", "b"]
    assert document["finite_sample_exact"] is False
    assert document["null_reason"] == schema.NullReason.INSUFFICIENT_N.value
    assert document["thresholds"]["a"] is None
    assert document["achievable_coverage"]["a"] is None
    assert set(document["group_reasons"].values()) == {"n_below_min_count"}
    with pytest.raises(
        calibration.InsufficientCalibrationError, match="not calibrated"
    ):
        flagged.apply([0.0], ["a"])

    # Even above min_count, alpha=0.01 needs n >= 99; n=50 has rank 51 > 50.
    with pytest.raises(
        calibration.InsufficientCalibrationError, match="cannot be calibrated"
    ):
        MondrianCalibrator(alpha=0.01, min_count=20).fit(
            np.random.default_rng(13).normal(size=50), np.array(["a"] * 50)
        )
    rank_flagged = MondrianCalibrator(
        alpha=0.01, min_count=20, on_insufficient="flag"
    ).fit(np.random.default_rng(13).normal(size=50), np.array(["a"] * 50))
    assert rank_flagged["group_reasons"]["a"] == "rank_exceeds_n"
    assert rank_flagged["null_reason"] == schema.NullReason.INSUFFICIENT_N.value

    with pytest.raises(ValueError, match="pooling"):
        MondrianCalibrator(pooling="nearest")
    with pytest.raises(ValueError, match="on_insufficient"):
        MondrianCalibrator(on_insufficient="ignore")


def test_stratum_metrics_reports_width_coverage_and_pooling():
    y = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    lo = np.full(6, -1.0)
    hi = np.full(6, 1.0)
    strata = np.array(["rgb"] * 3 + ["rc"] * 3)
    pooled = {"rgb": ["rgb", "rc"], "rc": ["rgb", "rc"]}
    rows = stratum_metrics(y, lo, hi, strata, pooled=pooled)
    by_stratum = {row["stratum"]: row for row in rows}
    assert by_stratum["rgb"]["n"] == 3
    assert by_stratum["rgb"]["coverage"] == pytest.approx(2.0 / 3.0)
    assert by_stratum["rgb"]["width"] == pytest.approx(2.0)
    assert by_stratum["rgb"]["median_width"] == pytest.approx(2.0)
    assert by_stratum["rgb"]["descriptive_only"] is True
    assert by_stratum["rgb"]["pooled_strata"] == ["rgb", "rc"]
    assert "pooled_group=rgb,rc" in by_stratum["rgb"]["notes"]
    assert by_stratum["rc"]["coverage"] == pytest.approx(1.0)
    assert by_stratum["rgb"]["null_reason"] == ""
    with pytest.raises(ValueError, match="one label per row"):
        stratum_metrics(y, lo, hi, np.array(["rgb"]))
    with pytest.raises(ValueError, match="equal length"):
        stratum_metrics(y, lo[:-1], hi, strata)


# ---------------------------------------------------------------------------
# theory dispersion and envelope
# ---------------------------------------------------------------------------
def test_theory_dispersion_and_envelope_contain_every_model():
    rng = np.random.default_rng(17)
    base = rng.normal(size=(1, 37, 2))
    models = base + 0.3 * rng.normal(size=(4, 1, 2))
    dispersion = theory_dispersion(models)
    np.testing.assert_allclose(dispersion["std"], np.std(models, axis=0))
    np.testing.assert_allclose(
        dispersion["range"], models.max(axis=0) - models.min(axis=0)
    )
    np.testing.assert_allclose(dispersion["median"], np.median(models, axis=0))
    assert dispersion["n_models"] == 4
    assert dispersion["n_rows"] == 37
    assert dispersion["model_families"] == THEORY_MODEL_ORDER

    envelope = theory_envelope(models, level=0.95)
    np.testing.assert_allclose(envelope["lower"], models.min(axis=0))
    np.testing.assert_allclose(envelope["upper"], models.max(axis=0))
    np.testing.assert_allclose(envelope["width"], dispersion["range"])
    assert np.all(models >= envelope["lower"])
    assert np.all(models <= envelope["upper"])
    assert envelope["contains_all_models"] is True
    assert envelope["kind"] == "theory_sensitivity_envelope"
    # ``level`` only narrows the central summary, never the full envelope.
    assert np.all(envelope["level_width"] <= envelope["width"] + 1e-12)
    full = theory_envelope(models, level=1.0)
    np.testing.assert_allclose(full["level_lower"], envelope["lower"])
    np.testing.assert_allclose(full["level_upper"], envelope["upper"])
    explicit = theory_envelope(
        models, model_families=("MIST", "PARSEC", "GARSTEC", "BASTI")
    )
    assert explicit["model_families"] == ("MIST", "PARSEC", "GARSTEC", "BASTI")

    with pytest.raises(ValueError, match="shape"):
        theory_dispersion(models[0, 0])
    with pytest.raises(ValueError, match="at least 2 theory models"):
        theory_dispersion(models[:1])
    bad = models.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        theory_dispersion(bad)
    with pytest.raises(ValueError, match="model_families has"):
        theory_dispersion(models, model_families=("MIST", "PARSEC"))
    with pytest.raises(ValueError, match="unique"):
        theory_dispersion(models, model_families=("MIST", "MIST", "PARSEC", "BASTI"))
    with pytest.raises(ValueError, match="level"):
        theory_envelope(models, level=0.0)


# ---------------------------------------------------------------------------
# theory-scaled conformal score
# ---------------------------------------------------------------------------
def test_theory_scaled_interval_widens_with_sigma_theo_and_denominator_is_safe():
    rng = np.random.default_rng(20261022)
    y, median, sigma_exp, sigma_theo = _theory_dgp(rng, 600)
    score = TheoryScaledScore(alpha=ALPHA, fixed_gamma=1.0)
    score.fit(y, median, sigma_exp, sigma_theo)
    assert score.gamma == 1.0

    lo, hi = score.interval(np.zeros(2), np.array([0.1, 0.1]), np.array([0.0, 0.5]))
    widths = hi - lo
    assert widths[1] > widths[0]
    # Same quoted error: the width ratio is exactly the denominator ratio.
    assert widths[1] / widths[0] == pytest.approx((0.1 + 1.0 * 0.5) / 0.1)

    narrow = score.evaluate(y, median, sigma_exp, sigma_theo * 0.1)
    wide = score.evaluate(y, median, sigma_exp, sigma_theo * 5.0)
    assert wide["mean_width"] > narrow["mean_width"]

    # A zero quoted error cannot produce inf/NaN: the denominator is floored and
    # the number of floored rows is reported.
    floored = TheoryScaledScore(alpha=ALPHA, fixed_gamma=0.0)
    floored.fit(y, median, sigma_exp, sigma_theo)
    denominator, n_floored = floored.denominator(np.zeros(4), np.zeros(4))
    np.testing.assert_allclose(denominator, floored.denominator_floor)
    assert n_floored == 4
    values = floored.score(
        [0.0, 1.0, 2.0, 3.0],
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        gamma=0.0,
    )
    assert np.isfinite(values).all()
    assert floored.document_["n_denominator_floored"] == int(
        np.sum(sigma_exp < floored.denominator_floor)
    )
    with pytest.raises(ValueError, match="sigma_exp must be non-negative"):
        floored.denominator([-0.1], [0.0])
    with pytest.raises(ValueError, match="non-negative"):
        floored.denominator([0.1], [-0.1])
    with pytest.raises(ValueError, match="denominator_floor"):
        TheoryScaledScore(denominator_floor=0.0)


def test_theory_scaled_empty_evaluation_is_an_explicit_null():
    rng = np.random.default_rng(5)
    y, median, sigma_exp, sigma_theo = _theory_dgp(rng, 120)
    score = TheoryScaledScore(alpha=ALPHA, fixed_gamma=1.0)
    score.fit(y, median, sigma_exp, sigma_theo)
    empty = score.evaluate([], [], [], [])
    assert empty["coverage"] is None
    assert empty["mean_width"] is None
    assert empty["n"] == 0
    assert empty["descriptive_only"] is True
    assert empty["null_reason"] == schema.NullReason.SPLIT_EMPTY.value


def test_theory_scaled_gamma_is_cross_fitted_on_calibration_only():
    rng = np.random.default_rng(20261022)
    y, median, sigma_exp, sigma_theo = _theory_dgp(rng, 1200)
    score = TheoryScaledScore(alpha=ALPHA, n_folds=5, fold_seed=0)
    document = score.fit(y, median, sigma_exp, sigma_theo)

    assert document["used_test_rows"] is False
    assert document["n_test_rows_used"] == 0
    assert not any(
        "test" in name for name in inspect.signature(TheoryScaledScore.fit).parameters
    )
    assert document["gamma"] > 0.0, "a real sigma_theo must be used by gamma"
    assert document["gamma"] in score.gamma_grid
    assert document["gamma_selection"] == "kfold_interval_score"
    assert len(document["gamma_sensitivity"]) == len(score.gamma_grid)
    for record in document["gamma_sensitivity"]:
        assert record["gamma"] in score.gamma_grid
        assert 0.75 <= record["oof_coverage"] <= 1.0
        assert record["n_oof"] == len(y)
        assert record["n_folds"] == 5
        assert math.isfinite(record["oof_interval_score"])

    expected_scores = np.abs(y - median) / (sigma_exp + document["gamma"] * sigma_theo)
    np.testing.assert_allclose(document["oof_scores"], expected_scores)
    assert document["threshold"] == pytest.approx(
        calibration.exact_quantile(expected_scores, ALPHA)
    )
    assert document["order_stat_index"] == calibration.finite_sample_rank(len(y), ALPHA)
    assert document["achievable_coverage"] == pytest.approx(
        document["order_stat_index"] / (len(y) + 1.0)
    )
    assert document["oof_coverage"] == pytest.approx(
        float(np.mean(document["oof_membership"]))
    )

    # Deterministic refit: same calibration rows -> same gamma and threshold.
    again = TheoryScaledScore(alpha=ALPHA, n_folds=5, fold_seed=0)
    again_document = again.fit(y, median, sigma_exp, sigma_theo)
    assert again_document["gamma"] == document["gamma"]
    assert again_document["threshold"] == document["threshold"]
    np.testing.assert_allclose(again_document["oof_scores"], document["oof_scores"])

    # Test rows only move the empirical evaluation, never the fitted state.
    y_a, median_a, exp_a, theo_a = _theory_dgp(rng, 500)
    y_b, median_b, exp_b, theo_b = _theory_dgp(rng, 500)
    first = score.evaluate(y_a, median_a, exp_a, theo_a)
    second = score.evaluate(y_b, median_b, exp_b, theo_b)
    assert first["coverage"] != second["coverage"]
    assert score.gamma == document["gamma"]
    assert score.threshold == document["threshold"]


def test_theory_scaled_empirical_coverage_over_replications():
    reps = 200
    n_cal, n_test = 500, 300
    rng = np.random.default_rng(31415)
    covers = np.empty(reps)
    for rep in range(reps):
        y, median, sigma_exp, sigma_theo = _theory_dgp(rng, n_cal)
        score = TheoryScaledScore(
            alpha=ALPHA,
            n_folds=4,
            fold_seed=rep,
            gamma_grid=(0.0, 0.5, 1.0, 2.0),
        )
        score.fit(y, median, sigma_exp, sigma_theo)
        y_test, median_test, exp_test, theo_test = _theory_dgp(rng, n_test)
        covers[rep] = score.evaluate(y_test, median_test, exp_test, theo_test)[
            "coverage"
        ]
    mean_coverage = float(covers.mean())
    standard_error = float(covers.std(ddof=1) / math.sqrt(reps))
    assert mean_coverage >= 1.0 - ALPHA - 3.0 * standard_error, (
        f"mean coverage {mean_coverage:.4f} < {1.0 - ALPHA} within MC error"
    )
    assert abs(mean_coverage - (1.0 - ALPHA)) <= 3.0 * standard_error + 0.01
    assert 0.0 < mean_coverage < 1.0


def test_theory_scaled_reports_envelope_apart_and_never_as_coverage():
    rng = np.random.default_rng(2718)
    y, median, sigma_exp, sigma_theo = _theory_dgp(rng, 800)
    score = TheoryScaledScore(alpha=ALPHA, n_folds=5, fold_seed=3)
    score.fit(y, median, sigma_exp, sigma_theo)

    n_test = 600
    y_test, median_test, exp_test, theo_test = _theory_dgp(rng, n_test)
    models = median_test[None, :] + rng.normal(size=(4, n_test)) * theo_test[None, :]
    envelope = theory_envelope(models, level=0.95)
    lo, hi = envelope["lower"], envelope["upper"]
    strata = np.asarray(STAGE_ORDER)[np.arange(n_test) % len(STAGE_ORDER)]

    evaluation = score.evaluate(
        y_test,
        median_test,
        exp_test,
        theo_test,
        envelope_low=lo,
        envelope_high=hi,
        strata=strata,
    )
    assert evaluation["coverage_kind"] == "marginal_conformal"
    assert evaluation["envelope_claim"] == "sensitivity_only"
    assert evaluation["coverage"] == pytest.approx(
        float(
            np.mean(
                np.abs(y_test - median_test) / (exp_test + score.gamma * theo_test)
                <= score.threshold
            )
        )
    )
    assert evaluation["theory_envelope_width"] == pytest.approx(float(np.mean(hi - lo)))
    assert evaluation["theory_envelope_low_mean"] == pytest.approx(float(np.mean(lo)))
    assert evaluation["theory_envelope_high_mean"] == pytest.approx(float(np.mean(hi)))
    assert evaluation["envelope_contains_y_fraction"] == pytest.approx(
        float(np.mean((y_test >= lo) & (y_test <= hi)))
    )
    # The two widths are separate quantities; the envelope is not a CI.
    assert evaluation["mean_width"] > 0.0
    assert evaluation["theory_envelope_width"] > 0.0
    assert evaluation["per_stratum_coverage_claim"] == "descriptive"
    assert len(evaluation["per_stratum"]) == len(STAGE_ORDER)
    for row in evaluation["per_stratum"]:
        assert {"n", "coverage", "width", "descriptive_only", "pooled_strata"} <= set(
            row
        )

    fields = score.metrics_fields(evaluation)
    assert fields["calibration_kind"] == CALIBRATION_KIND
    assert fields["gamma"] == pytest.approx(score.gamma)
    assert fields["theory_sigma"] == pytest.approx(float(np.mean(theo_test)))
    assert fields["theory_envelope_low"] == pytest.approx(float(np.mean(lo)))
    assert fields["theory_envelope_high"] == pytest.approx(float(np.mean(hi)))

    with pytest.raises(ValueError, match="supplied together"):
        score.evaluate(y_test, median_test, exp_test, theo_test, envelope_low=lo)


def test_no_leakage_from_test_rows_into_mondrian_or_theory():
    rng = np.random.default_rng(99)
    labels = np.array(["dwarf_ms"] * 60 + ["rgb"] * 60 + ["agb_other"] * 60)
    scores = rng.normal(size=labels.size)
    calibrator = MondrianCalibrator(alpha=ALPHA, min_count=50)
    document = calibrator.fit(scores, labels)
    thresholds = dict(document["thresholds"])

    # Altering the test rows cannot alter a threshold: fit never sees them.
    test_a = rng.normal(size=labels.size) + 0.5
    test_b = rng.normal(size=labels.size) - 3.0
    membership_a = calibrator.apply(test_a, labels)
    membership_b = calibrator.apply(test_b, labels)
    assert not np.array_equal(membership_a, membership_b)
    assert calibrator.thresholds == thresholds
    refit = MondrianCalibrator(alpha=ALPHA, min_count=50).fit(scores, labels)
    assert refit["thresholds"] == thresholds
    assert refit["groups"] == document["groups"]

    y, median, sigma_exp, sigma_theo = _theory_dgp(rng, 400)
    score = TheoryScaledScore(alpha=ALPHA, n_folds=4, fold_seed=0)
    fitted = score.fit(y, median, sigma_exp, sigma_theo)
    y_a, median_a, exp_a, theo_a = _theory_dgp(rng, 200)
    y_b, median_b, exp_b, theo_b = _theory_dgp(rng, 200)
    score.evaluate(y_a, median_a, exp_a, theo_a)
    assert score.gamma == fitted["gamma"]
    assert score.threshold == fitted["threshold"]
    score.evaluate(y_b, median_b, exp_b, theo_b)
    assert score.gamma == fitted["gamma"]
    assert score.threshold == fitted["threshold"]
    refit_score = TheoryScaledScore(alpha=ALPHA, n_folds=4, fold_seed=0)
    refitted = refit_score.fit(y, median, sigma_exp, sigma_theo)
    assert refitted["gamma"] == fitted["gamma"]
    assert refitted["threshold"] == fitted["threshold"]


# ---------------------------------------------------------------------------
# schema compatibility
# ---------------------------------------------------------------------------
def _metrics_row_with_theory(**overrides):
    values = dict(
        run_id="run-r1",
        model_id="mvgaussian",
        label="teff",
        unit="K",
        stratum="rgb",
        n=420,
        n_eff=420.0,
        mae=70.0,
        rmse=92.0,
        bias=-1.0,
        nmad=66.0,
        r2=0.97,
        atten_slope=0.98,
        outlier_frac=0.005,
        tail_mae=190.0,
        cov_68=0.69,
        cov_90=0.90,
        cov_95=0.95,
        cov_nominal=0.90,
        width_90=220.0,
        interval_score_90=260.0,
        calib_slope=0.02,
        calibration_kind=CALIBRATION_KIND,
        pooled_strata="rgb,rc",
        theory_sigma=35.0,
        gamma=1.0,
        theory_envelope_low=-80.0,
        theory_envelope_high=75.0,
        evidence_status="producer_only",
        bootstrap_kind="star",
        bootstrap_reps=1000,
        multiplicity="bonferroni_k=16",
        null_reason="not_computed",
        notes="R1 theory/Mondrian row",
    )
    values.update(overrides)
    return io.build_row("metrics", values)


def test_schema_round_trip_with_theory_and_mondrian_fields(tmp_path):
    row = _metrics_row_with_theory()
    assert row["calibration_kind"] == "theory_scaled_split_rank"
    assert row["gamma"] == pytest.approx(1.0)
    path = io.write_csv(tmp_path / "metrics.csv", "metrics", [row])
    text = path.read_text()
    assert "nan" not in text.lower()
    assert "inf" not in text.replace("infer", "").lower()
    assert io.read_csv(path, "metrics") == [row]

    assert schema.SCHEMA_VERSION == "1.1.0"
    assert schema.CONTRACT_ID == "C6"
    for name in (
        "calibration_kind",
        "pooled_strata",
        "theory_sigma",
        "gamma",
        "theory_envelope_low",
        "theory_envelope_high",
    ):
        assert name in schema.ROW_SCHEMAS["metrics"]
    document = schema.schema_document()
    fields = {
        field["name"]: field for field in document["tables"]["metrics"]["columns"]
    }
    assert set(fields["calibration_kind"]["enum"]) == set(schema.CALIBRATION_KINDS)
    assert fields["gamma"]["valid_range"] == [0.0, None]
    assert set(document["tables"]) == {"metrics", "joint", "distance", "age"}
    for table in ("joint", "distance", "age"):
        names = {field["name"] for field in document["tables"][table]["columns"]}
        assert "calibration_kind" in names
    for table in ("distance", "age"):
        names = {field["name"] for field in document["tables"][table]["columns"]}
        assert {
            "theory_sigma",
            "gamma",
            "theory_envelope_low",
            "theory_envelope_high",
        } <= names

    # 1.0.0-style producers keep validating: the new cells are simply null and
    # the row's existing null_reason covers them.
    legacy = io.validate_row(
        "metrics",
        {
            "run_id": "run-old",
            "model_id": "mvgaussian",
            "label": "fe_h",
            "unit": "dex",
            "stratum": "bulk",
            "n": 100,
            "evidence_status": "producer_only",
            "bootstrap_kind": "star",
            "multiplicity": "none",
            "null_reason": "not_computed",
        },
    )
    assert legacy["calibration_kind"] is None
    assert legacy["gamma"] is None
    assert legacy["theory_envelope_low"] is None

    with pytest.raises(io.ContractError, match="not in allowed values"):
        _metrics_row_with_theory(calibration_kind="posthoc")
    with pytest.raises(io.ContractError, match="valid_range"):
        _metrics_row_with_theory(gamma=-1.0)

    # Distance and age tables carry the optional theory envelope as well.
    distance_row = io.build_row(
        "distance",
        dict(
            run_id="run-r1",
            kind="kpc",
            stratum="bulk",
            n=100,
            ci_kind="predictive",
            ref_type="HIP",
            calibration_kind=CALIBRATION_KIND,
            theory_sigma=0.3,
            gamma=1.0,
            theory_envelope_low=1.8,
            theory_envelope_high=2.4,
            evidence_status="producer_only",
            null_reason="not_computed",
        ),
    )
    age_row = io.build_row(
        "age",
        dict(
            run_id="run-r1",
            stratum="rgb",
            n=50,
            calibration_kind=CALIBRATION_KIND,
            theory_sigma=1.5,
            gamma=0.5,
            theory_envelope_low=8.0,
            theory_envelope_high=13.0,
            evidence_status="producer_only",
            null_reason="not_computed",
        ),
    )
    distance_path = io.write_csv(tmp_path / "distance.csv", "distance", [distance_row])
    age_path = io.write_csv(tmp_path / "age.csv", "age", [age_row])
    assert io.read_csv(distance_path, "distance") == [distance_row]
    assert io.read_csv(age_path, "age") == [age_row]
