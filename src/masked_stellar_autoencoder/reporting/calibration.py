"""Exact split conformal, joint rank-score regions and Mondrian calibration.

The finite-sample guarantee of split conformal comes from *one order statistic*,
never from a plug-in sample quantile:

    q_hat = s_( ceil((n+1)(1-alpha)) )        (1-based index)

For exchangeable calibration/test scores this gives

    P( s_test <= q_hat ) >= 1 - alpha

with equality only when ``(n+1)(1-alpha)`` is an integer.  ``numpy.quantile``
with linear interpolation lands *below* that order statistic (e.g. at n=2046,
alpha=0.1 the offset is short by ~6.0e-4 in scaled label space) and silently
under-covers; :func:`exact_quantile` raises instead of degrading.

Two valid two-sided constructions are provided and must be declared:

* ``two_sided="per_tail"`` -- two one-sided intervals at level ``alpha/2``
  each (asymmetric offsets; union bound gives >= ``1 - alpha``);
* ``two_sided="one_level"`` -- the single score ``max(q_lo - y, y - q_hi)`` at
  level ``alpha`` and one shared offset (CQR; exact marginal coverage of the
  symmetrically inflated box).

The joint region is *primary* as a rank-score region (Mahalanobis radius or
``-log p_hat``), which uses the dependence and has finite-sample marginal joint
coverage; the Bonferroni box is the *secondary*, conservative construction.
"""

from __future__ import annotations

import math

import numpy as np

from .schema import NullReason

__all__ = [
    "InsufficientCalibrationError",
    "exact_quantile",
    "finite_sample_rank",
    "finite_sample_coverage_floor",
    "split_conformal_interval",
    "calibrate_mondrian_intervals",
    "mahalanobis_scores",
    "neg_log_p_scores",
    "joint_rank_score_region",
    "bonferroni_box_region",
    "mondrian_conformal",
    "mondrian_membership",
    "box_region_volume",
    "ellipsoid_region_volume",
    "region_volume",
    "levelset_method_for",
    "unbounded_reason",
]


class InsufficientCalibrationError(ValueError):
    """Raised when the requested finite-sample rank exceeds the calibration size."""


def _validate_alpha(alpha: float, *, name: str = "alpha") -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {alpha!r}")
    return alpha


def _as_vector(values, name: str, *, min_len: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite (a non-finite score is a failed arm)")
    return array


def finite_sample_rank(n: int, alpha: float) -> int:
    """The exact 1-based order-statistic index ``ceil((n+1)(1-alpha))``.

    This is the *only* admissible conformal threshold index.  It does not clamp:
    the caller must check the returned index against ``n`` (see
    :func:`exact_quantile` and :class:`InsufficientCalibrationError`).
    """
    n = int(n)
    alpha = _validate_alpha(alpha)
    if n < 1:
        raise ValueError("n must be a positive calibration size")
    return int(math.ceil((n + 1) * (1.0 - alpha)))


def finite_sample_coverage_floor(n: int, alpha: float) -> float:
    """Achievable coverage ``rank/(n+1)`` -- always ``>= 1 - alpha``.

    At small ``n`` the achievable coverage is discrete: report this number and
    the integer index, not a nominal level the data cannot express.
    """
    return finite_sample_rank(n, alpha) / float(int(n) + 1)


def exact_quantile(scores, alpha: float) -> float:
    """Split-conformal threshold: the exact finite-sample order statistic.

    Raises :class:`InsufficientCalibrationError` when
    ``ceil((n+1)(1-alpha)) > n``: there is no data-supported threshold, and
    returning a plug-in quantile or a zero offset would be a silent
    under-coverage bug.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise InsufficientCalibrationError("no conformity scores: cannot calibrate")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite (a non-finite score is a failed arm)")
    alpha = _validate_alpha(alpha)
    rank = finite_sample_rank(values.size, alpha)
    if rank > values.size:
        raise InsufficientCalibrationError(
            f"finite-sample rank {rank} exceeds n_cal={values.size} at alpha={alpha}: "
            "no calibration row supports this level; refusing a plug-in quantile"
        )
    ordered = np.sort(values)
    return float(ordered[rank - 1])


def _two_sided_scores(y: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    return np.maximum(0.0, lower - y), np.maximum(0.0, y - upper)


def split_conformal_interval(
    y_cal,
    lower_cal,
    upper_cal,
    alpha: float = 0.1,
    *,
    two_sided: str = "per_tail",
) -> dict:
    """Calibrate a box ``[lower - o_lo, upper + o_hi]`` on a calibration split.

    Parameters
    ----------
    y_cal : (n,) labels.
    lower_cal, upper_cal : (n,) predicted interval bounds.
    alpha : target miscoverage.
    two_sided : ``"per_tail"`` (default; level ``alpha/2`` per tail, asymmetric
        offsets) or ``"one_level"`` (single ``max(q_lo - y, y - q_hi)`` score at
        level ``alpha``, shared offset, exact marginal coverage).

    Returns
    -------
    dict with ``method="split_rank"``, ``two_sided``, ``alpha``, ``level_lower``,
    ``level_upper``, ``offsets_lower``, ``offsets_upper``,
    ``order_stat_index_lower``, ``order_stat_index_upper``, ``n_cal``,
    ``achievable_coverage``, ``finite_sample_exact``, ``null_reason``.
    """
    y = _as_vector(y_cal, "y_cal")
    lower = _as_vector(lower_cal, "lower_cal")
    upper = _as_vector(upper_cal, "upper_cal")
    if not (y.size == lower.size == upper.size):
        raise ValueError("y_cal, lower_cal and upper_cal must have equal length")
    if np.any(lower > upper):
        raise ValueError("lower_cal must not exceed upper_cal")
    alpha = _validate_alpha(alpha)
    if two_sided not in ("per_tail", "one_level"):
        raise ValueError("two_sided must be 'per_tail' or 'one_level'")
    n = int(y.size)

    if two_sided == "per_tail":
        level = 1.0 - alpha / 2.0
        s_lo, s_hi = _two_sided_scores(y, lower, upper)
        rank_lo = finite_sample_rank(n, alpha / 2.0)
        rank_hi = finite_sample_rank(n, alpha / 2.0)
        o_lo = exact_quantile(s_lo, alpha / 2.0)
        o_hi = exact_quantile(s_hi, alpha / 2.0)
        offsets_lower, offsets_upper = o_lo, o_hi
        ranks = (rank_lo, rank_hi)
    else:
        level = 1.0 - alpha
        combined = np.maximum(lower - y, y - upper)
        rank = finite_sample_rank(n, alpha)
        offset = exact_quantile(combined, alpha)
        offsets_lower = offsets_upper = offset
        ranks = (rank, rank)

    return {
        "method": "split_rank",
        "two_sided": two_sided,
        "alpha": alpha,
        "level_lower": level,
        "level_upper": level,
        "offsets_lower": float(offsets_lower),
        "offsets_upper": float(offsets_upper),
        "order_stat_index_lower": int(ranks[0]),
        "order_stat_index_upper": int(ranks[1]),
        "n_cal": n,
        # Guaranteed box coverage: the one-level rank floor, or the union-bound
        # combination of the two per-tail floors (2 * r/(n+1) - 1).
        "achievable_coverage": (
            2.0 * finite_sample_coverage_floor(n, alpha / 2.0) - 1.0
            if two_sided == "per_tail"
            else finite_sample_coverage_floor(n, alpha)
        ),
        "finite_sample_exact": True,
        "null_reason": "",
    }


def calibrate_mondrian_intervals(
    y_cal,
    lower_cal,
    upper_cal,
    strata,
    alpha: float = 0.1,
    *,
    two_sided: str = "per_tail",
) -> dict:
    """Mondrian wrapper: run :func:`split_conformal_interval` *within* strata.

    Strata are pre-registered (FeH < -2, logg branch, G bin, tier, ...).  Each
    stratum gets its own exact rank, ``n_cal`` and achievable coverage; a
    stratum too small to support the level raises -- it is never pooled
    silently into the marginal calibration.
    """
    y = _as_vector(y_cal, "y_cal")
    lower = _as_vector(lower_cal, "lower_cal")
    upper = _as_vector(upper_cal, "upper_cal")
    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size != y.size:
        raise ValueError("strata must be a 1-D label per calibration row")
    alpha = _validate_alpha(alpha)
    out: dict[str, dict] = {}
    for stratum in sorted({str(value) for value in labels.tolist()}):
        mask = np.asarray([str(value) == stratum for value in labels.tolist()])
        out[stratum] = split_conformal_interval(
            y[mask], lower[mask], upper[mask], alpha=alpha, two_sided=two_sided
        )
    return {
        "method": "mondrian_split_rank",
        "alpha": alpha,
        "two_sided": two_sided,
        "strata": out,
        "finite_sample_exact": True,
        "null_reason": "",
    }


def mahalanobis_scores(y, mean, cov) -> np.ndarray:
    """Rank-score ``(y - mu)^T Sigma^-1 (y - mu)`` with a frozen Gaussian.

    ``cov`` must be positive definite; jitter is forbidden (``+ eps I`` would
    change the score and is not a tolerance).  ``mu``/``cov`` must come from a
    *frozen* model: fitting them on the same calibration rows breaks
    exchangeability and voids the conformal guarantee.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if y_arr.ndim != 2:
        raise ValueError("y must have shape (n, d)")
    mean_arr = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov_arr = np.asarray(cov, dtype=np.float64)
    d = y_arr.shape[1]
    if mean_arr.size != d or cov_arr.shape != (d, d):
        raise ValueError("mean/cov dimensions must match y")
    if (
        not np.isfinite(y_arr).all()
        or not np.isfinite(mean_arr).all()
        or not np.isfinite(cov_arr).all()
    ):
        raise ValueError("y, mean and cov must be finite")
    if not np.allclose(cov_arr, cov_arr.T, rtol=0.0, atol=1e-12):
        raise ValueError("cov must be symmetric")
    try:
        chol = np.linalg.cholesky(cov_arr)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "cov is not positive definite; jitter (+eps I) is forbidden because it "
            "changes the rank score"
        ) from error
    centred = (y_arr - mean_arr).T
    solved = np.linalg.solve(chol, centred)
    return np.sum(solved**2, axis=0)


def neg_log_p_scores(log_prob) -> np.ndarray:
    """Rank score ``-log p_hat(y|x)`` from a frozen density (natural log)."""
    values = np.asarray(log_prob, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(
            "log_prob must be finite (a non-finite density is a failed arm)"
        )
    return -values


def joint_rank_score_region(
    cal_scores,
    test_scores,
    alpha: float = 0.1,
    *,
    score_kind: str = "rank_score_mahalanobis",
) -> dict:
    """Joint region ``{y: score(y) <= q_hat}`` from an exact rank-score rule.

    This is the *primary* joint construction: it uses the dependence between
    coordinates and has finite-sample **marginal joint** coverage
    ``>= 1 - alpha`` for exchangeable calibration/test scores.  It is not a
    rectangular interval and not a conditional-coverage claim.

    Returns ``threshold, rank, n_cal, n_test, achievable_coverage,
    finite_sample_exact, test_membership, test_coverage, unbounded,
    score_kind, null_reason``.
    """
    cal = _as_vector(cal_scores, "cal_scores")
    test = _as_vector(test_scores, "test_scores")
    if score_kind not in ("rank_score_mahalanobis", "rank_score_neg_log_p"):
        raise ValueError(
            "score_kind must be 'rank_score_mahalanobis' or 'rank_score_neg_log_p'"
        )
    alpha = _validate_alpha(alpha)
    threshold = exact_quantile(cal, alpha)
    membership = test <= threshold
    return {
        "threshold": float(threshold),
        "rank": int(finite_sample_rank(cal.size, alpha)),
        "n_cal": int(cal.size),
        "n_test": int(test.size),
        "achievable_coverage": finite_sample_coverage_floor(cal.size, alpha),
        "finite_sample_exact": True,
        "test_membership": membership,
        "test_coverage": float(np.mean(membership)),
        "unbounded": False,
        "score_kind": score_kind,
        "null_reason": "",
    }


def bonferroni_box_region(
    y_cal,
    lower_cal,
    upper_cal,
    y_test,
    lower_test,
    upper_test,
    *,
    alpha: float = 0.1,
    n_labels: int | None = None,
    two_sided: str = "per_tail",
) -> dict:
    """Secondary joint region: a Bonferroni box at ``alpha / n_labels`` per label.

    Per-label exact split-conformal offsets are calibrated on the calibration
    split and applied to the test box; joint membership requires all labels to
    be inside.  Conservative (union bound), but robust and rectangular -- it is
    what the existing 91.9% all / 29.1% tail joint numbers used and must be
    labelled as such.

    Returns ``joint_coverage, joint_membership, per_label_coverage,
    offsets_lower, offsets_upper, level, n_labels, n_cal, n_test,
    order_stat_index, finite_sample_exact, method, null_reason``.
    """
    y_cal_arr = _as_matrix(y_cal, "y_cal")
    lower_cal_arr = _as_matrix(lower_cal, "lower_cal")
    upper_cal_arr = _as_matrix(upper_cal, "upper_cal")
    y_test_arr = _as_matrix(y_test, "y_test")
    lower_test_arr = _as_matrix(lower_test, "lower_test")
    upper_test_arr = _as_matrix(upper_test, "upper_test")
    if not (y_cal_arr.shape == lower_cal_arr.shape == upper_cal_arr.shape):
        raise ValueError("calibration arrays must share shape (n_cal, d)")
    if not (y_test_arr.shape == lower_test_arr.shape == upper_test_arr.shape):
        raise ValueError("test arrays must share shape (n_test, d)")
    if y_cal_arr.shape[1] != y_test_arr.shape[1]:
        raise ValueError("calibration and test arrays must share the label dimension d")
    d = y_cal_arr.shape[1]
    if n_labels is None:
        n_labels = d
    n_labels = int(n_labels)
    if n_labels < d:
        raise ValueError("n_labels cannot be smaller than the number of scored labels")
    alpha = _validate_alpha(alpha)
    per_label = alpha / n_labels
    offsets_lower = np.empty(d, dtype=np.float64)
    offsets_upper = np.empty(d, dtype=np.float64)
    ranks = np.empty(d, dtype=np.int64)
    per_label_coverage = np.empty(d, dtype=np.float64)
    inside = np.ones(y_test_arr.shape, dtype=bool)
    for index in range(d):
        calibration = split_conformal_interval(
            y_cal_arr[:, index],
            lower_cal_arr[:, index],
            upper_cal_arr[:, index],
            alpha=per_label,
            two_sided=two_sided,
        )
        offsets_lower[index] = calibration["offsets_lower"]
        offsets_upper[index] = calibration["offsets_upper"]
        ranks[index] = calibration["order_stat_index_lower"]
        lo_test = lower_test_arr[:, index] - offsets_lower[index]
        hi_test = upper_test_arr[:, index] + offsets_upper[index]
        label_inside = (y_test_arr[:, index] >= lo_test) & (
            y_test_arr[:, index] <= hi_test
        )
        inside[:, index] = label_inside
        per_label_coverage[index] = float(np.mean(label_inside))
    joint = np.all(inside, axis=1)
    return {
        "method": "bonferroni_box",
        "alpha": alpha,
        "level": 1.0 - per_label,
        "n_labels": n_labels,
        "n_cal": int(y_cal_arr.shape[0]),
        "n_test": int(y_test_arr.shape[0]),
        "offsets_lower": [float(value) for value in offsets_lower],
        "offsets_upper": [float(value) for value in offsets_upper],
        "order_stat_index": [int(value) for value in ranks],
        "per_label_coverage": [float(value) for value in per_label_coverage],
        "joint_coverage": float(np.mean(joint)),
        "joint_membership": joint,
        "finite_sample_exact": True,
        "null_reason": "",
    }


def _as_matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n, d)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def mondrian_conformal(scores, strata, alpha: float = 0.1) -> dict:
    """Exact rank rule *within* each pre-registered stratum.

    Returns per-stratum ``thresholds``, integer ``order_stat_index``, ``n_cal``,
    ``achievable_coverage`` and the exact finite-sample level.  At
    ``n=141, alpha=0.1`` the index ``ceil(142*0.9)=128`` exists, so the
    achievable coverage is ``128/142`` -- discrete, and that is what must be
    reported.  A stratum that cannot support the level raises.
    """
    values = _as_vector(scores, "scores")
    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size != values.size:
        raise ValueError("strata must be a 1-D label per score")
    alpha = _validate_alpha(alpha)
    thresholds: dict[str, float] = {}
    ranks: dict[str, int] = {}
    counts: dict[str, int] = {}
    achievable: dict[str, float] = {}
    for stratum in sorted({str(value) for value in labels.tolist()}):
        mask = np.asarray([str(value) == stratum for value in labels.tolist()])
        subset = values[mask]
        rank = finite_sample_rank(subset.size, alpha)
        if rank > subset.size:
            raise InsufficientCalibrationError(
                f"stratum {stratum!r}: finite-sample rank {rank} exceeds n_cal={subset.size} "
                f"at alpha={alpha}; refusing a plug-in quantile"
            )
        thresholds[stratum] = exact_quantile(subset, alpha)
        ranks[stratum] = rank
        counts[stratum] = int(subset.size)
        achievable[stratum] = rank / float(subset.size + 1)
    return {
        "method": "mondrian_split_rank",
        "alpha": alpha,
        "thresholds": thresholds,
        "order_stat_index": ranks,
        "n_cal": counts,
        "achievable_coverage": achievable,
        "finite_sample_exact": True,
        "null_reason": "",
    }


def mondrian_membership(test_scores, strata, thresholds) -> np.ndarray:
    """Apply per-stratum thresholds: ``score <= threshold[stratum]``."""
    values = _as_vector(test_scores, "test_scores")
    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size != values.size:
        raise ValueError("strata must be a 1-D label per test score")
    missing = sorted(
        {str(value) for value in labels.tolist()} - {str(key) for key in thresholds}
    )
    if missing:
        raise ValueError(f"no calibration threshold for strata {missing}")
    limits = np.asarray([float(thresholds[str(value)]) for value in labels.tolist()])
    return values <= limits


# ---------------------------------------------------------------------------
# region size
# ---------------------------------------------------------------------------
def box_region_volume(lower, upper) -> np.ndarray:
    """Native-unit box volume ``prod_j (upper_j - lower_j)`` per row."""
    lo = _as_matrix(lower, "lower")
    hi = _as_matrix(upper, "upper")
    if lo.shape != hi.shape:
        raise ValueError("lower and upper must share shape")
    if np.any(hi < lo):
        raise ValueError("upper bounds must not be below lower bounds")
    return np.prod(hi - lo, axis=1)


def ellipsoid_region_volume(cov, threshold: float) -> float:
    """Volume of ``{y: (y-mu)^T Sigma^-1 (y-mu) <= threshold}``.

    ``V_d = pi^{d/2} / Gamma(d/2 + 1)``; volume ``= V_d sqrt(det Sigma)
    threshold^{d/2}``.  This is the analytic level-set size for a Gaussian
    rank-score region (``levelset_method="analytic_ellipsoid"``).
    """
    cov_arr = np.asarray(cov, dtype=np.float64)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not np.isfinite(cov_arr).all():
        raise ValueError("cov must be finite")
    if not np.allclose(cov_arr, cov_arr.T, rtol=0.0, atol=1e-12):
        raise ValueError("cov must be symmetric")
    threshold = float(threshold)
    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and non-negative")
    d = cov_arr.shape[0]
    sign, logdet = np.linalg.slogdet(cov_arr)
    if sign <= 0:
        raise ValueError("cov must be positive definite (no jitter)")
    unit_ball = math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)
    return float(unit_ball * math.exp(0.5 * logdet) * threshold ** (d / 2.0))


def region_volume(
    *,
    kind: str,
    lower=None,
    upper=None,
    cov=None,
    threshold: float | None = None,
) -> float | None:
    """Mean region volume for a box or an analytic ellipsoid.

    ``kind="box"`` needs ``lower``/``upper``; ``kind="ellipsoid"`` needs
    ``cov``/``threshold``.  Returns ``None`` with ``null_reason="not_computed"``
    when the inputs do not support the requested size -- never an invented
    number.
    """
    if kind == "box":
        if lower is None or upper is None:
            return None
        return float(np.mean(box_region_volume(lower, upper)))
    if kind == "ellipsoid":
        if cov is None or threshold is None:
            return None
        return ellipsoid_region_volume(cov, threshold)
    raise ValueError("kind must be 'box' or 'ellipsoid'")


def levelset_method_for(kind: str) -> str:
    """Map a region kind to the ``levelset_method`` schema value."""
    mapping = {
        "box": "box",
        "ellipsoid": "analytic_ellipsoid",
        "samples": "mc_samples",
        "saved": "saved_membership",
        "none": "none",
    }
    if kind not in mapping:
        raise ValueError(f"unknown region kind {kind!r}")
    return mapping[kind]


def unbounded_reason(threshold: float | None) -> str:
    """``null_reason`` for an infinite/unavailable threshold, else ``""``."""
    if threshold is None or not math.isfinite(float(threshold)):
        return NullReason.UNBOUNDED.value
    return ""
