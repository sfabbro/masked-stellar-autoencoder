"""
Split conformal / CQR-style offsets for quantile prediction intervals (scaled label space).

Offsets are computed on a calibration set and applied as:
  q_lo' = q_lo - offsets_lower
  q_hi' = q_hi + offsets_upper

The threshold is the **exact finite-sample order statistic**

    q_hat = s_( ceil((n + 1) (1 - alpha)) )        (1-based)

not a plug-in sample quantile.  ``numpy.quantile`` interpolates between order
statistics and lands *below* the conformal order statistic (e.g. ~6.0e-4 in
scaled label space at n=2046, alpha=0.1), which silently under-covers.  When
``ceil((n+1)(1-alpha)) > n`` this module raises instead of returning a zero
offset: at that size the calibration split simply cannot support the level.

Two-sided intervals (documented level, choose one and record it):

* ``two_sided="per_tail"`` (default) -- two one-sided intervals at level
  ``alpha/2`` each.  Asymmetric offsets are preserved; the union bound gives
  coverage >= ``1 - alpha``.
* ``two_sided="one_level"`` -- the single score ``max(q_lo - y, y - q_hi)`` at
  level ``alpha`` and one shared offset.  This is the classical CQR score; it
  gives exact marginal coverage of the symmetrically inflated box and, when all
  calibration points already fall inside the predicted box, the quantile is
  negative and the interval is *narrowed* (allowed).

This module is a self-contained mirror of
``masked_stellar_autoencoder.reporting.calibration`` so it can be vendored
without the reporting package; ``tests/test_conformal_ranks.py`` asserts the
two implementations agree.
"""

from typing import Any

import numpy as np


def exact_order_stat_index(n: int, alpha: float) -> int:
    """Exact 1-based conformal order-statistic index ``ceil((n+1)(1-alpha))``."""
    n = int(n)
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n < 1:
        raise ValueError("n must be a positive calibration size")
    return int(np.ceil((n + 1) * (1.0 - alpha)))


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Exact split-conformal threshold for one-sided conformity scores."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("no conformity scores: cannot calibrate")
    if not np.isfinite(values).all():
        raise ValueError("conformity scores must be finite")
    rank = exact_order_stat_index(values.size, alpha)
    if rank > values.size:
        raise ValueError(
            f"finite-sample rank {rank} exceeds n_cal={values.size} at alpha={alpha}: "
            "no calibration row supports this level (refusing a plug-in quantile)"
        )
    return float(np.sort(values)[rank - 1])


def conformity_scores_interval(
    y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per (sample, label) one-sided gaps outside [q_lo, q_hi]."""
    s_lo = np.maximum(0.0, q_lo - y)
    s_hi = np.maximum(0.0, y - q_hi)
    return s_lo, s_hi


def conformity_scores_two_sided(
    y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
) -> np.ndarray:
    """Classical CQR score ``max(q_lo - y, y - q_hi)`` (negative when inside)."""
    y = np.asarray(y, dtype=np.float64)
    q_lo = np.asarray(q_lo, dtype=np.float64)
    q_hi = np.asarray(q_hi, dtype=np.float64)
    return np.maximum(q_lo - y, y - q_hi)


def calibrate_cqr_offsets(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    alpha: float = 0.1,
    *,
    two_sided: str = "per_tail",
) -> dict[str, Any]:
    """
    Asymmetric conformal offsets per label from val residuals, exact rank.

    Parameters
    ----------
    y_val : (N, L) calibration labels (scaled space, same as preds).
    pred_val : (N, L, 3) with [:, :, 0]=lower, [:, :, 1]=median, [:, :, 2]=upper.
    alpha : miscoverage rate (target coverage 1 - alpha on calibration if exchangeable).
    two_sided : ``"per_tail"`` (level ``alpha/2`` per tail, default) or
        ``"one_level"`` (single ``max(q_lo - y, y - q_hi)`` score at level
        ``alpha``, shared offset).  The level actually used is recorded in
        ``levels``.

    Returns
    -------
    dict with keys ``version``, ``method``, ``alpha``, ``two_sided``, ``levels``,
    ``offsets_lower``, ``offsets_upper``, ``n_cal``,
    ``order_stat_index_lower``, ``order_stat_index_upper``,
    ``finite_sample_exact``.

    Raises
    ------
    ValueError
        If a label has no finite calibration rows, or if the exact finite-sample
        rank exceeds that label's calibration size.  A zero offset is never
        substituted: at that size the split cannot support the level.
    """
    if pred_val.ndim != 3 or pred_val.shape[2] != 3:
        raise ValueError("pred_val must have shape (N, L, 3)")
    y_val = np.asarray(y_val, dtype=np.float64)
    pred_val = np.asarray(pred_val, dtype=np.float64)
    n, ell = y_val.shape
    if pred_val.shape[:2] != (n, ell):
        raise ValueError("y_val and pred_val batch/label dims must match")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if two_sided not in ("per_tail", "one_level"):
        raise ValueError("two_sided must be 'per_tail' or 'one_level'")

    q_lo = pred_val[:, :, 0]
    q_hi = pred_val[:, :, 2]
    s_lo, s_hi = conformity_scores_interval(y_val, q_lo, q_hi)
    combined = conformity_scores_two_sided(y_val, q_lo, q_hi)

    m = np.isfinite(y_val) & np.isfinite(q_lo) & np.isfinite(q_hi)
    per_tail_alpha = alpha / 2.0 if two_sided == "per_tail" else alpha
    level = 1.0 - per_tail_alpha

    offsets_lower: list[float] = []
    offsets_upper: list[float] = []
    n_cal: list[int] = []
    index_lower: list[int] = []
    index_upper: list[int] = []
    for j in range(ell):
        mask = m[:, j]
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ValueError(f"label {j}: no finite calibration rows")
        rank = exact_order_stat_index(count, per_tail_alpha)
        if rank > count:
            raise ValueError(
                f"label {j}: finite-sample rank {rank} exceeds n_cal={count} "
                f"at alpha={per_tail_alpha:g}: refusing a zero offset / plug-in quantile"
            )
        if two_sided == "per_tail":
            lo_offset = float(np.sort(s_lo[mask, j])[rank - 1])
            hi_offset = float(np.sort(s_hi[mask, j])[rank - 1])
        else:
            offset = float(np.sort(combined[mask, j])[rank - 1])
            lo_offset = hi_offset = offset
        offsets_lower.append(lo_offset)
        offsets_upper.append(hi_offset)
        n_cal.append(count)
        index_lower.append(rank)
        index_upper.append(rank)

    return {
        "version": 2,
        "method": "cqr_asymmetric_quantile_offsets",
        "alpha": alpha,
        "two_sided": two_sided,
        "levels": {"lower": level, "upper": level},
        "offsets_lower": offsets_lower,
        "offsets_upper": offsets_upper,
        "n_cal": n_cal,
        "order_stat_index_lower": index_lower,
        "order_stat_index_upper": index_upper,
        "finite_sample_exact": True,
    }


def apply_cqr_offsets_inplace(pred: np.ndarray, calib: dict[str, Any]) -> np.ndarray:
    """Apply offsets to (N, L, 3) predictions in place; returns pred."""
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError("pred must have shape (N, L, 3)")
    o_lo = np.asarray(calib["offsets_lower"], dtype=np.float64).reshape(-1)
    o_hi = np.asarray(calib["offsets_upper"], dtype=np.float64).reshape(-1)
    ell = pred.shape[1]
    if o_lo.size != ell or o_hi.size != ell:
        raise ValueError(
            f"Calibration offsets length ({o_lo.size}, {o_hi.size}) != pred labels ({ell})"
        )
    pred[:, :, 0] -= o_lo.reshape(1, -1)
    pred[:, :, 2] += o_hi.reshape(1, -1)
    return pred


def interval_coverage(
    y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
) -> np.ndarray:
    """Per-label coverage rate in (0,1), ignoring non-finite y or bounds."""
    y_true = np.asarray(y_true, dtype=np.float64)
    q_lo = np.asarray(q_lo, dtype=np.float64)
    q_hi = np.asarray(q_hi, dtype=np.float64)

    m = np.isfinite(y_true) & np.isfinite(q_lo) & np.isfinite(q_hi) & (q_lo <= q_hi)

    # Suppress RuntimeWarnings for invalid comparisons with NaNs, which we explicitly handle
    with np.errstate(invalid="ignore"):
        inside = (y_true >= q_lo) & (y_true <= q_hi) & m

    valid_counts = m.sum(axis=0)
    inside_counts = inside.sum(axis=0)

    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    np.divide(inside_counts, valid_counts, out=out, where=valid_counts > 0)

    return out
