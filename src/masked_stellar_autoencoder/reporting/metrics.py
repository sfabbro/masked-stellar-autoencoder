"""Point, proper-score, interval and resampling metrics for the C6 contract.

Everything here is NumPy + float64 and never touches a checkpoint.  The rules
that matter for evidence quality:

* a metric that is undefined returns ``None`` with an explicit ``null_reason``
  (from :class:`masked_stellar_autoencoder.reporting.schema.NullReason`), never
  ``NaN``;
* ``density_kind`` is mandatory on every NLL/CRPS call: an intrinsic density and
  an observed-label density are different estimands and are never averaged
  together;
* CRPS comes from saved samples when the head is a flow/MDN; a Gaussian proxy is
  used only through the explicit :func:`crps_gaussian` helper;
* paired uncertainty resamples *groups* (unique ``source_id`` and/or HEALPix
  block), not rows: duplicated stars and spatial blocks make the row bootstrap
  anti-conservative.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .schema import BOOTSTRAP_KINDS, DENSITY_KINDS, LOSS_DENOMINATORS, NullReason

try:  # scipy is present in the scoring env; the module stays importable without it
    from scipy.special import erf as _erf_scipy
    from scipy.stats import chi2 as _chi2
except ImportError:  # pragma: no cover - optional soft dependency
    _erf_scipy = None
    _chi2 = None

#: Below this row count a slice is descriptive only (plan section 7).
MIN_COUNT = 20

#: Row block size for ``O(n * S^2 * d)`` multivariate scores.
_CHUNK_ROWS = 1024


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_1d(values, name: str, *, min_len: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} values")
    return array


def _as_aligned(*arrays: tuple[str, object], min_len: int = 0) -> list[np.ndarray]:
    out = [_as_1d(values, name, min_len=min_len) for name, values in arrays]
    sizes = {array.size for array in out}
    if len(sizes) != 1:
        raise ValueError(f"inputs must have equal length, got {sorted(sizes)}")
    return out


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_div(num: float, den: float) -> float | None:
    if not np.isfinite(den) or den == 0.0:
        return None
    return _safe_float(num / den)


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values))
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have a positive sum")
    return float(np.sum(weights * values) / total)


def _weighted_var(values: np.ndarray, weights: np.ndarray | None) -> float:
    mean = _weighted_mean(values, weights)
    if weights is None:
        return float(np.mean((values - mean) ** 2))
    total = float(np.sum(weights))
    return float(np.sum(weights * (values - mean) ** 2) / total)


def descriptive_only(n: int, *, min_count: int = MIN_COUNT) -> bool:
    """``True`` when a slice must be reported as descriptive only (n < 20)."""
    return int(n) < int(min_count)


def _reason(reasons: Sequence[str]) -> str:
    """Pick the single row-level ``null_reason`` from the causes that fired."""
    for reason in reasons:
        if reason:
            return reason
    return ""


def _require_density_kind(density_kind: str) -> str:
    if density_kind not in DENSITY_KINDS:
        raise ValueError(
            f"density_kind must be one of {DENSITY_KINDS} (intrinsic vs observed_label "
            "are different estimands and must never be mixed)"
        )
    return density_kind


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------
def point_metrics(
    y_true,
    y_pred,
    *,
    weights=None,
    outlier_k: float = 3.0,
    tail_quantile: float = 0.9,
    min_count: int = MIN_COUNT,
) -> dict:
    """Point-error block of one ``metrics.csv`` row.

    Returns ``n, n_eff, mae, rmse, bias, nmad, r2, pearson, atten_slope,
    outlier_frac, outlier_threshold, tail_mae, tail_count, descriptive_only,
    null_reason``.

    Definitions (native units, computed inside the slice):

    * ``bias = mean(y_pred - y_true)`` (signed);
    * ``nmad = 1.4826 * median|r - median(r)|``;
    * ``r2 = 1 - SS_res/SS_tot`` -- may be negative; ``None`` if ``SS_tot == 0``;
    * ``atten_slope`` -- OLS slope of ``y_true`` on ``y_pred`` (dilution
      diagnostic; never used to divide predictions, which would break
      calibration);
    * ``outlier_frac = mean(|r| > outlier_k * sd(y_true))``;
    * ``tail_mae`` -- mean ``|r|`` over the worst ``1 - tail_quantile`` of rows
      (at least one row).
    """
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail_quantile must be in (0, 1)")
    if outlier_k <= 0:
        raise ValueError("outlier_k must be positive")
    y = _as_1d(y_true, "y_true", min_len=0)
    p = _as_1d(y_pred, "y_pred", min_len=0)
    if y.size != p.size:
        raise ValueError("y_true and y_pred must have equal length")
    w = None
    if weights is not None:
        w = _as_1d(weights, "weights", min_len=0)
        if w.size != y.size:
            raise ValueError("weights must align with y_true")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    finite = np.isfinite(y) & np.isfinite(p)
    if w is not None:
        finite &= np.isfinite(w) & (w > 0)
    y, p = y[finite], p[finite]
    w = None if w is None else w[finite]
    n = int(y.size)
    if n == 0:
        return {
            "n": 0,
            "n_eff": 0.0,
            "mae": None,
            "rmse": None,
            "bias": None,
            "nmad": None,
            "r2": None,
            "pearson": None,
            "atten_slope": None,
            "outlier_frac": None,
            "outlier_threshold": None,
            "tail_mae": None,
            "tail_count": 0,
            "descriptive_only": True,
            "null_reason": NullReason.SPLIT_EMPTY.value,
        }
    if w is None:
        n_eff = float(n)
    else:
        n_eff = float(np.sum(w) ** 2 / np.sum(w**2))

    r = p - y
    mae = _weighted_mean(np.abs(r), w)
    rmse = float(np.sqrt(_weighted_mean(r**2, w)))
    bias = _weighted_mean(r, w)

    median_r = float(np.median(r))
    nmad = float(1.4826 * np.median(np.abs(r - median_r)))

    var_y = _weighted_var(y, w)
    if var_y > 0.0:
        r2 = _safe_float(1.0 - _weighted_mean(r**2, w) / var_y)
    else:
        r2 = None
    var_p = _weighted_var(p, w)
    if var_p > 0.0:
        cov = _weighted_mean((p - _weighted_mean(p, w)) * (y - _weighted_mean(y, w)), w)
        pearson = _safe_div(cov, math.sqrt(var_p * var_y)) if var_y > 0.0 else None
        atten_slope = _safe_div(cov, var_p)
    else:
        pearson = None
        atten_slope = None

    sd_y = float(np.sqrt(var_y * n / (n - 1))) if n > 1 else 0.0
    threshold = float(outlier_k * sd_y)
    outlier_frac = float(np.mean(np.abs(r) > threshold))

    k_tail = max(1, int(math.ceil((1.0 - tail_quantile) * n)))
    worst = np.sort(np.abs(r))[::-1][:k_tail]
    tail_mae = float(np.mean(worst))

    reasons = []
    if r2 is None or pearson is None or atten_slope is None:
        reasons.append(NullReason.NOT_APPLICABLE.value)
    return {
        "n": n,
        "n_eff": n_eff,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "nmad": nmad,
        "r2": r2,
        "pearson": pearson,
        "atten_slope": atten_slope,
        "outlier_frac": outlier_frac,
        "outlier_threshold": threshold,
        "tail_mae": tail_mae,
        "tail_count": k_tail,
        "descriptive_only": descriptive_only(n, min_count=min_count),
        "null_reason": _reason(reasons),
    }


# ---------------------------------------------------------------------------
# proper scores
# ---------------------------------------------------------------------------
def nll_from_log_prob(
    log_prob,
    *,
    density_kind: str,
    loss_denominator: str = "per_row",
    min_count: int = MIN_COUNT,
) -> dict:
    """Negative mean log density from saved per-row (or per-coordinate) values.

    ``log_prob`` are natural-log densities already in the reporting space.  The
    caller must state ``density_kind`` (``intrinsic`` or ``observed_label``) and
    ``loss_denominator``; NLLs from different values of either are never
    comparable and must never share a column.
    """
    _require_density_kind(density_kind)
    if loss_denominator not in LOSS_DENOMINATORS:
        raise ValueError(f"loss_denominator must be one of {LOSS_DENOMINATORS}")
    values = _as_1d(log_prob, "log_prob", min_len=0)
    finite = np.isfinite(values)
    n_valid = int(finite.sum())
    n_nonfinite = int(values.size - n_valid)
    if n_valid == 0:
        return {
            "n": 0,
            "nll": None,
            "density_kind": density_kind,
            "loss_denominator": loss_denominator,
            "descriptive_only": True,
            "null_reason": NullReason.SPLIT_EMPTY.value,
        }
    # A non-finite log density is a model failure, not a missing value: refuse
    # to drop it silently (the plan's huber->NaN density row is exactly this bug).
    if n_nonfinite:
        raise ValueError(
            f"log_prob contains {n_nonfinite} non-finite values; a non-finite density "
            "is a failed arm, not a row to drop"
        )
    return {
        "n": n_valid,
        "nll": float(-np.mean(values)),
        "density_kind": density_kind,
        "loss_denominator": loss_denominator,
        "descriptive_only": descriptive_only(n_valid, min_count=min_count),
        "null_reason": "",
    }


def crps_from_samples(
    samples,
    y,
    *,
    density_kind: str,
    min_count: int = MIN_COUNT,
) -> dict:
    """Unbiased empirical CRPS from saved samples (never a Gaussian proxy).

    ``samples`` has shape ``(n,)`` for one row or ``(n_rows, n_samples)``;
    samples along the last axis.  The estimator is

    ``CRPS = mean_s |s - y| - sum_k (2k - S + 1) x_(k) / (S (S-1))``

    with ``x_(k)`` the sorted samples (0-based), i.e. the unbiased
    ``0.5 * E|X - X'|`` correction.  At least two samples per row are required.
    """
    _require_density_kind(density_kind)
    y_arr = np.asarray(y, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        if y_arr.size != 1:
            raise ValueError("1-D samples only pair with a scalar y")
        samples = samples.reshape(1, -1)
        y_arr = y_arr.reshape(1)
    if samples.ndim != 2:
        raise ValueError("samples must have shape (n,) or (n_rows, n_samples)")
    if y_arr.ndim != 1 or y_arr.size != samples.shape[0]:
        raise ValueError("y must be 1-D and align with samples.shape[0]")
    n_samples = int(samples.shape[1])
    if n_samples < 2:
        raise ValueError("CRPS from samples needs at least 2 samples per row")
    if not np.isfinite(samples).all() or not np.isfinite(y_arr).all():
        raise ValueError("samples and y must be finite (save all fixed nodes)")

    ordered = np.sort(samples, axis=1)
    first = np.mean(np.abs(samples - y_arr[:, None]), axis=1)
    k = np.arange(n_samples, dtype=np.float64)
    coeff = 2.0 * k - n_samples + 1.0
    second = (ordered @ coeff) / (n_samples * (n_samples - 1))
    values = first - second
    return {
        "crps": float(np.mean(values)),
        "n": int(values.size),
        "n_samples": n_samples,
        "density_kind": density_kind,
        "descriptive_only": descriptive_only(values.size, min_count=min_count),
        "null_reason": "",
    }


def _aligned_or_scalar(y: np.ndarray, value, name: str) -> np.ndarray:
    array = _as_1d(value, name)
    if array.size == 1:
        return np.full(y.size, float(array[0]))
    if array.size != y.size:
        raise ValueError(f"{name} must be scalar or align with y")
    return array


def crps_gaussian(y, mu, sigma, *, density_kind: str) -> dict:
    """Analytic CRPS of ``N(mu, sigma^2)`` (explicit Gaussian proxy only).

    ``density_kind`` is mandatory: this is a *Gaussian* proxy and must never be
    substituted for sample-based CRPS of a flow/MDN arm.
    """
    _require_density_kind(density_kind)
    y = _as_1d(y, "y")
    mu = _aligned_or_scalar(y, mu, "mu")
    sigma = _aligned_or_scalar(y, sigma, "sigma")
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("sigma must be finite and positive")
    z = (y - mu) / sigma
    cdf = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    values = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return {
        "crps": float(np.mean(values)),
        "n": int(values.size),
        "density_kind": density_kind,
        "null_reason": "",
    }


def nll_gaussian(y, mu, sigma, *, density_kind: str) -> dict:
    """Analytic Gaussian NLL (explicit family; never mixed with other families)."""
    _require_density_kind(density_kind)
    y = _as_1d(y, "y")
    mu = _aligned_or_scalar(y, mu, "mu")
    sigma = _aligned_or_scalar(y, sigma, "sigma")
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError("sigma must be finite and positive")
    z = (y - mu) / sigma
    values = 0.5 * (np.log(2.0 * math.pi) + 2.0 * np.log(sigma)) + 0.5 * z**2
    return {
        "nll": float(np.mean(values)),
        "n": int(values.size),
        "density_kind": density_kind,
        "null_reason": "",
    }


def _erf(x):
    """Vectorised ``erf``; scipy when available, exact ``math.erf`` fallback."""
    values = np.asarray(x, dtype=np.float64)
    if _erf_scipy is not None:
        return _erf_scipy(values)
    flat = values.reshape(-1)
    out = np.empty(flat.size, dtype=np.float64)
    for index, item in enumerate(flat):
        out[index] = math.erf(float(item))
    return out.reshape(values.shape)


# ---------------------------------------------------------------------------
# PIT / rank histogram
# ---------------------------------------------------------------------------
def pit_from_samples(
    samples,
    y,
    *,
    randomized: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Probability integral transform ``u = P(S < y) (+ 0.5 P(S = y))``.

    With ``randomized=True`` ties are broken by a uniform draw (needs ``seed``
    for reproducibility); otherwise half the tie mass is used (a continuous
    approximation).  Returns a 1-D array of length ``n_rows``.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        if y_arr.size != 1:
            raise ValueError("1-D samples only pair with a scalar y")
        samples = samples.reshape(1, -1)
        y_arr = y_arr.reshape(1)
    if samples.ndim != 2 or samples.shape[0] != y_arr.size:
        raise ValueError("samples must be (n_rows, n_samples) and align with y")
    below = np.sum(samples < y_arr[:, None], axis=1)
    equal = np.sum(samples == y_arr[:, None], axis=1)
    if randomized:
        rng = np.random.default_rng(seed)
        jitter = rng.random(equal.size)
        u = (below + jitter * equal) / samples.shape[1]
    else:
        u = (below + 0.5 * equal) / samples.shape[1]
    return np.clip(u, 0.0, 1.0)


def _ks_uniform(sorted_u: np.ndarray) -> float:
    n = sorted_u.size
    index = np.arange(1, n + 1, dtype=np.float64)
    upper = index / n - sorted_u
    lower = sorted_u - (index - 1.0) / n
    return float(max(upper.max(), lower.max()))


def _ks_pvalue(statistic: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * statistic
    if lam <= 0.0:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-14:
            break
    return float(min(1.0, max(0.0, total)))


def pit_diagnostics(pit_values, *, n_bins: int = 20) -> dict:
    """Uniformity diagnostics of a PIT sample: KS, Cramer-von Mises, chi-square."""
    values = _as_1d(pit_values, "pit_values", min_len=0)
    finite = np.isfinite(values)
    values = values[finite]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "pit_ks": None,
            "pit_ks_pvalue": None,
            "pit_cvm": None,
            "pit_chi2": None,
            "pit_chi2_pvalue": None,
            "pit_bins": int(n_bins),
            "pit_mean": None,
            "pit_var": None,
            "null_reason": NullReason.SPLIT_EMPTY.value,
        }
    ordered = np.sort(values)
    ks = _ks_uniform(ordered)
    index = np.arange(1, n + 1, dtype=np.float64)
    cvm = float(
        1.0 / (12.0 * n) + np.sum((ordered - (2.0 * index - 1.0) / (2.0 * n)) ** 2)
    )
    counts, _ = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    expected = n / n_bins
    chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
    chi2_p = None if _chi2 is None else float(_chi2.sf(chi2_stat, n_bins - 1))
    return {
        "n": n,
        "pit_ks": ks,
        "pit_ks_pvalue": _ks_pvalue(ks, n),
        "pit_cvm": cvm,
        "pit_chi2": chi2_stat,
        "pit_chi2_pvalue": chi2_p,
        "pit_bins": int(n_bins),
        "pit_mean": float(np.mean(values)),
        "pit_var": float(np.var(values)),
        "null_reason": "" if chi2_p is not None else NullReason.NOT_COMPUTED.value,
    }


def rank_histogram(samples, y, *, min_count: int = MIN_COUNT) -> dict:
    """Rank histogram (SBC-style) from samples: counts, chi-square, p-value.

    ``samples`` is ``(n_rows, n_samples)``.  Rank ``r = #{s < y}`` lies in
    ``0..S``; a calibrated predictive distribution gives a uniform histogram.
    Needs thousands of rows per stratum to be informative (plan section 7).
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        if y_arr.size != 1:
            raise ValueError("1-D samples only pair with a scalar y")
        samples = samples.reshape(1, -1)
    if samples.ndim != 2 or samples.shape[0] != y_arr.size:
        raise ValueError("samples must be (n_rows, n_samples) and align with y")
    n_samples = int(samples.shape[1])
    ranks = np.sum(samples < y_arr[:, None], axis=1).astype(np.int64)
    counts = np.bincount(ranks, minlength=n_samples + 1)
    n = int(y_arr.size)
    expected = n / (n_samples + 1)
    chi2_stat = float(np.sum((counts - expected) ** 2 / expected))
    chi2_p = None if _chi2 is None else float(_chi2.sf(chi2_stat, n_samples))
    return {
        "n": n,
        "n_samples": n_samples,
        "counts": [int(value) for value in counts],
        "chi2": chi2_stat,
        "chi2_pvalue": chi2_p,
        "descriptive_only": descriptive_only(n, min_count=min_count),
        "null_reason": "" if chi2_p is not None else NullReason.NOT_COMPUTED.value,
    }


# ---------------------------------------------------------------------------
# multivariate scores
# ---------------------------------------------------------------------------
def _as_samples_3d(samples, y) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    if y_arr.ndim != 2:
        raise ValueError("y must have shape (n_rows, dim) or (dim,)")
    if samples.ndim == 2:
        samples = samples[None, ...]
    if samples.ndim != 3:
        raise ValueError("samples must have shape (n_rows, n_samples, dim)")
    if samples.shape[0] != y_arr.shape[0] or samples.shape[2] != y_arr.shape[1]:
        raise ValueError("samples and y disagree on rows or dimension")
    if samples.shape[1] < 2:
        raise ValueError("at least 2 samples per row are required")
    if not np.isfinite(samples).all() or not np.isfinite(y_arr).all():
        raise ValueError("samples and y must be finite")
    return samples, y_arr


def energy_score(samples, y, *, beta: float = 1.0, min_count: int = MIN_COUNT) -> dict:
    """Energy score ``E||S-y||^beta - 0.5 E||S-S'||^beta`` (proper, multivariate).

    ``samples`` is ``(n_rows, n_samples, dim)``, ``y`` is ``(n_rows, dim)``.
    Identical samples ``S == y`` give exactly ``0``.  ``beta=1`` is the
    standard energy score; ``0 < beta < 2`` is valid.  Rows are processed in
    blocks so the ``O(n * S^2 * d)`` pair term never materialises all at once.
    """
    if not 0.0 < beta < 2.0:
        raise ValueError("beta must be in (0, 2)")
    samples, y_arr = _as_samples_3d(samples, y)
    n_rows = int(samples.shape[0])
    total = 0.0
    for start in range(0, n_rows, _CHUNK_ROWS):
        block = samples[start : start + _CHUNK_ROWS]
        target = y_arr[start : start + _CHUNK_ROWS]
        first = np.mean(_norms(block - target[:, None, :]) ** beta, axis=1)
        pair = _norms(block[:, :, None, :] - block[:, None, :, :])
        second = 0.5 * np.mean(pair**beta, axis=(1, 2))
        total += float(np.sum(first - second))
    return {
        "energy_score": total / n_rows,
        "n": n_rows,
        "n_samples": int(samples.shape[1]),
        "dim": int(samples.shape[2]),
        "beta": float(beta),
        "descriptive_only": descriptive_only(n_rows, min_count=min_count),
        "null_reason": "",
    }


def _norms(deltas: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(deltas**2, axis=-1))


def variogram_score(
    samples,
    y,
    *,
    p: float = 0.5,
    weights=None,
    min_count: int = MIN_COUNT,
) -> dict:
    """Variogram score of Scheuerer & Hamill: the dependence diagnostic.

    ``VS = sum_{i<j} w_ij [ mean_s |s_i - s_j|^p - |y_i - y_j|^p ]^2``.

    Each unordered coordinate pair contributes once (default ``w_ij = 1``);
    the convention is documented because a factor-2 convention change is a
    different score, not a tolerance change.  Identical samples give ``0``.
    Native units per coordinate: never aggregate across unit conventions.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1] (0.5 is the standard choice)")
    samples, y_arr = _as_samples_3d(samples, y)
    dim = samples.shape[2]
    if dim < 2:
        raise ValueError("variogram score needs at least 2 coordinates")
    if weights is None:
        w = np.ones((dim, dim), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (dim, dim):
            raise ValueError("weights must have shape (dim, dim)")
    upper = np.triu_indices(dim, k=1)
    total = 0.0
    n_rows = int(samples.shape[0])
    for start in range(0, n_rows, _CHUNK_ROWS):
        block = samples[start : start + _CHUNK_ROWS]
        target = y_arr[start : start + _CHUNK_ROWS]
        empirical = np.abs(block[:, :, :, None] - block[:, :, None, :]) ** p
        empirical = np.mean(empirical, axis=1)
        observed = np.abs(target[:, :, None] - target[:, None, :]) ** p
        residual = empirical - observed
        total += float(np.sum(w[upper] * residual[:, upper[0], upper[1]] ** 2))
    return {
        "variogram_score": total / n_rows,
        "n": n_rows,
        "n_samples": int(samples.shape[1]),
        "dim": int(dim),
        "p": float(p),
        "n_pairs": int(upper[0].size),
        "descriptive_only": descriptive_only(n_rows, min_count=min_count),
        "null_reason": "",
    }


# ---------------------------------------------------------------------------
# intervals and coverage
# ---------------------------------------------------------------------------
def interval_score(
    y, lo, hi, *, alpha: float = 0.1, min_count: int = MIN_COUNT
) -> dict:
    """Mean interval score (Gneiting & Raftery) at level ``1 - alpha``.

    ``IS = (hi - lo) + (2/alpha)(lo - y) 1{y < lo} + (2/alpha)(y - hi) 1{y > hi}``.
    Rows with non-finite or inverted bounds are excluded and counted.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    y_arr, lo_arr, hi_arr = _as_aligned(("y", y), ("lo", lo), ("hi", hi))
    valid = (
        np.isfinite(y_arr)
        & np.isfinite(lo_arr)
        & np.isfinite(hi_arr)
        & (lo_arr <= hi_arr)
    )
    n_invalid = int((~valid).sum())
    if not valid.any():
        return {
            "interval_score": None,
            "width": None,
            "coverage": None,
            "alpha": float(alpha),
            "n": 0,
            "n_invalid": n_invalid,
            "null_reason": NullReason.SPLIT_EMPTY.value,
        }
    yv, lov, hiv = y_arr[valid], lo_arr[valid], hi_arr[valid]
    width = hiv - lov
    penalty_lo = np.where(yv < lov, (2.0 / alpha) * (lov - yv), 0.0)
    penalty_hi = np.where(yv > hiv, (2.0 / alpha) * (yv - hiv), 0.0)
    return {
        "interval_score": float(np.mean(width + penalty_lo + penalty_hi)),
        "width": float(np.mean(width)),
        "coverage": float(np.mean((yv >= lov) & (yv <= hiv))),
        "alpha": float(alpha),
        "n": int(yv.size),
        "n_invalid": n_invalid,
        "null_reason": "",
    }


def coverage(y, lo, hi, *, min_count: int = MIN_COUNT) -> dict:
    """Marginal coverage of ``[lo, hi]``; counts inverted bounds separately."""
    y_arr, lo_arr, hi_arr = _as_aligned(("y", y), ("lo", lo), ("hi", hi))
    finite = np.isfinite(y_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)
    inverted = int((finite & (lo_arr > hi_arr)).sum())
    valid = finite & (lo_arr <= hi_arr)
    n = int(valid.sum())
    if n == 0:
        return {
            "coverage": None,
            "n": 0,
            "n_inside": 0,
            "n_inverted": inverted,
            "null_reason": NullReason.SPLIT_EMPTY.value,
        }
    inside = int(
        np.sum((y_arr[valid] >= lo_arr[valid]) & (y_arr[valid] <= hi_arr[valid]))
    )
    return {
        "coverage": inside / n,
        "n": n,
        "n_inside": inside,
        "n_inverted": inverted,
        "null_reason": "",
    }


def mondrian_coverage(y, lo, hi, strata, *, min_count: int = MIN_COUNT) -> list[dict]:
    """Coverage per pre-registered stratum (Mondrian reporting).

    ``strata`` is a 1-D label per row.  Returns one dict per stratum with
    ``stratum, n, n_inside, coverage, descriptive_only, null_reason`` sorted by
    stratum label.  Small strata are marked descriptive only -- they are never
    pooled into a marginal number.
    """
    y_arr, lo_arr, hi_arr = _as_aligned(("y", y), ("lo", lo), ("hi", hi))
    labels = np.asarray(strata)
    if labels.ndim != 1 or labels.size != y_arr.size:
        raise ValueError("strata must be a 1-D label per row")
    rows = []
    for stratum in sorted({str(value) for value in labels.tolist()}):
        mask = np.asarray([str(value) == stratum for value in labels.tolist()])
        result = coverage(y_arr[mask], lo_arr[mask], hi_arr[mask], min_count=min_count)
        result["stratum"] = stratum
        result["descriptive_only"] = descriptive_only(result["n"], min_count=min_count)
        rows.append(result)
    return rows


def coverage_calibration_slope(covered, width, *, min_count: int = MIN_COUNT) -> dict:
    """OLS slope of the cover indicator on ``log(width)``.

    A negative slope means wide intervals under-cover (width is not tracking
    difficulty); a positive slope means width and difficulty agree.  Returned as
    ``calib_slope``; ``None`` when width is constant.
    """
    covered_arr, width_arr = _as_aligned(("covered", covered), ("width", width))
    finite = (
        np.isfinite(covered_arr)
        & np.isfinite(width_arr)
        & (width_arr > 0)
        & ((covered_arr == 0) | (covered_arr == 1))
    )
    n = int(finite.sum())
    if n < 2:
        return {
            "calib_slope": None,
            "n": n,
            "null_reason": NullReason.INSUFFICIENT_N.value,
        }
    x = np.log(width_arr[finite])
    c = covered_arr[finite]
    var_x = float(np.var(x))
    if var_x <= 0.0:
        return {
            "calib_slope": None,
            "n": n,
            "null_reason": NullReason.NOT_APPLICABLE.value,
        }
    slope = float(np.mean((x - x.mean()) * (c - c.mean())) / var_x)
    return {
        "calib_slope": slope,
        "n": n,
        "descriptive_only": descriptive_only(n, min_count=min_count),
        "null_reason": "",
    }


# ---------------------------------------------------------------------------
# grouped paired bootstrap
# ---------------------------------------------------------------------------
def _normalise_groups(groups, n: int) -> list[np.ndarray]:
    if isinstance(groups, np.ndarray):
        if groups.ndim == 1:
            candidates = [groups]
        elif groups.ndim == 2:
            candidates = [groups[index] for index in range(groups.shape[0])]
        else:
            raise ValueError("a groups array must be 1-D or 2-D")
    elif isinstance(groups, list | tuple):
        if not groups:
            raise ValueError("groups must not be empty")
        first = np.asarray(groups[0])
        if first.ndim == 0:
            # A plain sequence of scalar labels is ONE grouping factor.
            candidates = [np.asarray(groups)]
        elif first.ndim == 1:
            candidates = [np.asarray(item) for item in groups]
        else:
            raise ValueError("each grouping vector must be 1-D")
    else:
        raise ValueError("groups must be an array or a sequence of arrays")
    out = []
    for candidate in candidates:
        labels = np.asarray(candidate)
        if labels.ndim != 1 or labels.size != n:
            raise ValueError("every grouping vector must be 1-D and align with delta")
        # Encode as integer level codes so string/int IDs behave identically.
        _, codes = np.unique(labels.astype(str), return_inverse=True)
        out.append(codes.astype(np.int64))
    if not out:
        raise ValueError("at least one grouping vector is required")
    return out


def _weighted_mean_or_zero(delta: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0  # signals an empty resample; the caller retries
    return float(np.dot(weights, delta) / total)


def grouped_paired_bootstrap(
    delta,
    *,
    groups,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    bootstrap_kind: str | None = None,
) -> dict:
    """Paired bootstrap that resamples *groups*, not individual rows.

    Parameters
    ----------
    delta : (n,) per-row paired difference, e.g. ``|err_a| - |err_b|``.
        Positive means model A is worse.
    groups : one 1-D label vector (unique ``source_id`` *or* HEALPix block) or a
        sequence of them (e.g. ``(source_id, healpix64)``).  With two grouping
        vectors this is the two-way cluster bootstrap: each factor's clusters
        are drawn independently and a row's weight is the product of its two
        cluster multiplicities, which keeps the cross-factor dependence.
    n_boot, alpha, seed : 1,000 reps and a recorded seed per plan section 7.
    bootstrap_kind : ``star`` | ``block`` | ``two_way``; inferred from the
        number of grouping vectors when omitted.

    Returns ``point, ci_lo, ci_hi, alpha, n_boot, seed, bootstrap_kind,
    n_rows, n_groups, null_reason``.
    """
    values = _as_1d(delta, "delta", min_len=2)
    if not np.isfinite(values).all():
        raise ValueError("delta must be finite")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    codes = _normalise_groups(groups, values.size)
    if bootstrap_kind is None:
        bootstrap_kind = "two_way" if len(codes) >= 2 else "star"
    if bootstrap_kind not in BOOTSTRAP_KINDS:
        raise ValueError(f"bootstrap_kind must be one of {BOOTSTRAP_KINDS}")
    if bootstrap_kind == "two_way" and len(codes) < 2:
        raise ValueError("two_way bootstrap needs two grouping vectors")
    if bootstrap_kind in ("star", "block") and len(codes) != 1:
        raise ValueError(
            f"{bootstrap_kind} bootstrap needs exactly one grouping vector"
        )

    rng = np.random.default_rng(seed)
    sizes = [int(codes[i].max()) + 1 for i in range(len(codes))]
    stats = np.empty(n_boot, dtype=np.float64)
    for rep in range(n_boot):
        weights = np.ones(values.size, dtype=np.float64)
        for factor, size in zip(codes, sizes):
            draws = rng.integers(0, size, size=size)
            counts = np.bincount(draws, minlength=size).astype(np.float64)
            weights *= counts[factor]
        estimate = _weighted_mean_or_zero(values, weights)
        attempts = 0
        while weights.sum() == 0.0 and attempts < 100:
            weights = np.ones(values.size, dtype=np.float64)
            for factor, size in zip(codes, sizes):
                draws = rng.integers(0, size, size=size)
                counts = np.bincount(draws, minlength=size).astype(np.float64)
                weights *= counts[factor]
            estimate = _weighted_mean_or_zero(values, weights)
            attempts += 1
        if weights.sum() == 0.0:
            raise RuntimeError("bootstrap could not draw a non-empty resample")
        stats[rep] = estimate

    lo, hi = np.percentile(stats, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "point": float(np.mean(values)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "bootstrap_kind": bootstrap_kind,
        "n_rows": int(values.size),
        "n_groups": [int(size) for size in sizes],
        "null_reason": "",
    }


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------
def bonferroni(pvalues, *, alpha: float = 0.05, n_tests: int | None = None) -> dict:
    """Bonferroni correction across the pre-registered slice count."""
    p = _as_1d(pvalues, "pvalues")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    m = int(n_tests) if n_tests is not None else int(p.size)
    if m < p.size:
        raise ValueError("n_tests cannot be smaller than the number of p-values")
    adjusted = np.clip(p * m, 0.0, 1.0)
    return {
        "method": "bonferroni",
        "n_tests": m,
        "alpha": float(alpha),
        "threshold": float(alpha / m),
        "adjusted": [float(value) for value in adjusted],
        "reject": [bool(value <= alpha) for value in adjusted],
        "null_reason": "",
    }


def benjamini_hochberg(pvalues, *, q: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR control at level ``q`` (step-up over all slices)."""
    p = _as_1d(pvalues, "pvalues")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    m = int(p.size)
    order = np.argsort(p)
    ranked = p[order]
    scaled = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(monotone, 0.0, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return {
        "method": "benjamini_hochberg",
        "n_tests": m,
        "q": float(q),
        "adjusted": [float(value) for value in adjusted],
        "reject": [bool(value <= q) for value in adjusted],
        "null_reason": "",
    }


def multiplicity_note(kind: str, n_tests: int) -> str:
    """Canonical ``multiplicity`` cell: ``bonferroni_k=16`` / ``bh_fdr_q=0.05``."""
    if kind not in ("bonferroni", "bh_fdr", "none"):
        raise ValueError("kind must be bonferroni, bh_fdr or none")
    if kind == "none":
        return "none"
    if kind == "bonferroni":
        return f"bonferroni_k={int(n_tests)}"
    return "bh_fdr_q=0.05"
