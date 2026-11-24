"""Statistics and pre-registered strata of the QA suite (``docs/programme/qa.md`` §0, §5.2).

Metrics follow qa.md §0: bias = median(prediction - reference), NMAD =
1.4826 x MAD, coverage within Wilson 95 % intervals with a Bonferroni split
over the bins of one axis (``torchregress.inference.binned_conditional_coverage``),
bins with n < ``MIN_N`` descriptive only. PIT values come from the predictive
quantiles by monotone piecewise-linear interpolation of the CDF.
"""

from __future__ import annotations

import math

import numpy as np

MIN_N = 20
FEH_BINS = (-9.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 9.0)
CORE_G_BINS = (-math.inf, 13.5, 15.0, 16.0, 17.65)
# qa.md §5.2 axes: (edges, labels); categorical axes have edges None.
STRATA = {
    "G": ((-math.inf, 13.5, 15.0, 16.0, 17.0, 18.0, 19.0, math.inf),
          ("<=13.5", "13.5-15", "15-16", "16-17", "17-18", "18-19", ">19")),
    "bp_rp": ((-math.inf, 0.5, 0.8, 1.2, 1.8, 2.5, math.inf),
              ("<0.5", "0.5-0.8", "0.8-1.2", "1.2-1.8", "1.8-2.5", ">2.5")),
    "feh_ref": (FEH_BINS, ("<-3", "-3--2.5", "-2.5--2", "-2--1.5", "-1.5--1", "-1--0.5", ">-0.5")),
    "feh_pred": (FEH_BINS, ("<-3", "-3--2.5", "-2.5--2", "-2--1.5", "-1.5--1", "-1--0.5", ">-0.5")),
    "abs_b": ((0.0, 5.0, 15.0, 30.0, 60.0, 90.01), ("<5", "5-15", "15-30", "30-60", ">60")),
    "ebv_sfd": ((-math.inf, 0.05, 0.3, 1.0, math.inf), ("<0.05", "0.05-0.3", "0.3-1", ">1")),
    "ruwe": ((-math.inf, 1.4, math.inf), ("<1.4", ">=1.4")),
    "stage": (None, ("dwarf_ms", "turnoff_sgb", "rc", "rgb", "agb_other")),
    "input_class": (None, ("X", "P", "G", "N")),
}  # fmt: skip


def nmad(x) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def summary(d) -> dict:
    """bias (median), NMAD and n of residuals; descriptive below MIN_N."""
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    n = int(len(d))
    return dict(
        n=n,
        bias=float(np.median(d)) if n else float("nan"),
        nmad=nmad(d),
        descriptive=n < MIN_N,
    )


def assign(axis: str, values) -> np.ndarray:
    """Stratum label per row ('' when the value is missing)."""
    edges, labels = STRATA[axis]
    v = np.asarray(values)
    if edges is None:
        out = np.asarray(v, dtype=object).astype(str)
        return np.where(np.isin(out, labels), out, "")
    x = np.asarray(v, np.float64)
    idx = np.searchsorted(np.asarray(edges[1:-1]), x, side="right")
    out = np.asarray(labels, dtype=object)[np.clip(idx, 0, len(labels) - 1)]
    return np.where(np.isfinite(x), out, "").astype(str)


def stage(teff, logg) -> np.ndarray:
    """``reporting.strata.stage_stratum`` with '' for rows lacking Teff or log g."""
    from masked_stellar_autoencoder.reporting.strata import stage_stratum

    teff, logg = np.asarray(teff, np.float64), np.asarray(logg, np.float64)
    ok = np.isfinite(teff) & np.isfinite(logg)
    out = np.full(len(teff), "", dtype=object)
    if ok.any():
        out[ok] = stage_stratum(teff[ok], logg[ok])
    return out.astype(str)


def wilson_by_stratum(labels, covered, nominal: float, order=None) -> list[dict]:
    """Coverage per stratum with Bonferroni-split Wilson 95 % intervals."""
    import torch
    from torchregress.inference import binned_conditional_coverage

    labels = np.asarray(labels).astype(str)
    covered = np.asarray(covered, bool)
    keep = labels != ""
    names = [s for s in (order or sorted(set(labels[keep]))) if np.any(labels == s)]
    if not names or not keep.any():
        return []
    code = np.array([names.index(s) if s in names else -1 for s in labels])
    sel = code >= 0
    theta = torch.as_tensor(code[sel], dtype=torch.float64).reshape(-1, 1)
    edges = torch.arange(-0.5, len(names), 1.0, dtype=torch.float64)
    res = binned_conditional_coverage(
        theta, torch.as_tensor(covered[sel]), nominal=nominal, bins=[edges]
    )[0]
    out = []
    for k, name in enumerate(names):
        n = int(res.count[k])
        out.append(
            dict(
                stratum=name, n=n, coverage=float(res.coverage[k]),
                lower=float(res.lower[k]), upper=float(res.upper[k]),
                nominal=nominal, descriptive=n < MIN_N,
                under=bool(res.under_covered[k]) and n >= MIN_N,
                over=bool(res.over_covered[k]) and n >= MIN_N,
                within_003=bool(abs(float(res.coverage[k]) - nominal) <= 0.03),
            )
        )  # fmt: skip
    return out


def pit_from_quantiles(levels, q, y) -> np.ndarray:
    """Predictive CDF at ``y`` from quantiles ``q`` (N, K) at ``levels`` (K,).

    Linear between quantiles; outside the outermost quantiles the CDF is
    extended linearly with the slope of the outer segment and clipped to
    [0, 1], so a reference beyond them gets a PIT near 0 or 1.
    """
    levels = np.asarray(levels, np.float64)
    q = np.sort(np.asarray(q, np.float64), axis=1)
    y = np.asarray(y, np.float64)
    n, k = q.shape
    out = np.full(n, np.nan)
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    if k < 2:
        return out
    qq, yy = q[ok], y[ok]
    idx = np.clip((qq < yy[:, None]).sum(axis=1) - 1, 0, k - 2)
    lo = qq[np.arange(len(qq)), idx]
    hi = qq[np.arange(len(qq)), idx + 1]
    span = np.where(hi > lo, hi - lo, np.nan)
    t = (yy - lo) / span
    t = np.where(np.isfinite(t), t, 0.5)
    pit = levels[idx] + t * (levels[idx + 1] - levels[idx])
    out[ok] = np.clip(pit, 0.0, 1.0)
    return out


def central_interval(levels, q, mass: float):
    """(lower, upper) of the central ``mass`` interval, if both levels exist."""
    levels = np.asarray(levels, np.float64)
    lo_l, hi_l = 0.5 - mass / 2, 0.5 + mass / 2
    i = np.flatnonzero(np.isclose(levels, lo_l, atol=2e-3))
    j = np.flatnonzero(np.isclose(levels, hi_l, atol=2e-3))
    if not len(i) or not len(j):
        return None
    return q[:, i[0]], q[:, j[0]]


def paired_bootstrap(a, b, n_boot: int = 500, seed: int = 0) -> dict:
    """Median |a| - median |b| with a percentile bootstrap 95 % interval."""
    a, b = np.abs(np.asarray(a, np.float64)), np.abs(np.asarray(b, np.float64))
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < MIN_N:
        return dict(n=int(len(a)), delta=float("nan"), lo=float("nan"), hi=float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return dict(
        n=int(len(a)),
        delta=float(np.median(a) - np.median(b)),
        lo=float(np.percentile(d, 2.5)),
        hi=float(np.percentile(d, 97.5)),
    )
