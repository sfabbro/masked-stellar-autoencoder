"""CPU diagnostics for exported Gaussian mixtures; never fits or recalibrates."""

import numpy as np
from scipy.special import logsumexp, ndtr


def marginal_scores(weights, means, variances, values, errors=None):
    """Exact mixture marginal PIT, log score and CRPS.

    Optional errors explicitly assume independent Gaussian reference noise.
    Scores are marginal, not joint credible-region diagnostics.
    """
    w, m, v = (np.asarray(x, dtype=float) for x in (weights, means, variances))
    y = np.asarray(values, dtype=float)
    if (
        w.ndim != 2
        or m.shape != w.shape
        or v.shape != w.shape
        or y.shape != w.shape[:1]
    ):
        raise ValueError("Expected aligned N,K component arrays and N values")
    if not all(np.isfinite(x).all() for x in (w, m, v, y)):
        raise ValueError("Inputs must be finite")
    if (w < 0).any() or not np.allclose(w.sum(1), 1) or (v <= 0).any():
        raise ValueError("Invalid weights or variances")
    if errors is not None:
        e = np.asarray(errors, dtype=float)
        if e.shape != y.shape or not np.isfinite(e).all() or (e < 0).any():
            raise ValueError("Invalid reference error")
        v = v + e[:, None] ** 2
    s = np.sqrt(v)
    z = (y[:, None] - m) / s
    pit = (w * ndtr(z)).sum(1)
    with np.errstate(divide="ignore"):
        logp = logsumexp(
            np.log(w) - np.log(s) - 0.5 * z**2 - 0.5 * np.log(2 * np.pi), axis=1
        )

    def absolute_normal(delta, sigma):
        q = delta / sigma
        return 2 * sigma * np.exp(-0.5 * q**2) / np.sqrt(2 * np.pi) + delta * (
            2 * ndtr(q) - 1
        )

    a = absolute_normal(y[:, None] - m, s)
    b = absolute_normal(
        m[:, :, None] - m[:, None, :], np.sqrt(v[:, :, None] + v[:, None, :])
    )
    crps = (w * a).sum(1) - 0.5 * (w[:, :, None] * w[:, None, :] * b).sum((1, 2))
    return dict(pit=pit, nll=-logp, crps=crps)


def summarize_residuals(prediction, reference):
    r = np.asarray(prediction) - np.asarray(reference)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError("Expected finite nonempty residual vector")
    median = np.median(r)
    return dict(
        n=len(r),
        bias=float(median),
        mae=float(np.mean(abs(r))),
        nmad=float(1.4826 * np.median(abs(r - median))),
        rmse=float(np.sqrt(np.mean(r**2))),
        p05=float(np.quantile(r, 0.05)),
        p95=float(np.quantile(r, 0.95)),
    )


def summarize_scores(scores):
    p = scores["pit"]
    return dict(
        n=len(p),
        nll=float(np.mean(scores["nll"])),
        crps=float(np.mean(scores["crps"])),
        coverage68=float(np.mean((p >= 0.16) & (p <= 0.84))),
        coverage90=float(np.mean((p >= 0.05) & (p <= 0.95))),
        coverage95=float(np.mean((p >= 0.025) & (p <= 0.975))),
        pit_histogram=np.histogram(p, bins=np.linspace(0, 1, 11))[0].tolist(),
    )


def affine_survey_components(
    component_mean, component_lower, reference_mean, offsets, derivatives
):
    """Push every component through the exact affine observation model used in fitting.

    Derivatives refer to the correction: the full Jacobian is I + derivative.
    Survey coordinates are linearized at the reference mean, not the mixture mean.
    """
    mu, lower, ref, delta, deriv = map(
        np.asarray,
        (component_mean, component_lower, reference_mean, offsets, derivatives),
    )
    n, components, dim = mu.shape
    if lower.shape != (n, components, dim, dim) or ref.shape != (n, dim):
        raise ValueError("Component/reference layout differs")
    if (
        delta.ndim != 3
        or delta.shape[0] != n
        or delta.shape[2] != dim
        or deriv.shape != (*delta.shape[:2], dim, dim)
    ):
        raise ValueError("Survey correction layout differs")
    jacobian = deriv + np.eye(dim)[None, None]
    mapped = (
        ref[:, None, None]
        + delta[:, :, None]
        + np.einsum("bsij,bcj->bsci", jacobian, mu - ref[:, None])
    )
    factors = np.einsum("bsij,bcjk->bscik", jacobian, lower)
    return mapped, factors
