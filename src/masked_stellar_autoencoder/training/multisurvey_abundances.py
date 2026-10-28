"""Explicit abundance-coordinate conversions with covariance provenance."""

from __future__ import annotations

import numpy as np


def iron_from_global_and_relative(
    metallicity_mh, neutral_iron_over_m, *, reference_offset_dex, covariance=None
):
    """Form an Fe I/H estimate from [M/H] + [Fe I/M] in dex.

    The caller must explicitly supply the additive solar/reference-convention
    offset (zero only if compatible conventions are justified). This arithmetic
    does not homogenize spectral pipelines, ionization states, or systematic
    scales. A finite 2x2 covariance produces a propagated variance including
    the cross term; otherwise uncertainty is unknown, never silently zero.
    Published 16th/84th percentiles alone are not a joint covariance.
    """
    if isinstance(reference_offset_dex, bool | np.bool_):
        raise ValueError("An explicit numeric reference offset is required")
    try:
        offset = float(reference_offset_dex)
    except (TypeError, ValueError) as error:
        raise ValueError("An explicit finite reference offset is required") from error
    if not np.isfinite(offset):
        raise ValueError("An explicit finite reference offset is required")
    m, f = np.broadcast_arrays(
        np.asarray(metallicity_mh, dtype=np.float64),
        np.asarray(neutral_iron_over_m, dtype=np.float64),
    )
    observed = np.isfinite(m) & np.isfinite(f) & (m != -9999) & (f != -9999)
    value = np.where(observed, m + f + offset, np.nan)
    variance = np.full(m.shape, np.nan, dtype=np.float64)
    if covariance is None:
        return {"value": value, "observed": observed, "variance": variance}
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (*m.shape, 2, 2):
        raise ValueError("Covariance must be aligned [...,2,2]")
    if np.isinf(cov).any():
        raise ValueError("Unknown covariance must use NaN, not infinity")
    known = np.isfinite(cov).all(axis=(-2, -1))
    complete = cov[known]
    if len(complete):
        if not np.allclose(complete, complete.swapaxes(-1, -2), rtol=1e-12, atol=1e-12):
            raise ValueError("Covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(complete)
        tolerance = 1e-12 * np.maximum(1.0, np.max(np.abs(complete), axis=(-2, -1)))
        if np.any(eigenvalues[:, 0] < -tolerance):
            raise ValueError("Covariance must be positive semidefinite")
        propagated = complete[:, 0, 0] + complete[:, 1, 1] + 2 * complete[:, 0, 1]
        if np.any(propagated < -2 * tolerance):
            raise ValueError("Propagated variance cannot be negative")
        variance[known] = np.maximum(propagated, 0)
    variance = np.where(observed, variance, np.nan)
    return {"value": value, "observed": observed, "variance": variance}
