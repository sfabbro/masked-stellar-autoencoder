"""Six-target coordinates and physical-unit density conversions.

Optional density-scoring utilities, not a requirement that every compared
uncertainty method expose a normalized density or target-on-target queries.
"""

import numpy as np

TARGETS = ("teff", "logg", "fe_h", "alpha", "age", "distance")
UNITS = ("K", "dex(cm/s2)", "dex", "dex", "Gyr", "kpc")


def target_axes(targets):
    """Validate a declared subset of the six output quantities."""
    targets = tuple(targets)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Request nonempty unique target axes")
    if not set(targets) <= set(TARGETS):
        raise ValueError("Unknown stellar target quantity")
    return tuple(i for i, t in enumerate(TARGETS) if t in targets)


def standardized_coordinates(native, center, scale):
    """Linear first four, ln(age/Gyr), ln(distance/kpc); no clipping or imputation."""
    native, center, scale = (np.asarray(v, dtype="f8") for v in (native, center, scale))
    if native.shape[-1:] != (6,) or center.shape != (6,) or scale.shape != (6,):
        raise ValueError("Require native [...,6] and center/scale [6]")
    if (
        not all(np.isfinite(v).all() for v in (native, center, scale))
        or (scale <= 0).any()
    ):
        raise ValueError("Finite coordinates and strictly positive scales required")
    if (native[..., 4:] <= 0).any():
        raise ValueError("Age and distance must be positive in declared physical units")
    coordinate = native.copy()
    coordinate[..., 4:] = np.log(coordinate[..., 4:])
    log_jacobian = np.broadcast_to(-np.log(scale), native.shape).copy()
    log_jacobian[..., 4:] -= np.log(native[..., 4:])
    return (coordinate - center) / scale, log_jacobian


def physical_log_density(log_density_model, coordinate_log_jacobian, targets):
    """Apply physical-coordinate Jacobians to the specified density axes.

    Density scoring is optional for region methods. For a joint density,
    each transformed physical target contributes its Jacobian exactly once.
    """
    axes = target_axes(targets)
    jac = np.asarray(coordinate_log_jacobian, dtype="f8")
    if jac.shape[-1:] != (6,) or not np.isfinite(jac).all():
        raise ValueError("Require finite per-coordinate Jacobians [...,6]")
    logp = np.asarray(log_density_model, dtype="f8")
    if logp.shape != jac.shape[:-1]:
        raise ValueError("One model log density per query context")
    if np.isnan(logp).any() or np.isposinf(logp).any():
        raise ValueError("Invalid model log density")
    return logp + jac[..., axes].sum(-1)
