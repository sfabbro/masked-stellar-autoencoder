"""Exact NumPy propagation of exported monotone own-target map coefficients."""

import numpy as np


def map_axis(value, target, source, coefficients):
    x = np.asarray(value, dtype="f8")
    center, scale = coefficients["center"], coefficients["scale"]
    if target >= 4:
        if (x <= 0).any():
            raise ValueError("Teacher age/distance coordinates must be positive")
        x = np.log(x)
    x = (x - center[target]) / scale[target]
    b = coefficients["bias"][source, target]
    slopes = coefficients["slopes"][source, target]
    kind = str(coefficients["kind"])
    if kind == "offset":
        y = x + b
    elif kind == "affine":
        y = x * slopes[0] + b
    elif kind == "spline":
        width = 8 / len(slopes)
        index = np.clip(np.floor((x + 4) / width).astype("i8"), 0, len(slopes) - 1)
        cumulative = np.r_[0, np.cumsum(slopes) * width]
        y = (
            cumulative[index]
            + slopes[index] * (x - (-4 + index * width))
            - cumulative[len(slopes) // 2]
            + b
        )
    else:
        raise ValueError("Unknown exported map kind")
    y = y * scale[target] + center[target]
    return np.exp(y) if target >= 4 else y


def inverse_map_axis(value, target, source, coefficients):
    """Map a native observation into common coordinates, retaining measurement noise.

    This deterministic inverse is a reporting coordinate change, not an estimate of
    a noise-free latent truth or an inverse conditional probability distribution.
    Linear extrapolation uses the same positive end slopes as the forward map.
    """
    y = np.asarray(value, dtype="f8")
    center, scale = coefficients["center"], coefficients["scale"]
    if not np.isfinite(y).all():
        raise ValueError("Inverse map accepts observed finite coordinates only")
    if target >= 4:
        if (y <= 0).any():
            raise ValueError("Teacher age/distance coordinates must be positive")
        y = np.log(y)
    y = (y - center[target]) / scale[target]
    b = coefficients["bias"][source, target]
    slopes = np.asarray(coefficients["slopes"][source, target], dtype="f8")
    if not np.isfinite(slopes).all() or (slopes <= 0).any():
        raise ValueError("Observation map must have finite positive slopes")
    kind = str(coefficients["kind"])
    if kind == "offset":
        x = y - b
    elif kind == "affine":
        x = (y - b) / slopes[0]
    elif kind == "spline":
        width = 8 / len(slopes)
        cumulative = np.r_[0, np.cumsum(slopes) * width]
        knots = cumulative - cumulative[len(slopes) // 2] + b
        index = np.clip(np.searchsorted(knots, y, side="right") - 1, 0, len(slopes) - 1)
        x = -4 + index * width + (y - knots[index]) / slopes[index]
    else:
        raise ValueError("Unknown exported map kind")
    x = x * scale[target] + center[target]
    return np.exp(x) if target >= 4 else x
