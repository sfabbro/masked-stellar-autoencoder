"""Separate CAL-based central intervals from an unchanged predictive density."""

import numpy as np


def central_mass(pit, coverage):
    """Finite-sample split-conformal threshold for |2 F_hat(y|x)-1|.

    Exchangeability is required. Monte Carlo CDF scores must be constructed in
    the same way for CAL and evaluation. This does not calibrate a joint PDF.
    """
    p = np.asarray(pit, dtype=float)
    if p.ndim != 1 or not 0 < coverage < 1 or not len(p):
        raise ValueError("Nonempty scalar CAL scores and coverage in (0,1) required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("CDF scores must be finite in [0,1]")
    rank = int(np.ceil((len(p) + 1) * coverage))
    if rank > len(p):
        return float("inf")
    return float(np.sort(abs(2 * p - 1))[rank - 1])


def central_bounds(quantiles, grid, mass):
    """Interpolate saved marginal quantiles; an unattainable rank is unbounded."""
    q, g = np.asarray(quantiles), np.asarray(grid)
    if (
        q.ndim != 2
        or q.shape[0] != len(g)
        or g[0] != 0
        or g[-1] != 1
        or (np.diff(g) <= 0).any()
    ):
        raise ValueError("Require quantiles [grid,star] on a sorted complete CDF grid")
    if not np.isfinite(q).all() or (np.diff(q, axis=0) < 0).any():
        raise ValueError("Require finite ordered quantiles")
    # With an empirical CDF, mass=1 also admits values outside every draw:
    # |2 F_hat(y)-1| <= 1 holds on the entire real line.
    if np.isposinf(mass) or mass == 1:
        return np.full(q.shape[1], -np.inf), np.full(q.shape[1], np.inf)
    if not 0 <= mass <= 1:
        raise ValueError("Central mass must be in [0,1] or positive infinity")
    levels = ((1 - mass) / 2, (1 + mass) / 2)
    return tuple(
        np.array([np.interp(level, g, row) for row in q.T]) for level in levels
    )
