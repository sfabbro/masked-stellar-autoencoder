"""Sample-based marginal and joint reference-compatibility scores."""

import numpy as np


def sample_scores(draws, target, observed):
    x = np.asarray(draws, dtype="f8")
    y = np.asarray(target, dtype="f8")
    mask = np.asarray(observed, dtype=bool)
    if (
        x.ndim != 3
        or x.shape[1:] != y.shape
        or y.shape != mask.shape
        or x.shape[0] < 2
        or not np.isfinite(x).all()
    ):
        raise ValueError("Finite joint draws and aligned targets/masks required")
    safe = np.where(mask, y, 0)
    count = len(x)
    ordered = np.sort(x, axis=0)
    coefficients = (2 * np.arange(1, count + 1) - count - 1)[:, None, None]
    crps = np.mean(abs(x - safe[None]), axis=0) - np.sum(
        coefficients * ordered, axis=0
    ) / (count * count)
    half = count // 2
    first = x[:half]
    second = x[half : 2 * half]
    delta = (x - safe[None]) * mask[None]
    pairs = (first - second) * mask[None]
    energy = np.linalg.norm(delta, axis=2).mean(0) - 0.5 * np.linalg.norm(
        pairs, axis=2
    ).mean(0)
    quantiles = np.quantile(
        x, [0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975], axis=0
    )
    return dict(
        crps=np.where(mask, crps, np.nan),
        energy=np.where(mask.any(1), energy, np.nan),
        pit=np.where(mask, np.mean(x <= safe[None], axis=0), np.nan),
        quantiles=quantiles,
    )
