"""Full record passes and fixed-parent target coordinates for expanded training."""

import numpy as np


def epoch_record_batches(lengths, batch_size, rng):
    lengths = np.asarray(lengths, dtype="i8")
    if (
        lengths.ndim != 1
        or (lengths < 0).any()
        or not lengths.sum() > 0
        or batch_size < 1
    ):
        raise ValueError("Nonnegative source sizes, positive total and batch required")
    bounds = np.cumsum(lengths)
    offsets = np.r_[0, bounds[:-1]]
    order = rng.permutation(int(bounds[-1]))
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        source = np.searchsorted(bounds, indices, side="right")
        yield source, indices - offsets[source]


def fixed_coordinates(value, error, observed, known, center, scale):
    value = np.asarray(value, dtype="f8")
    error = np.asarray(error, dtype="f8")
    observed = np.asarray(observed, bool)
    known = np.asarray(known, bool)
    center = np.asarray(center)
    scale = np.asarray(scale)
    if (
        value.ndim != 2
        or value.shape[1] != 6
        or any(v.shape != value.shape for v in [error, observed, known])
        or center.shape != (6,)
        or scale.shape != (6,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "Aligned six-target records and fixed finite positive normalization required"
        )
    if (
        not np.isfinite(value[observed]).all()
        or (known & ~observed).any()
        or (known[:, 4:]).any()
        or not np.isfinite(error[known]).all()
        or (error[known] <= 0).any()
    ):
        raise ValueError(
            "Invalid observed labels, known errors, or posterior widths mislabelled as errors"
        )
    valid = observed.copy()
    valid[:, 4:] &= value[:, 4:] > 0
    coordinate = np.where(valid, value, 0.0)
    coordinate[:, 4:] = np.log(np.where(valid[:, 4:], value[:, 4:], 1.0))
    y = np.where(valid, (coordinate - center) / scale, 0).astype("f4")
    sigma = np.where(known, np.where(np.isfinite(error), error, 0) / scale, 0).astype(
        "f4"
    )
    return y, sigma, valid, known & valid


def reconstruction_block_start(length, batch_size, rng):
    """Sample a bounded contiguous mmap block across the full cache, not a prefix."""
    if batch_size < 1 or length < batch_size:
        raise ValueError("Reconstruction cache must fit the declared batch")
    return int(rng.integers(0, length - batch_size + 1))
