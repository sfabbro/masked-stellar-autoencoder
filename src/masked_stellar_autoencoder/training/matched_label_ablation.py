"""Keep sampling and non-dropped observations fixed when removing DESI logg."""

import numpy as np


def desi_logg_masks(observed, known, drop):
    observed = np.asarray(observed, dtype=bool)
    known = np.asarray(known, dtype=bool)
    if (
        observed.ndim != 2
        or observed.shape[1] != 6
        or known.shape != observed.shape
        or (known & ~observed).any()
    ):
        raise ValueError("Aligned six-target masks required")
    sampling_logg = observed[:, 1].copy()
    eligible = observed[:, [0, 2, 3, 4, 5]].any(1)
    mask = observed.copy()
    errors = known.copy()
    if drop:
        mask[:, 1] = False
        errors[:, 1] = False
    return mask, errors, sampling_logg, eligible


def gravity_pool(sampling_observed, physical_logg, update, group_count):
    sampling_observed = np.asarray(sampling_observed, bool)
    physical_logg = np.asarray(physical_logg)
    if sampling_observed.shape != physical_logg.shape or physical_logg.ndim != 1:
        raise ValueError("Aligned immutable sampling coordinates required")
    pool = np.arange(len(physical_logg))
    if sampling_observed.any() and update % 2:
        bins = np.searchsorted([1, 2, 3, 4, 4.5], physical_logg)
        bands = np.unique(bins[sampling_observed])
        band = bands[(update // group_count) % len(bands)]
        pool = pool[sampling_observed & (bins == band)]
    return pool
