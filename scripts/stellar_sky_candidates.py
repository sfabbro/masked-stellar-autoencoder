"""Auditable spherical coordinate candidates; never fabricate global Gaia identity."""

import numpy as np
from scipy.spatial import cKDTree


def unit_vectors(ra_degrees, dec_degrees):
    ra, dec = np.asarray(ra_degrees, dtype="f8"), np.asarray(dec_degrees, dtype="f8")
    if (
        ra.ndim != 1
        or dec.shape != ra.shape
        or not (np.isfinite(ra).all() and np.isfinite(dec).all())
    ):
        raise ValueError("Require finite one-dimensional coordinates")
    if (ra < 0).any() or (ra >= 360).any() or (np.abs(dec) > 90).any():
        raise ValueError(
            "ICRS candidate coordinates require RA[0,360) and Dec[-90,90] degrees"
        )
    a, d = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])


def nearest_two(reference_ra, reference_dec, query_ra, query_dec, *, workers=2):
    """Nearest two angular neighbors in the PROVIDED catalogue, not the full sky."""
    reference = unit_vectors(reference_ra, reference_dec)
    query = unit_vectors(query_ra, query_dec)
    if len(reference) < 2:
        raise ValueError("Second-neighbor audit requires at least two reference rows")
    if not len(query):
        return np.empty((0, 2), dtype="i8"), np.empty((0, 2), dtype="f8")
    distance, row = cKDTree(reference).query(query, k=2, workers=workers)
    separation = np.rad2deg(2 * np.arcsin(np.clip(distance / 2, 0, 1))) * 3600
    return row.astype("i8"), separation


def unambiguous_candidates(
    rows, separation, *, radius_arcsec=1.0, isolation_arcsec=2.0
):
    """Conservative local-catalogue candidates, including many-to-one rejection."""
    rows = np.asarray(rows)
    separation = np.asarray(separation)
    if rows.shape != separation.shape or rows.ndim != 2 or rows.shape[1] != 2:
        raise ValueError("Need matched nearest/second-neighbor arrays")
    near = separation[:, 0] <= radius_arcsec
    unique, counts = np.unique(rows[near, 0], return_counts=True)
    repeated = unique[counts > 1]
    return (
        near & (separation[:, 1] >= isolation_arcsec) & ~np.isin(rows[:, 0], repeated)
    )
