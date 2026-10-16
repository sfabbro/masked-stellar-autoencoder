"""Explicit Hipparcos reference eligibility and two-dimensional epoch checks."""

import numpy as np


def ordinary_reference_mask(sn, nc, f2, f1, parallax, error, neighbors, xm_flag):
    """Design QC, not calibrated accuracy. Negative parallaxes remain eligible.

    ESA xm_flag8 is ordinary five-parameter Gaia astrometry;16 is two-parameter
    and72 includes special treatment for highRUWE/ipd_gof_harmonic_amplitude.
    """
    return (
        (np.asarray(sn) == 5)
        & (np.asarray(nc) == 1)
        & np.isfinite(f2)
        & (np.asarray(f2) < 3)
        & (np.asarray(f1) <= 10)
        & np.isfinite(parallax)
        & np.isfinite(error)
        & (np.asarray(error) > 0)
        & (np.asarray(neighbors) == 1)
        & (np.asarray(xm_flag) == 8)
    )


def propagate_position(ra_radians, dec_radians, pmra_mas_year, pmdec_mas_year, years):
    """ICRS direction at new epoch, zero radial perspective (no distance assumed).

    pmra includes cos(dec). Output is a normalized Cartesian direction, not
    propagated covariance. The15Hipparcos weight elements are not treated as
    absolute covariance without verifying their normalization convention.
    """
    a, d = np.asarray(ra_radians), np.asarray(dec_radians)
    direction = np.column_stack(
        [np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]
    )
    tangent_ra = np.column_stack([-np.sin(a), np.cos(a), np.zeros(len(a))])
    tangent_dec = np.column_stack(
        [-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)]
    )
    motion = (
        np.asarray(pmra_mas_year)[:, None] * tangent_ra
        + np.asarray(pmdec_mas_year)[:, None] * tangent_dec
    )
    moved = direction + np.deg2rad(1 / 3600000) * years * motion
    return moved / np.linalg.norm(moved, axis=1)[:, None]
