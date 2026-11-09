"""Observation policy for a full-Gaia model; independent of XP availability.

This module is opt-in. It does not change the frozen input contract of existing
runs. The encoder/reconstruction use all declared photometric bands and XP;
auxiliary measurements are explicitly exposed to the predictive head.
"""

from dataclasses import dataclass
import numpy as np

PHOTOMETRY = (
    "W1",
    "W2",
    "G",
    "BP",
    "RP",
    "U_SMSS",
    "V_SMSS",
    "G_SMSS",
    "R_SMSS",
    "I_SMSS",
    "Z_SMSS",
    "U_SDSS",
    "G_SDSS",
    "R_SDSS",
    "I_SDSS",
    "Z_SDSS",
    "G_PS1",
    "R_PS1",
    "I_PS1",
    "Z_PS1",
    "Y_PS1",
    "J",
    "H",
    "KS",
)
AUXILIARY = ("PARALLAX", "EBV", "pmra", "pmdec", "RA", "DEC")
XP = tuple(f"{arm}_{j}" for arm in ("bp", "rp") for j in range(1, 56))


@dataclass(frozen=True)
class InputPolicy:
    feature_names: tuple
    encoder: tuple
    reconstruction: tuple
    auxiliary: tuple


def full_gaia_policy(feature_names):
    names = tuple(feature_names)
    if len(set(names)) != len(names):
        raise ValueError("Unique observation columns required")
    if not set(XP + PHOTOMETRY + AUXILIARY).issubset(names):
        raise ValueError(
            "Declare full photometry, XP and auxiliary layout; unavailable values may be missing"
        )
    enc = tuple(n in set(XP + PHOTOMETRY) for n in names)
    return InputPolicy(names, enc, enc, tuple(names.index(n) for n in AUXILIARY))


def visible_observations(values):
    a = np.asarray(values)
    return np.isfinite(a) & (a != -9999)


def input_eligible(values, policy):
    """Observed photometry OR XP admits a star; never intersect labels with XP."""
    a = np.asarray(values)
    if a.ndim != 2 or a.shape[1] != len(policy.feature_names):
        raise ValueError("Input layout differs")
    return visible_observations(a)[:, np.asarray(policy.encoder)].any(1)


def auxiliary_context(raw_values, raw_errors, feature_names):
    """Raw-unit auxiliary values/errors with masks and wrap-free sky position.

    Values must be normalized later using FIT-only statistics. Missing errors
    are masked, never represented as known zero. RA/DEC in degrees, PM in
    mas/year (Gaia pmra already includes cos(dec)), parallax in mas, EBV in mag.
    No catalogue age, distance or atmospheric target enters this context.
    """
    names = tuple(feature_names)
    a = np.asarray(raw_values, dtype="f8")
    e = np.asarray(raw_errors, dtype="f8")
    if a.ndim != 2 or e.shape != a.shape or a.shape[1] != len(names):
        raise ValueError("Aligned raw values and errors required")
    columns = []
    observed = []
    labels = []
    for name in AUXILIARY[:4]:
        j = names.index(name)
        ok = visible_observations(a[:, j])
        columns.append(np.where(ok, a[:, j], 0))
        observed.append(ok)
        labels.append(name)
        known = ok & np.isfinite(e[:, j]) & (e[:, j] > 0)
        columns.append(np.where(known, e[:, j], 0))
        observed.append(known)
        labels.append("sigma_" + name)
    ra, dec = a[:, names.index("RA")], a[:, names.index("DEC")]
    good = visible_observations(ra) & visible_observations(dec)
    if (abs(dec[good]) > 90).any():
        raise ValueError("Declination outside physical range")
    r = np.deg2rad(np.where(good, ra, 0))
    d = np.deg2rad(np.where(good, dec, 0))
    for name, v in zip(
        ["sky_x", "sky_y", "sky_z"],
        [np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)],
        strict=True,
    ):
        columns.append(np.where(good, v, 0))
        observed.append(good)
        labels.append(name)
    return (
        np.column_stack(columns).astype("f4"),
        np.column_stack(observed),
        tuple(labels),
    )
