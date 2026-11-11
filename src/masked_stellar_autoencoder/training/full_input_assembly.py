"""Full encoder inputs and head metadata context from staged ARC catalogues.

The pretrained encoder has 140 input slots (``configs/pretrain.canfar.yaml``):
W1, W2, G, BP, RP, 110 XP coefficients, SkyMapper, SDSS, PS1, 2MASS,
PARALLAX, EBV, pmra, pmdec, RA, DEC. Earlier labelled sources filled only XP
and five magnitudes (``evaluation_feature_lookup`` ``allowed``). This module
rebuilds every slot from what is already staged:

* the original pretraining rows (``andrae2023/sslset-realmags-full``), either
  through the photometry stage parquet (``raw-observations``) or by direct row
  pointers from a feature-recovery run;
* Gaia DR3 columns (``full_gaia_gaia_stage .../labelled-gaia``) for stars
  outside the XP sample: G, BP, RP, parallax, proper motions, RA/Dec.

Missing values are stored as NaN with an explicit ``visible`` mask. The encoder's
-9999 missing token is applied only where inputs are loaded for the model.
The head context is a small per-star metadata vector (parallax and error,
EBV, proper motions and errors, unit sky vector), each with a mask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MISSING_TOKEN = -9999.0

# Gaia DR3 column -> encoder input slot (same units as the pretraining data).
GAIA_TO_FEATURE = {
    "phot_g_mean_mag": "G",
    "phot_bp_mean_mag": "BP",
    "phot_rp_mean_mag": "RP",
    "parallax": "PARALLAX",
    "pmra": "pmra",
    "pmdec": "pmdec",
    "ra": "RA",
    "dec": "DEC",
}

CONTEXT_NAMES = (
    "parallax",
    "log10_parallax_error",
    "ebv",
    "pmra",
    "pmdec",
    "log10_pmra_error",
    "log10_pmdec_error",
    "sky_x",
    "sky_y",
    "sky_z",
)


def clean(values) -> np.ndarray:
    """Float array with native missing sentinels (-9999, non-finite) as NaN."""
    out = np.array(values, dtype="f8", copy=True)
    out[~np.isfinite(out) | (out == MISSING_TOKEN)] = np.nan
    return out


def standardize(raw, center, scale) -> np.ndarray:
    """Encoder-scaled inputs (raw - center) / scale with NaN for missing."""
    raw = clean(raw)
    center = np.asarray(center, dtype="f8")
    scale = np.asarray(scale, dtype="f8")
    if raw.shape[-1] != len(center) or np.any(~np.isfinite(scale) | (scale <= 0)):
        raise ValueError("Scaler must match the 140-slot layout with positive scale")
    return ((raw - center) / scale).astype("f4")


def to_encoder(values) -> np.ndarray:
    """Model-boundary conversion: NaN -> the encoder's -9999 missing token."""
    out = np.array(values, dtype="f4", copy=True)
    out[~np.isfinite(out)] = MISSING_TOKEN
    return out


def raw_frame_to_matrix(frame: pd.DataFrame, names) -> np.ndarray:
    """Raw-unit [N, len(names)] matrix from a staged parquet frame (NaN missing)."""
    out = np.full((len(frame), len(names)), np.nan)
    for j, name in enumerate(names):
        if name in frame:
            out[:, j] = clean(pd.to_numeric(frame[name], errors="coerce"))
    return out


def gaia_frame_to_matrix(frame: pd.DataFrame, names) -> np.ndarray:
    """Raw-unit matrix for the slots Gaia DR3 provides; other slots NaN."""
    out = np.full((len(frame), len(names)), np.nan)
    index = {n: j for j, n in enumerate(names)}
    for column, feature in GAIA_TO_FEATURE.items():
        if column in frame and feature in index:
            out[:, index[feature]] = clean(
                pd.to_numeric(frame[column], errors="coerce")
            )
    return out


def merge_rows(base, update, *, overwrite: bool) -> np.ndarray:
    """Combine standardized rows: ``overwrite`` replaces wherever update is visible,
    otherwise update only fills slots that are missing in ``base``."""
    base = np.asarray(base, dtype="f4")
    update = np.asarray(update, dtype="f4")
    take = (
        np.isfinite(update) if overwrite else np.isfinite(update) & ~np.isfinite(base)
    )
    return np.where(take, update, base)


def unit_sky(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(clean(ra_deg))
    dec = np.radians(clean(dec_deg))
    return np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1
    )


def context_raw(
    *, parallax, parallax_error, ebv, pmra, pmdec, pmra_error, pmdec_error, ra, dec
):
    """Unscaled metadata context [N, 10]; NaN where unavailable."""

    def log_error(e):
        e = clean(e)
        out = np.full(e.shape, np.nan)
        good = e > 0
        out[good] = np.log10(e[good])
        return out

    return np.column_stack(
        [
            clean(parallax),
            log_error(parallax_error),
            clean(ebv),
            clean(pmra),
            clean(pmdec),
            log_error(pmra_error),
            log_error(pmdec_error),
            unit_sky(ra, dec),
        ]
    )


def context_scaler(values, fit_rows):
    """Robust per-column center/scale (median, 1.4826 MAD) over FIT stars.

    Sky unit-vector components are left unscaled (center 0, scale 1).
    """
    values = np.asarray(values, dtype="f8")
    center = np.zeros(values.shape[1])
    scale = np.ones(values.shape[1])
    for j, name in enumerate(CONTEXT_NAMES):
        if name.startswith("sky_"):
            continue
        column = values[fit_rows, j]
        column = column[np.isfinite(column)]
        if len(column) == 0:
            continue
        center[j] = np.median(column)
        mad = 1.4826 * np.median(np.abs(column - center[j]))
        scale[j] = mad if mad > 0 else 1.0
    return center, scale
