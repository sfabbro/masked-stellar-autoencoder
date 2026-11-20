"""Data-driven forward model of Gaia XP spectra, photometry and parallax (F2).

A prior-free per-star likelihood for (Teff, log g, [Fe/H]) with the published noise
model, in the spirit of Zhang, Green & Rix (2023, MNRAS 524, 1855) and of the
learned intrinsic scatter of Laroche & Speagle (2023, 2024). Design note:
``docs/design-20261120-likelihood-inference.md`` (section 3, "Alternative for the faint
end", and section 4.1).

Emulator
--------
``g(theta, E) -> (c(theta, E), M(theta, E))`` with ``theta = (Teff, log g, [Fe/H])``
and ``E = E(B-V)``:

* ``c`` are the 110 XP coefficients (BP 1..55, RP 1..55) of the star normalised to
  G = 8.5 (the table's units: ``c = c_raw / f``, ``f = 10**((8.5 - G) / 2.5)``),
  divided by a fixed per-coefficient scale ``s_j`` (median ``|c_j|`` of the training
  stars) so the network works in O(1) units:
  ``c(theta, E) = h0(theta) + E * h1(theta, E)``;
* ``M_b`` are absolute magnitudes in the photometric bands:
  ``M_b(theta, E) = mu_b + sigma_b * h0_b(theta) + E * softplus(h1_b(theta, E))``,
  so the extinction ``A_b = E * softplus(.) >= 0`` but its wavelength dependence
  (the extinction law) and its dependence on the spectrum are learned from the data:
  no stellar-atmosphere grid and no fixed extinction law.

``h0`` and ``h1`` are SiLU MLPs (smooth, so exact Hessians exist for Newton
inference).

Likelihood per star
-------------------
``-log L = NLL_XP + NLL_phot + NLL_parallax`` with

* XP, per band (BP and RP are independent: Gaia publishes no cross covariance):
  ``c_obs - c(theta, E) ~ N(0, C_known + diag((a * c)^2 + b^2) + (c * V)(c * V)^T)``,
  ``C_known`` the per-star Gaia covariance in table units (sidecar) or its diagonal
  from the table's errors (fallback), ``a`` (fractional), ``b`` (floor) and the
  fractional rank-``k`` factor ``V`` learned
  (:class:`torchregress.losses.KnownCovarianceGaussianNLL`). The XP term is
  conditional on the observed G used for the normalisation; the G-flux error adds a
  rank-one ``x x^T / (flux_over_error)^2`` term (``< 1e-5`` relative for G < 17),
  which the learned fractional factor absorbs.
* photometry: ``m_b = M_b(theta, E) + 5 log10(100 / parallax[mas])``,
  ``m_obs - m ~ N(0, sigma_b^2 + s_b^2)`` per band, ``sigma_b`` the quoted error (0
  where the table has none: G, W1, W2) and ``s_b`` a learned per-band scatter (in
  magnitudes, i.e. a fractional flux scatter). BP and RP magnitudes are left out by
  default: they are integrals of the XP spectra and would double count them.
* parallax: ``parallax_obs - zp ~ N(parallax, sigma_parallax^2)``, ``zp`` from
  Lindegren et al. (2021) via the covariance sidecar, else a constant placeholder
  (``--parallax-zero-point constant``, -0.017 mas, the L21 quasar median), else 0.

Missing coefficients, bands and parallaxes are marginalised (masked dimensions drop
out of the Gaussian exactly).

Positions and kinematics. RA/Dec enter only through the SFD reddening bound on ``E``
(the table's ``EBV`` column) and the parallax zero point (ecliptic latitude). Proper
motions are *not* part of this likelihood: a kinematic term would be a population
prior on the stellar parameters (velocity distributions differ between populations of
different [Fe/H] and age), which this per-star measurement must not contain.

Nuisance parameters
-------------------
``E`` and the parallax are nuisance parameters of every star, mapped to the unit box
so training and inference share one parametrisation:
``E = u_E * E_max``, ``E_max = 1.2 * max(EBV_SFD, 0) + 0.05``, and
``log10 parallax = lo + u_p * (hi - lo)`` with ``[lo, hi]`` = the corrected parallax
+- 6 sigma clipped to ``[0.005 mas, ...]`` (``[0.005, 100]`` mas without a parallax).
During training they are free per-star parameters (``nn.Embedding`` tables in logit
space, initialised at SFD and at the observed parallax, sparse Adam), constrained by
the parallax likelihood and the box. At inference they are profiled.

Training and label noise
------------------------
Training maximises the joint likelihood of the labelled stars over the emulator, the
scatter terms and the per-star nuisances with the labels plugged in
(``label_noise="plugin"``). ``label_noise="marginal"`` is the documented hook for
integrating ``theta`` over its label error: ``-log mean_k L(theta_k)``,
``theta_k ~ N(label, e_label^2)`` (an importance-weighted lower bound of the marginal
likelihood that tightens with ``k``; it deconvolves the emulator from label noise).
The learned scatter terms are frozen for ``scatter_warmup_epochs`` epochs with fresh
optimiser state (a variance learned against an untrained mean absorbs the misfit).

Inference
---------
Per star, ``maximize_score`` over ``(Teff, log g, [Fe/H], u_E, u_p)`` in a box (the
labels' range for ``theta``), started from the best point of a coarse candidate bank
(labelled population ``theta`` x an ``E`` grid, scored with the covariance fixed at
the observed coefficients and the distance modulus profiled analytically), then
``profile_interval`` (Wilks 68.27 % and 95 %) for the three atmospheric parameters
with the other four profiled. Outputs: MLE, intervals, ``E`` and parallax at the
MLE, XP chi^2 and dof at the MLE, bound flags.

Data contracts
--------------
* ``labels.npz``: ``source_id, table_row, teff, e_teff, logg, e_logg, fe_h, e_fe_h,
  has_core_feh, split`` (split 0/1/2 = train/valid/test, -1 not kept; strings
  accepted). :func:`labels_from_table` builds an equivalent from the table.
* Covariance sidecar (``masked_stellar_autoencoder.data.covariance_sidecar``):
  ``CovarianceSidecar(path).xp_covariance(rows, band, units="table")`` ``[n, 55, 55]``,
  ``.parallax_zero_point(rows) -> (zp_mas, in_range)``, ``.read(["parallax",
  "parallax_error", "found_xp", "found_astrometry"], rows)``; aligned to the table's
  rows. Without a sidecar the diagonal errors of the table are used (the table's
  ``bpe_*/rpe_*`` are raw archive units: ``docs/xp-benchmark-usage.md``).
"""

from __future__ import annotations

import copy
import json
import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchregress.inference import maximize_score, profile_interval, wilks_threshold
from torchregress.losses import (
    KnownCovarianceGaussianNLL,
    known_covariance_gaussian_nll,
)

N_COEFF = 55
N_XP = 2 * N_COEFF
XP_COLUMNS = [f"bp_{i}" for i in range(1, N_COEFF + 1)] + [
    f"rp_{i}" for i in range(1, N_COEFF + 1)
]
XP_ERROR_COLUMNS = [f"bpe_{i}" for i in range(1, N_COEFF + 1)] + [
    f"rpe_{i}" for i in range(1, N_COEFF + 1)
]
G_REF = 8.5
PARAMS = ("teff", "logg", "fe_h")
PARAM_NAMES = ("teff", "logg", "fe_h", "ebv", "parallax")
LOG_2PI = math.log(2.0 * math.pi)
E_MAX_SCALE = 1.2
E_MAX_OFFSET = 0.05
PARALLAX_MIN_MAS = 0.005
PARALLAX_MAX_MAS = 100.0
PARALLAX_BOX_SIGMA = 6.0
#: Lindegren et al. (2021) median quasar parallax (mas): placeholder zero point.
L21_CONSTANT_ZP = -0.017


@dataclass(frozen=True)
class PhotBand:
    """A photometric band of the table: magnitude column, error column, bright limit.

    ``bright_limit``: magnitudes brighter than this are treated as missing (survey
    saturation; Pan-STARRS1 Magnier et al. 2020, SDSS Fukugita et al. 1996 /
    Ivezic et al. 2007 report saturation at r ~ 14, u ~ 13, z ~ 12.5).
    """

    name: str
    error: str | None = None
    bright_limit: float | None = None


DEFAULT_BANDS: tuple[PhotBand, ...] = (
    PhotBand("G"),
    PhotBand("J", "E_J"),
    PhotBand("H", "E_H"),
    PhotBand("KS", "E_KS"),
    PhotBand("W1"),
    PhotBand("W2"),
    PhotBand("G_PS1", "E_G_PS1", 14.0),
    PhotBand("R_PS1", "E_R_PS1", 14.0),
    PhotBand("I_PS1", "E_I_PS1", 14.0),
    PhotBand("Z_PS1", "E_Z_PS1", 13.5),
    PhotBand("Y_PS1", "E_Y_PS1", 12.5),
    PhotBand("U_SDSS", "E_U_SDSS", 13.0),
    PhotBand("G_SDSS", "E_G_SDSS", 14.0),
    PhotBand("R_SDSS", "E_R_SDSS", 14.0),
    PhotBand("I_SDSS", "E_I_SDSS", 14.0),
    PhotBand("Z_SDSS", "E_Z_SDSS", 12.5),
    PhotBand("U_SMSS", "E_U_SMSS"),
    PhotBand("V_SMSS", "E_V_SMSS"),
    PhotBand("G_SMSS", "E_G_SMSS"),
    PhotBand("R_SMSS", "E_R_SMSS"),
    PhotBand("I_SMSS", "E_I_SMSS"),
    PhotBand("Z_SMSS", "E_Z_SMSS"),
)


def bands_by_name(names: Sequence[str]) -> tuple[PhotBand, ...]:
    """Select bands of :data:`DEFAULT_BANDS` (or ``BP``/``RP`` without errors) by name."""
    known = {b.name: b for b in DEFAULT_BANDS}
    known.update({"BP": PhotBand("BP"), "RP": PhotBand("RP")})
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(f"unknown photometric bands {missing}")
    return tuple(known[n] for n in names)


# ─── data ────────────────────────────────────────────────────────────────────


def read_table_rows(path: str | Path, rows: np.ndarray, columns: Sequence[str]) -> dict:
    """Columns of a FITS binary table for ``rows`` (any order), as float64/int64 arrays.

    Rows are read once in sorted order from the memory-mapped table and returned in
    the requested order. Missing values (NaN, column ``TNULL``) become NaN for floats.
    """
    from astropy.io import fits

    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    uniq, inverse = np.unique(rows[order], return_inverse=True)
    out: dict[str, np.ndarray] = {}
    with fits.open(path, memmap=True) as hdul:
        hdu = next(h for h in hdul if isinstance(h, fits.BinTableHDU))
        names = set(hdu.columns.names)
        missing = [c for c in columns if c not in names]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        if len(uniq) and uniq[-1] >= hdu.data.shape[0]:
            raise ValueError(f"{path}: row {uniq[-1]} beyond {hdu.data.shape[0]} rows")
        block = hdu.data[uniq] if len(uniq) else hdu.data[:0]
        for name in columns:
            values = np.asarray(block[name])
            if values.dtype.kind == "f":
                values = values.astype(np.float64)
                null = hdu.columns[name].null
                if null is not None:
                    values[values == null] = np.nan
            elif values.dtype.kind in "iu":
                values = values.astype(np.int64)
            sorted_values = values[inverse]
            result = np.empty_like(sorted_values)
            result[order] = sorted_values
            out[name] = result
    return out


def packed_tril_index(n: int = N_COEFF) -> tuple[np.ndarray, np.ndarray]:
    """Row-major lower-triangle (with diagonal) indices of an ``n x n`` matrix."""
    return np.tril_indices(n)


def pack_covariance(cov: np.ndarray) -> np.ndarray:
    """``[..., n, n]`` symmetric -> ``[..., n(n+1)/2]`` lower triangle with diagonal."""
    i, j = packed_tril_index(cov.shape[-1])
    return cov[..., i, j]


def unpack_covariance(packed: Tensor) -> Tensor:
    """Inverse of :func:`pack_covariance` on torch tensors (symmetric output)."""
    length = packed.shape[-1]
    n = int(round((math.sqrt(8 * length + 1) - 1) / 2))
    if n * (n + 1) // 2 != length:
        raise ValueError(f"{length} is not a packed lower-triangle length")
    i, j = (torch.as_tensor(a, device=packed.device) for a in packed_tril_index(n))
    out = packed.new_zeros(packed.shape[:-1] + (n, n))
    out[..., i, j] = packed
    out[..., j, i] = packed
    return out


@dataclass
class StarData:
    """Per-star arrays (host, numpy) for the forward model; row order is arbitrary.

    XP in table units (G = 8.5 normalisation); ``xp_cov`` packed per band
    ``[N, 2, 1540]`` (dense sidecar covariance) or ``None`` with ``xp_var`` ``[N, 110]``
    (diagonal fallback). Magnitudes and errors ``[N, n_bands]`` (``mag_err = 0``:
    no quoted error). Labels ``theta``/``theta_err`` ``[N, 3]`` (NaN if unlabelled).
    """

    rows: np.ndarray
    source_id: np.ndarray
    xp: np.ndarray
    xp_mask: np.ndarray
    xp_cov: np.ndarray | None
    xp_var: np.ndarray | None
    mag: np.ndarray
    mag_err: np.ndarray
    mag_mask: np.ndarray
    parallax: np.ndarray
    parallax_err: np.ndarray
    parallax_zp: np.ndarray
    g_mag: np.ndarray
    ebv_sfd: np.ndarray
    theta: np.ndarray
    theta_err: np.ndarray
    band_names: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.rows)

    def subset(self, index: np.ndarray) -> StarData:
        index = np.asarray(index)
        kw = {}
        for name, value in self.__dict__.items():
            kw[name] = value[index] if isinstance(value, np.ndarray) else value
        return StarData(**kw)

    def batch(
        self, index: np.ndarray, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Tensors of rows ``index`` on ``device`` (dense XP covariances unpacked)."""
        index = np.asarray(index)

        def t(a: np.ndarray, dt: torch.dtype = dtype) -> Tensor:
            return torch.as_tensor(np.ascontiguousarray(a[index]), device=device).to(dt)

        out = {
            "xp": t(self.xp),
            "xp_mask": t(self.xp_mask, torch.bool),
            "mag": t(self.mag),
            "mag_err": t(self.mag_err),
            "mag_mask": t(self.mag_mask, torch.bool),
            "parallax": t(self.parallax),
            "parallax_err": t(self.parallax_err),
            "parallax_zp": t(self.parallax_zp),
            "theta": t(self.theta),
            "theta_err": t(self.theta_err),
        }
        if self.xp_cov is not None:
            out["xp_cov"] = unpack_covariance(t(self.xp_cov))  # [B, 2, 55, 55]
        else:
            assert self.xp_var is not None
            out["xp_var"] = t(self.xp_var)
        lo, hi, e_max = nuisance_box(
            self.ebv_sfd[index],
            self.parallax[index] - self.parallax_zp[index],
            self.parallax_err[index],
        )
        out["e_max"] = torch.as_tensor(e_max, device=device, dtype=dtype)
        out["log_plx_lo"] = torch.as_tensor(lo, device=device, dtype=dtype)
        out["log_plx_hi"] = torch.as_tensor(hi, device=device, dtype=dtype)
        return out


def nuisance_box(
    ebv_sfd: np.ndarray, parallax_corrected: np.ndarray, parallax_err: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(log10 parallax lo, hi, E_max)`` per star (see the module docstring)."""
    sfd = np.where(np.isfinite(ebv_sfd), np.maximum(ebv_sfd, 0.0), 0.0)
    e_max = E_MAX_SCALE * sfd + E_MAX_OFFSET
    plx = np.asarray(parallax_corrected, dtype=np.float64)
    err = np.asarray(parallax_err, dtype=np.float64)
    good = np.isfinite(plx) & np.isfinite(err) & (err > 0)
    k = PARALLAX_BOX_SIGMA
    with np.errstate(invalid="ignore"):
        hi = np.where(good, plx + k * err, PARALLAX_MAX_MAS)
        lo = np.where(good, plx - k * err, PARALLAX_MIN_MAS)
    hi = np.clip(hi, 10.0 * PARALLAX_MIN_MAS, None)
    lo = np.clip(lo, PARALLAX_MIN_MAS, None)
    lo = np.minimum(lo, hi / 1.01)
    return np.log10(lo), np.log10(hi), e_max


class DiagonalXPErrors:
    """Diagonal XP covariance from the table's ``bpe_*/rpe_*`` columns.

    ``error_convention="raw_xp"``: the errors are raw Gaia archive units and are divided
    by ``f = 10**((8.5 - G) / 2.5)`` (the case of the multisurvey tables, see
    ``docs/xp-benchmark-usage.md``); ``"g_normalized_xp"``: already in table units.
    """

    def __init__(self, error_convention: str = "raw_xp") -> None:
        if error_convention not in ("raw_xp", "g_normalized_xp"):
            raise ValueError(f"unknown XP error convention {error_convention!r}")
        self.error_convention = error_convention

    def variances(self, errors: np.ndarray, g_mag: np.ndarray) -> np.ndarray:
        err = np.asarray(errors, dtype=np.float64)
        if self.error_convention == "raw_xp":
            f = 10.0 ** ((G_REF - np.asarray(g_mag, dtype=np.float64)) / 2.5)
            err = err / f[:, None]
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(err) & (err > 0), err**2, np.nan)


class SidecarXPCovariance:
    """Dense per-band XP covariances (table units) and astrometry from the sidecar.

    Adapter over :class:`masked_stellar_autoencoder.data.covariance_sidecar.CovarianceSidecar`
    (rows aligned to the table). Covariances are at the level of the published errors
    (``formal=False``: the errors already include the solution's standard deviation),
    divided by ``f**2`` for the G = 8.5 table units; ``g_flux_term`` adds the band block
    of the rank-one G-flux term (its BP/RP cross term is dropped: the bands are scored
    independently).
    """

    def __init__(self, path: str | Path, *, g_flux_term: bool = False) -> None:
        from masked_stellar_autoencoder.data.covariance_sidecar import CovarianceSidecar

        self.path = Path(path)
        self.g_flux_term = bool(g_flux_term)
        self.reader = CovarianceSidecar(self.path)

    def close(self) -> None:
        self.reader.close()

    def packed_covariances(self, rows: np.ndarray, block: int = 65536) -> np.ndarray:
        """``[n, 2, 1540]`` float32 packed per-band covariances (NaN where no XP)."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), 2, N_COEFF * (N_COEFF + 1) // 2), dtype=np.float32)
        order = np.argsort(rows, kind="stable")
        for start in range(0, len(rows), block):
            sel = order[start : start + block]
            for k, band in enumerate(("bp", "rp")):
                cov = self.reader.xp_covariance(
                    rows[sel], band, units="table", g_flux_term=self.g_flux_term
                )
                out[sel, k] = pack_covariance(np.asarray(cov)).astype(np.float32)
        return out

    def astrometry(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Raw Gaia parallax, error and the stored Lindegren+2021 zero point (mas)."""
        cols = ["parallax", "parallax_error", "found_astrometry"]
        stored = "parallax_zpt_l21" in self.reader.columns
        r = self.reader.read(cols + (["parallax_zpt_l21"] if stored else []), rows)
        zp = (
            r["parallax_zpt_l21"]
            if stored
            else self.reader.parallax_zero_point(rows)[0]
        )
        return {
            "parallax": np.asarray(r["parallax"], dtype=np.float64),
            "parallax_error": np.asarray(r["parallax_error"], dtype=np.float64),
            "found": np.asarray(r["found_astrometry"], dtype=bool),
            "zero_point": np.asarray(zp, dtype=np.float64),
        }


def load_star_data(
    table: str | Path,
    rows: np.ndarray,
    *,
    labels: dict[str, np.ndarray] | None = None,
    bands: Sequence[PhotBand] = DEFAULT_BANDS,
    sidecar: SidecarXPCovariance | None = None,
    xp_error_convention: str = "raw_xp",
    zero_point: str = "sidecar",
    zero_point_constant: float = L21_CONSTANT_ZP,
    log: Callable[[str], None] = print,
) -> tuple[StarData, dict[str, Any]]:
    """Read the forward-model inputs of table ``rows`` (and their labels).

    ``zero_point``: ``"sidecar"`` (Lindegren+2021 from the sidecar; requires one),
    ``"constant"`` (``zero_point_constant`` for every star) or ``"none"``. With a
    sidecar, parallaxes and their errors come from the sidecar (raw Gaia values)
    where the star was found, else from the table. Returns the data and a
    diagnostics dict.
    """
    rows = np.asarray(rows, dtype=np.int64)
    mag_cols = [b.name for b in bands]
    err_cols = [b.error for b in bands if b.error is not None]
    columns = list(
        dict.fromkeys(
            [
                "source_id",
                "G",
                "PARALLAX",
                "e_parallax",
                "EBV",
                *XP_COLUMNS,
                *mag_cols,
                *err_cols,
            ]
            + ([] if sidecar is not None else XP_ERROR_COLUMNS)
        )
    )
    t0 = time.perf_counter()
    raw = read_table_rows(table, rows, columns)
    diag: dict[str, Any] = {
        "rows": int(len(rows)),
        "read_seconds": round(time.perf_counter() - t0, 1),
    }
    g_mag = raw["G"]
    xp = np.stack([raw[c] for c in XP_COLUMNS], axis=1)
    xp_var = None
    xp_cov = None
    if sidecar is None:
        errors = np.stack([raw[c] for c in XP_ERROR_COLUMNS], axis=1)
        xp_var = DiagonalXPErrors(xp_error_convention).variances(errors, g_mag)
        xp_mask = np.isfinite(xp) & np.isfinite(xp_var)
        diag["xp_covariance"] = f"diagonal table errors ({xp_error_convention})"
    else:
        xp_cov = sidecar.packed_covariances(rows)
        i, j = packed_tril_index()
        diag_idx = np.flatnonzero(i == j)
        var = xp_cov[:, :, diag_idx].reshape(len(rows), N_XP).astype(np.float64)
        xp_mask = np.isfinite(xp) & np.isfinite(var) & (var > 0)
        # Rows whose covariance is non-finite anywhere are masked entirely per band.
        for k in range(2):
            bad = ~np.isfinite(xp_cov[:, k]).all(axis=1)
            xp_mask[bad, k * N_COEFF : (k + 1) * N_COEFF] = False
            xp_cov[bad, k] = np.where(np.isfinite(xp_cov[bad, k]), xp_cov[bad, k], 0.0)
        diag["xp_covariance"] = f"sidecar {sidecar.path}"
    mag = np.stack([raw[b.name] for b in bands], axis=1)
    mag_err = np.zeros_like(mag)
    mag_mask = np.isfinite(mag) & (np.abs(mag) < 50)
    for k, b in enumerate(bands):
        if b.error is not None:
            e = raw[b.error]
            ok = np.isfinite(e) & (e > 0)
            mag_err[:, k] = np.where(ok, e, 0.0)
            mag_mask[:, k] &= ok
        if b.bright_limit is not None:
            with np.errstate(invalid="ignore"):
                mag_mask[:, k] &= ~(mag[:, k] < b.bright_limit)
    parallax = raw["PARALLAX"].copy()
    parallax_err = raw["e_parallax"].copy()
    zp = np.zeros(len(rows))
    if sidecar is not None:
        ast = sidecar.astrometry(rows)
        found = (
            ast["found"] & np.isfinite(ast["parallax"]) & (ast["parallax_error"] > 0)
        )
        diff = np.abs(ast["parallax"][found] - raw["PARALLAX"][found])
        diag["sidecar_parallax_found"] = int(found.sum())
        diag["sidecar_vs_table_parallax_median_abs_diff_mas"] = (
            float(np.nanmedian(diff)) if diff.size else None
        )
        parallax[found] = ast["parallax"][found]
        parallax_err[found] = ast["parallax_error"][found]
        if zero_point == "sidecar":
            zp = np.where(
                np.isfinite(ast["zero_point"]), ast["zero_point"], zero_point_constant
            )
            diag["zero_point_l21_rows"] = int(np.isfinite(ast["zero_point"]).sum())
    if zero_point == "sidecar" and sidecar is None:
        raise ValueError(
            "zero_point='sidecar' needs a covariance sidecar; use 'constant'"
        )
    if zero_point == "constant":
        zp = np.full(len(rows), float(zero_point_constant))
    elif zero_point not in ("sidecar", "none"):
        raise ValueError(f"unknown zero_point mode {zero_point!r}")
    diag["zero_point"] = zero_point
    theta = np.full((len(rows), 3), np.nan)
    theta_err = np.full((len(rows), 3), np.nan)
    if labels is not None:
        pos = _label_positions(labels, rows)
        for k, name in enumerate(PARAMS):
            theta[:, k] = np.asarray(labels[name], dtype=np.float64)[pos]
            theta_err[:, k] = np.asarray(labels[f"e_{name}"], dtype=np.float64)[pos]
    data = StarData(
        rows=rows,
        source_id=raw["source_id"].astype(np.int64),
        xp=np.where(xp_mask, xp, 0.0).astype(np.float32),
        xp_mask=xp_mask,
        xp_cov=xp_cov,
        xp_var=None
        if xp_var is None
        else np.where(xp_mask, xp_var, 1.0).astype(np.float32),
        mag=np.where(mag_mask, mag, 0.0).astype(np.float32),
        mag_err=mag_err.astype(np.float32),
        mag_mask=mag_mask,
        parallax=parallax,
        parallax_err=parallax_err,
        parallax_zp=zp,
        g_mag=g_mag,
        ebv_sfd=raw["EBV"],
        theta=theta,
        theta_err=theta_err,
        band_names=tuple(b.name for b in bands),
    )
    diag["xp_observed_fraction"] = float(xp_mask.mean()) if len(rows) else 0.0
    diag["band_observed_fraction"] = {
        b.name: float(mag_mask[:, k].mean()) if len(rows) else 0.0
        for k, b in enumerate(bands)
    }
    log(f"xp_forward: loaded {len(rows)} rows: {diag}")
    return data, diag


# ─── labels ─────────────────────────────────────────────────────────────────

LABEL_FIELDS = (
    "source_id", "table_row", "teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h",
    "has_core_feh", "split",
)  # fmt: skip
SPLIT_CODES = {"train": 0, "valid": 1, "test": 2}


def load_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``labels.npz`` (contract fields; ``split`` as ints 0/1/2/-1 or strings)."""
    with np.load(path, allow_pickle=False) as z:
        missing = [f for f in LABEL_FIELDS if f not in z.files]
        if missing:
            raise ValueError(f"{path}: labels contract fields missing: {missing}")
        out = {f: np.asarray(z[f]) for f in LABEL_FIELDS}
    out["split"] = split_codes(out["split"])
    return out


def split_codes(split: np.ndarray) -> np.ndarray:
    split = np.asarray(split)
    if split.dtype.kind in "iu":
        return split.astype(np.int8)
    names = np.char.lower(np.char.strip(split.astype(str)))
    codes = np.full(len(names), -1, dtype=np.int8)
    for name, code in SPLIT_CODES.items():
        codes[names == name] = code
    return codes


def _label_positions(labels: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    table_row = np.asarray(labels["table_row"], dtype=np.int64)
    order = np.argsort(table_row)
    pos = np.searchsorted(table_row[order], rows)
    pos = np.clip(pos, 0, len(order) - 1)
    found = table_row[order][pos] == rows
    if not found.all():
        raise ValueError(f"{int((~found).sum())} rows have no entry in the labels file")
    return order[pos]


def labels_from_table(
    table: str | Path,
    split_config: dict[str, Any] | None = None,
    core_rows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Development stand-in for ``labels.npz`` built from the table's own labels.

    Labels are the table's ``teff/logg/fe_h`` and errors; the split is the
    fine-tuning split (``finetune_data.split_row_positions`` over the rows kept by the
    config's metal-poor filters, all rows without a config).
    """
    from astropy.io import fits

    from .finetune_data import kept_rows, split_row_positions

    with fits.open(table, memmap=True) as hdul:
        n = int(next(h for h in hdul if isinstance(h, fits.BinTableHDU)).data.shape[0])
    cols = ["source_id", *PARAMS, *(f"e_{p}" for p in PARAMS)]
    raw = read_table_rows(table, np.arange(n), cols)
    split = np.full(n, -1, dtype=np.int8)
    if split_config is not None:
        kept = kept_rows(str(table), split_config)
        mp = split_config["finetuning"].get("metal_poor") or {}
    else:
        kept, mp = np.arange(n), {}
    for code, pos in enumerate(split_row_positions(len(kept), mp)):
        split[kept[pos]] = code
    has_core = np.zeros(n, dtype=bool)
    if core_rows is not None:
        has_core[np.asarray(core_rows, dtype=np.int64)] = True
    out = {"source_id": raw["source_id"], "table_row": np.arange(n), "split": split,
           "has_core_feh": has_core}  # fmt: skip
    for p in PARAMS:
        out[p] = raw[p]
        out[f"e_{p}"] = raw[f"e_{p}"]
    return out


# ─── model ──────────────────────────────────────────────────────────────────


def _mlp(n_in: int, hidden: int, depth: int, n_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = n_in
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


@dataclass
class ModelConfig:
    hidden: int = 512
    depth: int = 4
    rank: int = 2
    init_fraction: float = 0.01
    init_floor: float = 1e-3
    init_mag_scatter: float = 0.02
    max_jitter: float = 1e-4


class XPForwardModel(nn.Module):
    """Emulator ``g(theta, E)`` with learned XP and photometric model scatter."""

    def __init__(
        self,
        band_names: Sequence[str],
        *,
        theta_loc: Sequence[float],
        theta_scale: Sequence[float],
        xp_scale: np.ndarray,
        mag_loc: np.ndarray,
        mag_scale: np.ndarray,
        n_coeff: int = N_COEFF,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.band_names = tuple(band_names)
        self.n_coeff = int(n_coeff)
        nb = len(self.band_names)
        n_out = 2 * self.n_coeff + nb
        self.register_buffer(
            "theta_loc", torch.as_tensor(theta_loc, dtype=torch.float32)
        )
        self.register_buffer(
            "theta_scale", torch.as_tensor(theta_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "xp_scale", torch.as_tensor(np.asarray(xp_scale), dtype=torch.float32)
        )
        self.register_buffer(
            "mag_loc", torch.as_tensor(np.asarray(mag_loc), dtype=torch.float32)
        )
        self.register_buffer(
            "mag_scale", torch.as_tensor(np.asarray(mag_scale), dtype=torch.float32)
        )
        self.intrinsic = _mlp(3, cfg.hidden, cfg.depth, n_out)
        self.extinction = _mlp(4, cfg.hidden, cfg.depth, n_out)
        self.xp_losses = nn.ModuleList(
            KnownCovarianceGaussianNLL(
                self.n_coeff,
                scatter="fractional",
                rank=cfg.rank,
                init_fraction=cfg.init_fraction,
                init_floor=cfg.init_floor,
                max_jitter=cfg.max_jitter,
                reduction="none",
            )
            for _ in range(2)
        )
        self.log_mag_scatter = nn.Parameter(
            torch.full((nb,), math.log(cfg.init_mag_scatter))
        )

    def scatter_parameters(self) -> list[nn.Parameter]:
        return [*self.xp_losses.parameters(), self.log_mag_scatter]

    def network_parameters(self) -> list[nn.Parameter]:
        return [*self.intrinsic.parameters(), *self.extinction.parameters()]

    def forward(self, theta: Tensor, ebv: Tensor) -> tuple[Tensor, Tensor]:
        """Standardised XP ``[M, 110]`` (table units / ``xp_scale``) and ``M_b`` ``[M, nb]``."""
        t = (theta - self.theta_loc) / self.theta_scale
        e = ebv.unsqueeze(-1)
        h0 = self.intrinsic(t)
        h1 = self.extinction(torch.cat([t, e], dim=-1))
        n = 2 * self.n_coeff
        xp = h0[:, :n] + e * h1[:, :n]
        mag = self.mag_loc + self.mag_scale * h0[:, n:] + e * F.softplus(h1[:, n:])
        return xp, mag

    def describe(self) -> dict[str, Any]:
        """Summary of the learned scatter terms."""
        out: dict[str, Any] = {}
        for name, loss in zip(("bp", "rp"), self.xp_losses, strict=True):
            frac = torch.exp(loss.log_fraction).detach().cpu().numpy()
            floor = torch.exp(loss.log_floor).detach().cpu().numpy()
            out[f"{name}_fraction_median"] = float(np.median(frac))
            out[f"{name}_fraction_p90"] = float(np.quantile(frac, 0.9))
            out[f"{name}_floor_median"] = float(np.median(floor))
            if loss.factor_weight is not None:
                out[f"{name}_factor_norm"] = float(loss.factor_weight.detach().norm())
        scatter = torch.exp(self.log_mag_scatter).detach().cpu().numpy()
        out["mag_scatter"] = dict(
            zip(self.band_names, map(float, scatter), strict=True)
        )
        return out


def star_nll(
    model: XPForwardModel,
    theta: Tensor,
    ebv: Tensor,
    parallax: Tensor,
    batch: dict[str, Tensor],
    *,
    parts: bool = False,
) -> Tensor | dict[str, Tensor]:
    """Per-star negative log-likelihood ``[M]`` (or its parts with ``parts=True``).

    ``batch`` holds the stars' data for the same ``M`` rows (see :meth:`StarData.batch`);
    ``parallax`` in mas (> 0).
    """
    xp_hat, mag_hat = model(theta, ebv)
    scale = model.xp_scale.to(xp_hat.dtype)
    xp_obs = batch["xp"] / scale
    nc = model.n_coeff
    nll_xp = torch.zeros_like(ebv)
    chi2_xp = torch.zeros_like(ebv)
    dof_xp = torch.zeros_like(ebv)
    for k, loss in enumerate(model.xp_losses):
        sl = slice(k * nc, (k + 1) * nc)
        s = scale[sl]
        kw: dict[str, Tensor] = {}
        if "xp_cov" in batch:
            kw["known_covariance"] = batch["xp_cov"][:, k] / (
                s.unsqueeze(-1) * s.unsqueeze(-2)
            )
        else:
            kw["known_variance"] = batch["xp_var"][:, sl] / s.square()
        res = loss.evaluate(
            xp_hat[:, sl], xp_obs[:, sl], mask=batch["xp_mask"][:, sl], **kw
        )
        nll_xp = nll_xp + res.nll
        chi2_xp = chi2_xp + res.chi2
        dof_xp = dof_xp + res.dof
    dist_mod = 5.0 * torch.log10(100.0 / parallax)
    resid = batch["mag"] - (mag_hat + dist_mod.unsqueeze(-1))
    phot = known_covariance_gaussian_nll(
        resid,
        known_variance=batch["mag_err"].square(),
        model_scale=torch.exp(model.log_mag_scatter).to(resid.dtype),
        mask=batch["mag_mask"],
    )
    plx_obs = batch["parallax"] - batch["parallax_zp"]
    sig = batch["parallax_err"]
    has_plx = torch.isfinite(plx_obs) & torch.isfinite(sig) & (sig > 0)
    sig_safe = torch.where(has_plx, sig, torch.ones_like(sig))
    z = torch.where(has_plx, (plx_obs - parallax) / sig_safe, torch.zeros_like(sig))
    nll_plx = torch.where(
        has_plx,
        0.5 * (z.square() + LOG_2PI) + torch.log(sig_safe),
        torch.zeros_like(sig),
    )
    total = nll_xp + phot.nll + nll_plx
    if not parts:
        return total
    return {
        "nll": total,
        "nll_xp": nll_xp,
        "chi2_xp": chi2_xp,
        "dof_xp": dof_xp,
        "nll_phot": phot.nll,
        "chi2_phot": phot.chi2,
        "dof_phot": phot.dof,
        "nll_parallax": nll_plx,
    }


def unit_to_nuisance(u: Tensor, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """``(E, parallax [mas])`` from unit-box coordinates ``u [M, 2]`` of each star."""
    ebv = u[:, 0] * batch["e_max"]
    lo, hi = batch["log_plx_lo"], batch["log_plx_hi"]
    parallax = torch.pow(10.0, lo + u[:, 1] * (hi - lo))
    return ebv, parallax


def nuisance_to_unit(
    ebv: np.ndarray,
    parallax: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    e_max: np.ndarray,
) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        ue = np.asarray(ebv) / e_max
        up = (np.log10(np.clip(parallax, 10.0**lo, 10.0**hi)) - lo) / (hi - lo)
    ue = np.where(np.isfinite(ue), ue, 0.5)
    up = np.where(np.isfinite(up), up, 0.5)
    return np.stack([np.clip(ue, 0.0, 1.0), np.clip(up, 0.0, 1.0)], axis=1)


def initial_nuisance_unit(data: StarData) -> np.ndarray:
    """Unit-box nuisance start: ``E`` at SFD, parallax at the corrected observation."""
    plx_c = data.parallax - data.parallax_zp
    lo, hi, e_max = nuisance_box(data.ebv_sfd, plx_c, data.parallax_err)
    sfd = np.where(np.isfinite(data.ebv_sfd), np.maximum(data.ebv_sfd, 0.0), 0.0)
    return nuisance_to_unit(sfd, plx_c, lo, hi, e_max)


# ─── normalisation from the training stars ──────────────────────────────────


def fit_normalisation(data: StarData, n_coeff: int = N_COEFF) -> dict[str, np.ndarray]:
    """Fixed scales from the training stars: theta loc/scale, XP and magnitude scales."""
    theta = data.theta
    ok = np.isfinite(theta).all(axis=1)
    if ok.sum() < 2:
        raise ValueError("need labelled training stars to fit the normalisation")
    t = theta[ok]
    theta_loc = np.median(t, axis=0)
    theta_scale = np.maximum(
        np.subtract(*np.percentile(t, [84, 16], axis=0)) / 2.0, 1e-3
    )
    xp = np.where(data.xp_mask, np.abs(data.xp.astype(np.float64)), np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        xp_scale = np.nanmedian(xp, axis=0)
    fallback = np.nanmedian(xp_scale) if np.isfinite(xp_scale).any() else 1.0
    xp_scale = np.where(np.isfinite(xp_scale) & (xp_scale > 0), xp_scale, fallback)
    plx_c = data.parallax - data.parallax_zp
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        good = (
            np.isfinite(plx_c)
            & (plx_c > 5.0 * data.parallax_err)
            & (data.parallax_err > 0)
        )
        absmag = np.where(
            data.mag_mask & good[:, None],
            data.mag - 5.0 * np.log10(100.0 / np.where(good, plx_c, 1.0))[:, None],
            np.nan,
        )
        mag_loc = np.nanmedian(absmag, axis=0)
        mag_scale = np.subtract(*np.nanpercentile(absmag, [84, 16], axis=0)) / 2.0
    mag_loc = np.where(np.isfinite(mag_loc), mag_loc, 5.0)
    mag_scale = np.where(np.isfinite(mag_scale) & (mag_scale > 0), mag_scale, 2.0)
    return {
        "theta_loc": theta_loc,
        "theta_scale": theta_scale,
        "xp_scale": xp_scale.astype(np.float64),
        "mag_loc": mag_loc,
        "mag_scale": mag_scale,
    }


def build_model(data: StarData, config: ModelConfig | None = None) -> XPForwardModel:
    norm = fit_normalisation(data)
    return XPForwardModel(
        data.band_names,
        theta_loc=norm["theta_loc"],
        theta_scale=norm["theta_scale"],
        xp_scale=norm["xp_scale"],
        mag_loc=norm["mag_loc"],
        mag_scale=norm["mag_scale"],
        n_coeff=data.xp.shape[1] // 2,
        config=config,
    )


# ─── training ───────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 1e-3
    scatter_lr: float = 3e-3
    weight_decay: float = 1e-5
    nuisance_lr: float = 0.05
    scatter_warmup_epochs: int = 1
    grad_clip: float = 10.0
    label_noise: str = "plugin"
    label_noise_samples: int = 8
    valid_rows: int = 4096
    seed: int = 0
    log_every: int = 200


def _theta_draws(
    batch: dict[str, Tensor], cfg: TrainConfig, gen: torch.Generator
) -> Tensor:
    """``[K, B, 3]`` label draws for the label-noise hook (``K = 1`` for plug-in)."""
    theta = batch["theta"]
    if cfg.label_noise == "plugin":
        return theta.unsqueeze(0)
    if cfg.label_noise != "marginal":
        raise ValueError(f"unknown label_noise {cfg.label_noise!r}")
    err = torch.where(
        torch.isfinite(batch["theta_err"]), batch["theta_err"], torch.zeros_like(theta)
    )
    noise = torch.randn(
        (cfg.label_noise_samples, *theta.shape),
        generator=gen,
        device="cpu",
        dtype=torch.float32,
    ).to(device=theta.device, dtype=theta.dtype)
    return theta.unsqueeze(0) + noise * err.unsqueeze(0)


def batch_loss(
    model: XPForwardModel,
    batch: dict[str, Tensor],
    u: Tensor,
    cfg: TrainConfig,
    gen: torch.Generator,
) -> Tensor:
    """Per-star training NLL ``[B]`` at nuisances ``u`` (label-noise hook applied)."""
    ebv, parallax = unit_to_nuisance(u, batch)
    draws = _theta_draws(batch, cfg, gen)
    nlls = torch.stack([star_nll(model, th, ebv, parallax, batch) for th in draws])
    if nlls.shape[0] == 1:
        return nlls[0]
    return -(torch.logsumexp(-nlls, dim=0) - math.log(nlls.shape[0]))


def train_forward_model(
    model: XPForwardModel,
    train: StarData,
    valid: StarData | None,
    cfg: TrainConfig,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Fit emulator, scatter terms and per-star nuisances on labelled stars.

    Returns the history (per-epoch train/valid NLL, learned scatter) and the per-star
    nuisances of the training stars. The model keeps the parameters of the epoch with
    the lowest validation NLL (profiled nuisances), or of the last epoch without
    validation stars.
    """
    ok = np.isfinite(train.theta).all(axis=1)
    if not ok.all():
        raise ValueError("training stars must all be labelled")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model.to(device=device, dtype=dtype)
    n = len(train)
    init_u = np.clip(initial_nuisance_unit(train), 0.02, 0.98)
    nuisance = nn.Embedding(n, 2, sparse=True).to(device=device, dtype=dtype)
    with torch.no_grad():
        nuisance.weight.copy_(
            torch.logit(torch.as_tensor(init_u, device=device, dtype=dtype))
        )
    network = model.network_parameters()
    scatter = model.scatter_parameters()
    opt = torch.optim.AdamW(
        [
            {"params": network, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
            {"params": scatter, "weight_decay": 0.0, "lr": cfg.scatter_lr},
        ]
    )
    opt_n = torch.optim.SparseAdam(list(nuisance.parameters()), lr=cfg.nuisance_lr)
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs

    def cosine(s: int) -> float:
        return 0.5 * (
            1.0 + math.cos(math.pi * min(s, total_steps) / max(total_steps, 1))
        )

    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine)
    # Sparse Adam moves a row seen once per epoch by several times its lr (lazy moment
    # updates with a global bias correction), so the nuisance steps must anneal too.
    sched_n = torch.optim.lr_scheduler.LambdaLR(opt_n, cosine)
    history: dict[str, Any] = {"epochs": [], "config": asdict(cfg)}
    best_state, best_valid = None, math.inf
    valid_sub = None
    if valid is not None and len(valid):
        keep = np.isfinite(valid.theta).all(axis=1)
        vidx = np.flatnonzero(keep)
        if len(vidx) > cfg.valid_rows:
            vidx = np.sort(rng.choice(vidx, cfg.valid_rows, replace=False))
        valid_sub = valid.subset(vidx) if len(vidx) else None
    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        warm = epoch < cfg.scatter_warmup_epochs
        for p in scatter:
            p.requires_grad_(not warm)
        perm = rng.permutation(n)
        t0 = time.perf_counter()
        sums = {"nll": 0.0, "count": 0}
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            batch = train.batch(idx, device, dtype)
            idx_t = torch.as_tensor(idx, device=device)
            u = torch.sigmoid(nuisance(idx_t))
            loss_rows = batch_loss(model, batch, u, cfg, gen)
            loss = loss_rows.mean()
            opt.zero_grad(set_to_none=True)
            opt_n.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network + [p for p in scatter if p.requires_grad], cfg.grad_clip
            )
            opt.step()
            opt_n.step()
            sched.step()
            sched_n.step()
            step += 1
            sums["nll"] += float(loss_rows.detach().sum())
            sums["count"] += len(idx)
            if cfg.log_every and step % cfg.log_every == 0:
                log(
                    f"xp_forward: epoch {epoch} step {step}/{total_steps} nll {float(loss):.3f}"
                )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_nll": sums["nll"] / max(sums["count"], 1),
            "seconds": round(time.perf_counter() - t0, 1),
            "scatter_frozen": warm,
            "scatter": model.describe(),
        }
        if valid_sub is not None:
            model.eval()
            prof = profile_nuisance_at_labels(
                model, valid_sub, device=device, dtype=dtype
            )
            record["valid_nll"] = float(np.mean(prof["nll"]))
            record["valid_chi2_xp_per_dof"] = float(
                np.sum(prof["chi2_xp"]) / max(np.sum(prof["dof_xp"]), 1)
            )
            if record["valid_nll"] < best_valid:
                best_valid = record["valid_nll"]
                best_state = copy.deepcopy(model.state_dict())
                record["best"] = True
        history["epochs"].append(record)
        log(
            f"xp_forward: {json.dumps({k: v for k, v in record.items() if k != 'scatter'})}"
        )
    for p in scatter:
        p.requires_grad_(True)
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        u_final = torch.sigmoid(nuisance.weight).cpu().numpy().astype(np.float64)
    lo, hi, e_max = nuisance_box(
        train.ebv_sfd, train.parallax - train.parallax_zp, train.parallax_err
    )
    history["train_nuisance"] = {
        "ebv": u_final[:, 0] * e_max,
        "parallax": 10.0 ** (lo + u_final[:, 1] * (hi - lo)),
    }
    history["best_valid_nll"] = best_valid if math.isfinite(best_valid) else None
    return history


# ─── inference ──────────────────────────────────────────────────────────────


def _score_fn(
    model: XPForwardModel,
    batch: dict[str, Tensor],
    fixed_theta: Tensor | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """``score(params [M, P], x [M, 1]) -> [M]`` with ``x`` the star's local index."""

    def score(params: Tensor, x: Tensor) -> Tensor:
        idx = x[:, 0].round().long()
        sub = {k: v[idx] for k, v in batch.items()}
        if fixed_theta is None:
            theta, u = params[:, :3], params[:, 3:5]
        else:
            theta, u = fixed_theta[idx], params
        ebv, parallax = unit_to_nuisance(u, sub)
        return -star_nll(model, theta, ebv, parallax, sub)

    return score


def profile_nuisance_at_labels(
    model: XPForwardModel,
    data: StarData,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    chunk_rows: int = 1024,
    n_candidates: int = 16,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Likelihood at the labels with ``E`` and parallax profiled (emulator diagnostic)."""
    model_d = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    for p in model_d.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in ("nll", "chi2_xp", "dof_xp", "ebv", "parallax")
    }
    for start in range(0, len(data), chunk_rows):
        idx = np.arange(start, min(start + chunk_rows, len(data)))
        batch = data.batch(idx, device, dtype)
        x = torch.arange(len(idx), device=device, dtype=dtype).unsqueeze(1)
        init = torch.as_tensor(
            initial_nuisance_unit(data.subset(idx)), device=device, dtype=dtype
        )
        score = _score_fn(model_d, batch, fixed_theta=batch["theta"])
        res = maximize_score(
            score, x, [[0.0, 1.0], [0.0, 1.0]], n_candidates=n_candidates, n_starts=2,
            steps=30, init=init, generator=gen,
        )  # fmt: skip
        with torch.no_grad():
            ebv, parallax = unit_to_nuisance(res.theta, batch)
            parts = star_nll(model_d, batch["theta"], ebv, parallax, batch, parts=True)
        assert isinstance(parts, dict)
        for k in ("nll", "chi2_xp", "dof_xp"):
            out[k].append(parts[k].cpu().numpy())
        out["ebv"].append(ebv.cpu().numpy())
        out["parallax"].append(parallax.cpu().numpy())
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def candidate_bank(
    theta_pool: np.ndarray, n_theta: int, ebv_grid: Sequence[float], seed: int = 0
) -> np.ndarray:
    """``[K, 4]`` (theta, E) candidates: ``n_theta`` labelled-population draws x ``E`` grid."""
    pool = theta_pool[np.isfinite(theta_pool).all(axis=1)]
    rng = np.random.default_rng(seed)
    pick = pool[rng.choice(len(pool), size=min(n_theta, len(pool)), replace=False)]
    grid = np.asarray(ebv_grid, dtype=np.float64)
    return np.concatenate(
        [np.repeat(pick, len(grid), axis=0), np.tile(grid, len(pick))[:, None]], axis=1
    )


def coarse_init(
    model: XPForwardModel,
    batch: dict[str, Tensor],
    bank: Tensor,
    *,
    star_block: int = 32,
) -> Tensor:
    """Best bank point per star as a start ``[n, 5]`` (theta, u_E, u_parallax).

    Fixed-covariance approximation: the XP covariance is evaluated at the observed
    coefficients (one factorisation per star and band), the distance modulus is
    profiled analytically from the photometry and the parallax (delta method), and
    bank points with ``E > E_max`` are excluded.
    """
    dtype, device = bank.dtype, bank.device
    with torch.no_grad():
        xp_bank, mag_bank = model(bank[:, :3], bank[:, 3])  # [K, 110], [K, nb]
        scale = model.xp_scale.to(dtype)
        xp_obs = batch["xp"] / scale
        n, nc = xp_obs.shape[0], model.n_coeff
        chi2 = torch.zeros(n, bank.shape[0], dtype=dtype, device=device)
        for k, loss in enumerate(model.xp_losses):
            sl = slice(k * nc, (k + 1) * nc)
            s = scale[sl]
            mask = batch["xp_mask"][:, sl]
            var, factor = loss.model_covariance_terms(xp_obs[:, sl])
            if "xp_cov" in batch:
                cov = batch["xp_cov"][:, k] / (s.unsqueeze(-1) * s.unsqueeze(-2))
            else:
                cov = torch.diag_embed(batch["xp_var"][:, sl] / s.square()).to(
                    dtype=dtype
                )
            cov = cov + torch.diag_embed(var).to(dtype=dtype)
            if factor is not None:
                cov = cov + factor @ factor.mT
            pair = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            cov = torch.where(pair, cov, torch.zeros((), dtype=dtype, device=device))
            cov = cov + torch.diag_embed((~mask).to(dtype)).to(dtype=dtype)
            d = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(torch.finfo(dtype).tiny)
            q = torch.rsqrt(d)
            corr = cov * q.unsqueeze(-1) * q.unsqueeze(-2)
            eye = torch.eye(nc, dtype=dtype, device=device)
            factor_l, info = torch.linalg.cholesky_ex(corr)
            if bool((info != 0).any()):
                factor_l, _ = torch.linalg.cholesky_ex(corr + 1e-4 * eye)
            for start in range(0, n, star_block):
                rs = slice(start, min(start + star_block, n))
                resid = xp_obs[rs, sl].unsqueeze(-1) - xp_bank[:, sl].T.unsqueeze(
                    0
                )  # [b, nc, K]
                resid = resid * (q[rs] * mask[rs].to(dtype)).unsqueeze(-1)
                w = torch.linalg.solve_triangular(factor_l[rs], resid, upper=False)
                chi2[rs] += w.square().sum(1)
        # Photometry with the distance modulus profiled (weighted mean incl. parallax).
        var_m = batch["mag_err"].square() + torch.exp(2.0 * model.log_mag_scatter).to(
            dtype
        )
        w_m = batch["mag_mask"].to(dtype) / var_m  # [n, nb]
        plx_c = batch["parallax"] - batch["parallax_zp"]
        sig = batch["parallax_err"]
        good = (
            torch.isfinite(plx_c)
            & torch.isfinite(sig)
            & (sig > 0)
            & (plx_c > 2.0 * sig)
        )
        plx_safe = torch.where(good, plx_c, torch.ones_like(plx_c))
        mu_plx = 5.0 * torch.log10(100.0 / plx_safe)
        var_mu = (5.0 / math.log(10.0) * sig / plx_safe).square()
        w_plx = torch.where(good, 1.0 / var_mu, torch.zeros_like(var_mu))
        mag = torch.where(
            batch["mag_mask"], batch["mag"], torch.zeros_like(batch["mag"])
        )
        diff = mag.unsqueeze(1) - mag_bank.unsqueeze(0)  # [n, K, nb]
        num = (w_m.unsqueeze(1) * diff).sum(-1) + (w_plx * mu_plx).unsqueeze(1)
        den = w_m.sum(-1, keepdim=True) + w_plx.unsqueeze(1)
        mu = num / den.clamp_min(torch.finfo(dtype).tiny)
        lo, hi = batch["log_plx_lo"], batch["log_plx_hi"]
        mu_lo = 5.0 * (2.0 - hi)  # distance modulus range of the parallax box
        mu_hi = 5.0 * (2.0 - lo)
        mu = torch.minimum(torch.maximum(mu, mu_lo.unsqueeze(1)), mu_hi.unsqueeze(1))
        chi2_phot = (w_m.unsqueeze(1) * (diff - mu.unsqueeze(-1)).square()).sum(-1)
        chi2_plx = w_plx.unsqueeze(1) * (mu - mu_plx.unsqueeze(1)).square()
        total = chi2 + chi2_phot + chi2_plx
        allowed = bank[:, 3].unsqueeze(0) <= batch["e_max"].unsqueeze(1)
        total = torch.where(allowed, total, torch.full_like(total, math.inf))
        total = torch.nan_to_num(total, nan=math.inf)
        best = total.argmin(dim=1)
        theta = bank[best, :3]
        u_e = (bank[best, 3] / batch["e_max"]).clamp(0.0, 1.0)
        log_plx = 2.0 - mu[torch.arange(n, device=device), best] / 5.0
        u_p = ((log_plx - lo) / (hi - lo)).clamp(0.0, 1.0)
    return torch.cat([theta, u_e.unsqueeze(1), u_p.unsqueeze(1)], dim=1)


@dataclass
class InferenceConfig:
    chunk_rows: int = 1024
    chunk_size: int = 8192
    n_candidates: int = 64
    n_starts: int = 2
    steps: int = 40
    profile_steps: int = 15
    profile_scan: int = 6
    bank_theta: int = 1024
    ebv_grid: tuple[float, ...] = (
        0.0,
        0.03,
        0.08,
        0.15,
        0.25,
        0.4,
        0.6,
        0.9,
        1.3,
        2.0,
        3.0,
    )
    confidence: tuple[float, float] = (0.6827, 0.95)
    seed: int = 0


def infer_stars(
    model: XPForwardModel,
    data: StarData,
    theta_bounds: np.ndarray,
    theta_pool: np.ndarray,
    cfg: InferenceConfig,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    log: Callable[[str], None] = print,
) -> dict[str, np.ndarray]:
    """MLE, profile intervals, nuisances and goodness of fit for every star of ``data``."""
    model_d = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    for p in model_d.parameters():
        p.requires_grad_(False)
    bounds = np.concatenate(
        [np.asarray(theta_bounds, dtype=np.float64), [[0.0, 1.0], [0.0, 1.0]]]
    )
    bank = torch.as_tensor(
        candidate_bank(theta_pool, cfg.bank_theta, cfg.ebv_grid, cfg.seed),
        device=device,
        dtype=dtype,
    )
    bank[:, :3] = torch.maximum(torch.minimum(bank[:, :3], torch.as_tensor(bounds[:3, 1], device=device, dtype=dtype)),
                                torch.as_tensor(bounds[:3, 0], device=device, dtype=dtype))  # fmt: skip
    gen = torch.Generator().manual_seed(cfg.seed)
    thresholds = [wilks_threshold(c, df=1) for c in cfg.confidence]
    keys = ("mle", "lo68", "hi68", "lo95", "hi95", "at_bound", "interval_at_bound", "E_mle",
            "parallax_mle", "chi2_xp", "dof_xp", "chi2_phot", "dof_phot", "nll", "init")  # fmt: skip
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    t0 = time.perf_counter()
    for start in range(0, len(data), cfg.chunk_rows):
        idx = np.arange(start, min(start + cfg.chunk_rows, len(data)))
        batch = data.batch(idx, device, dtype)
        init = coarse_init(model_d, batch, bank)
        x = torch.arange(len(idx), device=device, dtype=dtype).unsqueeze(1)
        score = _score_fn(model_d, batch)
        res = maximize_score(
            score, x, bounds, n_candidates=cfg.n_candidates, n_starts=cfg.n_starts,
            steps=cfg.steps, init=init, chunk_size=cfg.chunk_size, generator=gen,
        )  # fmt: skip
        intervals = {}
        for name, thr in zip(("68", "95"), thresholds, strict=True):
            lo, hi, lo_b, hi_b = [], [], [], []
            for j in range(3):
                pi = profile_interval(
                    score, x, res.theta, res.score, j, thr, bounds, n_scan=cfg.profile_scan,
                    steps=cfg.profile_steps, chunk_size=cfg.chunk_size,
                )  # fmt: skip
                lo.append(pi.lower)
                hi.append(pi.upper)
                lo_b.append(pi.lower_at_bound)
                hi_b.append(pi.upper_at_bound)
            intervals[name] = (
                torch.stack(lo, 1),
                torch.stack(hi, 1),
                torch.stack(lo_b, 1) | torch.stack(hi_b, 1),
            )
        with torch.no_grad():
            ebv, parallax = unit_to_nuisance(res.theta[:, 3:5], batch)
            parts = star_nll(
                model_d, res.theta[:, :3], ebv, parallax, batch, parts=True
            )
        assert isinstance(parts, dict)
        out["mle"].append(res.theta[:, :3].cpu().numpy())
        out["at_bound"].append(res.at_bound.cpu().numpy())
        out["lo68"].append(intervals["68"][0].cpu().numpy())
        out["hi68"].append(intervals["68"][1].cpu().numpy())
        out["lo95"].append(intervals["95"][0].cpu().numpy())
        out["hi95"].append(intervals["95"][1].cpu().numpy())
        out["interval_at_bound"].append(intervals["95"][2].cpu().numpy())
        out["E_mle"].append(ebv.cpu().numpy())
        out["parallax_mle"].append(parallax.cpu().numpy())
        for k in ("chi2_xp", "dof_xp", "chi2_phot", "dof_phot", "nll"):
            out[k].append(parts[k].cpu().numpy())
        out["init"].append(init.cpu().numpy())
        done = idx[-1] + 1
        rate = done / max(time.perf_counter() - t0, 1e-9)
        log(f"xp_forward: inferred {done}/{len(data)} stars ({rate:.1f} stars/s)")
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def theta_bounds_from_labels(theta: np.ndarray, pad: float = 0.02) -> np.ndarray:
    """``[3, 2]`` box: labelled range padded by ``pad`` of its width."""
    t = theta[np.isfinite(theta).all(axis=1)]
    lo, hi = t.min(axis=0), t.max(axis=0)
    width = np.maximum(hi - lo, 1e-6)
    return np.stack([lo - pad * width, hi + pad * width], axis=1)


# ─── checkpoint and outputs ─────────────────────────────────────────────────


def save_model(
    model: XPForwardModel, path: str | Path, extra: dict[str, Any] | None = None
) -> None:
    state = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "band_names": list(model.band_names),
        "n_coeff": model.n_coeff,
        "model_config": asdict(model.config),
        "extra": extra or {},
    }
    torch.save(state, path)


def load_model(path: str | Path) -> tuple[XPForwardModel, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state["state_dict"]
    model = XPForwardModel(
        state["band_names"],
        theta_loc=sd["theta_loc"].numpy(),
        theta_scale=sd["theta_scale"].numpy(),
        xp_scale=sd["xp_scale"].numpy(),
        mag_loc=sd["mag_loc"].numpy(),
        mag_scale=sd["mag_scale"].numpy(),
        n_coeff=state["n_coeff"],
        config=ModelConfig(**state["model_config"]),
    )
    model.load_state_dict(sd)
    return model, state.get("extra", {})


def prediction_arrays(
    data: StarData, result: dict[str, np.ndarray], subset: np.ndarray
) -> dict[str, np.ndarray]:
    """Arrays of ``xp_forward_predictions.npz`` (the contract read by the evaluation)."""
    return {
        "source_id": data.source_id,
        "table_row": data.rows,
        "param_names": np.array(PARAMS),
        "mle": result["mle"],
        "lo68": result["lo68"],
        "hi68": result["hi68"],
        "lo95": result["lo95"],
        "hi95": result["hi95"],
        "E_mle": result["E_mle"],
        "parallax_mle": result["parallax_mle"],
        "chi2_xp": result["chi2_xp"],
        "dof_xp": result["dof_xp"],
        "chi2_phot": result["chi2_phot"],
        "dof_phot": result["dof_phot"],
        "at_bound": result["at_bound"],
        "at_bound_names": np.array(PARAM_NAMES),
        "interval_at_bound": result["interval_at_bound"],
        "subset": np.asarray(subset),
        "g_mag": data.g_mag,
        "label": data.theta,
        "label_err": data.theta_err,
    }


# ─── report ─────────────────────────────────────────────────────────────────

G_BINS = (-np.inf, 13.5, 15.0, 16.0, np.inf)
FEH_BINS = (-np.inf, -3.0, -2.5, -2.0, -1.0, 0.0, np.inf)


def _nmad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return (
        float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")
    )


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo:g}, {hi:g})"


def recovery_summary(pred: dict[str, np.ndarray]) -> dict[str, Any]:
    """Parameter recovery against labels: offsets, NMAD, interval coverage, chi2 by G."""
    label, mle = pred["label"], pred["mle"]
    have = np.isfinite(label).all(axis=1)
    g = pred["g_mag"]
    out: dict[str, Any] = {"n_labelled": int(have.sum())}
    for k, name in enumerate(PARAMS):
        d = (mle[:, k] - label[:, k])[have]
        inside68 = (
            (label[:, k] >= pred["lo68"][:, k]) & (label[:, k] <= pred["hi68"][:, k])
        )[have]
        inside95 = (
            (label[:, k] >= pred["lo95"][:, k]) & (label[:, k] <= pred["hi95"][:, k])
        )[have]
        out[name] = {
            "median_offset": float(np.median(d)) if d.size else None,
            "nmad": _nmad(d),
            "coverage68": float(inside68.mean()) if d.size else None,
            "coverage95": float(inside95.mean()) if d.size else None,
        }
    cells = []
    for glo, ghi in zip(G_BINS[:-1], G_BINS[1:], strict=True):
        for flo, fhi in zip(FEH_BINS[:-1], FEH_BINS[1:], strict=True):
            sel = (
                have
                & (g >= glo)
                & (g < ghi)
                & (label[:, 2] >= flo)
                & (label[:, 2] < fhi)
            )
            if not sel.any():
                continue
            d = mle[sel, 2] - label[sel, 2]
            cells.append({
                "g_bin": _bin_label(glo, ghi), "feh_bin": _bin_label(flo, fhi), "n": int(sel.sum()),
                "median_offset": float(np.median(d)), "nmad": _nmad(d),
                "coverage68": float(((label[sel, 2] >= pred["lo68"][sel, 2]) & (label[sel, 2] <= pred["hi68"][sel, 2])).mean()),
            })  # fmt: skip
    out["feh_by_g_and_feh"] = cells
    chi2 = []
    for glo, ghi in zip(G_BINS[:-1], G_BINS[1:], strict=True):
        sel = (g >= glo) & (g < ghi) & (pred["dof_xp"] > 0)
        if sel.any():
            chi2.append({"g_bin": _bin_label(glo, ghi), "n": int(sel.sum()),
                         "chi2_xp_per_dof": float(pred["chi2_xp"][sel].sum() / pred["dof_xp"][sel].sum()),
                         "median_chi2_xp_per_dof": float(np.median(pred["chi2_xp"][sel] / pred["dof_xp"][sel]))})  # fmt: skip
    out["chi2_xp_by_g"] = chi2
    out["at_bound_fraction"] = (
        dict(zip(PARAM_NAMES, map(float, pred["at_bound"].mean(axis=0)), strict=True))
        if len(g)
        else {}
    )
    return out


def write_report(
    out_dir: str | Path,
    history: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    label_chi2: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> None:
    """``report.json``, ``report.md`` and (with matplotlib) ``training_curves.png``."""
    out = Path(out_dir)
    epochs = [{k: v for k, v in e.items()} for e in history.get("epochs", [])]
    payload = {
        "manifest": manifest,
        "training": epochs,
        "recovery": summaries,
        "chi2_at_labels": label_chi2,
    }
    (out / "report.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n"
    )
    lines = ["# XP forward model report", "", f"Run: `{manifest.get('run', '')}`", ""]
    lines += ["## Training", "", "| epoch | train NLL | valid NLL (profiled) | valid XP chi2/dof | s |",
              "|--:|--:|--:|--:|--:|"]  # fmt: skip
    for e in epochs:
        lines.append(
            f"| {e['epoch']} | {e['train_nll']:.3f} | {e.get('valid_nll', float('nan')):.3f} | "
            f"{e.get('valid_chi2_xp_per_dof', float('nan')):.3f} | {e['seconds']} |"
        )
    if epochs:
        lines += [
            "",
            "Learned scatter (last epoch): `" + json.dumps(epochs[-1]["scatter"]) + "`",
        ]
    if label_chi2:
        lines += ["", "## Whitened XP residuals at the labels (E, parallax profiled)", "",
                  "| G bin | n | chi2/dof | median chi2/dof |", "|:--|--:|--:|--:|"]  # fmt: skip
        for row in label_chi2.get("chi2_xp_by_g", []):
            lines.append(
                f"| {row['g_bin']} | {row['n']} | {row['chi2_xp_per_dof']:.3f} | {row['median_chi2_xp_per_dof']:.3f} |"
            )
    for name, summ in summaries.items():
        lines += ["", f"## Inference: {name} ({summ['n_labelled']} labelled)", "",
                  "| parameter | median offset | NMAD | 68 % coverage | 95 % coverage |", "|:--|--:|--:|--:|--:|"]  # fmt: skip
        for p in PARAMS:
            s = summ[p]
            lines.append(
                f"| {p} | {s['median_offset']} | {s['nmad']:.4g} | {s['coverage68']} | {s['coverage95']} |"
            )
        lines += [
            "",
            "XP chi2/dof at the MLE by G:",
            "",
            "| G bin | n | chi2/dof | median |",
            "|:--|--:|--:|--:|",
        ]
        for row in summ["chi2_xp_by_g"]:
            lines.append(
                f"| {row['g_bin']} | {row['n']} | {row['chi2_xp_per_dof']:.3f} | {row['median_chi2_xp_per_dof']:.3f} |"
            )
        lines += ["", "[Fe/H] recovery by G and [Fe/H] bin:", "", "| G | [Fe/H] | n | offset | NMAD | 68 % cov. |",
                  "|:--|:--|--:|--:|--:|--:|"]  # fmt: skip
        for c in summ["feh_by_g_and_feh"]:
            lines.append(
                f"| {c['g_bin']} | {c['feh_bin']} | {c['n']} | {c['median_offset']:.3f} | {c['nmad']:.3f} | {c['coverage68']:.2f} |"
            )
        lines += ["", f"At bound: `{json.dumps(summ['at_bound_fraction'])}`"]
    lines += [
        "",
        "Interval coverage is against labels that carry their own errors (not included).",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    if epochs:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(
            [e["epoch"] for e in epochs],
            [e["train_nll"] for e in epochs],
            label="train",
        )
        if "valid_nll" in epochs[0]:
            ax.plot(
                [e["epoch"] for e in epochs],
                [e["valid_nll"] for e in epochs],
                label="valid (profiled)",
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("NLL per star")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=120)
        plt.close(fig)
