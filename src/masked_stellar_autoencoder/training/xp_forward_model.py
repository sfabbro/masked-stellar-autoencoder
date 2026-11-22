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
  so the extinction ``A_b = E * softplus(.) >= 0`` for ``E >= 0``; its wavelength
  dependence (the extinction law) and its dependence on the spectrum are learned from
  the data: no stellar-atmosphere grid and no fixed extinction law.

``h0`` and ``h1`` are SiLU MLPs (smooth, so exact Hessians exist for Newton
inference).

Likelihood per star
-------------------
``-log L = NLL_XP + NLL_phot + NLL_parallax`` with

* XP, per band (BP and RP are independent: Gaia publishes no cross covariance):
  ``c_obs - c(theta, E) ~ N(0, C_known + diag((a * c)^2 + b^2) + (c * V)(c * V)^T)``,
  ``c = c(theta, E)``, ``C_known`` the per-star Gaia covariance in table units
  (sidecar) or its diagonal from the table's errors (fallback), ``a`` (fractional),
  ``b`` (floor) and the fractional rank-``k`` factor ``V`` learned
  (:class:`torchregress.losses.KnownCovarianceGaussianNLL` holds them;
  :func:`torchregress.losses.known_covariance_gaussian_nll` evaluates the training
  loss). The learned terms are evaluated at the emulator's prediction ``c``, never
  at the observed coefficients: a covariance that depends on the observation is not
  a likelihood (its factor ``c_obs * V`` aligns with the residual ``c_obs - c`` and
  explains it away; run 2 of 20261122 collapsed to XP chi^2/dof 0.05 and lost the
  ``E`` constraint that way). The training loss evaluates them at ``c`` exactly.
  The nuisance fits and inference hold the covariance fixed at a reference
  prediction ``c(theta_ref, E_ref)`` and refresh it (iteratively reweighted: the
  previous visit's ``E`` in training, the coarse start then the first MLE at
  inference), so each star's covariance is factorised once per pass
  (:class:`PreparedStars`). published covariance. The XP term is conditional on the observed G used for the
  normalisation; the G-flux error adds a rank-one ``x x^T / (flux_over_error)^2``
  term (``< 1e-5`` relative for G < 17), which the learned fractional factor absorbs.
* photometry: ``m_b = M_b(theta, E) + mu``, ``mu = 5 log10(100 / parallax[mas])``,
  ``m_obs - m ~ N(0, sigma_b^2 + s_b^2)`` per band, ``sigma_b`` the quoted error (0
  where the table has none: G, W1, W2) and ``s_b`` a learned per-band scatter (in
  magnitudes, i.e. a fractional flux scatter). BP and RP magnitudes are left out by
  default: they are integrals of the XP spectra and would double count them. 2MASS
  J/H/Ks are left out of :data:`LIKELIHOOD_BANDS`: the fine-tune tables' 2MASS
  columns are matched to the wrong stars (``training.reddening_labels``; the first
  run learned a 1.0 mag scatter for them against 0.37 mag for every other band).
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

Extinction: units, zero point and identifiability
-------------------------------------------------
With a learned extinction term, ``E`` is defined only up to a gauge of the emulator:
``E -> a E`` together with ``h1 -> h1 / a`` leaves every likelihood unchanged, and
per-star ``E_i`` fitted freely absorb any misfit along the extinction direction
(label noise in Teff, emulator error). The first run (``xp_forward_model_20261120``)
showed the consequences: ``E`` on a bound for 42-54 % of the held-out stars, and an
extinction direction mixed with Teff and [Fe/H].

The scale and the zero point are fixed with data, not with a stellar model: training
stars carry the empirical reddening labels (``training.reddening_labels``,
``build_reddening_labels.py``: ``ebv_star``, ``e_ebv_star``; observed BP-RP minus an
intrinsic colour fitted on low-extinction stars as a function of the star's labels,
divided by the Casagrande et al. 2021 ratio ``R_BP - R_RP``; E(B-V) on the SFD scale)
as a Gaussian measurement of their nuisance ``E_i``:

``-log L_i += 1/2 ((E_i - ebv_star_i) / e_ebv_star_i)^2 + log e_ebv_star_i + 1/2 log 2 pi``.

The learned extinction coefficients (``dc/dE``, ``dM_b/dE = A_b / E``) are then in
E(B-V) units and ``E = 0`` sits at the intrinsic colours of the labels'
low-extinction calibration stars. No extinction law is imposed: the wavelength and
spectral-type dependence of the extinction remain learned; only one number per star
anchors the scale. Stars without a label (outside the calibration's (Teff, log g)
cover, 28 % of the table rows) get their ``E_i`` from the data within the box.

The label is a function of the star's spectroscopic labels and of its BP-RP colour
(an integral of the XP spectrum). It is therefore a *training-only* term: at
inference the spectroscopic labels are what is being measured (using them would be
circular) and the colour is already in the XP term (double counting). Within training
the label repeats the colour's share of the information that the XP term carries
about ``E_i`` at the plugged-in ``theta`` (at most a doubled weight for that share, so
each anchored ``E_i`` is at most ~sqrt(2) too tight); its purpose, the
population-level scale and zero point of ``E``, is information that only the labels'
calibration provides. The labels were computed at the fine-tune table's labels, not
at the ``labels.npz`` values plugged in here; the difference (mostly [Fe/H] of the
core stars) moves the intrinsic colour, hence ``E``, by of order 0.01 mag per 0.1 dex
of [Fe/H], small against the labels' 0.03 mag errors except for the stars whose
[Fe/H] moved most. A normalisation convention
(e.g. fixing ``dE(BP-RP)/dE`` to the Casagrande ratio) would be the minimum without
labels; with 72 % of the rows labelled it is redundant, and it would impose the
extinction law in one colour.

``E`` has the support ``[E_MIN, 1.2 * max(EBV_SFD, 0) + 0.05]``, ``E_MIN = -0.05``,
in training and at inference: the zero point is known to ~0.03 mag (the labels'
scatter at low SFD reddening; 11 % of the labels are negative), and a hard bound at
exactly 0 would put about half of the unreddened stars on the bound and bias their
``theta``. The upper bound allows the SFD calibration error (SFD over-predicts by
~14 %, Schlafly & Finkbeiner 2011) and stars behind all of the dust.

Per-star nuisances in training
------------------------------
The data likelihood is invariant under ``E -> g(theta, E)`` for any monotone ``g``
(the emulator absorbs it), so the labels are the only information on the scale and
zero point of ``E``. With ``ebv_mode="label"`` (default) the ``E_i`` of every labelled
training star is held at its label (unclipped), which fixes that gauge exactly; only
the distance modulus and the unlabelled stars' ``E_i`` are fitted. The label noise
then attenuates the learned ``dc/dE`` by ``var(E) / (var(E) + e_label^2)`` (a few per
cent; the validation slope of ``E`` against the labels reports it). With
``ebv_mode="anchor"`` the labels enter as the Gaussian term above and training
maximises ``sum_i min_{E_i, mu_i} [-log L_i(theta_i, E_i, mu_i) + anchor_i(E_i)]`` by
block coordinate descent; the anchor is then only a weak restoring force along the
flat gauge direction (the XP term pins each bright star's ``E_i`` to ~0.005 mag, the
label to ~0.03), and run 2 of 20261122 drifted to ``E = 2 x label + 0.05`` within six
epochs. During the first ``nuisance_warmup_epochs`` (default 1) ``E_i``
stays at its start (the reddening label, else the SFD value) while the emulator learns
a first approximation; then each batch first refines the batch stars' ``(E_i, mu_i)``
(:func:`fit_nuisances`: ``nuisance_steps`` projected Gauss-Newton steps with the
emulator fixed, warm-started from the values stored at the star's previous visit,
exact Jacobians in ``E`` by forward-mode differentiation, Armijo backtracking; the
distance modulus starts from the photometric/parallax estimate on the first visit),
then takes one gradient step of the emulator and the scatter terms at the refined
nuisances. At a nuisance optimum the gradient of the profiled objective equals the
partial gradient (envelope theorem), so nothing is differentiated through the inner
solve. Gauss-Newton is exact for the fixed-covariance XP term, the photometry and the
anchor (quadratic in their residuals); the parallax term enters with its exact
gradient and Gauss-Newton curvature. (The first run took one sparse optimiser step
per star and epoch, so the nuisances stayed near their starts: ``E_i`` at the SFD
value of the full line of sight, the distance at the observed parallax.)

Training and label noise
------------------------
The labels are plugged in (``label_noise="plugin"``). ``label_noise="marginal"`` is
the documented hook for integrating ``theta`` over its label error:
``-log mean_k L(theta_k)``, ``theta_k ~ N(label, e_label^2)`` (an importance-weighted
lower bound of the marginal likelihood that tightens with ``k``; it deconvolves the
emulator from label noise); the nuisances are then refined at the plugged-in labels.
The learned scatter terms are frozen for ``scatter_warmup_epochs`` epochs with fresh
optimiser state (a variance learned against an untrained mean absorbs the misfit).

Inference
---------
Per star, ``maximize_score`` over ``(Teff, log g, [Fe/H], u_E, u_p)`` in a box (the
labels' range for ``theta``), started from the best point of a candidate bank
(training-population ``theta`` x an ``E`` grid, ranked with the XP covariance at
the observation and the distance modulus profiled analytically), re-maximised in
``covariance_passes`` passes with the covariance fixed at the previous point's
prediction (the coarse start, then the previous MLE), then ``profile_interval``
(Wilks 68.27 % and 95 %) for the three atmospheric parameters with the other four
profiled. Every evaluation reuses the star's covariance factor (:class:`PreparedStars`),
so a candidate costs one emulator pass and two 55 x 55 triangular solves. Outputs:
MLE, intervals, ``E`` and parallax at the MLE (with the side of any ``E`` bound),
the inverse observed information of (Teff, log g, [Fe/H]), XP chi^2 and dof at the
MLE, bound flags. The profile scans use a per-star local box around the MLE
(:func:`profile_intervals`) and stack the 68 % and 95 % levels in one call.

Data contracts
--------------
* ``labels.npz``: ``source_id, table_row, teff, e_teff, logg, e_logg, fe_h, e_fe_h,
  has_core_feh, split`` (split 0/1/2 = train/valid/test, -1 not kept; strings
  accepted), optionally ``fe_h_ref`` (raw high-resolution reference, report only).
  :func:`labels_from_table` builds an equivalent from the table.
* Reddening labels (``build_reddening_labels.py``): ``source_id, ebv_star,
  e_ebv_star`` (NaN where not covered), matched by ``source_id``.
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
from dataclasses import asdict, dataclass, fields
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
LN10_OVER_5 = math.log(10.0) / 5.0
E_MAX_SCALE = 1.2
E_MAX_OFFSET = 0.05
#: Lower bound of E(B-V): the zero point's uncertainty (see the module docstring).
E_MIN = -0.05
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
#: Bands used by default: all of :data:`DEFAULT_BANDS` except the mismatched 2MASS.
LIKELIHOOD_BANDS: tuple[str, ...] = tuple(
    b.name for b in DEFAULT_BANDS if b.name not in ("J", "H", "KS")
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
    ``ebv_label``/``ebv_label_err`` are the empirical reddening labels (training
    anchor; NaN where none); ``fe_h_ref`` and ``split`` are carried for the report.
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
    ebv_label: np.ndarray | None = None
    ebv_label_err: np.ndarray | None = None
    fe_h_ref: np.ndarray | None = None
    split: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def subset(self, index: np.ndarray) -> StarData:
        index = np.asarray(index)
        kw = {}
        for name, value in self.__dict__.items():
            kw[name] = value[index] if isinstance(value, np.ndarray) else value
        return StarData(**kw)

    def anchored(self) -> np.ndarray:
        """Bool ``[N]``: rows with a usable reddening label."""
        if self.ebv_label is None or self.ebv_label_err is None:
            return np.zeros(len(self), dtype=bool)
        with np.errstate(invalid="ignore"):
            return (
                np.isfinite(self.ebv_label)
                & np.isfinite(self.ebv_label_err)
                & (self.ebv_label_err > 0)
            )

    def batch(
        self, index: np.ndarray, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Tensors of rows ``index`` on ``device`` (dense XP covariances unpacked)."""
        index = np.asarray(index)

        def t(a: np.ndarray, dt: torch.dtype = dtype) -> Tensor:
            return torch.as_tensor(np.ascontiguousarray(a[index]), device=device).to(dt)

        def full(a: np.ndarray) -> Tensor:
            return torch.as_tensor(a, device=device, dtype=dtype)

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
        out["e_min"] = full(np.full(len(index), E_MIN))
        out["e_max"] = full(e_max)
        out["log_plx_lo"] = full(lo)
        out["log_plx_hi"] = full(hi)
        ok = self.anchored()[index]
        lab = self.ebv_label[index] if self.ebv_label is not None else None
        err = self.ebv_label_err[index] if self.ebv_label_err is not None else None
        # Unanchored rows: label 0 with infinite error (zero weight in the anchor).
        out["ebv_label"] = full(np.where(ok, lab, 0.0) if lab is not None else 0.0 * lo)
        out["ebv_label_err"] = full(
            np.where(ok, err, np.inf)
            if err is not None
            else np.full(len(index), np.inf)
        )
        return out


def nuisance_box(
    ebv_sfd: np.ndarray, parallax_corrected: np.ndarray, parallax_err: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(log10 parallax lo, hi, E_max)`` per star (see the module docstring).

    The lower bound of ``E`` is :data:`E_MIN` for every star.
    """
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


class ReddeningLabels:
    """Empirical E(B-V) labels by Gaia ``source_id`` (``build_reddening_labels.py``).

    The npz holds ``source_id``, ``ebv_star`` and ``e_ebv_star`` (NaN where the
    intrinsic-colour calibration does not cover the star). Units: E(B-V) on the SFD
    scale of the Casagrande et al. (2021) Gaia extinction coefficients.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            missing = [k for k in ("source_id", "ebv_star", "e_ebv_star") if k not in z]
            if missing:
                raise ValueError(f"{path}: reddening label fields missing: {missing}")
            sid = np.asarray(z["source_id"], dtype=np.int64)
            ebv = np.asarray(z["ebv_star"], dtype=np.float64)
            err = np.asarray(z["e_ebv_star"], dtype=np.float64)
        order = np.argsort(sid, kind="stable")
        self.source_id, self.ebv, self.err = sid[order], ebv[order], err[order]
        if len(sid) > 1 and bool((np.diff(self.source_id) == 0).any()):
            raise ValueError(f"{path}: duplicate source_id in the reddening labels")

    def lookup(self, source_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(ebv, err)`` for ``source_id`` (NaN where absent or unusable)."""
        sid = np.asarray(source_id, dtype=np.int64)
        if not len(self.source_id):
            nan = np.full(len(sid), np.nan)
            return nan, nan.copy()
        pos = np.clip(np.searchsorted(self.source_id, sid), 0, len(self.source_id) - 1)
        found = self.source_id[pos] == sid
        ebv, err = self.ebv[pos], self.err[pos]
        with np.errstate(invalid="ignore"):
            ok = found & np.isfinite(ebv) & np.isfinite(err) & (err > 0)
        return np.where(ok, ebv, np.nan), np.where(ok, err, np.nan)

    def summary(self) -> dict[str, Any]:
        ok = np.isfinite(self.ebv) & np.isfinite(self.err) & (self.err > 0)
        return {
            "path": str(self.path),
            "rows": int(len(self.ebv)),
            "labelled": int(ok.sum()),
            "median_error": float(np.median(self.err[ok])) if ok.any() else None,
        }


def load_star_data(
    table: str | Path,
    rows: np.ndarray,
    *,
    labels: dict[str, np.ndarray] | None = None,
    bands: Sequence[PhotBand] = DEFAULT_BANDS,
    sidecar: SidecarXPCovariance | None = None,
    reddening: ReddeningLabels | None = None,
    xp_error_convention: str = "raw_xp",
    zero_point: str = "sidecar",
    zero_point_constant: float = L21_CONSTANT_ZP,
    log: Callable[[str], None] = print,
) -> tuple[StarData, dict[str, Any]]:
    """Read the forward-model inputs of table ``rows`` (and their labels).

    ``zero_point``: ``"sidecar"`` (Lindegren+2021 from the sidecar; requires one),
    ``"constant"`` (``zero_point_constant`` for every star) or ``"none"``. With a
    sidecar, parallaxes and their errors come from the sidecar (raw Gaia values)
    where the star was found, else from the table. ``reddening``: empirical E(B-V)
    labels matched by ``source_id`` (training anchor). Returns the data and a
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
    fe_h_ref = split = None
    if labels is not None:
        pos = _label_positions(labels, rows)
        for k, name in enumerate(PARAMS):
            theta[:, k] = np.asarray(labels[name], dtype=np.float64)[pos]
            theta_err[:, k] = np.asarray(labels[f"e_{name}"], dtype=np.float64)[pos]
        split = np.asarray(labels["split"])[pos].astype(np.int8)
        if "fe_h_ref" in labels:
            fe_h_ref = np.asarray(labels["fe_h_ref"], dtype=np.float64)[pos]
    source_id = raw["source_id"].astype(np.int64)
    ebv_label = ebv_label_err = None
    if reddening is not None:
        ebv_label, ebv_label_err = reddening.lookup(source_id)
        diag["ebv_label_fraction"] = (
            float(np.isfinite(ebv_label).mean()) if len(rows) else 0.0
        )
    data = StarData(
        rows=rows,
        source_id=source_id,
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
        ebv_label=ebv_label,
        ebv_label_err=ebv_label_err,
        fe_h_ref=fe_h_ref,
        split=split,
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
OPTIONAL_LABEL_FIELDS = ("fe_h_ref", "e_fe_h_ref")
SPLIT_CODES = {"train": 0, "valid": 1, "test": 2}


def load_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``labels.npz`` (contract fields; ``split`` as ints 0/1/2/-1 or strings)."""
    with np.load(path, allow_pickle=False) as z:
        missing = [f for f in LABEL_FIELDS if f not in z.files]
        if missing:
            raise ValueError(f"{path}: labels contract fields missing: {missing}")
        out = {f: np.asarray(z[f]) for f in LABEL_FIELDS}
        out.update({f: np.asarray(z[f]) for f in OPTIONAL_LABEL_FIELDS if f in z.files})
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


def _pick(rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) > n:
        return np.sort(rng.choice(rows, size=max(n, 0), replace=False))
    return np.sort(rows)


def select_rows(
    labels: dict[str, np.ndarray],
    core_rows: np.ndarray,
    *,
    max_train: int,
    max_valid: int,
    max_heldout: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Table rows of each role (sorted).

    A row is *core* if it is in ``core_rows`` (the frozen high-resolution core) or
    has ``has_core_feh``. ``train``: every labelled train-split core row, topped up
    to ``max_train`` with a random draw of the other labelled train-split rows (the
    core carries most of the metal-poor and giant coverage, so it is never
    sub-sampled). ``valid`` (model selection) and ``heldout`` (inference) are random
    labelled valid/test-split rows *outside* the core, so the held-out core (core
    rows of the valid and test splits) is used neither for training nor for model
    selection. ``core``: the frozen-core rows (all splits), inferred and reported by
    split. ``core_train``: the core rows in ``train``.
    """
    rng = np.random.default_rng(seed)
    theta = np.stack([np.asarray(labels[p], dtype=np.float64) for p in PARAMS], 1)
    labelled = np.isfinite(theta).all(axis=1)
    table_row = np.asarray(labels["table_row"], dtype=np.int64)
    split = split_codes(labels["split"])
    core_rows = np.unique(np.asarray(core_rows, dtype=np.int64))
    core = np.isin(table_row, core_rows)
    if "has_core_feh" in labels:
        core |= np.asarray(labels["has_core_feh"]).astype(bool)
    pool = labelled & (split == 0)
    core_train = np.sort(table_row[pool & core])
    rest = _pick(table_row[pool & ~core], max_train - len(core_train), rng)
    return {
        "train": np.sort(np.concatenate([core_train, rest])),
        "valid": _pick(table_row[labelled & (split == 1) & ~core], max_valid, rng),
        "heldout": _pick(table_row[labelled & (split == 2) & ~core], max_heldout, rng),
        "core": core_rows,
        "core_train": core_train,
    }


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

    def extinction_coefficients(
        self, theta: Tensor, ebv: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``(dc/dE [M, 110], dM_b/dE [M, nb])`` at ``(theta, E)`` (forward mode)."""
        with torch.no_grad():
            _, (dxp, dmag) = torch.func.jvp(
                lambda e: self(theta, e), (ebv,), (torch.ones_like(ebv),)
            )
        return dxp, dmag

    def describe(self) -> dict[str, Any]:
        """Summary of the learned scatter terms."""
        out: dict[str, Any] = {}
        for name, loss in zip(("bp", "rp"), self.xp_losses, strict=True):
            frac = torch.exp(loss.log_fraction).detach().cpu().numpy()
            floor = torch.exp(loss.log_floor).detach().cpu().numpy()
            out[f"{name}_fraction_median"] = float(np.median(frac))
            out[f"{name}_fraction_p90"] = float(np.quantile(frac, 0.9))
            out[f"{name}_fraction_first5"] = [float(v) for v in frac[:5]]
            out[f"{name}_floor_median"] = float(np.median(floor))
            if loss.factor_weight is not None:
                out[f"{name}_factor_norm"] = float(loss.factor_weight.detach().norm())
        scatter = torch.exp(self.log_mag_scatter).detach().cpu().numpy()
        out["mag_scatter"] = dict(
            zip(self.band_names, map(float, scatter), strict=True)
        )
        return out


# ─── likelihood ─────────────────────────────────────────────────────────────


def _known_xp(batch: dict[str, Tensor], k: int, s: Tensor, nc: int) -> Tensor:
    """Band ``k``'s known covariance ``[B, nc, nc]`` in standardised units."""
    if "xp_cov" in batch:
        cov = batch["xp_cov"][:, k].to(s.dtype)
        return cov / (s.unsqueeze(-1) * s.unsqueeze(-2))
    var = batch["xp_var"][:, k * nc : (k + 1) * nc].to(s.dtype)
    return torch.diag_embed(var / s.square())


def _photometry_nll(
    resid: Tensor, var: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Diagonal Gaussian ``(nll, chi2, dof)`` over the observed bands."""
    maskf = mask.to(resid.dtype)
    r = torch.where(mask, resid, torch.zeros_like(resid))
    v = torch.where(mask, var, torch.ones_like(var))
    chi2 = (r.square() / v * maskf).sum(-1)
    dof = maskf.sum(-1)
    nll = 0.5 * (chi2 + (torch.log(v) * maskf).sum(-1) + dof * LOG_2PI)
    return nll, chi2, dof


def _parallax_nll(
    parallax: Tensor, plx_obs: Tensor, sig: Tensor, has_plx: Tensor
) -> Tensor:
    sig_safe = torch.where(has_plx, sig, torch.ones_like(sig))
    obs = torch.where(has_plx, plx_obs, torch.zeros_like(plx_obs))
    z = torch.where(has_plx, (obs - parallax) / sig_safe, torch.zeros_like(sig))
    return torch.where(
        has_plx,
        0.5 * (z.square() + LOG_2PI) + torch.log(sig_safe),
        torch.zeros_like(sig),
    )


def star_nll(
    model: XPForwardModel,
    theta: Tensor,
    ebv: Tensor,
    parallax: Tensor,
    batch: dict[str, Tensor],
    *,
    parts: bool = False,
    quasi: bool = False,
) -> Tensor | dict[str, Tensor]:
    """Per-star negative log-likelihood ``[M]`` (or its parts with ``parts=True``).

    ``quasi``: the learned covariance is evaluated at the detached prediction, so
    the emulator's mean gets the quasi-likelihood (iteratively reweighted) gradient
    ``J^T C^{-1} r`` without the ``d log|C| / d c`` and ``r^T dC^{-1}/dc r`` terms
    (unbiased for the mean even when the variance model is wrong); the scatter
    parameters still get the full Gaussian gradient.

    The training loss: :func:`torchregress.losses.known_covariance_gaussian_nll` with
    the learned model covariance evaluated at the observed coefficients (gradients
    reach the emulator and every scatter term). ``batch`` holds the stars' data for
    the same ``M`` rows (see :meth:`StarData.batch`); ``parallax`` in mas (> 0). The
    anchor is not included (:func:`anchor_nll`).
    """
    xp_hat, mag_hat = model(theta, ebv)
    scale = model.xp_scale.to(xp_hat.dtype)
    xp_obs = batch["xp"].to(xp_hat.dtype) / scale
    nc = model.n_coeff
    nll_xp = torch.zeros_like(ebv)
    chi2_xp = torch.zeros_like(ebv)
    dof_xp = torch.zeros_like(ebv)
    for k, loss in enumerate(model.xp_losses):
        sl = slice(k * nc, (k + 1) * nc)
        at = xp_hat[:, sl].detach() if quasi else xp_hat[:, sl]
        var, factor = loss.model_covariance_terms(at)
        assert var is not None
        res = known_covariance_gaussian_nll(
            xp_obs[:, sl] - xp_hat[:, sl],
            _known_xp(batch, k, scale[sl], nc),
            model_scale=var.sqrt(),
            model_factor=factor,
            mask=batch["xp_mask"][:, sl],
            max_jitter=loss.max_jitter,
        )
        nll_xp = nll_xp + res.nll
        chi2_xp = chi2_xp + res.chi2
        dof_xp = dof_xp + res.dof
    dist_mod = 5.0 * torch.log10(100.0 / parallax)
    resid = batch["mag"] - (mag_hat + dist_mod.unsqueeze(-1))
    var_m = batch["mag_err"].square() + torch.exp(2.0 * model.log_mag_scatter).to(
        resid.dtype
    )
    nll_phot, chi2_phot, dof_phot = _photometry_nll(resid, var_m, batch["mag_mask"])
    plx_obs = batch["parallax"] - batch["parallax_zp"]
    sig = batch["parallax_err"]
    has_plx = torch.isfinite(plx_obs) & torch.isfinite(sig) & (sig > 0)
    nll_plx = _parallax_nll(parallax, plx_obs, sig, has_plx)
    total = nll_xp + nll_phot + nll_plx
    if not parts:
        return total
    return {
        "nll": total,
        "nll_xp": nll_xp,
        "chi2_xp": chi2_xp,
        "dof_xp": dof_xp,
        "nll_phot": nll_phot,
        "chi2_phot": chi2_phot,
        "dof_phot": dof_phot,
        "nll_parallax": nll_plx,
    }


def anchor_nll(ebv: Tensor, label: Tensor, label_err: Tensor) -> Tensor:
    """Gaussian NLL of the reddening labels ``[M]`` (0 where a star has none)."""
    ok = torch.isfinite(label_err) & (label_err > 0)
    err = torch.where(ok, label_err, torch.ones_like(label_err))
    z = (ebv - label) / err
    return torch.where(
        ok, 0.5 * (z.square() + LOG_2PI) + torch.log(err), torch.zeros_like(ebv)
    )


@dataclass
class PreparedStars:
    """Per-star tensors for repeated likelihood evaluations at a fixed covariance.

    The XP covariance (known + learned model terms at a reference prediction) is
    equilibrated, ``C = D^{1/2} R D^{1/2}``, and ``R = L L^T`` is factorised once per
    star and band (as in :func:`torchregress.losses.known_covariance_gaussian_nll`;
    masked coefficients get unit variance and ``q = 0``). A likelihood evaluation is
    then ``1/2 |L^{-1} (q * (y - c))|^2 + const``. Built by :func:`prepare_stars`.
    """

    xp_y: Tensor  # [n, 110] observed coefficients / xp_scale
    xp_q: Tensor  # [n, 110] D^{-1/2}, 0 where masked
    xp_factor: Tensor  # [n, 2, nc, nc] lower Cholesky factor of R
    xp_log_det: Tensor  # [n] log |C| over the observed coefficients
    xp_dof: Tensor  # [n]
    mag: Tensor
    mag_var: Tensor
    mag_mask: Tensor
    plx: Tensor  # zero-point corrected parallax (0 where none)
    plx_err: Tensor  # 1 where none
    has_plx: Tensor
    e_min: Tensor
    e_max: Tensor
    log_plx_lo: Tensor
    log_plx_hi: Tensor
    ebv_label: Tensor
    ebv_label_err: Tensor

    def __len__(self) -> int:
        return int(self.xp_y.shape[0])

    def take(self, rows: Tensor) -> PreparedStars:
        return PreparedStars(
            **{f.name: getattr(self, f.name)[rows] for f in fields(self)}
        )

    def mu_bounds(self) -> tuple[Tensor, Tensor]:
        """Distance-modulus box ``(lo, hi)`` (the parallax box)."""
        return 5.0 * (2.0 - self.log_plx_hi), 5.0 * (2.0 - self.log_plx_lo)


def _jittered_cholesky(
    corr: Tensor, maskf: Tensor, failed: Tensor, start: float
) -> Tensor:
    """Factor of ``corr + j diag(mask)`` with the smallest power-of-ten ``j`` per failing row."""
    factor = torch.linalg.cholesky_ex(corr)[0]
    pending = failed.clone()
    amount = max(start, 16.0 * torch.finfo(corr.dtype).eps)
    while bool(pending.any()):
        if amount > 1.0:
            raise ValueError("XP covariance could not be factorised with jitter <= 1")
        trial = corr[pending] + torch.diag_embed(maskf[pending] * amount)
        f_try, info = torch.linalg.cholesky_ex(trial)
        idx = pending.nonzero().squeeze(1)
        ok = info == 0
        factor[idx[ok]] = f_try[ok]
        pending[idx[ok]] = False
        amount *= 10.0
    return factor


def model_reference(model: XPForwardModel, theta: Tensor, ebv: Tensor) -> Tensor:
    """Standardised emulator prediction ``c(theta, E)`` ``[n, 110]`` (no gradient).

    The point at which :func:`prepare_stars` evaluates the learned XP covariance.
    """
    with torch.no_grad():
        return model(theta, ebv)[0].detach()


def observed_reference(model: XPForwardModel, batch: dict[str, Tensor]) -> Tensor:
    """Observed coefficients in standardised units, as a :func:`prepare_stars` reference.

    Only for ranking start candidates (:func:`coarse_init`): a covariance evaluated
    at the observation is not a likelihood (see the module docstring).
    """
    return batch["xp"] / model.xp_scale.to(batch["xp"].dtype)


def prepare_stars(
    model: XPForwardModel,
    batch: dict[str, Tensor],
    *,
    reference: Tensor,
    dtype: torch.dtype | None = None,
    factor_dtype: torch.dtype = torch.float64,
) -> PreparedStars:
    """Factorise each star's XP covariance once (``factor_dtype``), cast to ``dtype``.

    ``reference`` ``[n, 110]`` (standardised units): the prediction at which the
    learned model covariance is evaluated, usually :func:`model_reference` at the
    current ``(theta, E)``; the known covariance does not depend on it.
    """
    work = dtype or batch["xp"].dtype
    fd = factor_dtype
    nc = model.n_coeff
    with torch.no_grad():
        scale = model.xp_scale.to(fd)
        y = batch["xp"].to(fd) / scale
        mask = batch["xp_mask"]
        factors, qs = [], []
        log_det = torch.zeros(y.shape[0], dtype=fd, device=y.device)
        for k, loss in enumerate(model.xp_losses):
            sl = slice(k * nc, (k + 1) * nc)
            m = mask[:, sl]
            cov = _known_xp(batch, k, scale[sl], nc)
            var, factor = loss.model_covariance_terms(reference[:, sl].to(fd))
            assert var is not None
            cov = cov + torch.diag_embed(var)
            if factor is not None:
                cov = cov + factor @ factor.mT
            pair = m.unsqueeze(-1) & m.unsqueeze(-2)
            cov = torch.where(pair, cov, torch.zeros((), dtype=fd, device=y.device))
            cov = cov + torch.diag_embed((~m).to(fd))
            d = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(torch.finfo(fd).tiny)
            q = torch.rsqrt(d)
            corr = cov * q.unsqueeze(-1) * q.unsqueeze(-2)
            corr = 0.5 * (corr + corr.mT)
            chol, info = torch.linalg.cholesky_ex(corr)
            if bool((info != 0).any()):
                chol = _jittered_cholesky(
                    corr, m.to(fd), info != 0, max(loss.max_jitter / 1e3, 1e-12)
                )
            factors.append(chol)
            qs.append(q * m.to(fd))
            log_det = log_det + 2.0 * torch.log(
                torch.diagonal(chol, dim1=-2, dim2=-1)
            ).sum(-1)
            log_det = log_det + torch.where(m, torch.log(d), torch.zeros_like(d)).sum(
                -1
            )
        mag_var = batch["mag_err"].to(fd).square() + torch.exp(
            2.0 * model.log_mag_scatter.to(fd)
        )
        plx_obs = batch["parallax"].to(fd) - batch["parallax_zp"].to(fd)
        sig = batch["parallax_err"].to(fd)
        has_plx = torch.isfinite(plx_obs) & torch.isfinite(sig) & (sig > 0)
        return PreparedStars(
            xp_y=y.to(work),
            xp_q=torch.cat(qs, dim=1).to(work),
            xp_factor=torch.stack(factors, dim=1).to(work),
            xp_log_det=log_det.to(work),
            xp_dof=mask.to(work).sum(-1),
            mag=batch["mag"].to(work),
            mag_var=mag_var.to(work),
            mag_mask=batch["mag_mask"],
            plx=torch.where(has_plx, plx_obs, torch.zeros_like(plx_obs)).to(work),
            plx_err=torch.where(has_plx, sig, torch.ones_like(sig)).to(work),
            has_plx=has_plx,
            e_min=batch["e_min"].to(work),
            e_max=batch["e_max"].to(work),
            log_plx_lo=batch["log_plx_lo"].to(work),
            log_plx_hi=batch["log_plx_hi"].to(work),
            ebv_label=batch["ebv_label"].to(work),
            ebv_label_err=batch["ebv_label_err"].to(work),
        )


def _whiten(prep: PreparedStars, values: Tensor, nc: int) -> list[Tensor]:
    """``L_k^{-1} (q * values)`` per band; ``values`` ``[n, 110]`` or ``[n, 110, r]``."""
    out = []
    v = values * (prep.xp_q if values.dim() == 2 else prep.xp_q.unsqueeze(-1))
    for k in range(2):
        rhs = v[:, k * nc : (k + 1) * nc]
        vec = rhs.dim() == 2
        w = torch.linalg.solve_triangular(
            prep.xp_factor[:, k], rhs.unsqueeze(-1) if vec else rhs, upper=False
        )
        out.append(w.squeeze(-1) if vec else w)
    return out


def star_nll_prepared(
    model: XPForwardModel,
    theta: Tensor,
    ebv: Tensor,
    parallax: Tensor,
    prep: PreparedStars,
    *,
    parts: bool = False,
) -> Tensor | dict[str, Tensor]:
    """:func:`star_nll` at the fixed covariance of ``prep`` (same value, no factorisation)."""
    xp_hat, mag_hat = model(theta, ebv)
    nc = model.n_coeff
    chi2_xp = sum(w.square().sum(-1) for w in _whiten(prep, prep.xp_y - xp_hat, nc))
    nll_xp = 0.5 * (chi2_xp + prep.xp_log_det + prep.xp_dof * LOG_2PI)
    nll_xp = torch.where(prep.xp_dof > 0, nll_xp, torch.zeros_like(nll_xp))
    dist_mod = 5.0 * torch.log10(100.0 / parallax)
    resid = prep.mag - (mag_hat + dist_mod.unsqueeze(-1))
    nll_phot, chi2_phot, dof_phot = _photometry_nll(resid, prep.mag_var, prep.mag_mask)
    nll_plx = _parallax_nll(parallax, prep.plx, prep.plx_err, prep.has_plx)
    total = nll_xp + nll_phot + nll_plx
    if not parts:
        return total
    return {
        "nll": total,
        "nll_xp": nll_xp,
        "chi2_xp": chi2_xp,
        "dof_xp": prep.xp_dof,
        "nll_phot": nll_phot,
        "chi2_phot": chi2_phot,
        "dof_phot": dof_phot,
        "nll_parallax": nll_plx,
    }


def unit_to_nuisance(u: Tensor, batch: Any) -> tuple[Tensor, Tensor]:
    """``(E, parallax [mas])`` from unit-box coordinates ``u [M, 2]`` of each star.

    ``batch``: a :meth:`StarData.batch` dict or :class:`PreparedStars` (``e_min``
    defaults to :data:`E_MIN`).
    """
    get = batch.get if isinstance(batch, dict) else lambda k, d=None: getattr(batch, k)
    e_max = get("e_max")
    e_min = get("e_min", None)
    if e_min is None:
        e_min = torch.full_like(e_max, E_MIN)
    ebv = e_min + u[:, 0] * (e_max - e_min)
    lo, hi = get("log_plx_lo"), get("log_plx_hi")
    parallax = torch.pow(10.0, lo + u[:, 1] * (hi - lo))
    return ebv, parallax


def nuisance_to_unit(
    ebv: np.ndarray,
    parallax: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    e_max: np.ndarray,
    e_min: float | np.ndarray = E_MIN,
) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        ue = (np.asarray(ebv) - e_min) / (e_max - e_min)
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


# ─── per-star nuisances: (E, distance modulus) ──────────────────────────────


def distance_modulus_start(
    model: XPForwardModel, theta: Tensor, ebv: Tensor, prep: PreparedStars
) -> Tensor:
    """Distance modulus from the photometry and the parallax at ``(theta, E)``.

    Inverse-variance mean of ``m_b - M_b`` and of ``5 log10(100 / parallax)`` (delta
    method, parallaxes above 2 sigma), clipped to the box; the box centre for a star
    with neither.
    """
    with torch.no_grad():
        _, mag_hat = model(theta, ebv)
        iv = prep.mag_mask.to(mag_hat.dtype) / prep.mag_var
        num = (iv * (prep.mag - mag_hat)).sum(-1)
        den = iv.sum(-1)
        good = prep.has_plx & (prep.plx > 2.0 * prep.plx_err)
        plx = torch.where(good, prep.plx, torch.ones_like(prep.plx))
        mu_p = 5.0 * torch.log10(100.0 / plx)
        w_p = torch.where(
            good, (plx / (prep.plx_err / LN10_OVER_5)).square(), torch.zeros_like(plx)
        )
        lo, hi = prep.mu_bounds()
        mu = torch.where(
            den + w_p > 0,
            (num + w_p * mu_p) / (den + w_p).clamp_min(torch.finfo(plx.dtype).tiny),
            0.5 * (lo + hi),
        )
        return torch.minimum(torch.maximum(mu, lo), hi)


def _nuisance_objective(
    model: XPForwardModel,
    theta: Tensor,
    v: Tensor,
    prep: PreparedStars,
    *,
    anchor: bool,
    jac: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """NLL in ``v = (E, mu)`` without constant terms; with ``jac`` its gradient
    ``[n, 2]`` and Gauss-Newton curvature ``[n, 2, 2]``."""
    ebv, mu = v[:, 0], v[:, 1]
    nc = model.n_coeff
    if jac:
        (xp_hat, mag_hat), (dxp, dmag) = torch.func.jvp(
            lambda e: model(theta, e), (ebv,), (torch.ones_like(ebv),)
        )
    else:
        xp_hat, mag_hat = model(theta, ebv)
    n = ebv.shape[0]
    g = h = None
    if jac:
        both = torch.stack([prep.xp_y - xp_hat, -dxp], dim=-1)  # [n, 110, 2]
        ws = _whiten(prep, both, nc)
        wr = torch.cat([w[..., 0] for w in ws], dim=1)
        wj = torch.cat([w[..., 1] for w in ws], dim=1)
        f = 0.5 * wr.square().sum(-1)
        g = torch.zeros(n, 2, dtype=ebv.dtype, device=ebv.device)
        h = torch.zeros(n, 2, 2, dtype=ebv.dtype, device=ebv.device)
        g[:, 0] = (wr * wj).sum(-1)
        h[:, 0, 0] = wj.square().sum(-1)
    else:
        f = 0.5 * sum(w.square().sum(-1) for w in _whiten(prep, prep.xp_y - xp_hat, nc))
    maskf = prep.mag_mask.to(ebv.dtype)
    iv = maskf / prep.mag_var
    res = (prep.mag - mag_hat - mu.unsqueeze(-1)) * maskf
    f = f + 0.5 * (iv * res.square()).sum(-1)
    plx_model = 100.0 * torch.pow(10.0, -mu / 5.0)
    hp = prep.has_plx.to(ebv.dtype)
    z = (prep.plx - plx_model) / prep.plx_err * hp
    f = f + 0.5 * z.square()
    if anchor:
        iv_a = torch.where(
            torch.isfinite(prep.ebv_label_err),
            prep.ebv_label_err.square().reciprocal(),
            torch.zeros_like(ebv),
        )
        d = ebv - prep.ebv_label
        f = f + 0.5 * iv_a * d.square()
    if jac:
        assert g is not None and h is not None
        g[:, 0] = g[:, 0] - (iv * res * dmag).sum(-1)
        g[:, 1] = -(iv * res).sum(-1)
        h[:, 0, 0] = h[:, 0, 0] + (iv * dmag.square()).sum(-1)
        h[:, 0, 1] = (iv * dmag).sum(-1)
        h[:, 1, 1] = iv.sum(-1)
        dz = LN10_OVER_5 * plx_model / prep.plx_err * hp  # dz/dmu
        g[:, 1] = g[:, 1] + z * dz
        h[:, 1, 1] = h[:, 1, 1] + dz.square()
        if anchor:
            g[:, 0] = g[:, 0] + iv_a * d
            h[:, 0, 0] = h[:, 0, 0] + iv_a
        h[:, 1, 0] = h[:, 0, 1]
    return f, g, h


def fit_nuisances(
    model: XPForwardModel,
    theta: Tensor,
    prep: PreparedStars,
    ebv: Tensor,
    mu: Tensor,
    *,
    steps: int = 3,
    anchor: bool = True,
    hold_ebv: bool | Tensor = False,
    max_backtracks: int = 6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Projected Gauss-Newton refinement of ``(E, mu)`` per star with ``theta`` fixed.

    Box: ``E`` in ``[e_min, e_max]``, ``mu`` in the parallax box. ``hold_ebv`` (a
    bool, or one per star): only ``mu`` moves, and a held ``E`` is kept as given
    (not clipped to the box). Coordinates on a
    bound whose gradient points out of the box are held; the step is capped
    (0.5 mag in ``E``, 2 mag in ``mu``) and accepted by Armijo backtracking along the
    projection arc (monotone decrease). Returns ``(E, mu, objective)``; the objective
    omits terms constant in the nuisances.
    """
    mu_lo, mu_hi = prep.mu_bounds()
    lo = torch.stack([prep.e_min, mu_lo], dim=1)
    hi = torch.stack([prep.e_max, mu_hi], dim=1)
    if isinstance(hold_ebv, Tensor):
        hold = hold_ebv.to(device=ebv.device, dtype=torch.bool)
    else:
        hold = torch.full_like(ebv, bool(hold_ebv), dtype=torch.bool)
    if bool(hold.any()):
        lo[:, 0] = torch.where(hold, torch.full_like(ebv, -math.inf), lo[:, 0])
        hi[:, 0] = torch.where(hold, torch.full_like(ebv, math.inf), hi[:, 0])
    cap = torch.tensor([0.5, 2.0], dtype=ebv.dtype, device=ebv.device)
    v = torch.minimum(torch.maximum(torch.stack([ebv, mu], dim=1), lo), hi)
    tiny = torch.finfo(v.dtype).tiny
    with torch.no_grad():
        f = None
        for _ in range(steps):
            f, g, h = _nuisance_objective(
                model, theta, v, prep, anchor=anchor, jac=True
            )
            assert g is not None and h is not None
            free = ~(((v <= lo) & (g > 0)) | ((v >= hi) & (g < 0)))
            free[:, 0] &= ~hold
            freef = free.to(v.dtype)
            gf = g * freef
            hf = h * freef.unsqueeze(2) * freef.unsqueeze(1)
            hf = hf + torch.diag_embed(1.0 - freef)
            damp = 1e-10 * torch.diagonal(hf, dim1=1, dim2=2).abs().sum(-1) + tiny
            a, b, c = hf[:, 0, 0] + damp, hf[:, 0, 1], hf[:, 1, 1] + damp
            det = (a * c - b * b).clamp_min(tiny)
            step = -torch.stack(
                [
                    (c * gf[:, 0] - b * gf[:, 1]) / det,
                    (a * gf[:, 1] - b * gf[:, 0]) / det,
                ],
                dim=1,
            )
            step = step * freef
            step = step / (step.abs() / cap).amax(-1, keepdim=True).clamp_min(1.0)
            pending = torch.arange(v.shape[0], device=v.device)
            alpha = 1.0
            for _ in range(max_backtracks):
                if pending.numel() == 0:
                    break
                base = v[pending]
                trial = torch.minimum(
                    torch.maximum(base + alpha * step[pending], lo[pending]),
                    hi[pending],
                )
                ft, _, _ = _nuisance_objective(
                    model,
                    theta[pending],
                    trial,
                    prep.take(pending),
                    anchor=anchor,
                    jac=False,
                )
                gain = (g[pending] * (trial - base)).sum(-1)
                ok = torch.isfinite(ft) & (ft <= f[pending] + 1e-4 * gain)
                acc = pending[ok]
                v[acc] = trial[ok]
                f[acc] = ft[ok]
                pending = pending[~ok]
                alpha *= 0.5
        f, _, _ = _nuisance_objective(model, theta, v, prep, anchor=anchor, jac=False)
    return v[:, 0].contiguous(), v[:, 1].contiguous(), f


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
    nuisance_steps: int = 3
    nuisance_warmup_epochs: int = 1
    anchor: bool = True
    scatter_warmup_epochs: int = 1
    grad_clip: float = 10.0
    label_noise: str = "plugin"
    label_noise_samples: int = 8
    valid_rows: int = 4096
    seed: int = 0
    log_every: int = 200
    # "full": the Gaussian likelihood's gradient for the emulator's mean (through
    # the covariance's dependence on the prediction too); "quasi": the
    # quasi-likelihood gradient (covariance at the detached prediction).
    mean_gradient: str = "full"
    # How the reddening labels fix the E scale and zero point in training. The
    # data likelihood is invariant under E -> g(theta, E) (the emulator absorbs
    # it), so only the labels set the gauge. "label": E of every labelled star is
    # held at its label (the gauge is fixed exactly; label noise attenuates the
    # learned dc/dE by var(E) / (var(E) + e_label^2), a few per cent). "anchor":
    # E fitted with the labels as a Gaussian term; under alternating updates the
    # term is a weak restoring force along the flat gauge direction, and run 2 of
    # 20261122 drifted to E = 2 x label (slope) + 0.05 within 6 epochs.
    ebv_mode: str = "label"


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
    ebv: Tensor,
    parallax: Tensor,
    cfg: TrainConfig,
    gen: torch.Generator,
) -> Tensor:
    """Per-star training NLL ``[B]`` at the nuisances (label-noise hook applied).

    The data terms only: the anchor is constant in the emulator at fixed ``E``.
    """
    draws = _theta_draws(batch, cfg, gen)
    if cfg.mean_gradient not in ("full", "quasi"):
        raise ValueError(f"unknown mean_gradient {cfg.mean_gradient!r}")
    quasi = cfg.mean_gradient == "quasi"
    nlls = torch.stack(
        [star_nll(model, th, ebv, parallax, batch, quasi=quasi) for th in draws]
    )
    if nlls.shape[0] == 1:
        return nlls[0]
    return -(torch.logsumexp(-nlls, dim=0) - math.log(nlls.shape[0]))


def initial_ebv(
    data: StarData, anchor: bool = True, hold_labels: bool = False
) -> np.ndarray:
    """Training start of ``E``: the reddening label (or SFD without one), in the box.

    ``hold_labels``: labelled stars keep their label unclipped (it is held there).
    """
    lo, hi, e_max = nuisance_box(
        data.ebv_sfd, data.parallax - data.parallax_zp, data.parallax_err
    )
    sfd = np.where(np.isfinite(data.ebv_sfd), np.maximum(data.ebv_sfd, 0.0), 0.0)
    start = sfd.copy()
    if anchor and data.ebv_label is not None:
        ok = data.anchored()
        start[ok] = data.ebv_label[ok]
    clipped = np.clip(start, E_MIN, e_max)
    if anchor and hold_labels and data.ebv_label is not None:
        clipped = np.where(data.anchored(), start, clipped)
    return clipped


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

    Block coordinate descent on the profile likelihood (module docstring): per
    batch, :func:`fit_nuisances` refines the stars' ``(E, mu)`` (anchored by the
    reddening labels with ``cfg.anchor``), then one optimiser step updates the
    emulator and the scatter terms at those nuisances. Returns the history (per-epoch
    train/valid NLL, learned scatter, ``E`` recovery at the labels on the validation
    stars) and the per-star nuisances of the training stars. The model keeps the
    parameters of the epoch with the lowest validation NLL (nuisances profiled, no
    anchor), or of the last epoch without validation stars.
    """
    ok = np.isfinite(train.theta).all(axis=1)
    if not ok.all():
        raise ValueError("training stars must all be labelled")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    if cfg.ebv_mode not in ("label", "anchor"):
        raise ValueError(f"unknown ebv_mode {cfg.ebv_mode!r}")
    hold_labels = cfg.anchor and cfg.ebv_mode == "label"
    model.to(device=device, dtype=dtype)
    n = len(train)
    ebv_store = torch.as_tensor(
        initial_ebv(train, cfg.anchor, hold_labels=hold_labels),
        device=device,
        dtype=dtype,
    )
    mu_store = torch.zeros(n, device=device, dtype=dtype)
    visited = torch.zeros(n, device=device, dtype=torch.bool)
    network = model.network_parameters()
    scatter = model.scatter_parameters()
    opt = torch.optim.AdamW(
        [
            {"params": network, "weight_decay": cfg.weight_decay, "lr": cfg.lr},
            {"params": scatter, "weight_decay": 0.0, "lr": cfg.scatter_lr},
        ]
    )
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs

    def cosine(s: int) -> float:
        return 0.5 * (
            1.0 + math.cos(math.pi * min(s, total_steps) / max(total_steps, 1))
        )

    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine)
    history: dict[str, Any] = {
        "epochs": [],
        "config": asdict(cfg),
        "anchored_fraction": float(train.anchored().mean()) if n else 0.0,
    }
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
        sums = {"nll": 0.0, "anchor": 0.0, "count": 0}
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            batch = train.batch(idx, device, dtype)
            idx_t = torch.as_tensor(idx, device=device)
            theta = batch["theta"]
            ebv0 = ebv_store[idx_t]
            prep = prepare_stars(
                model, batch, reference=model_reference(model, theta, ebv0),
                dtype=dtype, factor_dtype=dtype,
            )  # fmt: skip
            new = ~visited[idx_t]
            mu0 = mu_store[idx_t]
            if bool(new.any()):
                mu0 = torch.where(
                    new, distance_modulus_start(model, theta, ebv0, prep), mu0
                )
            if epoch < cfg.nuisance_warmup_epochs:
                held = torch.ones_like(ebv0, dtype=torch.bool)
            elif hold_labels:
                held = torch.isfinite(batch["ebv_label_err"])
            else:
                held = torch.zeros_like(ebv0, dtype=torch.bool)
            ebv, mu, _ = fit_nuisances(
                model, theta, prep, ebv0, mu0, steps=cfg.nuisance_steps,
                anchor=cfg.anchor, hold_ebv=held,
            )  # fmt: skip
            ebv_store[idx_t] = ebv
            mu_store[idx_t] = mu
            visited[idx_t] = True
            parallax = 100.0 * torch.pow(10.0, -mu / 5.0)
            loss_rows = batch_loss(model, batch, ebv, parallax, cfg, gen)
            loss = loss_rows.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network + [p for p in scatter if p.requires_grad], cfg.grad_clip
            )
            opt.step()
            sched.step()
            step += 1
            sums["nll"] += float(loss_rows.detach().sum())
            if cfg.anchor:
                sums["anchor"] += float(
                    anchor_nll(ebv, batch["ebv_label"], batch["ebv_label_err"]).sum()
                )
            sums["count"] += len(idx)
            if cfg.log_every and step % cfg.log_every == 0:
                log(
                    f"xp_forward: epoch {epoch} step {step}/{total_steps} nll {float(loss):.3f}"
                )
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_nll": sums["nll"] / max(sums["count"], 1),
            "train_anchor_nll": sums["anchor"] / max(sums["count"], 1),
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
            record["valid_ebv_vs_label"] = ebv_recovery(
                prof["ebv"], valid_sub.ebv_label, valid_sub.ebv_label_err
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
    history["train_nuisance"] = {
        "ebv": ebv_store.detach().cpu().numpy().astype(np.float64),
        "parallax": (100.0 * 10.0 ** (-mu_store.detach().cpu().double().numpy() / 5.0)),
    }
    history["best_valid_nll"] = best_valid if math.isfinite(best_valid) else None
    return history


def ebv_recovery(
    ebv: np.ndarray, label: np.ndarray | None, label_err: np.ndarray | None
) -> dict[str, Any]:
    """``E`` against the reddening labels: median offset, NMAD, pull NMAD."""
    if label is None or label_err is None:
        return {"n": 0}
    with np.errstate(invalid="ignore"):
        ok = (
            np.isfinite(ebv)
            & np.isfinite(label)
            & np.isfinite(label_err)
            & (label_err > 0)
        )
    if not ok.any():
        return {"n": 0}
    d = ebv[ok] - label[ok]
    return {
        "n": int(ok.sum()),
        "median_offset": float(np.median(d)),
        "nmad": _nmad(d),
        "pull_nmad": _nmad(d / label_err[ok]),
        "slope": float(np.polyfit(label[ok], ebv[ok], 1)[0]) if ok.sum() > 2 else None,
    }


# ─── inference ──────────────────────────────────────────────────────────────


def _score_fn(
    model: XPForwardModel,
    prep: PreparedStars,
    fixed_theta: Tensor | None = None,
) -> Callable[[Tensor, Tensor], Tensor]:
    """``score(params [M, P], x [M, 1]) -> [M]`` with ``x`` the star's local index."""

    def score(params: Tensor, x: Tensor) -> Tensor:
        idx = x[:, 0].round().long()
        sub = prep.take(idx)
        if fixed_theta is None:
            theta, u = params[:, :3], params[:, 3:5]
        else:
            theta, u = fixed_theta[idx], params
        ebv, parallax = unit_to_nuisance(u, sub)
        out = star_nll_prepared(model, theta, ebv, parallax, sub)
        assert isinstance(out, Tensor)
        return -out

    return score


def profile_nuisance_at_labels(
    model: XPForwardModel,
    data: StarData,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    chunk_rows: int = 4096,
    steps: int = 12,
    anchor: bool = False,
) -> dict[str, np.ndarray]:
    """Likelihood at the labels with ``E`` and parallax profiled (emulator diagnostic).

    :func:`fit_nuisances` from three ``E`` starts (0, mid-box, the upper bound) with
    the distance modulus started from the photometry; the best end point is kept.
    Without ``anchor`` this is the data likelihood (validation NLL, ``E`` against the
    reddening labels).
    """
    model_d = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    for p in model_d.parameters():
        p.requires_grad_(False)
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in ("nll", "chi2_xp", "dof_xp", "ebv", "parallax")
    }
    for start in range(0, len(data), chunk_rows):
        idx = np.arange(start, min(start + chunk_rows, len(data)))
        batch = data.batch(idx, device, dtype)
        theta = batch["theta"]
        n = len(idx)
        e_min, e_max = batch["e_min"], batch["e_max"]
        starts = torch.stack(
            [torch.zeros_like(e_max).clamp(e_min, e_max), 0.5 * (e_min + e_max), e_max]
        )
        rows = torch.arange(n, device=device).repeat(3)
        batch3 = {k: v[rows] for k, v in batch.items()}
        theta3, ebv0 = theta[rows], starts.reshape(-1)
        prep3 = prepare_stars(
            model_d, batch3, reference=model_reference(model_d, theta3, ebv0),
            dtype=dtype,
        )  # fmt: skip
        mu0 = distance_modulus_start(model_d, theta3, ebv0, prep3)
        ebv, mu, f = fit_nuisances(
            model_d, theta3, prep3, ebv0, mu0, steps=steps, anchor=anchor
        )
        del prep3, batch3
        best = torch.nan_to_num(f.view(3, n), nan=math.inf).argmin(0)
        pick = best * n + torch.arange(n, device=device)
        ebv, mu = ebv[pick], mu[pick]
        # Refresh the covariance at the best end point and refine (one IRLS pass).
        prep = prepare_stars(
            model_d, batch, reference=model_reference(model_d, theta, ebv), dtype=dtype
        )
        ebv, mu, _ = fit_nuisances(
            model_d, theta, prep, ebv, mu, steps=steps, anchor=anchor
        )
        parallax = 100.0 * torch.pow(10.0, -mu / 5.0)
        with torch.no_grad():
            parts = star_nll_prepared(model_d, theta, ebv, parallax, prep, parts=True)
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
    prep: PreparedStars,
    bank: Tensor,
    *,
    star_block: int = 32,
) -> Tensor:
    """Best bank point per star as a start ``[n, 5]`` (theta, u_E, u_parallax).

    The XP term is exact (the star's fixed covariance factor); the distance modulus
    is profiled analytically from the photometry and the parallax (delta method);
    bank points outside the star's ``E`` box are excluded.
    """
    dtype, device = bank.dtype, bank.device
    with torch.no_grad():
        xp_bank, mag_bank = model(bank[:, :3], bank[:, 3])  # [K, 110], [K, nb]
        n, nc = prep.xp_y.shape[0], model.n_coeff
        chi2 = torch.zeros(n, bank.shape[0], dtype=dtype, device=device)
        for k in range(2):
            sl = slice(k * nc, (k + 1) * nc)
            for start in range(0, n, star_block):
                rs = slice(start, min(start + star_block, n))
                resid = prep.xp_y[rs, sl].unsqueeze(-1) - xp_bank[:, sl].T.unsqueeze(0)
                resid = resid * prep.xp_q[rs, sl].unsqueeze(-1)  # [b, nc, K]
                w = torch.linalg.solve_triangular(
                    prep.xp_factor[rs, k], resid, upper=False
                )
                chi2[rs] += w.square().sum(1)
        # Photometry with the distance modulus profiled (weighted mean incl. parallax),
        # expanded into [n, K] matrix products (no [n, K, bands] tensor), in float64.
        f64 = torch.float64
        w_m = prep.mag_mask.to(f64) / prep.mag_var.to(f64)  # [n, nb]
        mag = prep.mag.to(f64)
        bank_m = mag_bank.to(f64)  # [K, nb]
        plx, plx_err = prep.plx.to(f64), prep.plx_err.to(f64)
        good = prep.has_plx & (plx > 2.0 * plx_err)
        plx_safe = torch.where(good, plx, torch.ones_like(plx))
        mu_plx = 5.0 * torch.log10(100.0 / plx_safe)
        var_mu = (plx_err / LN10_OVER_5 / plx_safe).square()
        w_plx = torch.where(good, 1.0 / var_mu, torch.zeros_like(var_mu))
        sum_wd = (w_m * mag).sum(-1, keepdim=True) - w_m @ bank_m.T  # sum_b w (m - M)
        sum_wdd = (
            (w_m * mag.square()).sum(-1, keepdim=True)
            - 2.0 * (w_m * mag) @ bank_m.T
            + w_m @ bank_m.square().T
        )
        w_sum = w_m.sum(-1, keepdim=True)
        mu = (sum_wd + (w_plx * mu_plx).unsqueeze(1)) / (
            w_sum + w_plx.unsqueeze(1)
        ).clamp_min(torch.finfo(f64).tiny)
        mu_lo, mu_hi = (b.to(f64) for b in prep.mu_bounds())
        mu = torch.minimum(torch.maximum(mu, mu_lo.unsqueeze(1)), mu_hi.unsqueeze(1))
        chi2_phot = (sum_wdd - 2.0 * mu * sum_wd + mu.square() * w_sum).clamp_min(0.0)
        chi2_plx = w_plx.unsqueeze(1) * (mu - mu_plx.unsqueeze(1)).square()
        chi2_phot, chi2_plx, mu = (t.to(dtype) for t in (chi2_phot, chi2_plx, mu))
        total = chi2 + chi2_phot + chi2_plx
        e = bank[:, 3].unsqueeze(0)
        allowed = (e <= prep.e_max.unsqueeze(1)) & (e >= prep.e_min.unsqueeze(1))
        total = torch.where(allowed, total, torch.full_like(total, math.inf))
        total = torch.nan_to_num(total, nan=math.inf)
        best = total.argmin(dim=1)
        theta = bank[best, :3]
        u_e = ((bank[best, 3] - prep.e_min) / (prep.e_max - prep.e_min)).clamp(0.0, 1.0)
        log_plx = 2.0 - mu[torch.arange(n, device=device), best] / 5.0
        lo, hi = prep.log_plx_lo, prep.log_plx_hi
        u_p = ((log_plx - lo) / (hi - lo)).clamp(0.0, 1.0)
    return torch.cat([theta, u_e.unsqueeze(1), u_p.unsqueeze(1)], dim=1)


def observed_information(
    score: Callable[[Tensor, Tensor], Tensor],
    params: Tensor,
    x: Tensor,
    chunk_size: int = 8192,
) -> Tensor:
    """Negative Hessian of ``score`` in ``params`` ``[n, P, P]`` (rows independent)."""
    n_par = params.shape[1]
    out = []
    for start in range(0, params.shape[0], chunk_size):
        sl = slice(start, start + chunk_size)
        with torch.enable_grad():
            leaf = params[sl].detach().requires_grad_(True)
            (g,) = torch.autograd.grad(
                score(leaf, x[sl]).sum(), leaf, create_graph=True
            )
            rows = [
                torch.autograd.grad(g[:, i].sum(), leaf, retain_graph=i < n_par - 1)[0]
                for i in range(n_par)
            ]
        out.append(-torch.stack(rows, dim=1).detach())
    return torch.cat(out) if out else params.new_zeros(0, n_par, n_par)


def inverse_information(info: Tensor) -> Tensor:
    """Inverse of each observed-information matrix; NaN where not positive definite."""
    chol, bad = torch.linalg.cholesky_ex(info)
    eye = torch.eye(info.shape[1], dtype=info.dtype, device=info.device).expand_as(info)
    cov = torch.cholesky_solve(eye, chol)
    return torch.where((bad == 0).view(-1, 1, 1), cov, torch.full_like(cov, math.nan))


def profile_intervals(
    score: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    theta_hat: Tensor,
    s_max: Tensor,
    bounds: np.ndarray,
    thresholds: Sequence[float],
    cfg: InferenceConfig,
    n_profiled: int = 3,
    covariance: Tensor | None = None,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Profile intervals ``(lower, upper, at_bound)`` ``[n, n_profiled]`` per threshold.

    :func:`torchregress.inference.profile_interval` scans each side from the MLE to
    the box edge. With the labels' range as the box, every scan point of a typical
    star lies tens of sigma away and each needs a long inner maximisation. Here each
    star's profiled parameters get a local box, the MLE +- ``cfg.profile_local_sigma``
    marginal standard deviations (from the observed information, all five
    parameters free) intersected with the global box, through a per-star affine map
    to the unit box; the nuisance coordinates keep their global box. A side that
    reaches a local edge which is not a global edge is recomputed in the global box
    (as is any star whose observed information is not positive definite), so the
    intervals are those of the global box: the local box only moves the scan points
    next to the crossing.
    """
    glo = torch.as_tensor(bounds[:, 0], device=x.device, dtype=x.dtype)
    ghi = torch.as_tensor(bounds[:, 1], device=x.device, dtype=x.dtype)
    n = theta_hat.shape[0]
    k = n_profiled
    width = ghi[:k] - glo[:k]
    lo = glo[:k].expand(n, k).clone()
    hi = ghi[:k].expand(n, k).clone()
    if cfg.profile_local_sigma > 0:
        cov = covariance
        if cov is None:
            cov = inverse_information(
                observed_information(score, theta_hat, x, cfg.chunk_size)
            )
        sd = torch.diagonal(cov, dim1=1, dim2=2)[:, :k].clamp_min(0.0).sqrt()
        ok = torch.isfinite(sd)
        half = (cfg.profile_local_sigma * sd).clamp(min=1e-3 * width)
        half = torch.where(ok, half, width.expand(n, k))
        lo = torch.maximum(theta_hat[:, :k] - half, lo)
        hi = torch.minimum(theta_hat[:, :k] + half, hi)
    span = hi - lo

    def local_score(phi: Tensor, xx: Tensor) -> Tensor:
        idx = xx[:, 0].round().long()
        theta = lo[idx] + phi[:, :k] * span[idx]
        return score(torch.cat([theta, phi[:, k:]], dim=1), xx)

    phi_hat = torch.cat(
        [((theta_hat[:, :k] - lo) / span).clamp(0.0, 1.0), theta_hat[:, k:]], 1
    )
    unit = [[0.0, 1.0]] * theta_hat.shape[1]
    kw = dict(steps=cfg.profile_steps, tol=cfg.profile_tol, chunk_size=cfg.chunk_size)
    # All thresholds in one call per parameter (rows stacked, per-row threshold).
    m = len(thresholds)
    thr = torch.cat(
        [torch.full((n,), float(t), dtype=x.dtype, device=x.device) for t in thresholds]
    )
    xs, phis, smax = x.repeat(m, 1), phi_hat.repeat(m, 1), s_max.repeat(m)
    lo_s, hi_s, span_s = lo.repeat(m, 1), hi.repeat(m, 1), span.repeat(m, 1)
    lower = torch.empty(m * n, k, dtype=x.dtype, device=x.device)
    upper = torch.empty_like(lower)
    at_bound = torch.zeros(m * n, k, dtype=torch.bool, device=x.device)
    for j in range(k):
        pi = profile_interval(local_score, xs, phis, smax, j, thr, unit,
                              n_scan=cfg.profile_scan_local, **kw)  # fmt: skip
        lower[:, j] = lo_s[:, j] + pi.lower * span_s[:, j]
        upper[:, j] = lo_s[:, j] + pi.upper * span_s[:, j]
        lo_edge, hi_edge = pi.lower_at_bound, pi.upper_at_bound
        redo = (lo_edge & (lo_s[:, j] > glo[j])) | (hi_edge & (hi_s[:, j] < ghi[j]))
        redo = redo.nonzero().squeeze(1)
        at_bound[:, j] = lo_edge | hi_edge
        if redo.numel():
            star = redo % n
            g = profile_interval(score, x[star], theta_hat[star], s_max[star], j, thr[redo],
                                 bounds, n_scan=cfg.profile_scan, **kw)  # fmt: skip
            lower[redo, j], upper[redo, j] = g.lower, g.upper
            at_bound[redo, j] = g.lower_at_bound | g.upper_at_bound
    return [
        (
            lower[i * n : (i + 1) * n],
            upper[i * n : (i + 1) * n],
            at_bound[i * n : (i + 1) * n],
        )
        for i in range(m)
    ]


@dataclass
class InferenceConfig:
    chunk_rows: int = 2048
    chunk_size: int = 8192
    n_candidates: int = 8
    n_starts: int = 1
    steps: int = 30
    # MLE passes, each with the XP covariance fixed at the previous point's
    # prediction (coarse start, then the previous MLE); the intervals use the last.
    covariance_passes: int = 2
    profile_steps: int = 10
    profile_scan: int = 4
    profile_scan_local: int = 2
    profile_local_sigma: float = 8.0
    profile_tol: float = 1e-3
    bank_theta: int = 1024
    ebv_grid: tuple[float, ...] = (
        -0.03,
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
    """MLE, profile intervals, nuisances and goodness of fit for every star of ``data``.

    ``cov_theta`` ``[n, 3, 3]``: the (Teff, log g, [Fe/H]) block of the inverse
    observed information at the MLE (nuisances profiled; NaN where not positive
    definite), a Gaussian summary of the profile likelihood for population work.
    No reddening label enters (they are training-only). Covariance factors are
    computed in float64 and cast to ``dtype`` (float32 is safe for the Newton
    iterations and the profile scans: the score is a whitened quadratic form).
    """
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
            "E_bound", "E_box_max", "parallax_mle", "chi2_xp", "dof_xp", "chi2_phot", "dof_phot",
            "nll", "init", "cov_theta")  # fmt: skip
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    t0 = time.perf_counter()
    for start in range(0, len(data), cfg.chunk_rows):
        idx = np.arange(start, min(start + cfg.chunk_rows, len(data)))
        batch = data.batch(idx, device, dtype)
        # Candidates are ranked with the covariance at the observation (ranking only),
        # then each pass fixes the covariance at the previous point's prediction.
        prep = prepare_stars(
            model_d, batch, reference=observed_reference(model_d, batch), dtype=dtype
        )
        init = coarse_init(model_d, prep, bank)
        x = torch.arange(len(idx), device=device, dtype=dtype).unsqueeze(1)
        point = init
        for _ in range(max(1, cfg.covariance_passes)):
            ebv_ref, _ = unit_to_nuisance(point[:, 3:5], prep)
            prep = prepare_stars(
                model_d, batch, reference=model_reference(model_d, point[:, :3], ebv_ref),
                dtype=dtype,
            )  # fmt: skip
            score = _score_fn(model_d, prep)
            res = maximize_score(
                score, x, bounds, n_candidates=cfg.n_candidates, n_starts=cfg.n_starts,
                steps=cfg.steps, init=point, chunk_size=cfg.chunk_size, generator=gen,
            )  # fmt: skip
            point = res.theta
        cov = inverse_information(
            observed_information(score, res.theta, x, cfg.chunk_size)
        )
        intervals = profile_intervals(
            score, x, res.theta, res.score, bounds, thresholds, cfg, covariance=cov
        )
        with torch.no_grad():
            ebv, parallax = unit_to_nuisance(res.theta[:, 3:5], prep)
            parts = star_nll_prepared(
                model_d, res.theta[:, :3], ebv, parallax, prep, parts=True
            )
        assert isinstance(parts, dict)
        u_e = res.theta[:, 3]
        side = (u_e >= 1).to(torch.int8) - (u_e <= 0).to(torch.int8)
        out["mle"].append(res.theta[:, :3].cpu().numpy())
        out["at_bound"].append(res.at_bound.cpu().numpy())
        out["lo68"].append(intervals[0][0].cpu().numpy())
        out["hi68"].append(intervals[0][1].cpu().numpy())
        out["lo95"].append(intervals[1][0].cpu().numpy())
        out["hi95"].append(intervals[1][1].cpu().numpy())
        out["interval_at_bound"].append(intervals[1][2].cpu().numpy())
        out["E_mle"].append(ebv.cpu().numpy())
        out["E_bound"].append(side.cpu().numpy())
        out["E_box_max"].append(prep.e_max.cpu().numpy())
        out["parallax_mle"].append(parallax.cpu().numpy())
        for k in ("chi2_xp", "dof_xp", "chi2_phot", "dof_phot", "nll"):
            out[k].append(parts[k].cpu().numpy())
        out["init"].append(init.cpu().numpy())
        out["cov_theta"].append(cov[:, :3, :3].cpu().numpy())
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
    n = len(data)
    nan = np.full(n, np.nan)
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
        "E_bound": result["E_bound"],
        "E_box_max": result["E_box_max"],
        "parallax_mle": result["parallax_mle"],
        "chi2_xp": result["chi2_xp"],
        "dof_xp": result["dof_xp"],
        "chi2_phot": result["chi2_phot"],
        "dof_phot": result["dof_phot"],
        "at_bound": result["at_bound"],
        "cov_theta": result["cov_theta"],
        "at_bound_names": np.array(PARAM_NAMES),
        "interval_at_bound": result["interval_at_bound"],
        "subset": np.asarray(subset),
        "g_mag": data.g_mag,
        "label": data.theta,
        "label_err": data.theta_err,
        "ebv_label": data.ebv_label if data.ebv_label is not None else nan,
        "ebv_label_err": data.ebv_label_err if data.ebv_label_err is not None else nan,
        "fe_h_ref": data.fe_h_ref if data.fe_h_ref is not None else nan,
        "split": data.split if data.split is not None else np.full(n, -1, np.int8),
    }


# ─── report ─────────────────────────────────────────────────────────────────

G_BINS = (-np.inf, 13.5, 15.0, 16.0, np.inf)
FEH_BINS = (-np.inf, -3.0, -2.5, -2.0, -1.0, 0.0, np.inf)
#: Reference [Fe/H] bins of the frozen-core tables (as in the S1 report).
FEH_REF_BINS = (-9.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 9.0)


def _nmad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return (
        float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")
    )


def _bin_label(lo: float, hi: float) -> str:
    return f"[{lo:g}, {hi:g})"


def _cells(
    est: np.ndarray, ref: np.ndarray, g: np.ndarray, feh_bins: Sequence[float],
    lo: np.ndarray | None = None, hi: np.ndarray | None = None,
) -> list[dict[str, Any]]:  # fmt: skip
    """Median offset / NMAD (and coverage) of ``est - ref`` per G x reference bin."""
    have = np.isfinite(est) & np.isfinite(ref)
    cells = []
    for glo, ghi in zip(G_BINS[:-1], G_BINS[1:], strict=True):
        for flo, fhi in zip(feh_bins[:-1], feh_bins[1:], strict=True):
            sel = have & (g >= glo) & (g < ghi) & (ref >= flo) & (ref < fhi)
            if not sel.any():
                continue
            d = est[sel] - ref[sel]
            cell = {"g_bin": _bin_label(glo, ghi), "feh_bin": _bin_label(flo, fhi),
                    "n": int(sel.sum()), "median_offset": float(np.median(d)), "nmad": _nmad(d)}  # fmt: skip
            if lo is not None and hi is not None:
                cell["coverage68"] = float(
                    ((ref[sel] >= lo[sel]) & (ref[sel] <= hi[sel])).mean()
                )
            cells.append(cell)
    return cells


def recovery_summary(pred: dict[str, np.ndarray]) -> dict[str, Any]:
    """Parameter recovery against labels: offsets, NMAD, interval coverage, chi2 by G.

    Also, where present: ``E`` bound fractions by side and ``E`` against the
    reddening labels (overall and by G), and [Fe/H] against the raw high-resolution
    reference ``fe_h_ref`` per reference [Fe/H] x G bin.
    """
    label, mle = pred["label"], pred["mle"]
    have = np.isfinite(label).all(axis=1)
    g = pred["g_mag"]
    out: dict[str, Any] = {"n": int(len(g)), "n_labelled": int(have.sum())}
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
    lab = np.where(have[:, None], label, np.nan)
    out["feh_by_g_and_feh"] = _cells(
        mle[:, 2], lab[:, 2], g, FEH_BINS, pred["lo68"][:, 2], pred["hi68"][:, 2]
    )
    ref = pred.get("fe_h_ref")
    if ref is not None and np.isfinite(ref).any():
        out["n_fe_h_ref"] = int(np.isfinite(ref).sum())
        out["feh_vs_ref_by_g_and_feh"] = _cells(
            mle[:, 2], ref, g, FEH_REF_BINS, pred["lo68"][:, 2], pred["hi68"][:, 2]
        )
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
    if "E_bound" in pred and len(g):
        side = pred["E_bound"]
        out["ebv_at_lower_bound"] = float((side < 0).mean())
        out["ebv_at_upper_bound"] = float((side > 0).mean())
    if "E_mle" in pred and pred.get("ebv_label") is not None:
        e, lab_e, err_e = pred["E_mle"], pred["ebv_label"], pred["ebv_label_err"]
        out["ebv_vs_label"] = ebv_recovery(e, lab_e, err_e)
        out["ebv_vs_label_by_g"] = []
        for glo, ghi in zip(G_BINS[:-1], G_BINS[1:], strict=True):
            sel = (g >= glo) & (g < ghi)
            rec = ebv_recovery(e[sel], lab_e[sel], err_e[sel])
            if rec["n"]:
                out["ebv_vs_label_by_g"].append({"g_bin": _bin_label(glo, ghi), **rec})
    return out


def _fmt(v: Any, spec: str = ".3f") -> str:
    return "nan" if v is None or not np.isfinite(v) else format(v, spec)


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
    if "anchored_fraction" in history:
        lines += [f"Training stars with a reddening label (E anchor): "
                  f"{history['anchored_fraction']:.3f}", ""]  # fmt: skip
    lines += ["## Training", "", "| epoch | train NLL | anchor NLL | valid NLL (profiled) | valid XP chi2/dof "
              "| valid E - label (offset / NMAD / slope) | s |", "|--:|--:|--:|--:|--:|--:|--:|"]  # fmt: skip
    for e in epochs:
        ev = e.get("valid_ebv_vs_label") or {}
        lines.append(
            f"| {e['epoch']} | {e['train_nll']:.3f} | {e.get('train_anchor_nll', float('nan')):.3f} | "
            f"{e.get('valid_nll', float('nan')):.3f} | {e.get('valid_chi2_xp_per_dof', float('nan')):.3f} | "
            f"{_fmt(ev.get('median_offset'))} / {_fmt(ev.get('nmad'))} / {_fmt(ev.get('slope'))} | {e['seconds']} |"
        )
    if epochs:
        lines += [
            "",
            "Learned scatter (last epoch): `" + json.dumps(epochs[-1]["scatter"]) + "`",
        ]
    if label_chi2:
        lines += ["", "## At the labels (E, parallax profiled, no anchor)", "",
                  "| G bin | n | chi2/dof | median chi2/dof |", "|:--|--:|--:|--:|"]  # fmt: skip
        for row in label_chi2.get("chi2_xp_by_g", []):
            lines.append(
                f"| {row['g_bin']} | {row['n']} | {row['chi2_xp_per_dof']:.3f} | {row['median_chi2_xp_per_dof']:.3f} |"
            )
        if label_chi2.get("ebv_vs_label"):
            lines += ["", "E at the labels vs reddening labels: `"
                      + json.dumps(label_chi2["ebv_vs_label"]) + "`"]  # fmt: skip
    for name, summ in summaries.items():
        lines += ["", f"## Inference: {name} ({summ['n']} stars, {summ['n_labelled']} labelled)", "",
                  "| parameter | median offset | NMAD | 68 % coverage | 95 % coverage |", "|:--|--:|--:|--:|--:|"]  # fmt: skip
        for p in PARAMS:
            s = summ[p]
            lines.append(
                f"| {p} | {_fmt(s['median_offset'], '.4g')} | {_fmt(s['nmad'], '.4g')} | "
                f"{_fmt(s['coverage68'])} | {_fmt(s['coverage95'])} |"
            )
        lines += ["", f"At bound: `{json.dumps(summ['at_bound_fraction'])}`; E at the lower bound "
                  f"{_fmt(summ.get('ebv_at_lower_bound'))}, upper {_fmt(summ.get('ebv_at_upper_bound'))}"]  # fmt: skip
        if summ.get("ebv_vs_label", {}).get("n"):
            lines += ["", "E at the MLE vs reddening labels (labels not used at inference):", "",
                      "| G bin | n | offset | NMAD | pull NMAD | slope |", "|:--|--:|--:|--:|--:|--:|"]  # fmt: skip
            for r in [{"g_bin": "all", **summ["ebv_vs_label"]}] + summ[
                "ebv_vs_label_by_g"
            ]:
                lines.append(
                    f"| {r['g_bin']} | {r['n']} | {_fmt(r['median_offset'])} | {_fmt(r['nmad'])} | "
                    f"{_fmt(r['pull_nmad'])} | {_fmt(r.get('slope'))} |"
                )
        lines += ["", "XP chi2/dof at the MLE by G:", "", "| G bin | n | chi2/dof | median |",
                  "|:--|--:|--:|--:|"]  # fmt: skip
        for row in summ["chi2_xp_by_g"]:
            lines.append(
                f"| {row['g_bin']} | {row['n']} | {row['chi2_xp_per_dof']:.3f} | {row['median_chi2_xp_per_dof']:.3f} |"
            )
        tables = [
            (
                "[Fe/H] recovery against the labels by G and [Fe/H] bin:",
                "feh_by_g_and_feh",
            )
        ]
        if summ.get("feh_vs_ref_by_g_and_feh"):
            tables.append(("[Fe/H] against the raw high-resolution reference (`fe_h_ref`) by G and "
                           "reference [Fe/H]:", "feh_vs_ref_by_g_and_feh"))  # fmt: skip
        for title, key in tables:
            lines += ["", title, "", "| G | [Fe/H] | n | offset | NMAD | 68 % cov. |",
                      "|:--|:--|--:|--:|--:|--:|"]  # fmt: skip
            for c in summ[key]:
                lines.append(
                    f"| {c['g_bin']} | {c['feh_bin']} | {c['n']} | {c['median_offset']:.3f} | "
                    f"{c['nmad']:.3f} | {c.get('coverage68', float('nan')):.2f} |"
                )
    lines += [
        "",
        "Interval coverage is against labels (or references) that carry their own errors "
        "(not included).",
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
