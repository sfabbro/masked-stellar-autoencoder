"""Passbands of the filter-integral term: band selection, linear XP operators, exact
variances.

**Integral.** For a passband ``T`` the photon-weighted mean flux of an XP spectrum is
torchgaiaxp 0.3.0's ``XPSpectra.integrate(passband, wl)``::

    <f_lambda>_b = int f(l) T(l) l dl / int T(l) l dl

and the AB magnitude is ``m_b = -2.5 log10(<f_lambda>_b l_p^2 / c) - 56.10`` with the
pivot ``l_p^2 = int T l dl / int T / l dl``. ``f`` is linear in the 110 coefficients
``c`` (BP 55, then RP 55), so the integral is a fixed vector ``a_b``: ``F_b = a_b . c``.
We obtain ``a_b`` exactly by calling ``XPSpectra.integrate`` on the 110 unit coefficient
vectors (tests check ``a_b . c`` and its propagated error against ``integrate`` on random
spectra with full covariance). For systems that torchgaiaxp packages (GaiaXPy's
PanSTARRS1, SDSS, SkyMapper, DECam, JPLUS, Gaia_Vega, ...), ``a_b`` is the packaged
design merged over BP/RP, exactly what ``torchgaiaxp.generate`` uses, with its zero
point; other bands (BASS, MzLS, OmegaCAM, MegaCam, S-PLUS) integrate the SVO curves
stored in ``filters/`` (provenance in ``filters/provenance.json``).

**Units.** We store ``a'_b = a_b 10**(-0.4 (zp_b - 8.5))`` so that, for raw coefficients,
``m_b = 8.5 - 2.5 log10(a'_b . c_raw)``; for G-normalised coefficients
``c_norm = c_raw / f_G`` (``f_G = 10**((8.5 - G) / 2.5)``), ``m_b = G - 2.5
log10(a'_b . c_norm)``. Relative fluxes ``a'_b . c_norm`` are O(1).

**Exact variance.** ``Var(a'_b . c) = sum_{BP,RP} a^T diag(e) R diag(e) a`` with the
published errors ``e`` (they already include the solution standard deviation) and the
packed correlations ``R`` (Gaia ``tril(-1)`` order) - the covariance torchgaiaxp
propagates. BP and RP are independent. :func:`exact_band_variance` evaluates it with one
``[N, 1485] @ [1485, n_bands]`` product per band block.

**Selection.** A magnitude band joins the term when its SVO curve's support (``T >
threshold max T``) lies inside ``wavelength_range_nm`` (default 336-1020 nm) and it is
not a dereddened product.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .schema import N_XP, BandRegistry

FILTER_DIR = Path(__file__).with_name("filters")
C_NM_PER_S = 2.99792458e17
AB_ZP_SI = -56.10  # m_AB = -2.5 log10(f_nu / W m^-2 Hz^-1) - 56.10 (3631 Jy)
COLOUR_SYSTEM = "DR3_Gaia_Vega"
COLOUR_BANDS = ("BP", "RP")
XP_RANGE_NM = (330.0, 1050.0)


@lru_cache(maxsize=1)
def provenance() -> dict:
    return json.loads((FILTER_DIR / "provenance.json").read_text())


def load_svo_curve(filter_id: str) -> tuple[np.ndarray, np.ndarray]:
    """``(wavelength_nm, transmission)`` of a stored SVO curve."""
    meta = provenance()["filters"].get(filter_id)
    if meta is None:
        raise KeyError(
            f"SVO curve {filter_id!r} not stored; add it to scripts/fetch_svo_filters.py"
        )
    data = np.loadtxt(FILTER_DIR / meta["file"], delimiter=",", skiprows=1)
    return data[:, 0].copy(), data[:, 1].copy()


def support(filter_id: str, threshold: float) -> tuple[float, float]:
    wl, t = load_svo_curve(filter_id)
    s = wl[t > threshold * t.max()]
    return float(s.min()), float(s.max())


def pivot_nm(wl: np.ndarray, t: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(t * wl, wl) / np.trapezoid(t / wl, wl)))


def select_fi_bands(
    registry: BandRegistry,
    wavelength_range_nm=(336.0, 1020.0),
    threshold: float = 0.01,
    candidates: list[str] | None = None,
    strict: bool = True,
) -> list[str]:
    """Magnitude bands whose passband support lies inside the range (registry order).

    With ``strict``, every named candidate must qualify (an explicit band list);
    otherwise candidates that do not qualify are skipped.
    """
    lo, hi = wavelength_range_nm
    out = []
    names = None if candidates is None else set(candidates)
    for b in registry.bands:
        if b.kind != "mag" or b.svo is None or b.dereddened:
            continue
        if names is not None and b.name not in names:
            continue
        s_lo, s_hi = support(b.svo, threshold)
        if s_lo >= lo and s_hi <= hi:
            out.append(b.name)
    if names is not None and strict:
        missing = sorted(names - set(out))
        if missing:
            raise ValueError(
                f"filter_integral.bands {missing} are not magnitude bands fully inside "
                f"{lo}-{hi} nm at threshold {threshold}"
            )
    return out


def _packaged_vector(system: str, band: str) -> tuple[np.ndarray, float]:
    """Merged design ``[110]`` (BP then RP) and zero point of a packaged band."""
    from torchgaiaxp import _assets
    from torchgaiaxp.design import XPDesign

    design = XPDesign.photometric(system, dtype=torch.float64)
    if band not in design.bands:
        raise KeyError(f"{system} has no band {band!r}; bands {design.bands}")
    j = design.bands.index(band)
    d = design.design  # [2, nb, 55]
    w = design.merge  # [2, nb]
    vec = torch.cat([w[0, j] * d[0, j], w[1, j] * d[1, j]]).numpy()
    zp = float(_assets.tensor("systems", f"{design.system}/zero_points")[j])
    off = _assets.tensor("systems", f"{design.system}/offsets")
    if float(off[j]) != 0.0:
        raise ValueError(f"{system}:{band} is a standardised band (offset); not linear")
    return vec, zp


def svo_integrate_vector(filter_id: str, step_nm: float = 0.25) -> np.ndarray:
    """``a_b`` [110] in f_nu (W m^-2 Hz^-1 per unit coefficient), via XPSpectra.integrate."""
    from torchgaiaxp import Sampled, XPSpectra

    wl, t = load_svo_curve(filter_id)
    lo = max(float(wl.min()), XP_RANGE_NM[0])
    hi = min(float(wl.max()), XP_RANGE_NM[1])
    n = int(math.ceil((hi - lo) / step_nm)) + 1
    grid = torch.linspace(lo, hi, n, dtype=torch.float64)
    passband = Sampled(torch.as_tensor(wl), torch.as_tensor(t), fill=0.0)
    unit = torch.eye(2 * N_XP, dtype=torch.float64).reshape(2 * N_XP, 2, N_XP)
    flux_lambda, _ = XPSpectra(coefficients=unit).integrate(passband, grid)
    tg = passband(grid)
    lam2 = torch.trapezoid(tg * grid, grid) / torch.trapezoid(tg / grid, grid)
    a = flux_lambda * lam2 / C_NM_PER_S
    if not torch.isfinite(a).all():
        raise ValueError(f"{filter_id}: non-finite integral (outside the XP range?)")
    return a.numpy()


@dataclass(frozen=True)
class FilterOperator:
    """``a'`` vectors of the filter-integral bands plus the synthetic BP/RP colour.

    ``matrix``: [110, n_bands + 2], columns = ``bands`` then synthetic BP, RP.
    """

    bands: tuple[str, ...]
    band_index: np.ndarray  # registry indices of ``bands``
    matrix: np.ndarray  # float64
    routes: tuple[str, ...]  # "packaged:<system>:<band>" | "svo:<filter id>"
    zero_points: np.ndarray  # zp of each column (before folding into a')

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def colour_columns(self) -> tuple[int, int]:
        return self.n_bands, self.n_bands + 1

    def relative_flux(self, coefficients) -> torch.Tensor:
        """``a' . c`` for coefficients ``[..., 110]`` (raw or G-normalised)."""
        c = torch.as_tensor(coefficients)
        return c @ torch.as_tensor(self.matrix, dtype=c.dtype, device=c.device)

    def magnitudes_raw(self, coefficients_raw) -> torch.Tensor:
        """``m = 8.5 - 2.5 log10(a' . c_raw)`` (NaN for non-positive flux)."""
        s = self.relative_flux(coefficients_raw)
        return torch.where(
            s > 0, 8.5 - 2.5 * torch.log10(s.clamp_min(1e-300)), torch.nan
        )

    def to_dict(self) -> dict:
        return {
            "bands": list(self.bands),
            "routes": list(self.routes),
            "zero_points": self.zero_points.tolist(),
            "colour": f"{COLOUR_SYSTEM}:{COLOUR_BANDS[0]}-{COLOUR_BANDS[1]}",
        }


def build_operator(
    registry: BandRegistry, band_names: list[str], step_nm: float = 0.25
) -> FilterOperator:
    cols, routes, zps = [], [], []
    for name in band_names:
        b = registry[name]
        if b.xp_system:
            system, band = b.xp_system.split(":")
            vec, zp = _packaged_vector(system, band)
            routes.append(f"packaged:{b.xp_system}")
        elif b.svo:
            vec, zp = svo_integrate_vector(b.svo, step_nm), AB_ZP_SI
            routes.append(f"svo:{b.svo}")
        else:
            raise ValueError(f"{name}: no passband (xp_system or svo)")
        cols.append(vec)
        zps.append(zp)
    for band in COLOUR_BANDS:
        vec, zp = _packaged_vector(COLOUR_SYSTEM, band)
        cols.append(vec)
        zps.append(zp)
        routes.append(f"packaged:{COLOUR_SYSTEM}:{band}")
    zps = np.asarray(zps, np.float64)
    matrix = np.stack(cols, axis=1) * np.power(10.0, -0.4 * (zps - 8.5))[None, :]
    return FilterOperator(
        bands=tuple(band_names),
        band_index=np.asarray([registry.index(n) for n in band_names], np.int64),
        matrix=matrix,
        routes=tuple(routes),
        zero_points=zps,
    )


@lru_cache(maxsize=4)
def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    r, c = np.tril_indices(n, -1)
    return r, c


def exact_band_variance(
    matrix: torch.Tensor,
    errors: torch.Tensor,
    correlations: torch.Tensor | None,
) -> torch.Tensor:
    """``Var(a . c)`` per star and column, exact under the Gaia XP covariance.

    matrix ``[110, K]``; errors ``[N, 2, 55]``; correlations ``[N, 2, 1485]`` packed
    (``None`` = diagonal covariance). Rows with a missing band give NaN.
    """
    rows, cols = (torch.as_tensor(x, device=errors.device) for x in _pairs(N_XP))
    out = None
    for blk in range(2):
        a = matrix[blk * N_XP : (blk + 1) * N_XP].to(errors.dtype)  # [55, K]
        e = errors[:, blk]  # [N, 55]
        v = (e * e) @ (a * a)
        if correlations is not None:
            r = correlations[:, blk].to(errors.dtype)
            s = e[:, rows] * e[:, cols] * r  # [N, 1485]
            v = v + 2.0 * (s @ (a[rows] * a[cols]))
        out = v if out is None else out + v
    return out
