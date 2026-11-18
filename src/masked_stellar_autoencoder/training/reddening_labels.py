"""Per-star reddening E(B-V) labels from empirical intrinsic colours.

No stellar model is involved: the intrinsic colours come from the data.

1. Calibration stars: training stars where the full line-of-sight dust map
   is nearly empty (map E(B-V) < ``max_map_ebv``), so their observed colours
   are essentially intrinsic (the small map value is still removed).
2. Intrinsic colour model: for each colour (BP-RP, RP-Ks), a robust
   polynomial in theta = 5040/Teff, log g and [Fe/H] of the star's labels,
   fitted to the calibration stars with iterative sigma clipping.
3. Labels: E_c = (observed colour - intrinsic colour) / (R_a - R_b), with the
   Gaia DR3 extinction coefficients of Casagrande et al. (2021, MNRAS 507,
   2684; Fitzpatrick/Schlafly law), which depend on intrinsic BP-RP. The two
   colours are combined by inverse variance; when 2MASS disagrees with Gaia
   by more than ``max_disagreement`` sigma, only BP-RP is used.

Values are E(B-V) on the Schlegel et al. (1998) scale used by that law, so
they compare directly with the usual dust maps. They are left NaN where the
calibration stars do not cover the star's (Teff, log g) cell, and may be
slightly negative for unreddened stars (noise is kept unbiased).

Errors combine the calibration scatter of each colour (per Teff bin; it
includes photometric noise of typical stars), the propagated label errors
(Teff, log g, [Fe/H]) and the 2MASS error.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

COLOURS = ("bp_rp", "rp_ks")
# Casagrande et al. 2021 (colte, Fitzpatrick/Schlafly): R as a function of
# intrinsic BP-RP; 2MASS coefficients are constant.
_R_BP = (2.998, -0.140, -0.175, 0.062)
_R_RP = (1.689, -0.059)
_R_KS = 0.306


def extinction_coefficients(bprp0: np.ndarray) -> np.ndarray:
    """(N, 2) colour-excess ratios E(colour)/E(B-V) for BP-RP and RP-Ks."""
    x = np.asarray(bprp0, dtype="f8")
    r_bp = _R_BP[0] + _R_BP[1] * x + _R_BP[2] * x * x + _R_BP[3] * x**3
    r_rp = _R_RP[0] + _R_RP[1] * x
    return np.column_stack([r_bp - r_rp, r_rp - _R_KS])


def _design(teff, logg, feh, degree):
    theta = 5040.0 / np.asarray(teff, dtype="f8")
    z = np.column_stack([theta - 0.9, np.asarray(logg) - 3.5, np.asarray(feh) + 0.3])
    cols = [np.ones(len(z))]
    for d in range(1, degree + 1):
        for idx in combinations_with_replacement(range(3), d):
            cols.append(np.prod(z[:, idx], axis=1))
    return np.column_stack(cols)


def observed_colours(bp, rp, ks, e_ks=None):
    """(N, 2) BP-RP and RP-Ks; RP-Ks NaN for poor 2MASS (as in colte)."""
    bp, rp, ks = (np.asarray(a, dtype="f8") for a in (bp, rp, ks))
    rpk = rp - ks
    bad = ~np.isfinite(ks) | (ks < 4.2)
    if e_ks is not None:
        e_ks = np.asarray(e_ks, dtype="f8")
        bad |= ~np.isfinite(e_ks) | (e_ks > 0.05)
    rpk[bad] = np.nan
    return np.column_stack([bp - rp, rpk])


@dataclass
class IntrinsicColours:
    """Fitted intrinsic-colour polynomials, scatter and coverage."""

    coef: np.ndarray  # (2, n_terms)
    degree: int
    scatter_edges: np.ndarray  # Teff bin edges
    scatter: np.ndarray  # (2, n_bins) robust residual scatter
    cover_teff: np.ndarray  # Teff edges of the coverage grid
    cover_logg: np.ndarray  # log g edges
    covered: np.ndarray  # (n_teff, n_logg) bool
    feh_range: np.ndarray  # (2,) calibration [Fe/H] range

    def predict(self, teff, logg, feh) -> np.ndarray:
        """(N, 2) intrinsic BP-RP, RP-Ks; NaN outside the calibration cover."""
        teff, logg, feh = (np.asarray(a, dtype="f8") for a in (teff, logg, feh))
        out = np.full((len(teff), len(COLOURS)), np.nan)
        ok = self.inside(teff, logg, feh)
        if ok.any():
            out[ok] = _design(teff[ok], logg[ok], feh[ok], self.degree) @ self.coef.T
        return out

    def inside(self, teff, logg, feh) -> np.ndarray:
        ok = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh) & (teff > 0)
        ok &= (feh >= self.feh_range[0]) & (feh <= self.feh_range[1])
        i = np.searchsorted(self.cover_teff, teff, side="right") - 1
        j = np.searchsorted(self.cover_logg, logg, side="right") - 1
        ok &= (i >= 0) & (i < len(self.cover_teff) - 1)
        ok &= (j >= 0) & (j < len(self.cover_logg) - 1)
        good = np.zeros(len(teff), bool)
        good[ok] = self.covered[i[ok], j[ok]]
        return good

    def colour_scatter(self, teff) -> np.ndarray:
        """(N, 2) calibration scatter of each colour at Teff."""
        k = np.searchsorted(self.scatter_edges, np.asarray(teff, "f8"), side="right")
        k = np.clip(k - 1, 0, self.scatter.shape[1] - 1)
        return self.scatter[:, k].T

    def to_dict(self) -> dict:
        return {f"intrinsic_{k}": np.asarray(v) for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, d) -> IntrinsicColours:
        get = {k[len("intrinsic_") :]: d[k] for k in d if k.startswith("intrinsic_")}
        get["degree"] = int(get["degree"])
        return cls(**get)


def fit_intrinsic_colours(
    teff, logg, feh, colours, map_ebv, *, max_map_ebv=0.02, degree=4,
    clip=3.0, iterations=5, min_per_cell=20,
) -> IntrinsicColours:  # fmt: skip
    """Fit intrinsic colours on calibration stars (map E(B-V) < max_map_ebv)."""
    teff, logg, feh, map_ebv = (
        np.asarray(a, dtype="f8") for a in (teff, logg, feh, map_ebv)
    )
    colours = np.asarray(colours, dtype="f8")
    cal = np.isfinite(teff) & np.isfinite(logg) & np.isfinite(feh) & (teff > 0)
    cal &= np.isfinite(map_ebv) & (map_ebv >= 0) & (map_ebv < max_map_ebv)
    # Remove the small map reddening (first pass with BP-RP itself; refined
    # once an intrinsic BP-RP exists).
    coef = np.zeros(
        (len(COLOURS), len(_design(teff[:1], logg[:1], feh[:1], degree)[0]))
    )
    keep_all = []
    for c in range(len(COLOURS)):
        use = cal & np.isfinite(colours[:, c])
        X = _design(teff[use], logg[use], feh[use], degree)
        ratio = extinction_coefficients(colours[use, 0] - 0.8 * map_ebv[use])[:, c]
        y = colours[use, c] - ratio * map_ebv[use]
        keep = np.ones(len(y), bool)
        for _ in range(iterations):
            beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
            r = y - X @ beta
            s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
            new = np.abs(r) < clip * max(s, 1e-4)
            if (new == keep).all():
                break
            keep = new
        coef[c] = beta
        idx = np.flatnonzero(use)
        keep_all.append((idx[keep], r[keep]))
    # Coverage from calibration stars kept in the BP-RP fit.
    idx0 = keep_all[0][0]
    cover_teff = np.arange(3000.0, 10001.0, 100.0)
    cover_logg = np.arange(-0.5, 5.51, 0.25)
    n, _, _ = np.histogram2d(teff[idx0], logg[idx0], bins=[cover_teff, cover_logg])
    covered = n >= min_per_cell
    edges = np.arange(3000.0, 10001.0, 250.0)
    scatter = np.full((len(COLOURS), len(edges) - 1), np.nan)
    for c, (idx, r) in enumerate(keep_all):
        b = np.searchsorted(edges, teff[idx], side="right") - 1
        for k in range(len(edges) - 1):
            rk = r[b == k]
            if len(rk) >= 30:
                scatter[c, k] = 1.4826 * np.median(np.abs(rk - np.median(rk)))
        good = np.isfinite(scatter[c])
        scatter[c, ~good] = np.nanmax(scatter[c]) if good.any() else 0.05
    feh_range = np.percentile(feh[idx0], [0.1, 99.9])
    return IntrinsicColours(
        coef, degree, edges, scatter, cover_teff, cover_logg, covered, feh_range
    )


def reddening_labels(
    model: IntrinsicColours, teff, logg, feh, colours, *, e_teff=None, e_logg=None,
    e_feh=None, e_ks=None, max_disagreement=4.0,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """E(B-V) and its error per star (NaN where not covered)."""
    teff, logg, feh = (np.asarray(a, dtype="f8") for a in (teff, logg, feh))
    colours = np.asarray(colours, dtype="f8")
    n = len(teff)
    zero = np.zeros(n)
    e_teff, e_logg, e_feh, e_ks = (
        zero if a is None else np.nan_to_num(np.asarray(a, "f8"), nan=0.0)
        for a in (e_teff, e_logg, e_feh, e_ks)
    )
    c0 = model.predict(teff, logg, feh)
    ratio = extinction_coefficients(c0[:, 0])
    # Label-error propagation by finite differences of the intrinsic colours.
    var = model.colour_scatter(teff) ** 2
    for step, err, which in ((50.0, e_teff, 0), (0.1, e_logg, 1), (0.1, e_feh, 2)):
        args = [teff.copy(), logg.copy(), feh.copy()]
        args[which] = args[which] + step
        d = (model.predict(*args) - c0) / step
        var = var + np.nan_to_num(d * err[:, None]) ** 2
    var[:, 1] = var[:, 1] + e_ks**2
    est = (colours - c0) / ratio
    sig = np.sqrt(var) / ratio
    ok = np.isfinite(est) & np.isfinite(sig) & (sig > 0)
    both = ok[:, 0] & ok[:, 1]
    disagree = both & (
        np.abs(est[:, 0] - est[:, 1])
        > max_disagreement * np.hypot(sig[:, 0], sig[:, 1])
    )
    ok[disagree, 1] = False
    w = np.where(ok, 1.0 / np.where(ok, sig, 1.0) ** 2, 0.0)
    wsum = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ebv = (w * np.nan_to_num(est)).sum(axis=1) / wsum
        err = 1.0 / np.sqrt(wsum)
    bad = ~ok[:, 0]  # BP-RP is required
    ebv[bad] = np.nan
    err[bad] = np.nan
    return ebv, err
