"""Photometric Teff on the InfraRed Flux Method scale (Casagrande et al. 2021).

A physical Teff anchor, as asteroseismic log g is for gravity: the colour-Teff
relations of Casagrande et al. (2021, MNRAS 507, 2684) come from the IRFM,
which is nearly independent of spectroscopic model atmospheres, and are
validated on interferometric benchmarks. They are evaluated with the authors'
``colte`` code (vendored unchanged in ``masked_stellar_autoencoder.external``).

Use is restricted to stars where the relations are reliable:

* low reddening (``max_ebv``; the table's EBV is the full line-of-sight
  map value, so it bounds the star's own reddening from above);
* BP-RP Teff must exist; near-infrared colours (2MASS J, H, Ks, which are
  mismatched for some very bright stars) are kept only when they agree with
  BP-RP within ``tol`` K;
* colte's own cuts (G and BP/RP > 5, 0 <= log g <= 5, -8 <= [Fe/H] <= 0.6).

The adopted Teff is the median of the agreeing colour Teffs; its error is
their robust scatter plus colte's 20 K zero-point term, at least ``floor``.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

COLOURS = (
    "bprp", "bpj", "bph", "bpk", "rpj", "rph", "rpk", "gj", "gh", "gk", "gbp", "grp",
)  # fmt: skip


def colour_teffs(g, bp, rp, j, h, ks, ebv, logg, feh) -> np.ndarray:
    """colte Teff in each of the 12 colours, (N, 12), NaN where unavailable."""
    from masked_stellar_autoencoder.external.colte import colte

    n = len(g)
    out = np.full((n, len(COLOURS)), np.nan)
    if not n:
        return out
    sid = np.arange(n)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "colte.csv")
        colte(
            sid, logg, feh, g, bp, rp, j, h, ks, ebv, DR3=True, outfile=path
        )  # fmt: skip
        rows = []
        with open(path) as f:
            for line in f:
                if line.startswith(("#", "*")) or not line.strip():
                    continue
                rows.append(line.strip().split(","))
    if rows:
        table = np.array(rows, dtype="f8")
        idx = table[:, 0].astype("i8")
        out[idx] = table[:, 4 : 4 + len(COLOURS)]
    return out


def photometric_teff(
    g, bp, rp, j, h, ks, ebv, logg, feh, *, max_ebv=0.03, tol=250.0, floor=50.0
):
    """IRFM-scale Teff and error per star (NaN where not usable)."""
    arrays = [np.asarray(a, dtype="f8") for a in (g, bp, rp, j, h, ks, ebv, logg, feh)]
    g, bp, rp, j, h, ks, ebv, logg, feh = arrays
    teff = np.full(len(g), np.nan)
    err = np.full(len(g), np.nan)
    use = np.isfinite(ebv) & (ebv >= 0) & (ebv < max_ebv)
    use &= np.isfinite(bp) & np.isfinite(rp) & np.isfinite(logg) & np.isfinite(feh)
    if not use.any():
        return teff, err
    k = np.flatnonzero(use)
    t = colour_teffs(g[k], bp[k], rp[k], j[k], h[k], ks[k], ebv[k], logg[k], feh[k])
    ref = t[:, 0]  # BP-RP
    agree = np.isfinite(t) & (np.abs(t - ref[:, None]) <= tol)
    agree &= np.isfinite(ref)[:, None]
    vals = np.where(agree, t, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(vals, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(vals - med[:, None]), axis=1)
    n = agree.sum(axis=1)
    e = np.where(n > 1, mad / np.sqrt(np.maximum(n, 1)), 100.0) + 20.0
    ok = np.isfinite(med)
    teff[k[ok]] = med[ok]
    err[k[ok]] = np.maximum(e[ok], floor)
    return teff, err
