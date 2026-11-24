"""A small synthetic sky in the ``gaia_keyed_v1a`` contract, written as HATS-layout
trees (anchor + XP + msa-split-v1 assignment) so smoke runs and tests exercise the real
streaming loader.

Each star has a smooth SED ``f(l) = B(l; Teff) x 10**(-0.4 A(l)) x (1 - beta g(l))``
(blackbody, power-law extinction ``A = 3.1 E (l / 550 nm)**-1.3``, a blue absorption
trough of depth ``beta`` standing in for metallicity). Its XP coefficients are the
torchgaiaxp least-squares fit (``XPSpectra.fit``) scaled to the star's G; Gaia BP/RP and
every in-range survey magnitude are the XP synthetic magnitudes of :mod:`.filters`
plus a per-survey offset (the filter-integral zero points should recover these
offsets) and noise; bands outside the XP range use the SED at the band's pivot. XP
noise is drawn from an AR(1)-correlated covariance whose packed correlations are
stored, like the archive's. Coverage is random per survey.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import split_partition, split_unit
from .filters import build_operator
from .schema import N_XP, BandRegistry

SURVEY_COVER = {
    "sdss": 0.35,
    "skymapper": 0.5,
    "ps1": 0.7,
    "twomass": 0.8,
    "catwise": 0.75,
}
SURVEY_OFFSET = {"sdss": 0.02, "skymapper": -0.03, "ps1": 0.015}  # injected zero points
SURVEY_DEPTH = {
    "sdss": 21.5,
    "skymapper": 19.5,
    "ps1": 21.5,
    "twomass": 15.5,
    "catwise": 16.5,
    "gaia": 20.5,
}
V1A = {
    "sdss": [("sdss_u", "u", "err_u", "flags_u"), ("sdss_g", "g", "err_g", "flags_g"),
             ("sdss_r", "r", "err_r", "flags_r"), ("sdss_i", "i", "err_i", "flags_i"),
             ("sdss_z", "z", "err_z", "flags_z")],
    "skymapper": [(f"skymapper_{b}", f"{b}_psf", f"e_{b}_psf", None) for b in "uvgriz"],
    "ps1": [(f"ps1_{b}", f"{b}_mean_psf_mag", f"{b}_mean_psf_mag_error", None) for b in "grizy"],
    "twomass": [("tmass_j", "j_m", "j_msigcom", None), ("tmass_h", "h_m", "h_msigcom", None),
                ("tmass_ks", "ks_m", "ks_msigcom", None)],
    "catwise": [("catwise_w1", "w1mpro", "w1sigmpro", None), ("catwise_w2", "w2mpro", "w2sigmpro", None)],
}  # fmt: skip


def _sed(wl_nm, teff, ebv, beta):
    h, c, k = 6.62607e-34, 2.99792e8, 1.380649e-23
    lam = wl_nm[None, :] * 1e-9
    x = np.clip(h * c / (lam * k * teff[:, None]), None, 600)
    bb = 1.0 / (lam**5 * np.expm1(x))
    ext = 10 ** (-0.4 * 3.1 * ebv[:, None] * (wl_nm[None, :] / 550.0) ** -1.3)
    trough = 1 - beta[:, None] * np.exp(-0.5 * ((wl_nm[None, :] - 410.0) / 35.0) ** 2)
    return bb * ext * trough


def _ar1_packed(rho: np.ndarray) -> np.ndarray:
    r, c = np.tril_indices(N_XP, -1)
    return (rho[..., None] ** np.abs(r - c)[None, None, :]).astype(np.float32)


def synthetic_stars(
    n: int, seed: int, registry: BandRegistry, xp_fraction: float = 0.4
):
    """Columns of the anchor table and of the XP table (dicts of numpy arrays)."""
    from torchgaiaxp import Sampled, XPSpectra

    rng = np.random.default_rng(seed)
    teff = 10 ** rng.uniform(np.log10(3400), np.log10(9500), n)
    ebv = rng.exponential(0.25, n)
    beta = rng.uniform(0.0, 0.35, n)
    g = rng.uniform(10.0, 19.5, n)
    wl = np.linspace(330.0, 1050.0, 361)
    f = _sed(wl, teff, ebv, beta)
    f = f / f.mean(1, keepdims=True)
    xp = XPSpectra.fit(
        Sampled(torch.as_tensor(wl), torch.as_tensor(f)), torch.as_tensor(wl)
    )
    shape = xp.coefficients.reshape(n, 2 * N_XP).numpy()
    phot_names = [
        b.name
        for b in registry.bands
        if b.kind == "mag" and b.svo and b.xp_system and b.survey in SURVEY_OFFSET
    ]
    op = build_operator(registry, phot_names)
    gaia = build_operator(registry, [])  # just the BP/RP colour columns
    # scale to the requested G: the Gaia_Vega G column of torchgaiaxp
    from .filters import _packaged_vector

    gv, gzp = _packaged_vector("DR3_Gaia_Vega", "G")
    gs = shape @ (gv * 10 ** (-0.4 * (gzp - 8.5)))
    scale = 10 ** (-0.4 * (g - (8.5 - 2.5 * np.log10(gs))))
    c_true = shape * scale[:, None]
    mags = 8.5 - 2.5 * np.log10(np.clip(c_true @ op.matrix, 1e-30, None))
    bprp = 8.5 - 2.5 * np.log10(np.clip(c_true @ gaia.matrix, 1e-30, None))
    anchor: dict[str, np.ndarray] = {}
    anchor["phot_g_mean_mag"] = g.astype(np.float32)
    ge = 0.0003 + 0.002 * 10 ** (0.4 * (g - 19.0))
    anchor["phot_g_mean_flux_over_error"] = (1.0857 / ge).astype(np.float32)
    for j, col in enumerate(("phot_bp_mean_mag", "phot_rp_mean_mag")):
        has = rng.random(n) < 0.92
        e = 0.001 + 0.01 * 10 ** (0.4 * (g - 19.0))
        anchor[col] = np.where(has, bprp[:, j] + rng.normal(0, e), np.nan).astype(
            np.float32
        )
        anchor[col.replace("_mag", "_flux_over_error")] = np.where(
            has, 1.0857 / e, np.nan
        ).astype(np.float32)
    d_kpc = rng.uniform(0.1, 6.0, n)
    pe = 0.02 * 10 ** (0.2 * (g - 15.0))
    anchor["parallax"] = (1.0 / d_kpc + rng.normal(0, pe)).astype(np.float32)
    anchor["parallax_error"] = pe.astype(np.float32)
    for col in ("pmra", "pmdec"):
        anchor[col] = rng.normal(0, 8, n).astype(np.float32)
        anchor[col + "_error"] = (pe * 1.2).astype(np.float32)
    anchor["sfd_ebv"] = (ebv * rng.uniform(1.0, 1.3, n)).astype(np.float32)
    # survey magnitudes
    pivots = {b.name: b.wavelength_nm for b in registry.bands if b.kind == "mag"}
    g_pivot = pivots["gaia_g"]
    f_g = _sed(np.asarray([g_pivot]), teff, ebv, beta)[:, 0]
    for survey, cols in V1A.items():
        cover = rng.random(n) < SURVEY_COVER[survey]
        depth = SURVEY_DEPTH[survey]
        for name, vcol, ecol, fcol in cols:
            if name in phot_names:
                m = mags[:, phot_names.index(name)] + SURVEY_OFFSET[survey]
            else:
                fb = _sed(np.asarray([pivots[name]]), teff, ebv, beta)[:, 0]
                m = (
                    g
                    - 2.5 * np.log10(fb / f_g)
                    + 0.3 * np.log10(pivots[name] / g_pivot)
                )
            e = 0.008 + 0.05 * 10 ** (0.4 * (m - depth))
            ok = cover & (e < 0.4)
            anchor[vcol] = np.where(ok, m + rng.normal(0, e), np.nan).astype(np.float32)
            anchor[ecol] = np.where(ok, e, np.nan).astype(np.float32)
            if fcol is not None:
                anchor[fcol] = np.zeros(n, np.int64)
    anchor["flags"] = np.zeros(n, np.int64)  # SkyMapper
    anchor["quality_flag"] = np.full(n, 4, np.int64)  # PS1 QF_OBJ_GOOD
    anchor["ph_qual"] = np.asarray(
        rng.choice(["AAA", "AAB", "ABC", "AAU"], n, p=[0.7, 0.15, 0.1, 0.05])
    )
    # XP: present for a fraction of the brighter stars, AR(1)-correlated noise
    has_xp = (g < 17.65) & (
        rng.random(n) < min(1.0, xp_fraction / max((g < 17.65).mean(), 1e-9))
    )
    cb = c_true.reshape(n, 2, N_XP)
    err = np.abs(cb) * 0.01 + np.abs(cb).max(-1, keepdims=True) * 0.0015 * 10 ** (
        0.2 * (g[:, None, None] - 15)
    )
    rho = rng.uniform(-0.2, 0.3, (n, 2))
    packed = _ar1_packed(rho)
    lag = np.abs(np.arange(N_XP)[:, None] - np.arange(N_XP)[None, :])
    noise = np.empty_like(cb)
    for i in np.nonzero(has_xp)[0]:
        for b in range(2):
            chol = np.linalg.cholesky(rho[i, b] ** lag)
            noise[i, b] = err[i, b] * (chol @ rng.standard_normal(N_XP))
    c_obs = cb + noise
    xp_rows = np.nonzero(has_xp)[0]
    xp_cols = {
        "bp_coefficients": c_obs[xp_rows, 0],
        "rp_coefficients": c_obs[xp_rows, 1],
        "bp_coefficient_errors": err[xp_rows, 0].astype(np.float32),
        "rp_coefficient_errors": err[xp_rows, 1].astype(np.float32),
        "bp_coefficient_correlations": packed[xp_rows, 0],
        "rp_coefficient_correlations": packed[xp_rows, 1],
    }
    truth = {
        "teff": teff,
        "ebv": ebv,
        "beta": beta,
        "has_xp": has_xp,
        "offsets": dict(SURVEY_OFFSET),
    }
    return anchor, xp_rows, xp_cols, truth


def _list_array(m: np.ndarray):
    import pyarrow as pa

    flat = pa.array(np.ascontiguousarray(m).ravel())
    return pa.FixedSizeListArray.from_arrays(flat, m.shape[1]).cast(pa.list_(flat.type))


def write_synthetic_trees(
    root: str | Path,
    registry: BandRegistry,
    n_rows: int = 4096,
    n_partitions: int = 4,
    xp_fraction: float = 0.4,
    seed: int = 0,
    reserved_fraction: float = 0.01,
    shuffle_xp_rows: bool = False,
) -> dict:
    """Write ``anchor/``, ``xp/`` and ``assignment/`` HATS-layout trees; returns paths."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    rng = np.random.default_rng(seed + 1)
    anchor, xp_rows, xp_cols, truth = synthetic_stars(
        n_rows, seed, registry, xp_fraction
    )
    order = 1
    pix1 = rng.integers(0, n_partitions, n_rows)  # order-1 pixels 0..n_partitions-1
    pix12 = (pix1 << 22) + rng.integers(0, 4**11, n_rows)
    serial = rng.integers(0, 2**34, n_rows)
    sid = (pix12.astype(np.int64) << 35) + serial
    part = split_partition(split_unit(sid))
    part = np.where(rng.random(n_rows) < reserved_fraction, 4, part).astype(np.uint8)
    has_xp = np.zeros(n_rows, bool)
    has_xp[xp_rows] = True
    nb = np.isfinite(anchor["phot_bp_mean_mag"]) & np.isfinite(
        anchor["phot_rp_mean_mag"]
    )
    cls = np.where(has_xp, 0, np.where(nb, 1, 2)).astype(np.uint8)
    xp_pos = np.full(n_rows, -1)
    xp_pos[xp_rows] = np.arange(len(xp_rows))
    paths = {k: root / k for k in ("anchor", "xp", "assignment")}
    info_lines = ["Norder,Npix"]
    for p in range(n_partitions):
        rows = np.nonzero(pix1 == p)[0]
        rows = rows[np.argsort(sid[rows])]
        if not len(rows):
            continue
        info_lines.append(f"{order},{p}")
        rel = (
            Path(f"Norder={order}") / f"Dir={p // 10000 * 10000}" / f"Npix={p}.parquet"
        )
        cols = {"source_id": pa.array(sid[rows])}
        for k, v in anchor.items():
            cols[k] = pa.array(v[rows])
        _write(paths["anchor"] / "dataset" / rel, pa.table(cols), pq)
        xr = rows[has_xp[rows]]
        if shuffle_xp_rows:
            xr = rng.permutation(xr)
        if len(xr):
            xc = {"source_id": pa.array(sid[xr])}
            for k, v in xp_cols.items():
                m = v[xp_pos[xr]]
                xc[k] = _list_array(m)
            _write(paths["xp"] / "dataset" / rel, pa.table(xc), pq, row_group_size=97)
        at = pa.table(
            {
                "source_id": pa.array(sid[rows]),
                "partition": pa.array(part[rows]),
                "group_id": pa.array(np.full(len(rows), -1, np.int32)),
                "input_class": pa.array(cls[rows]),
            }
        )
        _write(paths["assignment"] / rel, at, pq)
    for k in ("anchor", "xp", "assignment"):
        paths[k].mkdir(parents=True, exist_ok=True)
        (paths[k] / "partition_info.csv").write_text("\n".join(info_lines) + "\n")
        (paths[k] / "properties").write_text(
            f"obs_collection=msa_v11_synthetic_{k}\nhats_nrows={n_rows}\n"
        )
    return {
        "anchor": str(paths["anchor"]),
        "xp": str(paths["xp"]),
        "assignment": str(paths["assignment"]),
        "truth": truth,
        "source_id": sid,
        "partition": part,
    }


def _write(path: Path, table, pq, row_group_size: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group_size)
