"""Physical reference labels for the multi-survey gauge and for validation.

Seismic catalogues give log g only and anchor the common log g scale (their
records bypass the survey maps). Their coverage is uneven: APOKASC-3 and
APO-K2 are red giants and red-clump stars; dwarfs and subgiants come only
from the APOKASC dwarf sample (Serenelli+2017) and Kepler LEGACY, a few
hundred stars. Elsewhere the log g scale is set by the survey consensus.

The Gaia FGK benchmark stars (GBS v3, Soubiran+2024) have fundamental Teff
and log g; they are returned with ``role="validation"`` and must not enter
training, so they stay an independent check (as in SoS DR3).

All outputs use the master-record columns (survey, tier, source_id, teff,
logg, fe_h, alpha and errors, fe_h_kind, alpha_kind, fe_h_corrected, fit_ok)
plus ``role`` (``anchor`` or ``validation``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CATALOGUES = Path("/arc/projects/k-pop/catalogues")
SEISMIC = (
    "anchor_apokasc3",
    "anchor_apok2",
    "anchor_apokasc_dwarfs",
    "anchor_kepler_legacy",
)


def _table(path):
    from astropy.table import Table

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = Table.read(path)
    names = [n for n in t.colnames if len(t[n].shape) <= 1]
    return t[names].to_pandas()


def _frame(survey, role, source_id, **cols):
    n = len(source_id)
    out = pd.DataFrame(
        dict(
            survey=survey,
            tier="benchmark" if role == "validation" else "anchor",
            role=role,
            source_id=np.asarray(source_id, dtype="i8"),
            fit_ok=True,
            fe_h_kind="fe_h",
            alpha_kind="",
            fe_h_corrected=np.nan,
        ),
        index=range(n),
    )
    for name in ("teff", "logg", "fe_h", "alpha"):
        for key in (name, "e_" + name):
            out[key] = np.asarray(cols.get(key, np.full(n, np.nan)), dtype="f8")
    keep = (out.source_id > 0) & np.isfinite(out[["teff", "logg", "fe_h"]]).any(axis=1)
    return out[keep].reset_index(drop=True)


def _kic_to_gaia(root):
    k = _table(root / "kepler_dr3_good.fits")
    k = k.sort_values("kepler_gaia_ang_dist").drop_duplicates("kepid")
    return dict(zip(k.kepid.astype("i8"), k.source_id.astype("i8"), strict=True))


def apokasc3(root):
    t = _table(root / "apokasc-3-2024.fits")
    logg = t["APOKASC3_LOGG"].to_numpy("f8")
    err = t["APOKASC3_LOGG_ERR"].to_numpy("f8")
    sid = pd.to_numeric(t["GAIAEDR3_SOURCE_ID"], errors="coerce").fillna(0)
    ok = np.isfinite(logg) & (logg > 0.5)  # a few unphysical values near 0
    return _frame("anchor_apokasc3", "anchor", sid[ok], logg=logg[ok], e_logg=err[ok])


def apok2(root):
    t = _table(root / "apo_k2_2024.fits")
    logg = t["logg-seis"].to_numpy("f8")
    ok = np.isfinite(logg)
    sid = pd.to_numeric(t["GaiaEDR3"], errors="coerce").fillna(0)
    return _frame(
        "anchor_apok2", "anchor", sid[ok], logg=logg[ok],
        e_logg=t["e_logg-seis"].to_numpy("f8")[ok],
    )  # fmt: skip


def apokasc_dwarfs(root, kic):
    t = _table(root / "apokasc-1-dwarfs.fits")
    sid = t["KIC"].astype("i8").map(kic).fillna(0)
    err = 0.5 * (t["E_logg"].to_numpy("f8") + t["e_logg"].to_numpy("f8"))
    return _frame(
        "anchor_apokasc_dwarfs", "anchor", sid, logg=t["logg"].to_numpy("f8"),
        e_logg=err,
    )  # fmt: skip


def kepler_legacy(root, kic):
    t = _table(root / "kepler-legacy-dwarfs.fits")
    t["err"] = 0.5 * (t["E_log(g)"] + t["e_log(g)"])
    g = t.groupby("KIC").agg(logg=("log(g)", "median"), err=("err", "median"))
    sid = pd.Series(g.index.astype("i8")).map(kic).fillna(0)
    return _frame(
        "anchor_kepler_legacy", "anchor", sid, logg=g["logg"].to_numpy("f8"),
        e_logg=g["err"].to_numpy("f8"),
    )  # fmt: skip


def gaia_benchmarks(root):
    t = pd.read_parquet(root / "gaia_fgk_benchmarks.parquet")
    sid = pd.to_numeric(t["GaiaDR3"], errors="coerce").fillna(0)
    teff = t["Teff"].to_numpy("f8")
    # GBS v3 gives no per-star errors in this table; nominal values (1.5 %,
    # 0.05 dex, 0.1 dex) are for plotting only, the stars are never trained on.
    return _frame(
        "gbs_v3", "validation", sid, teff=teff, e_teff=0.015 * teff,
        logg=t["logg"].to_numpy("f8"), e_logg=np.full(len(t), 0.05),
        fe_h=t["[Fe/H]"].to_numpy("f8"), e_fe_h=np.full(len(t), 0.1),
    )  # fmt: skip


def load_reference_labels(root: Path = CATALOGUES) -> pd.DataFrame:
    """Seismic anchors (role=anchor) and GBS (role=validation)."""
    root = Path(root)
    kic = _kic_to_gaia(root)
    parts = [
        apokasc3(root),
        apok2(root),
        apokasc_dwarfs(root, kic),
        kepler_legacy(root, kic),
        gaia_benchmarks(root),
    ]
    out = pd.concat(parts, ignore_index=True)
    # One record per (catalogue, star): duplicated Gaia ids keep the first.
    return out.drop_duplicates(["survey", "source_id"]).reset_index(drop=True)
