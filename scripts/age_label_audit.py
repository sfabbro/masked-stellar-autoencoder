#!/usr/bin/env python3
"""Age-label audit: inventory, Gaia DR3 crossmatch, pairwise and co-natal systematics.

usage:
  age_label_audit.py ingest  --raw RAW --out OUT     # catalogues -> OUT/records.parquet
  age_label_audit.py gaia    --out OUT               # Gaia archive TAP: 2MASS -> DR3, FLAME
  age_label_audit.py compare --raw RAW --out OUT     # inventory, pairs, clusters, binaries
  age_label_audit.py plots   --out OUT --plots DIR

Design note: ``docs/design-20261122-age-consensus.md``. No fitting, no labels
are imputed, nothing is called truth: every catalogue is a method-biased
measurement of log age, and this script only measures how the measurements
disagree.

``RAW`` mirrors the arc layout (``/arc/projects/k-pop/catalogues``): the file
names in ``FILES`` relative to RAW; missing files are reported and skipped. The
``gaia`` stage queries the Gaia archive (astroquery) for (a) the Gaia DR3
source_id of every 2MASS / APOGEE identifier found in ``ingest`` (best
neighbour table) and (b) Gaia DR3 FLAME ages (``age_flame`` and
``age_flame_spec``) for the union of Gaia IDs: catalogue stars and the
Hunt & Reffert (2023) cluster members. Re-run ``ingest`` afterwards so those
identifiers resolve and FLAME joins as a catalogue.

Conventions:
* ``lage`` = log10(age / Gyr) of the catalogue median; ``sig`` = half the
  16-84 % width in log10 (dex), from the catalogue's own percentiles or
  quoted errors (statistical part only).
* A pair difference is ``lage_A - lage_B`` (dex). Bins in age use the pair
  mean ``(lage_A + lage_B) / 2`` (Bland-Altman), so that neither catalogue's
  scatter defines the x axis; bins in [Fe/H] use the mean of the available
  values; the evolutionary class comes from the highest-ranked catalogue of
  the pair (seismic > spectroscopic > photometric).
* NMAD = 1.4826 median |x - median x|. ``inflation`` = NMAD / median
  sqrt(sig_A^2 + sig_B^2): above 1 the quoted errors are too small or the
  methods disagree beyond their noise; below 1 the two share inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Catalogue registry
# --------------------------------------------------------------------------- #
STARHORSE_SURVEYS = (
    "apogee17", "galahdr3", "gesdr5", "lamostm7", "lamostl7", "ravedr6",
    "sdssdr12", "rvs",
)  # fmt: skip
WILLETT_FILES = {
    "willett26_kepler": "cat_Kepler_APG.csv",
    "willett26_k2_bhm": "cat_K2_APG_BHM.csv",
    "willett26_k2_cor": "cat_K2_APG_COR.csv",
    "willett26_k2_gal": "cat_K2_GAL_BHM.csv",
    "willett26_tess": "cat_TESS_APG.csv",
}
LEGACY_PIPES = ("AST", "BASTA", "C2kSMO", "GOE", "V&A", "YMCM")

FILES = {
    **{f"starhorse23_{s}": f"starhorse_ages/{s}.fits" for s in STARHORSE_SURVEYS},
    "apokasc3_vac": "apokasc-3-2024.fits",
    "apok2": "apo_k2_2024.fits",
    "apok2_kepler": "apo_k2_2024_recal.fits",
    "legacy17": "kepler-legacy-dwarfs.fits",
    "chaplin14": "apokasc-1-dwarfs.fits",
    "martig16": "martig-kepler-preds.fits",
    "leung23": "nn_latent_age_dr17.csv.gz",
    "ambre": "ambre-ages.fits",
    "galah_dr4": "galah_dr4_allstar_240705.fits",
    "xiang24": "LAMOST_Gaia_subgiants_Xiangetal2024.fits",
    "kepler_gaia": "kepler_dr3_good.fits",
    "hr23_clusters": "OpenClusters/clusters.fits",
    "hr23_crossmatch": "OpenClusters/crossma.fits",
    "consistency_targets": "consistency_targets_20261118.npz",
    **{k: f"willett2026/{v}" for k, v in WILLETT_FILES.items()},
}


@dataclass(frozen=True)
class Catalogue:
    """Inventory metadata (method, models/priors, ID route); not used in maths."""

    family: str
    method: str
    models: str
    gaia_route: str
    reference: str
    rank: int  # evolutionary-class authority in a pair (higher wins)


_SH = Catalogue(
    "starhorse23",
    "Bayesian isochrone (spectro-photometric-astrometric), MSTO/SGB only",
    "PARSEC 1.2S grid; StarHorse priors (IMF, density/age/[M/H] of Galactic "
    "components); spectroscopic Teff/logg/[M/H] + Gaia EDR3 parallax + photometry",
    "native Gaia EDR3 = DR3 source_id",
    "Queiroz et al. 2023, A&A 673, A155",
    2,
)
_WIL = Catalogue(
    "willett26",
    "asteroseismic grid (PARAM; numax, Dnu or L) + spectroscopy",
    "PARAM grid, ages up to about 20 Gyr; APOGEE DR17 / GALAH DR3 "
    "Teff and [Fe/H]; Gaia DR3 luminosity in the _L variant",
    "released GAIADR3_ID constant (bug); KIC -> Gaia (kepler_dr3_good), "
    "EPIC/APOGEE -> APO-K2 / 2MASS best neighbour",
    "Willett et al. 2026, Zenodo 20428851",
    4,
)
CATALOGUES: dict[str, Catalogue] = {
    **{f"starhorse23_{s}": _SH for s in STARHORSE_SURVEYS},
    "apokasc3": Catalogue(
        "apokasc",
        "asteroseismic scaling masses (10 pipelines) -> grid age",
        "stellar-model grid (see paper); APOGEE DR17 inputs; no 13.8 Gyr cap",
        "native GAIAEDR3_SOURCE_ID",
        "Pinsonneault et al. 2025, ApJS 276, 69",
        5,
    ),
    "apokasc2": Catalogue(
        "apokasc",
        "asteroseismic scaling masses -> grid age",
        "stellar-model grid (see paper); APOGEE DR14; APOKASC-2 seismic calibration",
        "APOKASC-3 VAC row (native Gaia ID)",
        "Pinsonneault et al. 2018, ApJS 239, 32",
        5,
    ),
    "apokasc_dw": Catalogue(
        "apokasc",
        "VAC AGE_DW column (dwarfs/subgiants); equals Chaplin et al. 2014",
        "as Chaplin et al. 2014",
        "APOKASC-3 VAC row (native Gaia ID)",
        "APOKASC-3 VAC; duplicate of chaplin14",
        5,
    ),
    "apok2": Catalogue(
        "apok2",
        "asteroseismic K2 GAP giants, grid-based (seismic + APOGEE)",
        "grid-based (see paper); K2 GAP DR3 seismology, APOGEE DR17",
        "native Gaia EDR3 ID",
        "Schonhut-Stasik et al. 2024, AJ 167, 50",
        4,
    ),
    "apok2_kepler": Catalogue(
        "apok2",
        "same pipeline as APO-K2 on Kepler giants (recalibrated scale)",
        "as APO-K2",
        "native Gaia EDR3 ID",
        "Schonhut-Stasik et al. 2024, table 2",
        4,
    ),
    **{k: _WIL for k in WILLETT_FILES},
    **{
        f"legacy17_{p}": Catalogue(
            "legacy17",
            f"asteroseismic individual-frequency modelling ({p})",
            "pipeline-specific stellar models (ASTEC/BaSTI/GARSTEC/MESA/YREC...)",
            "KIC -> Gaia DR3 (kepler_dr3_good)",
            "Silva Aguirre et al. 2017, ApJ 835, 173",
            6,
        )
        for p in LEGACY_PIPES
    },  # fmt: skip
    "chaplin14": Catalogue(
        "chaplin14",
        "asteroseismic dwarfs (global params) grid fit",
        "several grids (Chaplin et al. 2014 table 5)",
        "KIC -> Gaia DR3",
        "Chaplin et al. 2014, ApJS 210, 1",
        5,
    ),
    "martig16": Catalogue(
        "martig16",
        "data-driven [C/N] age (The Cannon trained on APOKASC)",
        "inherits APOKASC-1/2 masses and grid ages; APOGEE DR12",
        "2MASS -> Gaia DR3 best neighbour",
        "Martig et al. 2016, MNRAS 456, 3655",
        1,
    ),
    "leung23": Catalogue(
        "leung23",
        "data-driven latent-space NN age (APOGEE spectra)",
        "trained on Kepler seismic (APOKASC) ages; APOGEE DR17",
        "APOGEE_ID -> Gaia DR3 best neighbour",
        "Leung et al. 2023, MNRAS 522, 4577",
        1,
    ),
    "ambre": Catalogue(
        "ambre",
        "isochrone ages, AMBRE [Mg/Fe] sample",
        "isochrone grid (paper-specific)",
        "Gaia DR2 ID (not remapped)",
        "AMBRE project (Santos-Peral et al. 2021)",
        1,
    ),
    "galah_dr4": Catalogue(
        "galah_dr4",
        "in-pipeline Bayesian isochrone age (spectra + Gaia + 2MASS)",
        "isochrone grid and priors of the DR4 pipeline",
        "native gaiadr3_source_id",
        "Buder et al. 2025 (GALAH DR4)",
        2,
    ),
    "xiang24": Catalogue(
        "xiang24",
        "Bayesian isochrone fit of subgiants (Teff, MK, [Fe/H], [alpha/Fe])",
        "Yonsei-Yale (Y2) isochrones with alpha enhancement; LAMOST DR7 spectra, "
        "Gaia EDR3 parallax",
        "native GAIAEDR3_SOURCE_ID",
        "Xiang, Rix & Liu 2024 (NatAs; China-VO 10.12149/101467); Xiang & Rix 2022",
        3,
    ),
    "flame_dr3": Catalogue(
        "flame",
        "Gaia FLAME: L, Teff -> mass/age on tracks",
        "BaSTI solar-metallicity tracks; GSP-Phot Teff/logg; G-band BC",
        "native",
        "Creevey et al. 2023, A&A 674, A26",
        0,
    ),
    "flame_spec_dr3": Catalogue(
        "flame",
        "Gaia FLAME on GSP-Spec Teff/logg",
        "BaSTI solar-metallicity tracks; GSP-Spec inputs",
        "native",
        "Creevey et al. 2023, A&A 674, A26",
        0,
    ),
}

#: Sources that reproduce another catalogue exactly on their shared stars (found by
#: this audit); kept in the inventory with ``good=False`` and left out of comparisons.
DUPLICATE_SOURCES = {
    "apokasc_dw": "AGE_DW of the APOKASC-3 VAC equals Chaplin et al. 2014 table 5 (413/413)",
    "willett26_k2_gal": "age_Dnu equals K2 APG BHM for every row (4163/4163) although "
    "[Fe/H] differs; only the _L variant is a separate GALAH-based estimate",
}
STAGE_CLASSES = ("dwarf", "subgiant", "giant")
AGE_EDGES_GYR = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 100.0)
FEH_EDGES = (-5.0, -1.0, -0.5, -0.2, 0.1, 1.0)
TEFF_EDGES = (3000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 9000.0)
LOGG_EDGES = (0.0, 2.0, 2.5, 3.0, 3.5, 3.9, 4.2, 6.0)
RECORD_COLUMNS = (
    "source", "source_id", "id_route", "alt_kind", "alt_id", "age", "age16",
    "age84", "teff", "logg", "feh", "stage", "good", "l", "b",
)  # fmt: skip


# --------------------------------------------------------------------------- #
# Small helpers (unit-tested)
# --------------------------------------------------------------------------- #
def nmad(x) -> float:
    x = np.asarray(x, dtype="f8")
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def summarize(delta, sig=None) -> dict:
    """n, median, NMAD, robust error of the median, 16/84 % and inflation."""
    delta = np.asarray(delta, dtype="f8")
    ok = np.isfinite(delta)
    d = delta[ok]
    out = {"n": int(len(d))}
    if not len(d):
        return out | {"median": np.nan, "nmad": np.nan, "err_median": np.nan,
                      "p16": np.nan, "p84": np.nan, "inflation": np.nan}  # fmt: skip
    out |= {
        "median": float(np.median(d)),
        "nmad": nmad(d),
        "err_median": float(1.2533 * nmad(d) / np.sqrt(len(d))),
        "p16": float(np.percentile(d, 16)),
        "p84": float(np.percentile(d, 84)),
    }
    if sig is not None:
        s = np.asarray(sig, dtype="f8")[ok]
        s = s[np.isfinite(s) & (s > 0)]
        out["inflation"] = float(out["nmad"] / np.median(s)) if len(s) else np.nan
    return out


def binned(delta, x, edges, sig=None, min_n: int = 10) -> pd.DataFrame:
    """``summarize`` of ``delta`` in bins of ``x`` (left-closed, last bin closed)."""
    delta, x = np.asarray(delta, dtype="f8"), np.asarray(x, dtype="f8")
    edges = np.asarray(edges, dtype="f8")
    idx = np.digitize(x, edges[1:-1], right=False)
    rows = []
    for k in range(len(edges) - 1):
        m = (idx == k) & np.isfinite(x) & np.isfinite(delta)
        if m.sum() < min_n:
            continue
        s = summarize(delta[m], None if sig is None else np.asarray(sig)[m])
        rows.append({"lo": edges[k], "hi": edges[k + 1], **s})
    return pd.DataFrame(rows)


def log_age_columns(age, age16, age84):
    """(lage, sig) in dex; sig from the 16/84 bounds when both are positive."""
    age, lo, hi = (np.asarray(v, dtype="f8") for v in (age, age16, age84))
    with np.errstate(divide="ignore", invalid="ignore"):
        lage = np.where(age > 0, np.log10(age), np.nan)
        ok = (lo > 0) & (hi > lo)
        sig = np.where(ok, 0.5 * (np.log10(hi) - np.log10(np.where(ok, lo, 1))), np.nan)
    return lage, sig


def normalise_2mass(values) -> np.ndarray:
    """'2M00000002+7417074' / 'J00000002+7417074' / '00000002+7417074' -> digits form."""
    s = pd.Series(np.asarray(values).astype(str)).str.strip()
    s = s.str.replace(r"^(2M|J)", "", regex=True)
    ok = s.str.fullmatch(r"\d{8}[+-]\d{7}")
    return np.where(ok, s, "").astype(object)


def stage_from_teff_logg(teff, logg) -> np.ndarray:
    """Coarse class for binning only: dwarf / subgiant / giant / unknown."""
    teff, logg = np.asarray(teff, dtype="f8"), np.asarray(logg, dtype="f8")
    out = np.full(len(logg), "unknown", dtype=object)
    out[logg >= 4.1] = "dwarf"
    out[(logg >= 3.5) & (logg < 4.1)] = "subgiant"
    out[logg < 3.5] = "giant"
    out[(logg >= 3.5) & (logg < 3.8) & (teff < 4800)] = "giant"
    return out


STAGE_CLASS = {
    "MS": "dwarf", "MSTO": "dwarf", "dwarf": "dwarf", "SGB": "subgiant",
    "subgiant": "subgiant", "RGB": "giant", "RC": "giant", "AGB": "giant",
    "giant": "giant",
}  # fmt: skip


class IdMap:
    """Identifier -> Gaia DR3 source_id maps; ambiguous keys are dropped."""

    def __init__(self):
        self.maps: dict[str, pd.Series] = {}
        self.routes: dict[str, str] = {}

    def add(self, kind: str, keys, gaia, route: str) -> None:
        # Positional arrays: a filtered Series input must not align on its index.
        keys = pd.Series(np.asarray(keys).astype(str)).str.strip()
        gaia = pd.to_numeric(pd.Series(np.asarray(gaia)), errors="coerce").astype(
            "Int64"
        )
        if len(keys) != len(gaia):
            raise ValueError(f"{kind}: {len(keys)} keys for {len(gaia)} Gaia IDs")
        new = pd.DataFrame({"k": keys, "g": gaia})
        new = new[(new.k != "") & (new.k != "nan") & new.g.notna() & (new.g > 0)]
        if kind in self.maps:
            old = self.maps[kind]
            new = pd.concat([pd.DataFrame({"k": old.index, "g": old.values}), new])
            self.routes[kind] += "+" + route
        else:
            self.routes[kind] = route
        new = new.drop_duplicates()
        counts = new.groupby("k").g.nunique()
        keep = new[new.k.isin(counts.index[counts == 1])].drop_duplicates("k")
        self.maps[kind] = pd.Series(keep.g.values, index=keep.k.values)

    def resolve(self, kind: str, keys) -> np.ndarray:
        if kind not in self.maps:
            return np.zeros(len(keys), dtype="i8")
        keys = pd.Series(np.asarray(keys).astype(str)).str.strip()
        out = keys.map(self.maps[kind])
        return out.fillna(0).astype("i8").to_numpy()


def dedupe(records: pd.DataFrame) -> pd.DataFrame:
    """One record per (source, star): keep the smallest quoted sig, then the first."""
    rec = records.copy()
    key = np.where(
        rec.source_id > 0, rec.source_id.astype(str), "alt:" + rec.alt_id.astype(str)
    )
    rec["_key"] = key
    rec["_s"] = rec["sig"].fillna(np.inf)
    rec = rec.sort_values(["source", "_key", "_s"], kind="stable")
    rec = rec.drop_duplicates(["source", "_key"]).drop(columns=["_key", "_s"])
    return rec.reset_index(drop=True)


def crossmatch(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Inner match of two record sets by Gaia DR3 source_id, else by shared alt ID."""
    cols = [
        "source_id",
        "alt_kind",
        "alt_id",
        "lage",
        "sig",
        "teff",
        "logg",
        "feh",
        "stage",
    ]
    a, b = a[cols], b[cols]
    by_gaia = a[a.source_id > 0].merge(
        b[b.source_id > 0], on="source_id", suffixes=("_a", "_b")
    )
    by_gaia["match"] = "gaia_dr3"
    rest_a = a[a.source_id <= 0]
    rest_b = b[b.source_id <= 0]
    by_alt = rest_a.merge(rest_b, on=["alt_kind", "alt_id"], suffixes=("_a", "_b"))
    by_alt = by_alt[by_alt.alt_id.astype(str).str.len() > 0]
    by_alt["match"] = "alt:" + by_alt.alt_kind.astype(str)
    for frame in (by_gaia, by_alt):
        for c in ("alt_kind", "alt_id"):
            if c in frame:
                frame.drop(columns=c, inplace=True)
        for c in ("alt_kind_a", "alt_id_a", "alt_kind_b", "alt_id_b"):
            if c in frame:
                frame.drop(columns=c, inplace=True)
    out = pd.concat([by_gaia, by_alt], ignore_index=True)
    out["delta"] = out.lage_a - out.lage_b
    out["lage_mean"] = 0.5 * (out.lage_a + out.lage_b)
    out["sig_pair"] = np.sqrt(out.sig_a**2 + out.sig_b**2)
    out["feh_mean"] = out[["feh_a", "feh_b"]].mean(axis=1)
    return out


def pair_stage(stage_a, stage_b, rank_a: int, rank_b: int) -> np.ndarray:
    """Evolutionary class of a pair: the higher-ranked catalogue's when known."""
    first, second = (stage_a, stage_b) if rank_a >= rank_b else (stage_b, stage_a)
    first = pd.Series(np.asarray(first, dtype=object)).map(STAGE_CLASS)
    second = pd.Series(np.asarray(second, dtype=object)).map(STAGE_CLASS)
    return first.fillna(second).fillna("unknown").to_numpy()


def sky_fraction(
    l_deg, b_deg, n_lon: int = 36, n_lat: int = 18, min_n: int = 5
) -> float:
    """Fraction of equal-area (l, sin b) cells holding >= min_n stars."""
    l_deg, b_deg = np.asarray(l_deg, dtype="f8"), np.asarray(b_deg, dtype="f8")
    ok = np.isfinite(l_deg) & np.isfinite(b_deg)
    if not ok.any():
        return float("nan")
    h, _, _ = np.histogram2d(
        np.mod(l_deg[ok], 360.0), np.sin(np.radians(b_deg[ok])),
        bins=[n_lon, n_lat], range=[[0, 360], [-1, 1]],
    )  # fmt: skip
    return float((h >= min_n).mean())


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def _fits(path: Path, columns=None) -> pd.DataFrame:
    from astropy.table import Table

    t = Table.read(path, hdu=1, format="fits")
    if columns is not None:
        t = t[[c for c in columns if c in t.colnames]]
    out = {}
    for name in t.colnames:
        col = t[name]
        v = np.asarray(col.filled(np.nan) if hasattr(col, "filled") and col.dtype.kind == "f"
                       else col)  # fmt: skip
        if v.dtype.kind in "SU":
            v = np.char.strip(v.astype(str))
        elif v.dtype.kind in "iu" and hasattr(col, "mask") and np.any(col.mask):
            v = np.where(col.mask, 0, v)
        out[name] = v
    return pd.DataFrame(out)


def _frame(source, **cols) -> pd.DataFrame:
    cols = {k: v.to_numpy() if isinstance(v, pd.Series) else v for k, v in cols.items()}
    n = max(len(v) for v in cols.values() if not isinstance(v, str) and np.ndim(v) > 0)
    base = {
        c: np.full(n, np.nan)
        for c in ("age16", "age84", "teff", "logg", "feh", "l", "b")
    }
    base |= {"source_id": np.zeros(n, "i8"), "id_route": "", "alt_kind": "", "alt_id": "",
             "stage": "unknown", "good": True}  # fmt: skip
    base |= cols
    out = pd.DataFrame(base)
    out["source"] = source
    out["source_id"] = (
        pd.to_numeric(out.source_id, errors="coerce").fillna(0).astype("i8")
    )
    return out[list(RECORD_COLUMNS)]


def _num(frame, name):
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype="f8")


def _clean_bad(v, bad=-9000.0):
    v = np.asarray(v, dtype="f8")
    return np.where(v < bad, np.nan, v)


def read_starhorse(path: Path, survey: str) -> pd.DataFrame:
    d = _fits(path)
    good = ~pd.Series(d.OutputFlag).str.contains("BAD").to_numpy()
    good &= pd.Series(d.AgeInOut).str.len().to_numpy() == 0
    return _frame(
        f"starhorse23_{survey}", source_id=d.GaiaEDR3, id_route="native",
        alt_kind="apogee" if "APOGEE" in d else "", alt_id=d.APOGEE if "APOGEE" in d else "",
        age=d.Age50, age16=d.Age16, age84=d.Age84, teff=d.Teff50, logg=d.logg50,
        feh=d.Met50, stage=d.AgeFlag, good=good, l=d.GLON, b=d.GLAT,
    )  # fmt: skip


def read_apokasc_vac(path: Path) -> list[pd.DataFrame]:
    cols = [
        "KEPLER_ID", "2MASS_ID", "GAIAEDR3_SOURCE_ID", "GAIA_L", "GAIA_B",
        "DR17_TEFF_COR", "DR17_LOGG_COR", "DR17_FE_H", "APOKASC3_CAT_AGE",
        "APOKASC3_EVSTATE", "APOKASC3_CAT_TAB", "APOKASC3_AGE_RGB",
        "APOKASC3_AGE_RGB_PERR", "APOKASC3_AGE_RGB_MERR", "APOKASC3_AGE_RCAGB",
        "APOKASC3_AGE_RCAGB_PERR", "APOKASC3_AGE_RCAGB_MERR", "APOKASC2_AGE",
        "APOKASC2_AGE_PERR", "APOKASC2_AGE_MERR", "APOKASC2_EVSTATES", "AGE_DW",
        "AGE_DW_PERR", "AGE_DW_MERR",
    ]  # fmt: skip
    d = _fits(path, cols)
    teff, logg, feh = (
        _clean_bad(d[c]) for c in ("DR17_TEFF_COR", "DR17_LOGG_COR", "DR17_FE_H")
    )
    common = dict(source_id=d.GAIAEDR3_SOURCE_ID, id_route="native", alt_kind="kic",
                  alt_id=d.KEPLER_ID.astype(str), teff=teff, logg=logg, feh=feh,
                  l=d.GAIA_L, b=d.GAIA_B)  # fmt: skip
    cat = pd.Series(d.APOKASC3_CAT_AGE)
    rgb = cat.isin(["RGB", "RGB_AGB"]).to_numpy()
    rc = cat.isin(["RC", "RC/RGB"]).to_numpy()
    age = np.where(
        rgb,
        _clean_bad(d.APOKASC3_AGE_RGB),
        np.where(rc, _clean_bad(d.APOKASC3_AGE_RCAGB), np.nan),
    )
    perr = np.where(
        rgb, _clean_bad(d.APOKASC3_AGE_RGB_PERR), _clean_bad(d.APOKASC3_AGE_RCAGB_PERR)
    )
    merr = np.where(
        rgb, _clean_bad(d.APOKASC3_AGE_RGB_MERR), _clean_bad(d.APOKASC3_AGE_RCAGB_MERR)
    )
    stage3 = np.where(rgb, "RGB", np.where(rc, "RC", "unknown"))
    out = [_frame("apokasc3", age=age, age16=age - np.abs(merr), age84=age + np.abs(perr),
                  stage=stage3, good=np.isin(d.APOKASC3_CAT_TAB, ["Gold", "Silver"]),
                  **common)]  # fmt: skip
    es2 = pd.Series(d.APOKASC2_EVSTATES).str.replace(r"\(S\)", "", regex=True)
    a2 = _clean_bad(d.APOKASC2_AGE)
    out.append(_frame("apokasc2", age=a2, age16=a2 - np.abs(_clean_bad(d.APOKASC2_AGE_MERR)),
                      age84=a2 + np.abs(_clean_bad(d.APOKASC2_AGE_PERR)),
                      stage=np.where(es2.isin(["RGB", "RC"]), es2, "unknown"),
                      good=es2.isin(["RGB", "RC"]).to_numpy(), **common))  # fmt: skip
    dw = _clean_bad(d.AGE_DW)
    out.append(_frame("apokasc_dw", age=dw, age16=dw - np.abs(_clean_bad(d.AGE_DW_MERR)),
                      age84=dw + np.abs(_clean_bad(d.AGE_DW_PERR)),
                      stage=stage_from_teff_logg(teff, logg), **common))  # fmt: skip
    return out


def read_apok2(path: Path, source: str) -> pd.DataFrame:
    d = _fits(path)
    alt = ("epic", d.EPIC.astype(str)) if "EPIC" in d else ("kic", d.KIC.astype(str))
    age = _num(d, "Age")
    return _frame(
        source, source_id=d.GaiaEDR3, id_route="native", alt_kind=alt[0], alt_id=alt[1],
        age=age, age16=age - _num(d, "e_Age"), age84=age + _num(d, "E_Age"),
        teff=_num(d, "Teff"), logg=_num(d, "logg-seis"), feh=_num(d, "[Fe/H]"),
        stage="giant", l=_num(d, "GLON"), b=_num(d, "GLAT"),
    )  # fmt: skip


def read_willett(path: Path, source: str, ids: IdMap) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    n = len(d)
    sid = np.zeros(n, "i8")
    route = np.full(n, "", dtype=object)
    alt_kind, alt_id = "", np.full(n, "", dtype=object)

    def fill(kind, keys, label):
        got = ids.resolve(kind, keys)
        new = (sid == 0) & (got > 0)
        sid[new], route[new] = got[new], label

    if "KIC_ID" in d:
        alt_kind, alt_id = "kic", d.KIC_ID.astype("Int64").astype(str).to_numpy()
        fill("kic", alt_id, "kic")
    if "K2_ID" in d:
        alt_kind, alt_id = "epic", d.K2_ID.astype("Int64").astype(str).to_numpy()
        fill("epic", alt_id, "epic")
    if "TESS_ID" in d:
        alt_kind, alt_id = "tic", d.TESS_ID.astype("Int64").astype(str).to_numpy()
    if "APOGEE_ID" in d:
        fill("2mass", normalise_2mass(d.APOGEE_ID), "2mass")
    if "GALAH_ID" in d:
        fill("sobject", d.GALAH_ID.astype("Int64").astype(str).to_numpy(), "sobject")
    good = d.RelAge_Dnu.astype(bool).to_numpy().copy()
    if "drop_K2_dup" in d:
        good &= ~d.drop_K2_dup.astype("boolean").fillna(False).to_numpy(dtype=bool)
    stage = "giant"
    if "EvoState" in d:
        stage = d.EvoState.map({1: "RGB", 2: "RC"}).fillna("giant").to_numpy()
    feh = _num(d, "Fe_H")
    frames = []
    for variant, label in (("Dnu", ""), ("L", "_L")):
        frames.append(_frame(
            source + label, source_id=sid, id_route=route, alt_kind=alt_kind, alt_id=alt_id,
            age=_num(d, f"age_{variant}"), age16=_num(d, f"age_16_{variant}"),
            age84=_num(d, f"age_84_{variant}"), feh=feh, stage=stage,
            good=good if variant == "Dnu" else d.RelAge_L.astype(bool).to_numpy(),
        ))  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def read_legacy(path: Path, ids: IdMap) -> pd.DataFrame:
    d = _fits(path)
    kic = d.KIC.astype(str).to_numpy()
    age = _num(d, "Age")
    frames = []
    for pipe in LEGACY_PIPES:
        m = (d.Pipe == pipe).to_numpy()
        frames.append(_frame(
            f"legacy17_{pipe}", source_id=ids.resolve("kic", kic[m]), id_route="kic",
            alt_kind="kic", alt_id=kic[m], age=age[m], age16=age[m] - _num(d, "e_Age")[m],
            age84=age[m] + _num(d, "E_Age")[m], stage="MS",
        ))  # fmt: skip
    return pd.concat(frames, ignore_index=True)


def read_chaplin(path: Path, ids: IdMap) -> pd.DataFrame:
    d = _fits(path)
    kic = d.KIC.astype(str).to_numpy()
    age, teff, logg = _num(d, "Age"), _num(d, "Teff"), _num(d, "logg")
    return _frame(
        "chaplin14", source_id=ids.resolve("kic", kic), id_route="kic", alt_kind="kic",
        alt_id=kic, age=age, age16=age - _num(d, "e_Age"), age84=age + _num(d, "E_Age"),
        teff=teff, logg=logg, feh=_num(d, "[M/H]"), stage=stage_from_teff_logg(teff, logg),
    )  # fmt: skip


def read_martig(path: Path, ids: IdMap) -> pd.DataFrame:
    d = _fits(path)
    tm = normalise_2mass(d["2MASS"])
    return _frame(
        "martig16", source_id=ids.resolve("2mass", tm), id_route="2mass", alt_kind="2mass",
        alt_id=tm, age=_num(d, "Ageout"), teff=_num(d, "Teff"), logg=_num(d, "logg"),
        feh=_num(d, "[M/H]"), stage="giant",
    )  # fmt: skip


def read_leung(path: Path, ids: IdMap) -> pd.DataFrame:
    d = pd.read_csv(
        path, usecols=["APOGEE_ID", "STARFLAG", "Age", "LogAge", "LogAge_Error"]
    )
    d = d[np.isfinite(d.LogAge)]
    tm = normalise_2mass(d.APOGEE_ID)
    lg, el = d.LogAge.to_numpy(), d.LogAge_Error.to_numpy()
    return _frame(
        "leung23", source_id=ids.resolve("2mass", tm), id_route="2mass", alt_kind="2mass",
        alt_id=tm, age=10**lg, age16=10 ** (lg - el), age84=10 ** (lg + el), stage="unknown",
    )  # fmt: skip


def read_ambre(path: Path) -> pd.DataFrame:
    d = _fits(path)
    age = _num(d, "Age")
    age = np.where(age > 0, age, np.nan)
    return _frame(
        "ambre", alt_kind="gaia_dr2", alt_id=d.GaiaDR2.astype(str), age=age,
        age16=age - _num(d, "e_Age"), age84=age + _num(d, "e_Age"), feh=_num(d, "[M/H]"),
    )  # fmt: skip


def read_galah_dr4(path: Path) -> pd.DataFrame:
    cols = ["sobject_id", "gaiadr3_source_id", "teff", "logg", "fe_h", "age", "e_age",
            "flag_sp", "ra", "dec"]  # fmt: skip
    d = _fits(path, cols)
    age = _num(d, "age")
    e = _num(d, "e_age") if "e_age" in d else np.full(len(d), np.nan)
    teff, logg = _num(d, "teff"), _num(d, "logg")
    l_deg, b_deg = _galactic(_num(d, "ra"), _num(d, "dec"))
    return _frame(
        "galah_dr4", source_id=d.gaiadr3_source_id, id_route="native", alt_kind="sobject",
        alt_id=d.sobject_id.astype(str), age=age, age16=age - e, age84=age + e, teff=teff,
        logg=logg, feh=_num(d, "fe_h"), stage=stage_from_teff_logg(teff, logg),
        good=(_num(d, "flag_sp") == 0), l=l_deg, b=b_deg,
    )  # fmt: skip


def read_xiang(path: Path) -> pd.DataFrame:
    cols = [
        "GAIAEDR3_SOURCE_ID",
        "RAJ2000",
        "DEJ2000",
        "AGE",
        "E_AGE",
        "TEFF",
        "LOGG",
        "FEH",
    ]
    d = _fits(path, cols)
    age, e = _num(d, "AGE"), _num(d, "E_AGE")
    l_deg, b_deg = _galactic(_num(d, "RAJ2000"), _num(d, "DEJ2000"))
    return _frame(
        "xiang24", source_id=d.GAIAEDR3_SOURCE_ID, id_route="native", age=age,
        age16=age - e, age84=age + e, teff=_num(d, "TEFF"), logg=_num(d, "LOGG"),
        feh=_num(d, "FEH"), stage="SGB", l=l_deg, b=b_deg,
    )  # fmt: skip


def _galactic(ra, dec):
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    ok = np.isfinite(ra) & np.isfinite(dec)
    l_deg, b_deg = np.full(len(ra), np.nan), np.full(len(ra), np.nan)
    c = SkyCoord(ra[ok] * u.deg, dec[ok] * u.deg).galactic
    l_deg[ok], b_deg[ok] = c.l.deg, c.b.deg
    return l_deg, b_deg


def flame_stage(evol) -> np.ndarray:
    """FLAME evolstage (100 ZAMS, 360 IAMS, 420 TAMS, 490 base RGB) -> class."""
    e = np.asarray(evol, dtype="f8")
    out = np.full(len(e), "unknown", dtype=object)
    out[e < 420] = "MS"
    out[(e >= 420) & (e < 490)] = "SGB"
    out[e >= 490] = "giant"
    return out


def read_flame(path: Path) -> list[pd.DataFrame]:
    d = pd.read_csv(path)
    d = d.drop_duplicates("source_id")
    out = []
    for col, source in (
        ("age_flame", "flame_dr3"),
        ("age_flame_spec", "flame_spec_dr3"),
    ):
        if col not in d:
            continue
        m = d[col].notna().to_numpy()
        e = d[m]
        stage = flame_stage(
            e["evolstage_flame" if col == "age_flame" else "evolstage_flame_spec"]
        )
        out.append(_frame(
            source, source_id=e.source_id, id_route="native", age=e[col],
            age16=e[f"{col}_lower"], age84=e[f"{col}_upper"], teff=e.get("teff_gspphot"),
            logg=e.get("logg_gspphot"), feh=e.get("mh_gspphot"), stage=stage,
        ))  # fmt: skip
    return out


def build_id_map(raw: Path, gaia_dir: Path) -> IdMap:
    ids = IdMap()
    vac = raw / FILES["apokasc3_vac"]
    if vac.exists():
        d = _fits(vac, ["KEPLER_ID", "2MASS_ID", "GAIAEDR3_SOURCE_ID"])
        ids.add("kic", d.KEPLER_ID.astype(str), d.GAIAEDR3_SOURCE_ID, "apokasc3_vac")
        ids.add(
            "2mass",
            normalise_2mass(d["2MASS_ID"]),
            d.GAIAEDR3_SOURCE_ID,
            "apokasc3_vac",
        )
    kg = raw / FILES["kepler_gaia"]
    if kg.exists():
        d = _fits(kg, ["kepid", "source_id", "tm_designation", "kepler_gaia_ang_dist"])
        d = d[_num(d, "kepler_gaia_ang_dist") < 1.0]
        ids.add("kic", d.kepid.astype(str), d.source_id, "kepler_dr3_good<1as")
        ids.add(
            "2mass", normalise_2mass(d.tm_designation), d.source_id, "kepler_dr3_good"
        )
    k2 = raw / FILES["apok2"]
    if k2.exists():
        d = _fits(k2, ["EPIC", "APOGEE", "GaiaEDR3"])
        ids.add("epic", d.EPIC.astype(str), d.GaiaEDR3, "apo_k2")
        ids.add("2mass", normalise_2mass(d.APOGEE), d.GaiaEDR3, "apo_k2")
    sh = raw / FILES["starhorse23_apogee17"]
    if sh.exists():
        d = _fits(sh, ["APOGEE", "GaiaEDR3"])
        ids.add("2mass", normalise_2mass(d.APOGEE), d.GaiaEDR3, "starhorse_apogee17")
    tm = gaia_dir / "tmass_dr3.csv"
    if tm.exists():
        d = pd.read_csv(tm, dtype={"designation": str})
        d = d[d.number_of_neighbours == 1]
        ids.add("2mass", d.designation, d.source_id, "gaia_dr3_tmass_best_neighbour")
    gd = raw / FILES["galah_dr4"]
    if gd.exists():
        d = _fits(gd, ["sobject_id", "gaiadr3_source_id"])
        ids.add("sobject", d.sobject_id.astype(str), d.gaiadr3_source_id, "galah_dr4")
    return ids


def ingest(raw: Path, out: Path) -> pd.DataFrame:
    gaia_dir = out / "gaia"
    ids = build_id_map(raw, gaia_dir)
    frames, files = [], []

    def have(key):
        p = raw / FILES[key]
        ok = p.exists()
        files.append({"key": key, "path": str(p), "exists": ok,
                      "bytes": p.stat().st_size if ok else 0})  # fmt: skip
        if not ok:
            print(f"[skip] {key}: {p} missing")
        return ok

    for s in STARHORSE_SURVEYS:
        if have(f"starhorse23_{s}"):
            frames.append(read_starhorse(raw / FILES[f"starhorse23_{s}"], s))
    if have("apokasc3_vac"):
        frames += read_apokasc_vac(raw / FILES["apokasc3_vac"])
    for key in ("apok2", "apok2_kepler"):
        if have(key):
            frames.append(read_apok2(raw / FILES[key], key))
    for key in WILLETT_FILES:
        if have(key):
            frames.append(read_willett(raw / FILES[key], key, ids))
    if have("legacy17"):
        frames.append(read_legacy(raw / FILES["legacy17"], ids))
    if have("chaplin14"):
        frames.append(read_chaplin(raw / FILES["chaplin14"], ids))
    if have("martig16"):
        frames.append(read_martig(raw / FILES["martig16"], ids))
    if have("leung23"):
        frames.append(read_leung(raw / FILES["leung23"], ids))
    if have("ambre"):
        frames.append(read_ambre(raw / FILES["ambre"]))
    if have("galah_dr4"):
        frames.append(read_galah_dr4(raw / FILES["galah_dr4"]))
    if have("xiang24"):
        frames.append(read_xiang(raw / FILES["xiang24"]))
    flame = gaia_dir / "flame_dr3.csv.gz"
    if flame.exists():
        frames += read_flame(flame)
    rec = pd.concat(frames, ignore_index=True)
    rec["lage"], rec["sig"] = log_age_columns(rec.age, rec.age16, rec.age84)
    rec["good"] = (
        rec.good.astype(bool)
        & np.isfinite(rec.lage)
        & ~rec.source.isin(DUPLICATE_SOURCES)
    )
    rec["stage_class"] = rec.stage.map(STAGE_CLASS).fillna("unknown")
    for c in ("age", "age16", "age84", "teff", "logg", "feh", "l", "b", "lage", "sig"):
        rec[c] = rec[c].astype("float32")
    for c in ("source", "id_route", "alt_kind", "stage", "stage_class"):
        rec[c] = rec[c].astype(str).astype("category")
    out.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(out / "records.parquet", index=False)
    (out / "id_routes.json").write_text(json.dumps(
        {k: {"route": ids.routes[k], "keys": int(len(v))} for k, v in ids.maps.items()},
        indent=2))  # fmt: skip
    pd.DataFrame(files).to_csv(out / "files.csv", index=False)
    print(
        f"records {len(rec)} from {rec.source.nunique()} sources -> {out / 'records.parquet'}"
    )
    return rec


# --------------------------------------------------------------------------- #
# Gaia archive stage
# --------------------------------------------------------------------------- #
TMASS_ADQL = (
    "SELECT u.designation, b.source_id, b.angular_distance, b.number_of_neighbours "
    "FROM TAP_UPLOAD.ids AS u JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS b "
    "ON b.original_ext_source_id = u.designation"
)
FLAME_ADQL = (
    "SELECT u.source_id, a.age_flame, a.age_flame_lower, a.age_flame_upper, "
    "a.evolstage_flame, a.flags_flame, a.teff_gspphot, a.logg_gspphot, a.mh_gspphot, "
    "s.age_flame_spec, s.age_flame_spec_lower, s.age_flame_spec_upper, "
    "s.evolstage_flame_spec FROM TAP_UPLOAD.ids AS u "
    "LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = u.source_id "
    "LEFT OUTER JOIN gaiadr3.astrophysical_parameters_supp AS s ON s.source_id = u.source_id "
    "WHERE a.age_flame IS NOT NULL OR s.age_flame_spec IS NOT NULL"
)


GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/async"


def _tap_upload(query: str, column: str, values, workdir: Path, tag: str,
                chunk: int = 200_000, maxrec: int = 5_000_000) -> pd.DataFrame:  # fmt: skip
    """Run ``query`` (uploaded table TAP_UPLOAD.ids) in chunks on the Gaia archive.

    Plain TAP over HTTPS (requests, an astroquery dependency) with an explicit
    MAXREC: the archive otherwise truncates async results at 50,000 rows.
    """
    import io
    import time

    import requests
    from astropy.table import Table

    parts = []
    for i in range(0, len(values), chunk):
        part = workdir / f"{tag}_{i // chunk:03d}.csv"
        if not part.exists():
            buf = io.BytesIO()
            Table({column: values[i : i + chunk]}).write(buf, format="votable")
            r = requests.post(
                GAIA_TAP,
                data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "PHASE": "RUN",
                      "MAXREC": str(maxrec), "UPLOAD": "ids,param:t1", "QUERY": query},
                files={"t1": ("ids.xml", buf.getvalue(), "application/x-votable+xml")},
                allow_redirects=False, timeout=600,
            )  # fmt: skip
            job = r.headers.get("Location")
            if not job:
                raise RuntimeError(
                    f"TAP submit failed ({r.status_code}): {r.text[:500]}"
                )
            while (
                phase := requests.get(f"{job}/phase", timeout=60).text.strip()
            ) not in ("COMPLETED", "ERROR", "ABORTED"):
                time.sleep(10)
            if phase != "COMPLETED":
                raise RuntimeError(requests.get(f"{job}/error", timeout=60).text[:1000])
            res = requests.get(f"{job}/results/result", timeout=600)
            res.raise_for_status()
            got = pd.read_csv(io.BytesIO(res.content), dtype={"designation": str})
            if len(got) >= maxrec:
                raise RuntimeError(f"{tag} chunk {i // chunk} hit MAXREC={maxrec}")
            got.to_csv(part, index=False)
        parts.append(pd.read_csv(part, dtype={"designation": str}))
        print(f"{tag}: chunk {i // chunk} rows {len(parts[-1])}", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def gaia_stage(raw: Path, out: Path) -> None:
    """2MASS -> DR3 IDs, then FLAME for every catalogue star with a Gaia DR3 ID
    and every co-natal target (cluster members, wide-binary components).
    Incremental: IDs already queried (``flame_queried.npy``) are not uploaded
    again, so adding a catalogue only queries its new stars (about 1 min per
    200,000 IDs on the archive)."""
    rec = pd.read_parquet(out / "records.parquet")
    work = out / "gaia"
    work.mkdir(parents=True, exist_ok=True)
    tm = rec.loc[
        (rec.alt_kind == "2mass") & (rec.alt_id.astype(str).str.len() == 16), "alt_id"
    ]
    tm = np.unique(tm.astype(str).to_numpy())
    if not (work / "tmass_dr3.csv").exists():
        _tap_upload(TMASS_ADQL, "designation", tm, work, "tmass").to_csv(
            work / "tmass_dr3.csv", index=False)  # fmt: skip
    has = (rec.source_id > 0) & ~rec.source.str.startswith("flame")
    want = [rec.loc[has, "source_id"].unique()]
    tgt = raw / FILES["consistency_targets"]
    if tgt.exists():
        z = np.load(tgt)
        want += [z["member"], z["pair1"], z["pair2"]]
    want = np.unique(np.concatenate(want).astype("i8"))
    want = want[want > 0]
    done_path = work / "flame_queried.npy"
    flame_path = work / "flame_dr3.csv.gz"
    done = np.load(done_path) if done_path.exists() else np.zeros(0, "i8")
    if flame_path.exists() and not done_path.exists():
        raise RuntimeError(
            f"{flame_path} exists without {done_path}; remove it to re-query"
        )
    new = np.setdiff1d(want, done)
    if len(new):
        tag = f"flame{len(list(work.glob('flame*_000.csv'))):02d}"
        got = _tap_upload(FLAME_ADQL, "source_id", new, work, tag)
        old = pd.read_csv(flame_path) if flame_path.exists() else None
        pd.concat([old, got], ignore_index=True).to_csv(flame_path, index=False)
        np.save(done_path, np.union1d(done, new))
    (work / "manifest.json").write_text(json.dumps(
        {"tmass_uploaded": int(len(tm)), "flame_queried": int(len(np.union1d(done, new))),
         "flame_new_this_run": int(len(new)),
         "tmass_adql": TMASS_ADQL, "flame_adql": FLAME_ADQL}, indent=2))  # fmt: skip


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #
def inventory(rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, g in rec.groupby("source", sort=False):
        meta = CATALOGUES.get(source.removesuffix("_L"))
        good = g[g.good]
        stages = good.stage_class.value_counts().to_dict()
        top = float(np.nanmax(good.age)) if len(good) else np.nan
        rows.append({
            "source": source,
            "duplicate_of": DUPLICATE_SOURCES.get(source, ""),
            "family": meta.family if meta else "",
            "method": meta.method if meta else "",
            "models_priors": meta.models if meta else "",
            "gaia_route": meta.gaia_route if meta else "",
            "reference": meta.reference if meta else "",
            "rows": int(len(g)),
            "with_age": int(np.isfinite(g.lage).sum()),
            "with_age_gaia": int((np.isfinite(g.lage) & (g.source_id > 0)).sum()),
            "good": int(len(good)),
            "with_gaia_dr3": int((good.source_id > 0).sum()),
            "unique_gaia_dr3": int(good.loc[good.source_id > 0, "source_id"].nunique()),
            **{f"n_{c}": int(stages.get(c, 0)) for c in (*STAGE_CLASSES, "unknown")},
            "age_p05": float(np.nanpercentile(good.age, 5)) if len(good) else np.nan,
            "age_median": float(np.nanmedian(good.age)) if len(good) else np.nan,
            "age_p95": float(np.nanpercentile(good.age, 95)) if len(good) else np.nan,
            "age_max": top,
            "frac_at_max": float(np.mean(good.age >= 0.99 * top)) if len(good) else np.nan,
            "frac_gt_13p8": float(np.mean(good.age > 13.8)) if len(good) else np.nan,
            "sig_median_dex": float(np.nanmedian(good.sig)) if len(good) else np.nan,
            "feh_p05": float(np.nanpercentile(good.feh, 5)) if good.feh.notna().any() else np.nan,
            "sky_fraction": sky_fraction(good.l, good.b),
        })  # fmt: skip
    return pd.DataFrame(rows)


def pairwise(rec: pd.DataFrame, min_pairs: int = 30, min_bin: int = 10):
    """Summary and binned offsets for every pair of sources sharing >= min_pairs stars."""
    good = dedupe(rec[rec.good])
    sources = list(dict.fromkeys(good.source))
    by_source = {s: g for s, g in good.groupby("source")}
    summary, bins, matched = [], [], {}
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            m = crossmatch(by_source[a], by_source[b])
            if len(m) < min_pairs:
                continue
            ra = CATALOGUES.get(a.removesuffix("_L"))
            rb = CATALOGUES.get(b.removesuffix("_L"))
            m["stage_pair"] = pair_stage(m.stage_a, m.stage_b, ra.rank if ra else 0,
                                         rb.rank if rb else 0)  # fmt: skip
            matched[(a, b)] = m
            s = summarize(m.delta, m.sig_pair)
            s |= {"a": a, "b": b, "matched_gaia": int((m.match == "gaia_dr3").sum()),
                  "spearman": float(m[["lage_a", "lage_b"]].corr("spearman").iloc[0, 1])}  # fmt: skip
            summary.append(s)
            dims = (
                ("age_gyr", 10**m.lage_mean, AGE_EDGES_GYR),
                ("feh", m.feh_mean, FEH_EDGES),
                ("teff", m[["teff_a", "teff_b"]].mean(axis=1), TEFF_EDGES),
                ("logg", m[["logg_a", "logg_b"]].mean(axis=1), LOGG_EDGES),
            )
            for dim, x, edges in dims:
                t = binned(m.delta, x, edges, m.sig_pair, min_bin)
                if len(t):
                    bins.append(t.assign(a=a, b=b, dim=dim))
            hi_rank = (ra.rank if ra else 0) >= (rb.rank if rb else 0)
            state = np.where(hi_rank, m.stage_a, m.stage_b)
            groups = [(f"stage:{c}", (m.stage_pair == c).to_numpy())
                      for c in (*STAGE_CLASSES, "unknown")]  # fmt: skip
            groups += [
                (f"state:{c}", state == c) for c in ("MS", "MSTO", "SGB", "RGB", "RC")
            ]
            for k, (dim, sel) in enumerate(groups):
                if sel.sum() >= min_bin:
                    bins.append(pd.DataFrame([{**summarize(m.delta[sel], m.sig_pair[sel]),
                                               "lo": k, "hi": k + 1, "a": a, "b": b,
                                               "dim": dim}]))  # fmt: skip
    summary = pd.DataFrame(summary)
    if len(summary):
        summary = summary[
            ["a", "b", *[c for c in summary.columns if c not in ("a", "b")]]
        ]
    bins = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()
    if len(bins):
        bins = bins[
            ["a", "b", "dim", *[c for c in bins.columns if c not in ("a", "b", "dim")]]
        ]
    return summary, bins, matched


def read_clusters(raw: Path) -> pd.DataFrame:
    d = _fits(raw / FILES["hr23_clusters"],
              ["Name", "ID", "Type", "N", "CST", "logAge16", "logAge50", "logAge84",
               "AV50", "dist50", "CMDCl50", "Plx"])  # fmt: skip
    d["lage_cl"] = _num(d, "logAge50") - 9.0
    d["sig_cl"] = 0.5 * (_num(d, "logAge84") - _num(d, "logAge16"))
    return d


def cluster_consistency(rec: pd.DataFrame, raw: Path, min_members: int = 3,
                        min_prob: float = 0.7):  # fmt: skip
    """Per (catalogue, cluster): member median log age vs the cluster's age and spread."""
    tgt = raw / FILES["consistency_targets"]
    if not (tgt.exists() and (raw / FILES["hr23_clusters"]).exists()):
        return pd.DataFrame(), pd.DataFrame()
    z = np.load(tgt)
    mem = pd.DataFrame({"source_id": z["member"], "cid": z["member_cluster"],
                        "prob": z["member_prob"]})  # fmt: skip
    mem = mem[mem.prob >= min_prob].drop_duplicates("source_id")
    cl = read_clusters(raw).set_index("ID")
    good = dedupe(rec[rec.good & (rec.source_id > 0)])
    j = good.merge(mem, on="source_id")
    rows = []
    for (source, cid), g in j.groupby(["source", "cid"]):
        if len(g) < min_members or cid not in cl.index:
            continue
        c = cl.loc[cid]
        med = float(np.median(g.lage))
        rows.append({
            "source": source, "cid": int(cid), "cluster": c.Name, "type": c.Type,
            "n": int(len(g)), "lage_med": med, "nmad_members": nmad(g.lage),
            "sig_med": float(np.nanmedian(g.sig)), "lage_cl": float(c.lage_cl),
            "sig_cl": float(c.sig_cl), "delta_cl": med - float(c.lage_cl),
            "feh_med": float(np.nanmedian(g.feh)),
            "frac_giant": float(np.mean(g.stage_class == "giant")),
            "cmd_class": float(c.CMDCl50), "n_cluster": int(c.N),
        })  # fmt: skip
    per = pd.DataFrame(rows)
    if not len(per):
        return per, pd.DataFrame()
    summ = []
    subsets = {
        "all": per.cmd_class > -1,
        "cmd>=0.5,age>=1Gyr": (per.cmd_class >= 0.5) & (per.lage_cl >= 0.0),
        "cmd>=0.5,age<1Gyr": (per.cmd_class >= 0.5) & (per.lage_cl < 0.0),
    }
    for label, sel in subsets.items():
        for source, g in per[sel & per.lage_cl.notna()].groupby("source"):
            s = summarize(g.delta_cl)
            summ.append({"subset": label, "source": source, "clusters": int(len(g)),
                         "members": int(g.n.sum()), "median_delta_vs_cluster": s["median"],
                         "nmad_delta_vs_cluster": s["nmad"],
                         "median_within_nmad": float(np.nanmedian(g.nmad_members)),
                         "median_quoted_sig": float(np.nanmedian(g.sig_med)),
                         "within_over_quoted": float(np.nanmedian(g.nmad_members / g.sig_med))})  # fmt: skip
    return per, pd.DataFrame(summ)


def binary_consistency(
    rec: pd.DataFrame, raw: Path, min_pairs: int = 20
) -> pd.DataFrame:
    """Wide-binary components (El-Badry+2021 MSMS) in the same catalogue: age differences."""
    tgt = raw / FILES["consistency_targets"]
    if not tgt.exists():
        return pd.DataFrame()
    z = np.load(tgt)
    pairs = pd.DataFrame({"s1": z["pair1"], "s2": z["pair2"]})
    good = dedupe(rec[rec.good & (rec.source_id > 0)])
    rows = []
    for source, g in good.groupby("source"):
        g = g.set_index("source_id")
        ok = pairs.s1.isin(g.index) & pairs.s2.isin(g.index)
        if ok.sum() < min_pairs:
            continue
        p = pairs[ok]
        d = g.loc[p.s1, "lage"].to_numpy() - g.loc[p.s2, "lage"].to_numpy()
        sp = np.sqrt(
            g.loc[p.s1, "sig"].to_numpy() ** 2 + g.loc[p.s2, "sig"].to_numpy() ** 2
        )
        s = summarize(d, sp)
        rows.append({"source": source, "pairs": s["n"], "median": s["median"],
                     "nmad": s["nmad"], "nmad_over_sqrt2": s["nmad"] / np.sqrt(2),
                     "inflation": s.get("inflation", np.nan)})  # fmt: skip
    return pd.DataFrame(rows)


def compare(raw: Path, out: Path, min_pairs: int = 30) -> None:
    rec = pd.read_parquet(out / "records.parquet")
    inv = inventory(rec)
    inv.to_csv(out / "inventory.csv", index=False)
    summary, bins, matched = pairwise(rec, min_pairs=min_pairs)
    summary.to_csv(out / "pairwise_summary.csv", index=False)
    bins.to_csv(out / "pairwise_bins.csv", index=False)
    main = {frozenset(p) for p in MAIN_PAIRS}
    keep = {f"{a}|{b}": m[["source_id", "lage_a", "lage_b", "sig_a", "sig_b", "feh_mean",
                           "stage_pair", "match"]]
            for (a, b), m in matched.items() if frozenset((a, b)) in main}  # fmt: skip
    if keep:
        pd.concat([m.assign(pair=k) for k, m in keep.items()], ignore_index=True).to_parquet(
            out / "pairs_matched.parquet", index=False)  # fmt: skip
    per, cl_summary = cluster_consistency(rec, raw)
    per.to_csv(out / "cluster_members_by_catalogue.csv", index=False)
    if len(per):
        bench = per[per.cluster.isin(BENCHMARK_CLUSTERS)]
        tab = bench.pivot_table(index="cluster", columns="source", values="lage_med")
        cnt = bench.pivot_table(index="cluster", columns="source", values="n")
        cell = (
            (10**tab).round(2).astype(str)
            + " ("
            + cnt.fillna(0).astype(int).astype(str)
            + ")"
        )
        cell = cell.where(tab.notna(), "")
        hr = bench.groupby("cluster").agg(
            hr23_gyr=("lage_cl", "first"), cmd=("cmd_class", "first")
        )
        cell.insert(0, "hr23_gyr", (10**hr.hr23_gyr).round(2))
        cell.insert(1, "hr23_cmd_class", hr.cmd.round(2))
        cell.to_csv(out / "cluster_benchmarks.csv")
    cl_summary.to_csv(out / "cluster_summary.csv", index=False)
    binary_consistency(rec, raw).to_csv(out / "binary_summary.csv", index=False)
    meta = {"records_sha256": hashlib.sha256((out / "records.parquet").read_bytes()).hexdigest(),
            "sources": int(rec.source.nunique()), "pairs": int(len(summary)),
            "min_pairs": min_pairs}  # fmt: skip
    (out / "compare_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f2328", "#5b6470", "#e6e6e3"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MAIN_PAIRS = (
    ("apokasc3", "willett26_kepler"),
    ("apokasc3", "apokasc2"),
    ("willett26_k2_bhm", "apok2"),
    ("apokasc3", "leung23"),
    ("starhorse23_lamostl7", "xiang24"),
    ("starhorse23_galahdr3", "galah_dr4"),
    ("starhorse23_rvs", "flame_dr3"),
    ("apokasc3", "flame_dr3"),
)
#: Pairs on the offset-vs-age / [Fe/H] plot: seismic-seismic, seismic-data-driven,
#: isochrone-isochrone (same LAMOST spectra), isochrone-FLAME.
OFFSET_PAIRS = (
    ("apokasc3", "willett26_kepler"),
    ("apokasc3", "leung23"),
    ("starhorse23_lamostl7", "xiang24"),
    ("starhorse23_rvs", "flame_dr3"),
)


def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 9.5, "legend.frameon": False, "lines.linewidth": 2,
    })  # fmt: skip


def plots(out: Path, plot_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    plot_dir.mkdir(parents=True, exist_ok=True)
    pm = pd.read_parquet(out / "pairs_matched.parquet")
    bins = pd.read_csv(out / "pairwise_bins.csv")
    summary = pd.read_csv(out / "pairwise_summary.csv")
    present = [p for p in MAIN_PAIRS if f"{p[0]}|{p[1]}" in set(pm.pair)
               or f"{p[1]}|{p[0]}" in set(pm.pair)]  # fmt: skip

    def get(a, b):
        key = f"{a}|{b}"
        if key in set(pm.pair):
            return pm[pm.pair == key], 1.0
        return pm[pm.pair == f"{b}|{a}"], -1.0

    # 1. pairwise hexbins
    n = len(present)
    cols = 4
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows_, cols, figsize=(3.1 * cols, 3.0 * rows_), squeeze=False
    )
    for ax, (a, b) in zip(axes.flat, present, strict=False):
        m, sgn = get(a, b)
        x, y = (m.lage_a, m.lage_b) if sgn > 0 else (m.lage_b, m.lage_a)
        ax.hexbin(10**x, 10**y, gridsize=40, xscale="log", yscale="log", bins="log",
                  cmap="Blues", mincnt=1, extent=(-1.3, 1.4, -1.3, 1.4), linewidths=0)  # fmt: skip
        ax.plot([0.05, 25], [0.05, 25], color=INK, lw=1, ls="--")
        d = (x - y).to_numpy()
        ax.set_title(f"{a}\nvs {b}", loc="left")
        ax.text(0.04, 0.96, f"n={len(d)}\nmed={np.median(d):+.3f} dex\nNMAD={nmad(d):.3f} dex",
                transform=ax.transAxes, va="top", fontsize=8, color=INK)  # fmt: skip
        ax.set_xlabel(f"{a} age [Gyr]")
        ax.set_ylabel(f"{b} age [Gyr]")
        ax.set_xlim(0.05, 25)
        ax.set_ylim(0.05, 25)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(plot_dir / "pairwise_hexbin.png", dpi=110)
    plt.close(fig)

    # 2. offset vs age and vs [Fe/H] (four main seismic/spectroscopic pairs)
    focus = list(OFFSET_PAIRS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for dim, ax, xl in (("age_gyr", axes[0], "pair mean age [Gyr]"),
                        ("feh", axes[1], "pair mean [Fe/H] [dex]")):  # fmt: skip
        for color, (a, b) in zip(SERIES, focus, strict=False):
            t = bins[((bins.a == a) & (bins.b == b)) & (bins.dim == dim)]
            sgn = 1.0
            if not len(t):
                t, sgn = bins[(bins.a == b) & (bins.b == a) & (bins.dim == dim)], -1.0
            if not len(t):
                continue
            hi = (
                np.where(t.hi > 50, 20.0, t.hi)
                if dim == "age_gyr"
                else np.minimum(t.hi, 0.6)
            )
            lo = np.maximum(t.lo, -2.5) if dim == "feh" else t.lo
            xc = 0.5 * (lo + hi)
            ax.errorbar(xc, sgn * t["median"], yerr=t.nmad, color=color, marker="o", ms=5,
                        capsize=0, elinewidth=1, alpha=0.9, label=f"{a} − {b}")  # fmt: skip
        ax.axhline(0, color=INK, lw=1)
        ax.set_xlabel(xl)
        ax.set_ylabel("median Δ log10 age [dex] (bars: NMAD)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    axes[0].set_title("Offset vs age (pair mean)", loc="left")
    axes[1].set_title("Offset vs [Fe/H]", loc="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(plot_dir / "offset_vs_age_feh.png", dpi=110)
    plt.close(fig)

    # 3. co-natal test: member median vs cluster age, one panel per catalogue
    per_path = out / "cluster_members_by_catalogue.csv"
    if per_path.exists() and per_path.stat().st_size > 10:
        per = pd.read_csv(per_path)
        per = per[(per.cmd_class >= 0.5) & (per.n >= 5) & per.lage_cl.notna()]
        fams = [s for s in CLUSTER_PANELS if s in set(per.source)]
        fig, axes = plt.subplots(
            1, len(fams), figsize=(3.2 * len(fams), 3.4), squeeze=False
        )
        for ax, color, s in zip(axes[0], SERIES, fams, strict=False):
            g = per[per.source == s]
            if len(g) > 300:  # dense: points only, no error bars
                ax.scatter(
                    10**g.lage_cl, 10**g.lage_med, s=5, color=color, alpha=0.25, lw=0
                )
            else:
                err = 1.2533 * g.nmad_members / np.sqrt(g.n)
                ax.errorbar(10**g.lage_cl, 10**g.lage_med, yerr=np.log(10) * 10**g.lage_med * err,
                            fmt="o", ms=3.5, color=color, alpha=0.6, elinewidth=0.7)  # fmt: skip
            ax.plot([0.005, 20], [0.005, 20], color=INK, lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.005, 20)
            ax.set_ylim(0.05, 20)
            d = (g.lage_med - g.lage_cl).to_numpy()
            old = (g.lage_cl >= 0).to_numpy()
            ax.set_title(f"{s}: {len(g)} clusters", loc="left")
            ax.text(0.97, 0.04, "median Δ (catalogue − HR23)\n"
                    f"≥1 Gyr: {np.median(d[old]) if old.any() else np.nan:+.2f} dex (n={old.sum()})\n"
                    f"<1 Gyr: {np.median(d[~old]) if (~old).any() else np.nan:+.2f} dex (n={(~old).sum()})",
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85))  # fmt: skip
            ax.set_xlabel("Hunt & Reffert 2023 cluster age [Gyr]")
        axes[0][0].set_ylabel("median member age [Gyr]")
        fig.tight_layout()
        fig.savefig(plot_dir / "cluster_consistency.png", dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        bins = np.linspace(0, 6, 25)
        for color, s in zip(SERIES, fams, strict=False):
            g = per[per.source == s]
            ratio = (
                (g.nmad_members / g.sig_med).replace([np.inf, -np.inf], np.nan).dropna()
            )
            if len(ratio) >= 5:
                ax.hist(np.clip(ratio, 0, 5.99), bins=bins, histtype="step", color=color, lw=2,
                        density=True, label=f"{s} (median {np.median(ratio):.2f}, "
                        f"{len(ratio)} clusters)")  # fmt: skip
        ax.axvline(1, color=INK, lw=1, ls="--")
        ax.set_xlabel(
            "within-cluster NMAD / median quoted σ of log age (last bin: ≥ 6)"
        )
        ax.set_ylabel("fraction of clusters per unit")
        ax.set_title(
            "Co-natal spread vs quoted errors (members p ≥ 0.7, n ≥ 5)", loc="left"
        )
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(plot_dir / "cluster_spread.png", dpi=110)
        plt.close(fig)

    # 4. pair matrix: NMAD and median offset for the representative catalogues
    rep = [r for r in MATRIX_SOURCES if r in set(summary.a) | set(summary.b)]
    k = len(rep)
    nm, md, nn = (np.full((k, k), np.nan) for _ in range(3))
    for r in summary.itertuples():
        if r.a in rep and r.b in rep and r.n >= 30:
            i, j = rep.index(r.a), rep.index(r.b)
            nm[i, j] = nm[j, i] = r.nmad
            md[i, j], md[j, i] = r.median, -r.median
            nn[i, j] = nn[j, i] = r.n
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    seq = LinearSegmentedColormap.from_list("seq", ["#eaf2fb", SERIES[0], "#123f73"])
    div = LinearSegmentedColormap.from_list("div", [SERIES[0], "#e9e9e6", SERIES[1]])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    panels = ((axes[0], nm, seq, None, "NMAD of Δ log10 age [dex] (row vs column)"),
              (axes[1], md, div, TwoSlopeNorm(0.0, -0.3, 0.3),
               "median Δ log10 age, row − column [dex]"))  # fmt: skip
    for ax, mat, cmap, norm, title in panels:
        im = ax.imshow(mat, cmap=cmap, norm=norm, vmin=None if norm else 0.0,
                       vmax=None if norm else 0.4)  # fmt: skip
        for i in range(k):
            for j in range(k):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if (norm is None and mat[i, j] > 0.22) else INK)  # fmt: skip
        ax.set_xticks(range(k), rep, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(k), rep, fontsize=7)
        ax.grid(False)
        ax.set_title(title + " — blank: < 30 shared stars", loc="left")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(plot_dir / "pair_matrix.png", dpi=110)
    plt.close(fig)


#: Well-studied clusters for the per-catalogue age table (median member age, n).
BENCHMARK_CLUSTERS = (
    "NGC_6791", "NGC_6819", "NGC_2682", "NGC_188", "NGC_752", "Ruprecht_147", "NGC_6811",
    "NGC_2420", "NGC_2243", "Berkeley_39", "Trumpler_5", "NGC_6866", "Melotte_22",
    "Melotte_25", "NGC_2632", "NGC_7789", "NGC_2158", "Collinder_261", "NGC_6253",
)  # fmt: skip
CLUSTER_PANELS = ("flame_dr3", "starhorse23_lamostl7", "galah_dr4", "leung23")
MATRIX_SOURCES = (
    "apokasc3", "apokasc2", "apok2_kepler", "willett26_kepler", "willett26_kepler_L",
    "apok2", "willett26_k2_bhm", "willett26_k2_cor", "willett26_tess", "legacy17_BASTA",
    "chaplin14", "leung23", "martig16", "xiang24", "starhorse23_lamostl7",
    "starhorse23_apogee17", "starhorse23_galahdr3", "starhorse23_rvs", "galah_dr4",
    "flame_dr3", "flame_spec_dr3",
)  # fmt: skip


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("stage", choices=("ingest", "gaia", "compare", "plots"))
    p.add_argument("--raw", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plots", type=Path)
    p.add_argument("--min-pairs", type=int, default=30)
    a = p.parse_args(argv)
    if a.stage == "ingest":
        ingest(a.raw, a.out)
    elif a.stage == "gaia":
        gaia_stage(a.raw, a.out)
    elif a.stage == "compare":
        compare(a.raw, a.out, a.min_pairs)
    else:
        plots(a.out, a.plots or a.out / "plots")


if __name__ == "__main__":
    main()
