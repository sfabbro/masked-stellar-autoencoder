"""Master spectroscopic label union: native adapters, Gaia identity and census.

The master list has two levels:

* an **observation table** (long format): one row per native catalogue row,
  keeping native values/errors/flags, the native row identity, the Gaia DR3
  identity and how it was obtained, the pipeline dependency group, and whether
  the metallicity sits on a model-grid edge;
* a **star table**: one row per Gaia DR3 ``source_id`` with per-source presence
  and the minimum quality-passing [Fe/H] (by tier).

Nothing here assigns FIT/TUNE/CAL/DEV roles or enables training. Literature
high-resolution samples are tagged ``highres_reference`` and are meant for
calibration/QA reservation, not silent FIT supervision.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FEH_EDGES = (-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0)
TIERS = ("survey", "metal_poor_survey", "highres_reference")
VIZIER_MIRRORS = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",
)
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"

OBS_COLUMNS = [
    "source",
    "tier",
    "release",
    "pipeline",
    "dependency_group",
    "native_row",
    "native_id",
    "source_id",
    "id_method",
    "match_sep_arcsec",
    "match_n_within",
    "ra",
    "dec",
    "position_epoch",
    "teff",
    "e_teff",
    "logg",
    "e_logg",
    "fe_h",
    "e_fe_h",
    "fe_h_kind",
    "alpha",
    "e_alpha",
    "alpha_kind",
    "snr",
    "quality",
    "fe_h_grid_edge",
]


def sha256(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _f8(values) -> np.ndarray:
    out = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy(
        "f8", copy=True
    )
    out[np.isin(out, (-9999.0, 9999.0))] = np.nan  # native missing sentinels
    return out


def _exact_ids(values) -> np.ndarray:
    """Positive int64 Gaia IDs from integers or decimal strings; 0 for missing."""
    series = pd.Series(np.asarray(values, dtype=object))
    out = np.zeros(len(series), dtype="i8")
    for i, value in enumerate(series):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        if isinstance(value, float):
            raise ValueError("Gaia IDs must not pass through floating point")
        text = str(value).strip()
        if text in ("", "--", "0", "-1", "<NA>", "nan"):
            continue
        if not text.isdecimal():
            raise ValueError(f"Non-decimal Gaia ID {text!r}")
        out[i] = int(text)
    return out


def empty_frame(n: int, **constants) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(n))
    for name in OBS_COLUMNS:
        frame[name] = np.nan
    frame["source_id"] = np.zeros(n, dtype="i8")
    frame["match_n_within"] = np.zeros(n, dtype="i8")
    frame["quality"] = np.zeros(n, dtype=bool)
    frame["fe_h_grid_edge"] = np.zeros(n, dtype=bool)
    for name in ("native_id", "id_method", "fe_h_kind", "alpha_kind"):
        frame[name] = ""
    for key, value in constants.items():
        frame[key] = value
    return frame


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[OBS_COLUMNS].copy()
    frame["native_row"] = frame["native_row"].astype("i8")
    frame["source_id"] = frame["source_id"].astype("i8")
    frame["match_n_within"] = frame["match_n_within"].astype("i8")
    for name in ("quality", "fe_h_grid_edge"):
        frame[name] = frame[name].astype(bool)
    for name in (
        "source",
        "tier",
        "release",
        "pipeline",
        "dependency_group",
        "native_id",
        "id_method",
        "fe_h_kind",
        "alpha_kind",
    ):
        frame[name] = frame[name].astype(str)
    if not set(frame.tier) <= set(TIERS):
        raise ValueError("Unknown tier")
    if (frame.source_id < 0).any():
        raise ValueError("Negative Gaia ID")
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# SDSS-V Astra ASPCAP (DR19, Astra 0.6.0). DR20 carries no new APOGEE products.
# --------------------------------------------------------------------------

ASTRA_ASPCAP = dict(
    source="sdss_dr19_astra_aspcap",
    release="SDSS DR19 Astra 0.6.0 astraAllStarASPCAP",
    pipeline="ASPCAP (FERRE) in Astra; calibrated teff/logg, fe_h abundance, alpha_m_atm",
    dependency_group="apogee_spectra_aspcap",
    url="https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/astraAllStarASPCAP-0.6.0.fits.gz",
    sha1="b415f591ae88180e9b1e15dcfdf51cc62827d9ca",
    quality_policy="flag_bad false; finite calibrated teff/logg; snr>=20",
    grid_policy=(
        "fe_h or m_h_atm <= -2.45 marks the lower ASPCAP grid edge "
        "(DR19 values pile up at -2.50; censored, not measured)"
    ),
)


def astra_aspcap(path: str | Path) -> pd.DataFrame:
    from astropy.io import fits

    cols = [
        "sdss_id",
        "gaia_dr3_source_id",
        "ra",
        "dec",
        "telescope",
        "release",
        "snr",
        "teff",
        "e_teff",
        "logg",
        "e_logg",
        "fe_h",
        "e_fe_h",
        "m_h_atm",
        "e_m_h_atm",
        "alpha_m_atm",
        "e_alpha_m_atm",
        "flag_bad",
    ]
    with fits.open(path, memmap=True) as hdus:
        table = next(h for h in hdus[1:] if h.header.get("NAXIS2", 0) > 0).data
        data = {c: np.asarray(table[c]) for c in cols}
    n = len(data["sdss_id"])
    out = empty_frame(n, tier="survey", **_meta(ASTRA_ASPCAP))
    out["native_row"] = np.arange(n)
    out["native_id"] = [
        f"{int(s)}:{t.strip()}:{r.strip()}"
        for s, t, r in zip(
            data["sdss_id"],
            data["telescope"].astype(str),
            data["release"].astype(str),
            strict=True,
        )
    ]
    gaia = np.asarray(data["gaia_dr3_source_id"]).astype("i8")
    out["source_id"] = np.where(gaia > 0, gaia, 0)
    out["id_method"] = np.where(gaia > 0, "publisher_dr3", "")
    out["ra"], out["dec"], out["position_epoch"] = data["ra"], data["dec"], 2016.0
    for name in ("teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h", "snr"):
        out[name] = _f8(data[name])
    m_h = _f8(data["m_h_atm"])
    fe_h = out["fe_h"].to_numpy()
    out["fe_h_kind"] = np.where(np.isfinite(fe_h), "fe_h", "")
    out["alpha"], out["e_alpha"] = (
        _f8(data["alpha_m_atm"]),
        _f8(data["e_alpha_m_atm"]),
    )
    out["alpha_kind"] = "alpha_m_atm"
    out["fe_h_grid_edge"] = (np.isfinite(m_h) & (m_h <= -2.45)) | (
        np.isfinite(fe_h) & (fe_h <= -2.45)
    )
    out["quality"] = (
        ~np.asarray(data["flag_bad"], dtype=bool)
        & np.isfinite(out.teff)
        & np.isfinite(out.logg)
        & np.isfinite(out.fe_h)
        & (out.snr >= 20)
    )
    return finalize(out)


# --------------------------------------------------------------------------
# Other Astra pipelines on the same spectra (DR19 APOGEE; DR20 BOSS SLAM).
# They add pipelines, not stars; each is its own source and dependency group
# shared with the spectra (apogee_spectra_* / boss_spectra_*).
# --------------------------------------------------------------------------

_DR19 = "https://data.sdss.org/sas/dr19/spectro/astra/0.6.0/summary/"
_DR20 = "https://data.sdss.org/sas/dr20/spectro/astra/0.8.1/summary/"
ASTRA_SUMMARIES = (
    dict(
        source="sdss_dr19_astra_apogeenet",
        release="SDSS DR19 Astra 0.6.0 astraAllStarApogeeNet",
        pipeline="APOGEENet (neural network) on APOGEE spectra",
        dependency_group="apogee_spectra_aspcap",
        url=_DR19 + "astraAllStarApogeeNet-0.6.0.fits.gz",
        alpha=None,
        quality_policy="flag_bad false; finite teff/logg/fe_h",
    ),
    dict(
        source="sdss_dr19_astra_astronn",
        release="SDSS DR19 Astra 0.6.0 astraAllStarAstroNN",
        pipeline="AstroNN (Bayesian neural network) on APOGEE spectra",
        dependency_group="apogee_spectra_aspcap",
        url=_DR19 + "astraAllStarAstroNN-0.6.0.fits.gz",
        alpha=("mg_h", "e_mg_h", "mg_h"),
        quality_policy="flag_bad false; finite teff/logg/fe_h",
    ),
    dict(
        source="sdss_dr19_astra_thepayne",
        release="SDSS DR19 Astra 0.6.0 astraAllStarThePayne",
        pipeline="The Payne on APOGEE spectra",
        dependency_group="apogee_spectra_aspcap",
        url=_DR19 + "astraAllStarThePayne-0.6.0.fits.gz",
        alpha=("mg_h", "e_mg_h", "mg_h"),
        quality_policy="flag_bad false; finite teff/logg/fe_h",
    ),
    dict(
        source="sdss_dr20_astra_slam",
        release="SDSS DR20 Astra 0.8.1 astraAllStarSlam",
        pipeline="SLAM on BOSS spectra",
        dependency_group="boss_spectra_astra",
        url=_DR20 + "astraAllStarSlam-0.8.1.fits.gz",
        alpha=("alpha_fe", "e_alpha_fe", "alpha_fe"),
        quality_policy="flag_bad false; finite teff/logg/fe_h",
    ),
)


def astra_summary(path: str | Path, spec: dict) -> pd.DataFrame:
    """Rows of an Astra astraAllStar<Pipeline> summary (first non-empty HDU)."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdus:
        table = next(h for h in hdus[1:] if h.header.get("NAXIS2", 0) > 0).data
        names = set(table.columns.names)
        want = ["sdss_id", "gaia_dr3_source_id", "ra", "dec", "telescope"]
        want += ["release", "snr", "teff", "e_teff", "logg", "e_logg", "fe_h"]
        want += ["e_fe_h", "flag_bad", *(spec["alpha"] or ())[:2]]
        data = {c: np.asarray(table[c]) for c in want if c in names}
    n = len(data["sdss_id"])
    out = empty_frame(n, tier="survey", **_meta(spec))
    out["native_row"] = np.arange(n)
    tel = data.get("telescope", np.full(n, "")).astype(str)
    rel = data.get("release", np.full(n, "")).astype(str)
    out["native_id"] = [
        f"{int(s)}:{t.strip()}:{r.strip()}"
        for s, t, r in zip(data["sdss_id"], tel, rel, strict=True)
    ]
    gaia = np.asarray(data["gaia_dr3_source_id"]).astype("i8")
    out["source_id"] = np.where(gaia > 0, gaia, 0)
    out["id_method"] = np.where(gaia > 0, "publisher_dr3", "")
    out["ra"], out["dec"], out["position_epoch"] = data["ra"], data["dec"], 2016.0
    for name in ("teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h", "snr"):
        if name in data:
            out[name] = _f8(data[name])
    out["fe_h_kind"] = np.where(np.isfinite(out.fe_h), "fe_h", "")
    if spec["alpha"]:
        value, error, kind = spec["alpha"]
        alpha = _f8(data[value])
        if kind == "mg_h":  # [Mg/Fe] = [Mg/H] - [Fe/H]
            alpha = alpha - out.fe_h.to_numpy()
            kind = "mg_fe"
        out["alpha"], out["e_alpha"], out["alpha_kind"] = alpha, _f8(data[error]), kind
    out["quality"] = (
        ~np.asarray(data["flag_bad"], dtype=bool)
        & np.isfinite(out.teff)
        & np.isfinite(out.logg)
        & np.isfinite(out.fe_h)
    )
    return finalize(out)


MASTAR = dict(
    source="sdss_dr17_mastar_params",
    release="SDSS DR17 MaStar stellar parameters VAC v2 (goodstars, median of 4 methods)",
    pipeline="MaStar parameter VAC: median of DL/JI/LH/YC; calibrated [Fe/H]",
    dependency_group="mastar_spectra",
    url=(
        "https://data.sdss.org/sas/dr17/manga/spectro/mastar/v3_1_1/v1_7_7/vac/"
        "parameters/v2/mastar-goodstars-v3_1_1-v1_7_7-params-v2.fits"
    ),
    bytes=7888320,
    quality_policy="finite TEFF_MED, LOGG_MED and FEH_CAL_MED",
)


def mastar_params(path: str | Path) -> pd.DataFrame:
    from astropy.io import fits

    with fits.open(path, memmap=True) as f:
        t = f[1].data
        col = lambda k: np.asarray(t[k])  # noqa: E731
        n = len(t)
        out = empty_frame(n, tier="survey", **_meta(MASTAR))
        out["native_row"] = np.arange(n)
        out["native_id"] = np.char.strip(col("MANGAID").astype(str))
        out["ra"], out["dec"] = _f8(col("RA")), _f8(col("DEC"))
        epoch = _f8(col("EPOCH"))
        out["position_epoch"] = np.where(
            np.isfinite(epoch) & (epoch > 0), epoch, 2000.0
        )
        out["teff"], out["e_teff"] = _f8(col("TEFF_MED")), _f8(col("TEFF_MED_ERR"))
        out["logg"], out["e_logg"] = _f8(col("LOGG_MED")), _f8(col("LOGG_MED_ERR"))
        out["fe_h"], out["e_fe_h"] = (
            _f8(col("FEH_CAL_MED")),
            _f8(col("FEH_CAL_MED_ERR")),
        )
        out["alpha"], out["e_alpha"] = _f8(col("ALPHA_MED")), _f8(col("ALPHA_MED_ERR"))
        for name in (
            "teff",
            "e_teff",
            "logg",
            "e_logg",
            "fe_h",
            "e_fe_h",
            "alpha",
            "e_alpha",
        ):
            v = out[name].to_numpy("f8", copy=True)
            v[v <= -999] = np.nan  # VAC missing-value sentinel
            out[name] = v
    out["fe_h_kind"] = np.where(np.isfinite(out.fe_h), "fe_h", "")
    out["alpha_kind"] = "alpha_fe"
    out["quality"] = (
        np.isfinite(out.teff) & np.isfinite(out.logg) & np.isfinite(out.fe_h)
    )
    return finalize(out)


# --------------------------------------------------------------------------
# SDSS legacy + SEGUE SSPP (final DR12 ssppOut). Positions need Gaia matching.
# --------------------------------------------------------------------------

SSPP = dict(
    source="sdss_dr12_sspp",
    release="SDSS DR12 ssppOut (final SSPP; SEGUE-1/2 + legacy stars)",
    pipeline="SSPP adopted TEFF/LOGG/FEH_ADOP and AFE",
    dependency_group="sdss_optical_spectra_sspp",
    url="https://data.sdss.org/sas/dr12/sdss/sspp/ssppOut-dr12.fits",
    bytes=1913284800,
    quality_policy=(
        "FEH_ADOP valid; SNR>=10; 4000<=TEFF_ADOP<=10000; FLAG first char 'n'; "
        "ZWARNING==0; rows without any adopted label dropped; one row per "
        "BESTOBJID kept by quality then highest SNR"
    ),
    grid_policy="none flagged; SSPP metallicity estimators extend to about -4",
)


def sdss_sspp(path: str | Path) -> pd.DataFrame:
    from astropy.io import fits

    cols = [
        "SPECOBJID",
        "BESTOBJID",
        "RA",
        "DEC",
        "SNR",
        "FLAG",
        "ZWARNING",
        "TEFF_ADOP",
        "TEFF_ADOP_UNC",
        "LOGG_ADOP",
        "LOGG_ADOP_UNC",
        "FEH_ADOP",
        "FEH_ADOP_UNC",
        "AFE",
        "AFE_UNC",
    ]
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1].data
        if (
            hdus[1].header["NAXIS2"] * hdus[1].header["NAXIS1"]
            > Path(path).stat().st_size
        ):
            raise ValueError("Truncated SSPP file")
        data = {c: np.asarray(table[c]) for c in cols}
    n = len(data["SPECOBJID"])
    out = empty_frame(n, tier="survey", **_meta(SSPP))
    out["native_row"] = np.arange(n)
    spec_ids = np.char.strip(data["SPECOBJID"].astype(str))
    best_ids = np.char.strip(data["BESTOBJID"].astype(str))
    out["native_id"] = np.char.add(np.char.add(spec_ids, ":"), best_ids)
    out["ra"], out["dec"] = _f8(data["RA"]), _f8(data["DEC"])
    out["position_epoch"] = 2003.0  # SDSS imaging; epoch stability is tested
    for dst, src in (
        ("teff", "TEFF_ADOP"),
        ("e_teff", "TEFF_ADOP_UNC"),
        ("logg", "LOGG_ADOP"),
        ("e_logg", "LOGG_ADOP_UNC"),
        ("fe_h", "FEH_ADOP"),
        ("e_fe_h", "FEH_ADOP_UNC"),
        ("alpha", "AFE"),
        ("e_alpha", "AFE_UNC"),
        ("snr", "SNR"),
    ):
        out[dst] = _f8(data[src])
    out["fe_h_kind"], out["alpha_kind"] = "fe_h", "alpha_fe"
    flag = np.char.strip(data["FLAG"].astype(str))
    out["quality"] = (
        np.isfinite(out.fe_h)
        & np.isfinite(out.teff)
        & (out.snr >= 10)
        & (out.teff >= 4000)
        & (out.teff <= 10000)
        & np.char.startswith(flag, "n")
        & (np.asarray(data["ZWARNING"]) == 0)
    )
    # One row per photometric object (spectra without BESTOBJID stay separate)
    best = out.assign(best=np.where(best_ids != "", best_ids, "spec:" + spec_ids))
    best = best.sort_values(["best", "quality", "snr"], ascending=[True, False, False])
    best = best[np.isfinite(best.fe_h) | np.isfinite(best.teff)]  # any label
    keep = best.drop_duplicates("best").index
    return finalize(out.loc[np.sort(keep)])


# --------------------------------------------------------------------------
# VizieR-hosted catalogues (Gaia-ESO, S5, PIGS, metal-poor follow-up, high-res)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VizierSource:
    source: str
    table: str
    tier: str
    release: str
    pipeline: str
    dependency_group: str
    teff: str | None
    logg: str | None
    fe_h: str
    e_teff: str | None = None
    e_logg: str | None = None
    e_fe_h: str | None = None
    fe_h_kind: str = "fe_h"
    alpha: str | None = None
    alpha_kind: str = ""
    gaia: str | None = None
    gaia_release: str = "dr3"
    native_id: str | None = None
    snr: str | None = None
    epoch: float = 2000.0
    radius_arcsec: float = 1.5
    search_arcsec: float = 10.0
    expected_rows: int | None = None
    quality: dict = field(default_factory=dict)
    fe_h_floor: float | None = None
    position_from_name: bool = False
    notes: str = ""
    # Optional second VizieR table merged on a key (e.g. positions, HIP).
    join_table: str | None = None
    join_key: str | None = None
    join_right_key: str | None = None


VIZIER_SOURCES: tuple[VizierSource, ...] = (
    VizierSource(
        "gaia_eso_dr5_1",
        "J/A+A/676/A129/catalog",
        "survey",
        "Gaia-ESO DR5.1 final (Hourihane+2023)",
        "GES WG homogenised recommended parameters",
        "gaia_eso_flames",
        "Teff",
        "logg",
        "[Fe/H]",
        "e_Teff",
        "e_logg",
        "e_[Fe/H]",
        native_id="Object",
        snr="SNR",
        epoch=2000.0,
        expected_rows=114916,
        notes="Cluster/benchmark targets must be reserved; positions from 2MASS-based targeting",
    ),
    VizierSource(
        "s5_dr1",
        "J/MNRAS/490/3508/s5dr1",
        "metal_poor_survey",
        "S5 DR1 (Li+2019)",
        "rvspecfit posterior medians (50th percentile)",
        "s5_aat_2df",
        "teff50",
        "logg50",
        "feh50",
        e_fe_h="fehcalib_std",
        gaia="Gaia",
        gaia_release="edr3",
        native_id="Name",
        snr="sn_1700d",
        quality={"good_star": ("==", 1), "Prim": ("==", 1)},
        notes="Stream/halo fields; EDR3 IDs are identical to DR3 IDs",
    ),
    VizierSource(
        "pigs_aat",
        "J/MNRAS/530/3391/tabled2",
        "metal_poor_survey",
        "PIGS VIII AAT (Ardern-Arentsen+2024)",
        "FERRE on AAOmega blue+CaT",
        "pigs_aat",
        "TeffF",
        "loggF",
        "[Fe/H]F",
        "e_TeffFtot",
        "e_loggFtot",
        "e_[Fe/H]Ftot",
        gaia="GaiaDR3",
        native_id="Star",
        snr="SNRCaT",
        expected_rows=None,
        notes="ULySS alternative retained natively on ARC; FERRE adopted here",
    ),
    VizierSource(
        "lamost_vmp_li2018",
        "J/ApJS/238/16/table2",
        "metal_poor_survey",
        "LAMOST DR3 VMP catalogue (Li+2018)",
        "Li+2018 VMP template matching on LAMOST LRS",
        "lamost_lrs_spectra",
        "Teff",
        "log(g)",
        "[Fe/H]",
        "e_Teff",
        "e_log(g)",
        "e_[Fe/H]",
        native_id="LAMOST",
        snr="sng",
        notes="Same spectra as LASP; separate pipeline extending below the LASP floor",
    ),
    VizierSource(
        "lamost_cat_li2025",
        "J/ApJS/279/53/table2",
        "metal_poor_survey",
        "LAMOST DR10 CaT VMP metallicities (Li+2025)",
        "Ca II triplet calibration",
        "lamost_lrs_spectra",
        None,
        None,
        "[Fe/H]",
        e_fe_h="e_[Fe/H]",
        fe_h_kind="fe_h_cat",
        gaia="Gaia",
        native_id="obsid",
        snr="snr",
        notes="Teff/logg columns here are LASP, not this pipeline; not adopted",
    ),
    VizierSource(
        "pristine_aguado2019",
        "J/MNRAS/490/2241/table4",
        "metal_poor_survey",
        "Pristine INT/IDS follow-up (Aguado+2019)",
        "FERRE",
        "pristine_int_ids",
        "Teff",
        "logg",
        "[Fe/H]F",
        "e_Teff",
        "e_logg",
        "e_[Fe/H]F",
        native_id="Name",
        epoch=2015.0,
    ),
    VizierSource(
        "smss_dacosta2019_t2",
        "J/MNRAS/489/5900/table2",
        "metal_poor_survey",
        "SkyMapper EMP 2.3m/WiFeS (Da Costa+2019) table2",
        "FITTER spectrophotometric",
        "smss_wifes",
        "Teff",
        "logg",
        "[Fe/H]f",
        native_id="Name",
        epoch=2015.0,
    ),
    VizierSource(
        "smss_dacosta2019_t3",
        "J/MNRAS/489/5900/table3",
        "metal_poor_survey",
        "SkyMapper EMP 2.3m/WiFeS (Da Costa+2019) table3",
        "FITTER spectrophotometric",
        "smss_wifes",
        "Teff",
        "logg",
        "[Fe/H]f",
        native_id="Name",
        epoch=2015.0,
    ),
    VizierSource(
        "bestbrightest_placco2019",
        "J/ApJ/870/122/stars",
        "metal_poor_survey",
        "Best & Brightest medium-res (Placco+2019)",
        "n-SSPP",
        "bb_medres",
        "Teff",
        "logg",
        "[Fe/H]",
        alpha="[A/Fe]",
        alpha_kind="alpha_fe",
        native_id="WISEA",
        epoch=2010.5,
        position_from_name=True,
        notes="VizieR decimal positions are rounded to 0.001 deg; WISEA name parsed",
    ),
    VizierSource(
        "rpa_hansen2018",
        "J/ApJ/858/92/stars",
        "highres_reference",
        "RPA southern pilot (Hansen+2018)",
        "1D LTE high-res",
        "rpa_hansen2018",
        "Teff",
        "log(g)",
        "[Fe/H]",
        e_fe_h="e_[Fe/H]",
        native_id="2MASS",
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "rpa_holmbeck2020",
        "J/ApJS/249/30/stars",
        "highres_reference",
        "RPA (Holmbeck+2020)",
        "1D LTE high-res",
        "rpa_holmbeck2020",
        "Teff",
        "logg",
        "[Fe/H]",
        e_fe_h="e_[Fe/H]",
        native_id="2MASS",
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "smss_yong2021",
        "J/MNRAS/507/4102/sup",
        "highres_reference",
        "SkyMapper high-res follow-up (Yong+2021)",
        "1D LTE high-res",
        "smss_yong2021",
        "Teff",
        "logg",
        "[Fe/H]",
        native_id="SMSS",
        epoch=2015.0,
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "yong2013_program",
        "J/ApJ/762/26/stars",
        "highres_reference",
        "Most metal-poor stars II program (Yong+2013)",
        "1D LTE high-res",
        "yong2013",
        "Teff",
        "logg",
        "[Fe/H]",
        native_id="Name",
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "yong2013_literature",
        "J/ApJ/762/26/sample",
        "highres_reference",
        "Most metal-poor stars II homogenised literature (Yong+2013)",
        "1D LTE high-res reanalysis",
        "yong2013",
        "Teff",
        "logg",
        "[Fe/H]",
        native_id="Name",
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "roederer2014",
        "J/AJ/147/136/stars",
        "highres_reference",
        "Stars of very low metal abundance VI (Roederer+2014)",
        "1D LTE high-res",
        "roederer2014",
        "Teff",
        "logg",
        "[M/H]",
        fe_h_kind="m_h_model",
        native_id="Name",
        radius_arcsec=6.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "pastel_2020",
        "B/pastel/pastel",
        "highres_reference",
        "PASTEL 2020-01-30 (Soubiran+2016)",
        "heterogeneous high-res literature compilation",
        "pastel_literature",
        "Teff",
        "logg",
        "[Fe/H]",
        e_fe_h="e_[Fe/H]",
        native_id="ID",
        radius_arcsec=3.0,
        search_arcsec=30.0,
        notes="Multiple rows per star from different papers; not independent",
    ),
    # ---- Added 2026-11-11: AMBRE, RPA, dwarf galaxies, LAMOST-Subaru, SEGUE
    # high-res follow-up, solar-neighbourhood dwarfs.
    VizierSource(
        "ambre_feros",
        "J/A+A/542/A48/ambredr1",
        "survey",
        "AMBRE FEROS archive spectra (Worley+2012)",
        "MATISSE (AMBRE)",
        "ambre_feros",
        "Teff",
        "logg",
        "[M/H]",
        e_teff="e_Teff",
        e_logg="e_logg",
        e_fe_h="e_[M/H]",
        fe_h_kind="m_h",
        alpha="[alpha/Fe]",
        alpha_kind="alpha_fe",
        native_id="Ident",
        snr="SNR",
        radius_arcsec=3.0,
        search_arcsec=30.0,
        notes="One row per archived spectrum; [M/H] is a global metallicity",
    ),
    VizierSource(
        "ezzeddine2020_rpa",
        "J/ApJ/898/150/abund",
        "highres_reference",
        "RPA MIKE metal-poor stars (Ezzeddine+2020)",
        "1D LTE high-res (corrected LTE parameters)",
        "rpa_ezzeddine2020",
        "TeffLTEc",
        "loggLTEc",
        "[Fe/H]LTEc",
        e_fe_h="e_[Fe/H]LTEc",
        native_id="2MASS",
        position_from_name=True,
        radius_arcsec=3.0,
        search_arcsec=30.0,
        notes="Positions parsed from the 2MASS designation; NLTE values in the native row",
    ),
    VizierSource(
        "sakari2018_rpa",
        "J/ApJ/868/110/table3",
        "highres_reference",
        "RPA first northern/southern halo release (Sakari+2018)",
        "1D LTE high-res",
        "rpa_sakari2018",
        "Teff",
        "log(g)",
        "[FeII/H]",
        e_fe_h="e_[FeII/H]",
        native_id="ID",
        radius_arcsec=3.0,
        search_arcsec=30.0,
        join_table="J/ApJ/868/110/table1",
        join_key="ID",
        notes="Positions from table1 joined on ID; [FeI/H] in the native row",
    ),
    VizierSource(
        "kirby2010_dsph",
        "J/ApJS/191/352/abun",
        "metal_poor_survey",
        "Milky Way dSph DEIMOS abundances (Kirby+2010)",
        "DEIMOS medium-resolution spectral synthesis",
        "kirby2010_deimos",
        "Teff",
        "logg",
        "[Fe/H]",
        e_teff="e_Teff",
        e_logg="e_logg",
        e_fe_h="e_[Fe/H]",
        alpha="[Mg/Fe]",
        alpha_kind="mg_fe",
        native_id="Name",
        radius_arcsec=1.5,
        search_arcsec=10.0,
        notes="Stars in eight dwarf spheroidal galaxies (dSph column)",
    ),
    VizierSource(
        "kirby2010_highres",
        "J/ApJS/191/352/table7",
        "highres_reference",
        "dSph high-res literature comparison set (Kirby+2010 table 7)",
        "literature high-resolution",
        "kirby2010_literature",
        "TeffH",
        "loggH",
        "[Fe/H]H",
        e_fe_h="e_[Fe/H]H",
        alpha="[Mg/Fe]H",
        alpha_kind="mg_fe",
        native_id="Name",
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "li2022_lamost_subaru",
        "J/ApJ/931/147/table2",
        "highres_reference",
        "LAMOST VMP stars with Subaru high-res (Li+2022)",
        "1D LTE high-res",
        "lamost_subaru",
        "Teff",
        "logg",
        "FeH",
        e_teff="e_Teff",
        e_logg="e_logg",
        e_fe_h="e_FeH",
        native_id="ID",
        snr="SNR",
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "aoki2013_segue",
        "J/AJ/145/13/stars",
        "highres_reference",
        "SEGUE extremely metal-poor follow-up (Aoki+2013)",
        "1D LTE high-res (Subaru)",
        "aoki2013",
        "Teff",
        "logg",
        "[Fe/H]",
        native_id="SDSS",
        radius_arcsec=3.0,
        search_arcsec=30.0,
    ),
    VizierSource(
        "bensby2014",
        "J/A+A/562/A71/tablec3",
        "highres_reference",
        "Solar-neighbourhood F/G dwarfs (Bensby+2014)",
        "1D LTE high-res",
        "bensby2014",
        "Teff",
        "logg",
        "Fe/H",
        e_teff="e_Teff",
        e_logg="e_logg",
        e_fe_h="e_Fe/H",
        alpha="Mg/Fe",
        alpha_kind="mg_fe",
        native_id="HIP",
        epoch=1991.25,
        radius_arcsec=1.5,
        search_arcsec=20.0,
        join_table="I/311/hip2",
        join_key="HIP",
        notes="Positions from Hipparcos (van Leeuwen 2007) joined on HIP, epoch 1991.25",
    ),
)


def _meta(spec: dict) -> dict:
    return {k: spec[k] for k in ("source", "release", "pipeline", "dependency_group")}


def fetch_vizier(
    spec: VizierSource, cache: Path, mirrors=VIZIER_MIRRORS, timeout=300
) -> tuple[pd.DataFrame, dict]:
    """Download the full table once (ASU-TSV + VizieR J2000 positions); cache it."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (spec.source + ".tsv")
    receipt = cache / (spec.source + ".json")
    if not path.exists():
        query = urllib.parse.urlencode(
            {
                "-source": spec.table,
                "-out.max": "unlimited",
                "-out": "**",
                "-out.add": "_RAJ,_DEJ",
                "-oc.form": "d",
            }
        )
        errors = []
        # A mirror that refuses the tunnel (403) is skipped at once; a mirror
        # that answers without a table (rate limiting) is retried with backoff.
        for base in mirrors:
            for attempt in range(5):
                try:
                    if attempt:
                        time.sleep(min(300, 30 * 2 ** (attempt - 1)))
                    with urllib.request.urlopen(
                        base + "?" + query, timeout=timeout
                    ) as r:
                        payload = r.read()
                    if b"#Column" not in payload:
                        head = payload[:300].decode("utf-8", "replace")
                        head = " ".join(ln for ln in head.splitlines() if ln.strip())
                        raise ValueError(f"No table schema in response: {head[:200]}")
                    break
                except urllib.error.URLError as error:
                    errors.append(f"{base}: URLError: {error.reason}")
                    payload = None
                    refused = isinstance(error, urllib.error.HTTPError) and (
                        400 <= error.code < 500
                    )
                    if refused or "403" in str(error.reason):
                        break  # blocked tunnel or missing table: next mirror
                except Exception as error:
                    errors.append(f"{base}: {type(error).__name__}: {error}")
                    payload = None
            if payload is None:
                continue
            try:
                tmp = path.with_suffix(".part")
                tmp.write_bytes(payload)
                tmp.replace(path)
                receipt.write_text(
                    json.dumps(
                        dict(
                            table=spec.table,
                            url=base + "?" + query,
                            sha256=sha256(path),
                            bytes=len(payload),
                            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        ),
                        indent=2,
                    )
                )
                break
            except Exception as error:  # try the next mirror
                errors.append(f"{base}: {type(error).__name__}: {error}")
        else:
            raise RuntimeError("; ".join(errors))
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    frame = pd.read_csv(
        io.StringIO("\n".join([lines[0]] + lines[3:])),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    frame = frame.map(lambda v: v.strip())
    if spec.expected_rows is not None and len(frame) != spec.expected_rows:
        raise ValueError(f"{spec.source}: {len(frame)} rows != {spec.expected_rows}")
    meta = json.loads(receipt.read_text())
    if spec.join_table:
        frame, meta["join"] = _join_vizier(spec, frame, cache, mirrors, timeout)
    return frame, meta


def _join_vizier(spec, frame, cache, mirrors, timeout):
    """Left-join a second VizieR table on a key (one row per key kept)."""
    from dataclasses import replace

    side = replace(
        spec,
        source=spec.source + "__join",
        table=spec.join_table,
        join_table=None,
        expected_rows=None,
    )
    other, meta = fetch_vizier(side, cache, mirrors=mirrors, timeout=timeout)
    right = spec.join_right_key or spec.join_key
    other = other.drop_duplicates(right)
    keep = [c for c in other.columns if c == right or c not in frame.columns]
    merged = frame.merge(other[keep], how="left", left_on=spec.join_key, right_on=right)
    if len(merged) != len(frame):
        raise ValueError(f"{spec.source}: join changed the row count")
    return merged, meta


_DESIGNATION = re.compile(
    r"J(\d{2})(\d{2})(\d{2})(\.?\d*)([+-])(\d{2})(\d{2})(\d{2})(\.?\d*)"
)


def _seconds(whole: str, frac: str) -> float:
    """``42`` + ``.77`` or 2MASS-style ``42`` + ``77`` (implied decimal point)."""
    frac = frac.lstrip(".")
    return float(whole + ("." + frac if frac else ""))


def position_from_designation(names) -> tuple[np.ndarray, np.ndarray]:
    """RA/Dec (deg) from ``JHHMMSS.ss+DDMMSS.s`` or 2MASS ``JHHMMSSss+DDMMSSs``."""
    ra = np.full(len(names), np.nan)
    dec = np.full(len(names), np.nan)
    for i, name in enumerate(names):
        m = _DESIGNATION.search(str(name).replace(" ", ""))
        if not m:
            continue
        h, mi, se, sf, sign, d, dm, ds, df = m.groups()
        ra[i] = 15 * (int(h) + int(mi) / 60 + _seconds(se, sf) / 3600)
        sign = 1 if sign == "+" else -1
        dec[i] = sign * (int(d) + int(dm) / 60 + _seconds(ds, df) / 3600)
    return ra, dec


def parse_sexagesimal(ra_text, dec_text) -> tuple[np.ndarray, np.ndarray]:
    """Degrees from ``HH MM SS.s`` / ``±DD MM SS.s`` strings; NaN when blank."""
    ra = np.full(len(ra_text), np.nan)
    dec = np.full(len(dec_text), np.nan)
    for i, (r, d) in enumerate(zip(ra_text, dec_text, strict=True)):
        rp, dp = str(r).split(), str(d).split()
        if len(rp) != 3 or len(dp) != 3:
            continue
        try:
            ra[i] = 15 * (float(rp[0]) + float(rp[1]) / 60 + float(rp[2]) / 3600)
            sign = -1 if dp[0].startswith("-") else 1
            deg = abs(float(dp[0]))
            dec[i] = sign * (deg + float(dp[1]) / 60 + float(dp[2]) / 3600)
        except ValueError:
            continue
    return ra, dec


def vizier_observations(spec: VizierSource, frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    out = empty_frame(
        n,
        tier=spec.tier,
        source=spec.source,
        release=spec.release,
        pipeline=spec.pipeline,
        dependency_group=spec.dependency_group,
    )
    out["native_row"] = np.arange(n)
    if spec.native_id:
        out["native_id"] = frame[spec.native_id].to_numpy(str)
    ra_col = "_RAJ2000" if "_RAJ2000" in frame else "RAJ2000"
    de_col = "_DEJ2000" if "_DEJ2000" in frame else "DEJ2000"
    if ra_col in frame and de_col in frame:
        out["ra"], out["dec"] = _f8(frame[ra_col]), _f8(frame[de_col])
    if spec.position_from_name:
        out["ra"], out["dec"] = position_from_designation(frame[spec.native_id])
    out["position_epoch"] = spec.epoch
    if spec.gaia:
        ids = _exact_ids(frame[spec.gaia])
        out["source_id"] = ids
        out["id_method"] = np.where(ids > 0, "publisher_" + spec.gaia_release, "")
    for dst, src in (
        ("teff", spec.teff),
        ("e_teff", spec.e_teff),
        ("logg", spec.logg),
        ("e_logg", spec.e_logg),
        ("fe_h", spec.fe_h),
        ("e_fe_h", spec.e_fe_h),
        ("alpha", spec.alpha),
        ("snr", spec.snr),
    ):
        if src:
            out[dst] = _f8(frame[src])
    out["fe_h_kind"] = np.where(np.isfinite(out.fe_h), spec.fe_h_kind, "")
    out["alpha_kind"] = spec.alpha_kind if spec.alpha else ""
    quality = np.isfinite(out.fe_h.to_numpy())
    for column, (op, value) in spec.quality.items():
        native = _f8(frame[column])
        quality &= {"==": native == value, ">=": native >= value}[op]
    if spec.fe_h_floor is not None:
        out["fe_h_grid_edge"] = out.fe_h <= spec.fe_h_floor
    out["quality"] = quality
    return finalize(out)


# --------------------------------------------------------------------------
# Metal-poor literature compilations: SAGA (Milky Way) and JINAbase
# --------------------------------------------------------------------------

SAGA = dict(
    source="saga_mw",
    release="SAGA database, Milky Way retrieval system (database update 2026-01-20)",
    pipeline="literature high-resolution compilation; [Fe/H] on Asplund+2009 solar scale",
    dependency_group="literature_highres_compilation",
    url="http://sagadatabase.jp/Retrieval/db.cgi",
    fe_h_bins=tuple(np.round(np.arange(-8.0, 1.51, 0.5), 2)),
    query=(
        "x=[Fe/H] (metal), y=Teff, w1=logg; DATA_CHOICE=data_same (all three from "
        "the same paper); every paper row kept (DATA_SAME_ONE=0); SOLAR_REF=A09; "
        "downloaded in 0.5 dex [Fe/H] slices because the server gateway times out "
        "after 60 s; rows repeated at slice edges are dropped"
    ),
    notes=(
        "Rows are per (star, paper). Positions missing for a row are filled from "
        "another row of the same object, then from a J2000 designation in the name"
    ),
)
SAGA_COLUMNS = (
    "Object",
    "Reference",
    "Population",
    "Facility",
    "RA",
    "Decl",
    "Teff",
    "logg",
    "Metallicity",
    "metal",
    "Teff_2",
    "logg_2",
)


def _saga_form(lo, hi) -> list[tuple[str, str]]:
    return [
        ("OUTPUT_OPTION", "single_selected"),
        ("xcategory", "Atmosphere"),
        ("xaxis", "metal"),
        ("AX:xaxis_text", "[Fe/H]"),
        ("AX:xaxis_from", lo),
        ("AX:xaxis_to", hi),
        ("xupper", "include"),
        ("GP:xerrorbars", "with"),
        ("ycategory", "Atmosphere"),
        ("yaxis", "Teff"),
        ("AX:yaxis_text", "Teff"),
        ("AX:yaxis_from", ""),
        ("AX:yaxis_to", ""),
        ("yupper", "include"),
        ("GP:yerrorbars", "with"),
        ("w1category", "Atmosphere"),
        ("w1axis", "logg"),
        ("AX:w1axis_text", "logg"),
        ("AX:w1axis_from", ""),
        ("AX:w1axis_to", ""),
        ("w1upper", "include"),
        ("SE:star_name", ""),
        ("auth_mode", "or"),
        ("DATA_CHOICE", "data_same"),
        ("DATA_SAME_ONE", "0"),
        ("DATA_SOL", "sol_any"),
        ("SOLAR_REF", "A09"),
        ("DATA_REPRESENTATIVE", "Mean"),
        ("SE:n_display", "10"),
        ("OUTPUT_DATA_PLOT", "0"),
        ("SE:output_text_data", "1"),
        ("SE:output_search_result", "0"),
        ("DM:output_data_comment", "0"),
        ("SE:output_data_error", "0"),
        ("button", "search"),
    ]


def _multipart(fields) -> dict:
    return {"files": [(k, (None, v)) for k, v in fields]}


def fetch_saga(cache: Path, log=print) -> tuple[pd.DataFrame, dict]:
    """Query SAGA once per [Fe/H] slice (search, then tab-separated download)."""
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    edges = SAGA["fe_h_bins"]
    parts, files = [], {}
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        path = cache / f"saga-mw-{lo:+.1f}-{hi:+.1f}.tsv"
        if not path.exists():
            for attempt in range(4):
                try:
                    with requests.Session() as session:
                        session.post(
                            SAGA["url"],
                            timeout=300,
                            **_multipart(_saga_form(f"{lo}", f"{hi}")),
                        ).raise_for_status()
                        download = [("button", "text_download")]
                        download += [("TDM:columns", c) for c in SAGA_COLUMNS[:10]]
                        download += [("TDM:columns", "Teff"), ("TDM:columns", "logg")]
                        download += [("TDM_ncolumn", "0")]
                        r = session.post(
                            SAGA["url"], timeout=300, **_multipart(download)
                        )
                        r.raise_for_status()
                    text = r.content.decode("utf-8", errors="replace")
                    if not text.startswith("# Object"):
                        raise ValueError("SAGA did not return a data table")
                    tmp = path.with_suffix(".part")
                    tmp.write_text(text)
                    tmp.replace(path)
                    break
                except Exception as error:
                    log(f"saga slice {lo}..{hi}: attempt {attempt}: {error}")
                    if attempt == 3:
                        raise
                    time.sleep(10 * (attempt + 1))
        files[path.name] = sha256(path)
        lines = [ln for ln in path.read_text().splitlines() if ln and ln[0] != "#"]
        rows = [ln.split("\t") for ln in lines]
        if any(len(r) != len(SAGA_COLUMNS) for r in rows):
            raise ValueError(f"Unexpected SAGA column count in {path.name}")
        parts.append(pd.DataFrame(rows, columns=list(SAGA_COLUMNS)))
        log(f"saga slice {lo:+.1f}..{hi:+.1f}: {len(rows)} rows")
    raw = pd.concat(parts, ignore_index=True).map(str.strip)
    before = len(raw)
    raw = raw.drop_duplicates(ignore_index=True)
    receipt = dict(
        url=SAGA["url"],
        query=SAGA["query"],
        slices=files,
        rows_before_edge_dedup=before,
        rows=len(raw),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return raw, receipt


def _fill_positions(names: pd.Series, ra: np.ndarray, dec: np.ndarray):
    """Fill missing positions from other rows of the same object, then names."""
    frame = pd.DataFrame(dict(name=names.to_numpy(), ra=ra, dec=dec))
    known = frame.dropna().drop_duplicates("name").set_index("name")
    miss = frame.ra.isna()
    frame.loc[miss, "ra"] = frame.loc[miss, "name"].map(known.ra)
    frame.loc[miss, "dec"] = frame.loc[miss, "name"].map(known.dec)
    miss = frame.ra.isna().to_numpy()
    nra, ndec = position_from_designation(frame.name[miss])
    frame.loc[miss, "ra"], frame.loc[miss, "dec"] = nra, ndec
    return frame.ra.to_numpy("f8"), frame.dec.to_numpy("f8")


def saga_observations(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    out = empty_frame(n, tier="highres_reference", **_meta(SAGA))
    out["native_row"] = np.arange(n)
    out["native_id"] = (raw.Object + "|" + raw.Reference).to_numpy(str)
    ra, dec = parse_sexagesimal(raw.RA, raw.Decl)
    out["ra"], out["dec"] = _fill_positions(raw.Object, ra, dec)
    out["position_epoch"] = 2000.0
    out["teff"], out["logg"] = _f8(raw.Teff), _f8(raw.logg)
    out["fe_h"] = _f8(raw.metal)
    out["fe_h_kind"] = np.where(np.isfinite(out.fe_h), "fe_h_a09", "")
    out["quality"] = np.isfinite(out.fe_h) & raw.Population.eq("MILKYWAY").to_numpy()
    return finalize(out)


JINABASE = dict(
    source="jinabase_2021",
    release="JINAbase 2021 CSV (Abohalima & Frebel 2018; Mardini et al. updates)",
    pipeline="literature high-resolution compilation (paper values; Priority kept)",
    dependency_group="literature_highres_compilation",
    commit="e034a8c0bea71b3a657046ae622d75750c661319",
    url=(
        "https://raw.githubusercontent.com/Mohammad-Mardini/JINAbase/"
        "e034a8c0bea71b3a657046ae622d75750c661319/JINAbase_2021.csv"
    ),
    quality_policy="finite Fe/H; Priority==1 marks the compilers' preferred paper",
)


def fetch_jinabase(cache: Path) -> tuple[pd.DataFrame, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "JINAbase_2021.csv"
    if not path.exists():
        with urllib.request.urlopen(JINABASE["url"], timeout=300) as r:
            payload = r.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(payload)
        tmp.replace(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False).map(str.strip)
    receipt = dict(
        url=JINABASE["url"],
        commit=JINABASE["commit"],
        sha256=sha256(path),
        rows=len(raw),
    )
    return raw, receipt


def jinabase_observations(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)
    out = empty_frame(n, tier="highres_reference", **_meta(JINABASE))
    out["native_row"] = np.arange(n)
    out["native_id"] = (raw.Name + "|" + raw.Ref).to_numpy(str)
    ra, dec = parse_sexagesimal(raw.RA, raw.DEC)
    out["ra"], out["dec"] = _fill_positions(raw.Name, ra, dec)
    out["position_epoch"] = 2000.0
    out["teff"], out["logg"] = _f8(raw.Teff), _f8(raw.logg)
    out["fe_h"] = _f8(raw["Fe/H"])
    out["fe_h_kind"] = np.where(np.isfinite(out.fe_h), "fe_h", "")
    out["quality"] = np.isfinite(out.fe_h) & (_f8(raw.Priority) == 1)
    return finalize(out)


SIMBAD_TAP = (
    "https://simbad.cfa.harvard.edu/simbad/sim-tap",
    "https://simbad.cds.unistra.fr/simbad/sim-tap",
)


def name_variants(name: str) -> list[str]:
    """SIMBAD spellings for compact literature names (``HD2665``, ``CD-30_298``)."""
    base = re.sub(r"\s+", " ", str(name).replace("_", " ")).strip()
    out = [base]
    spaced = re.sub(r"^(2MASS|SDSS|SMSS|Gaia ?DR[23])J", r"\1 J", base)
    spaced = re.sub(r"^([A-Za-z]+)(?=[+\-]?\d)", r"\1 ", spaced)
    spaced = re.sub(r"^(BD|CD|CPD) ([+-])", r"\1\2", spaced)
    out.append(spaced)
    cs = re.match(r"^CS ?(\d{5})-(\d{1,4})$", spaced)
    if cs:
        out.append(f"BPS CS {cs.group(1)}-{int(cs.group(2)):04d}")
    bd = re.match(r"^(BD|CD|CPD)([+-])(\d{1,2}) (\d+)$", spaced)
    if bd:
        out.append(f"{bd.group(1)}{bd.group(2)}{int(bd.group(3)):02d} {bd.group(4)}")
    seen, unique = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def simbad_resolve(names, cache: Path, log=print) -> pd.DataFrame:
    """Gaia DR3 ID and J2000 position per name through SIMBAD identifiers.

    Returns name, simbad_main_id, ra, dec, gaia_dr3 (0 if SIMBAD lists none).
    Names SIMBAD does not know are absent. Cached per name set.
    """
    import pyvo
    from astropy.table import Table

    names = sorted(set(map(str, names)))
    rows = [
        (k, r, v) for k, n in enumerate(names) for r, v in enumerate(name_variants(n))
    ]
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    path = cache / f"simbad-{digest}.parquet"
    cache.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        query = (
            "SELECT t.k, t.r, b.main_id, b.ra, b.dec, g.id AS gaia "
            "FROM TAP_UPLOAD.t AS t JOIN ident AS i ON i.id = t.v "
            "JOIN basic AS b ON b.oid = i.oidref "
            "LEFT JOIN ident AS g ON g.oidref = i.oidref AND g.id LIKE 'Gaia DR3 %'"
        )
        parts = []
        for start in range(0, len(rows), 5000):
            chunk = rows[start : start + 5000]
            upload = Table(
                dict(
                    k=np.array([c[0] for c in chunk], "i8"),
                    r=np.array([c[1] for c in chunk], "i8"),
                    v=np.array([c[2] for c in chunk], str),
                )
            )
            for attempt, url in enumerate(SIMBAD_TAP * 3):
                try:
                    table = (
                        pyvo.dal.TAPService(url)
                        .run_sync(query, uploads={"t": upload}, maxrec=100000)
                        .to_table()
                    )
                    break
                except Exception as error:
                    log(f"simbad batch {start}: {url}: {error}")
                    if attempt == 3 * len(SIMBAD_TAP) - 1:
                        raise
                    time.sleep(5 * (attempt + 1))
            parts.append(
                pd.DataFrame(
                    dict(
                        k=np.asarray(table["k"], "i8"),
                        r=np.asarray(table["r"], "i8"),
                        simbad_main_id=np.asarray(table["main_id"]).astype(str),
                        ra=np.ma.filled(np.ma.asarray(table["ra"], float), np.nan),
                        dec=np.ma.filled(np.ma.asarray(table["dec"], float), np.nan),
                        gaia=np.ma.filled(np.ma.asarray(table["gaia"]), "").astype(str),
                    )
                )
            )
        found = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        found.to_parquet(path, index=False)
    found = pd.read_parquet(path)
    if found.empty:
        return pd.DataFrame(columns=["name", "simbad_main_id", "ra", "dec", "gaia_dr3"])
    found["gaia_dr3"] = _exact_ids(
        [g.replace("Gaia DR3", "").strip() if g else "" for g in found.gaia]
    )
    # Best spelling per name; an object with several Gaia DR3 idents is ambiguous
    found = found.sort_values(["k", "r"])
    first = found.groupby("k").r.transform("min")
    found = found[found.r == first]
    counts = found.groupby("k").gaia_dr3.nunique()
    found = found.drop_duplicates("k").set_index("k")
    found.loc[counts[counts > 1].index, "gaia_dr3"] = 0
    found["name"] = [names[k] for k in found.index]
    return found[["name", "simbad_main_id", "ra", "dec", "gaia_dr3"]].reset_index(
        drop=True
    )


def apply_name_resolution(obs: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    """Give literature rows SIMBAD Gaia DR3 IDs (exact int64, never via float).

    Rows whose name SIMBAD maps to exactly one Gaia DR3 source get that ID;
    other rows lacking a position get the SIMBAD J2000 position instead.
    """
    obs = obs.copy()
    names = obs.native_id.str.split("|").str[0].to_numpy()
    ids = dict(zip(resolved.name, resolved.gaia_dr3.to_numpy("i8"), strict=True))
    ras = dict(zip(resolved.name, resolved.ra.to_numpy("f8"), strict=True))
    decs = dict(zip(resolved.name, resolved.dec.to_numpy("f8"), strict=True))
    gaia = np.array([ids.get(n, 0) for n in names], dtype="i8")
    hit = (obs.source_id.to_numpy("i8") == 0) & (gaia > 0)
    obs.loc[hit, "source_id"] = gaia[hit]
    obs.loc[hit, "id_method"] = "simbad_dr3"
    fill = obs.ra.isna().to_numpy() & ~hit
    obs.loc[fill, "ra"] = [ras.get(n, np.nan) for n in names[fill]]
    obs.loc[fill, "dec"] = [decs.get(n, np.nan) for n in names[fill]]
    obs["source_id"] = obs.source_id.astype("i8")
    return obs


def resolve_by_name(obs: pd.DataFrame, cache: Path, log=print) -> pd.DataFrame:
    """SIMBAD identity for literature rows (``native_id`` = ``name|reference``)."""
    names = obs.native_id.str.split("|").str[0]
    resolved = simbad_resolve(names.unique(), cache, log=log)
    out = apply_name_resolution(obs, resolved)
    log(
        f"simbad: {len(resolved)} of {names.nunique()} names known; "
        f"{int((out.id_method == 'simbad_dr3').sum())} rows given Gaia DR3 IDs"
    )
    return out


# --------------------------------------------------------------------------
# Positional Gaia DR3 identity through the ESA Gaia archive (public positions)
# --------------------------------------------------------------------------


def propagate(ra, dec, pmra, pmdec, dt_years):
    """Linear proper-motion propagation in degrees (adequate for <1 arcmin)."""
    pmra = np.nan_to_num(pmra)
    pmdec = np.nan_to_num(pmdec)
    dec_t = dec + pmdec * dt_years / 3.6e6
    ra_t = ra + pmra * dt_years / 3.6e6 / np.cos(np.radians(dec))
    return ra_t, dec_t


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    s = (
        np.sin((dec2 - dec1) / 2) ** 2
        + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    )
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600


def choose_matches(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    radius_arcsec: float,
    epochs: tuple[float, ...],
) -> pd.DataFrame:
    """Best Gaia candidate per target, stable across the assumed epochs.

    ``candidates`` has key, source_id, ra, dec, pmra, pmdec, ref_epoch;
    ``targets`` has key, ra, dec. A target is matched only if the nearest
    candidate is within the radius at every assumed epoch and is the same
    source at every epoch. ``n_within`` counts distinct candidates inside the
    radius at any epoch (>1 means a crowded/ambiguous identity).
    """
    if candidates.empty:
        return pd.DataFrame(
            dict(key=[], source_id=[], sep_arcsec=[], n_within=[])
        ).astype(dict(key="i8", source_id="i8", sep_arcsec="f8", n_within="i8"))
    pairs = candidates.merge(
        targets[["key", "ra", "dec"]].rename(columns=dict(ra="t_ra", dec="t_dec")),
        on="key",
    )
    best, within = [], []
    for epoch in epochs:
        ra, dec = propagate(
            pairs.ra.to_numpy(),
            pairs.dec.to_numpy(),
            pairs.pmra.to_numpy(),
            pairs.pmdec.to_numpy(),
            epoch - pairs.ref_epoch.to_numpy(),
        )
        sep = angular_sep_arcsec(ra, dec, pairs.t_ra.to_numpy(), pairs.t_dec.to_numpy())
        p = pairs[["key", "source_id"]].assign(sep=sep)
        within.append(p[p.sep <= radius_arcsec][["key", "source_id"]])
        nearest = p.sort_values(["key", "sep"]).drop_duplicates("key")
        best.append(nearest[nearest.sep <= radius_arcsec].set_index("key"))
    stable = best[0][["source_id", "sep"]].rename(columns=dict(sep="sep_arcsec"))
    for other in best[1:]:
        joined = stable.join(
            other[["source_id"]].rename(columns=dict(source_id="source_id_o")),
            how="inner",
        )
        stable = joined[joined.source_id == joined.source_id_o][
            ["source_id", "sep_arcsec"]
        ]
    n_within = (
        pd.concat(within).drop_duplicates().groupby("key").size().rename("n_within")
    )
    out = stable.join(n_within, how="left").reset_index()
    out["n_within"] = out.n_within.fillna(1).astype("i8")
    return out


def _phase(job, log, label):
    """Job phase, tolerating transient network errors while polling."""
    for attempt in range(10):
        try:
            return job.phase
        except Exception as error:
            log(f"gaia batch {label}: phase poll retry {attempt}: {error}")
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError("Gaia archive job phase unavailable")


def _run_upload_job(service, query, upload, *, log, label, attempts=6):
    """Submit one upload query; resubmit only when the job itself is lost."""
    for attempt in range(attempts):
        try:
            job = service.submit_job(query, uploads={"t": upload}, maxrec=3_000_000)
            job.run()
            deadline = time.monotonic() + 3600
            while _phase(job, log, label) in ("PENDING", "QUEUED", "EXECUTING", "RUN"):
                if time.monotonic() > deadline:
                    raise TimeoutError("Gaia archive job exceeded 1 h")
                time.sleep(5)
            if job.phase != "COMPLETED":
                job.raise_if_error()
                raise RuntimeError("Gaia archive job " + job.phase)
            for fetch in range(6):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = job.fetch_result().to_table()
                    break
                except Exception as error:
                    if fetch == 5:
                        raise
                    log(f"gaia batch {label}: fetch retry {fetch}: {error}")
                    time.sleep(10 * (fetch + 1))
            try:
                job.delete()
            except Exception:
                pass
            return result
        except Exception as error:
            log(f"gaia batch {label}: attempt {attempt}: {error}")
            if attempt == attempts - 1:
                raise
            time.sleep(min(120, 2 ** (attempt + 3)))
    raise AssertionError("unreachable")


def gaia_candidates(
    targets: pd.DataFrame,
    cache: Path,
    *,
    search_arcsec: float = 5.0,
    batch: int = 20000,
    service_url: str = GAIA_TAP,
    log=print,
) -> pd.DataFrame:
    """Candidate Gaia DR3 sources near each (key, ra, dec), cached per batch."""
    import pyvo
    from astropy.table import Table

    cache.mkdir(parents=True, exist_ok=True)
    service = pyvo.dal.TAPService(service_url)
    radius = search_arcsec / 3600.0
    query = (
        "SELECT t.key, g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.ref_epoch, "
        "g.phot_g_mean_mag FROM TAP_UPLOAD.t AS t JOIN gaiadr3.gaia_source AS g "
        # POINT(upload) inside CIRCLE(gaia) lets q3c use the gaia_source index
        f"ON 1=CONTAINS(POINT('ICRS',t.ra,t.dec),CIRCLE('ICRS',g.ra,g.dec,{radius!r}))"
    )
    parts = []
    targets = targets[np.isfinite(targets.ra) & np.isfinite(targets.dec)]
    for start in range(0, len(targets), batch):
        chunk = targets.iloc[start : start + batch]
        digest = hashlib.sha256(
            chunk[["key", "ra", "dec"]].to_numpy("f8").tobytes() + query.encode()
        ).hexdigest()[:16]
        path = cache / f"gaia-candidates-{start:09d}-{digest}.parquet"
        if not path.exists():
            upload = Table(
                dict(
                    key=chunk.key.to_numpy("i8"),
                    ra=chunk.ra.to_numpy("f8"),
                    dec=chunk.dec.to_numpy("f8"),
                )
            )
            result = _run_upload_job(service, query, upload, log=log, label=start)
            if len(result) >= 3_000_000:
                raise RuntimeError("Possibly truncated Gaia candidate batch")
            frame = pd.DataFrame(
                {
                    c: np.ma.filled(np.ma.asarray(result[c]), np.nan)
                    if result[c].dtype.kind == "f"
                    else np.asarray(result[c])
                    for c in result.colnames
                }
            )
            frame["key"] = frame.key.astype("i8")
            frame["source_id"] = frame.source_id.astype("i8")
            frame.to_parquet(path.with_suffix(".part"), index=False)
            path.with_suffix(".part").replace(path)
            log(f"gaia batch {start}: {len(chunk)} targets, {len(frame)} candidates")
        parts.append(pd.read_parquet(path))
    if not parts:
        return pd.DataFrame(
            columns=["key", "source_id", "ra", "dec", "pmra", "pmdec", "ref_epoch"]
        )
    return pd.concat(parts, ignore_index=True)


def resolve_positional(
    obs: pd.DataFrame,
    cache: Path,
    *,
    radius_arcsec: float,
    epochs: tuple[float, ...],
    log=print,
    **kwargs,
) -> pd.DataFrame:
    """Fill missing source_id by epoch-stable nearest Gaia DR3 match."""
    obs = obs.copy()
    need = (obs.source_id == 0) & np.isfinite(obs.ra) & np.isfinite(obs.dec)
    targets = obs.loc[need, ["ra", "dec"]].copy()
    targets["key"] = targets.index.to_numpy("i8")
    candidates = gaia_candidates(targets, cache, log=log, **kwargs)
    matches = choose_matches(
        candidates, targets, radius_arcsec=radius_arcsec, epochs=epochs
    )
    idx = matches.key.to_numpy("i8")
    obs.loc[idx, "source_id"] = matches.source_id.to_numpy("i8")
    obs.loc[idx, "match_sep_arcsec"] = matches.sep_arcsec.to_numpy("f8")
    obs.loc[idx, "match_n_within"] = matches.n_within.to_numpy("i8")
    obs.loc[idx, "id_method"] = np.where(
        matches.n_within.to_numpy() > 1, "positional_ambiguous", "positional"
    )
    obs.loc[need & (obs.source_id == 0), "id_method"] = "positional_unmatched"
    return obs


# --------------------------------------------------------------------------
# Census
# --------------------------------------------------------------------------


def _feh_bins(values: np.ndarray) -> dict[str, int]:
    return {f"lt_{edge:+.1f}": int((values < edge).sum()) for edge in FEH_EDGES}


def star_table(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per Gaia star with quality-passing, edge-free minimum [Fe/H]."""
    linked = obs[obs.source_id > 0]
    good = linked[linked.quality & ~linked.fe_h_grid_edge & np.isfinite(linked.fe_h)]
    stars = linked.groupby("source_id").agg(
        n_rows=("source", "size"),
        n_sources=("source", "nunique"),
        sources=("source", lambda s: ",".join(sorted(set(s)))),
    )
    for tier in TIERS:
        part = good[good.tier == tier].groupby("source_id").fe_h
        stars["fe_h_median_" + tier] = part.median()
    stars["fe_h_min_any"] = good.groupby("source_id").fe_h.min()
    stars["fe_h_median_any"] = good.groupby("source_id").fe_h.median()
    stars["edge_censored_only"] = (
        linked[linked.fe_h_grid_edge].groupby("source_id").size().gt(0)
        & stars.fe_h_median_any.isna()
    ).reindex(stars.index, fill_value=False)
    return stars.reset_index()


def census(obs: pd.DataFrame, existing_ids: dict[str, np.ndarray] | None = None):
    """Per-source and union counts; metal-poor bins use quality, edge-free rows."""
    result = dict(feh_edges=list(FEH_EDGES), sources={}, union={})
    union_ids = []
    for name, part in obs.groupby("source", sort=True):
        linked = part[part.source_id > 0]
        good = linked[linked.quality & np.isfinite(linked.fe_h)]
        good_free = good[~good.fe_h_grid_edge]
        per_star = good_free.groupby("source_id").fe_h.median().to_numpy()
        result["sources"][name] = dict(
            tier=str(part.tier.iloc[0]),
            dependency_group=str(part.dependency_group.iloc[0]),
            rows=int(len(part)),
            gaia_linked_rows=int(len(linked)),
            unique_gaia=int(linked.source_id.nunique()),
            id_methods={k: int(v) for k, v in part.id_method.value_counts().items()},
            quality_unique_gaia=int(good.source_id.nunique()),
            grid_edge_unique_gaia=int(good[good.fe_h_grid_edge].source_id.nunique()),
            fe_h_min=float(np.nanmin(per_star)) if len(per_star) else None,
            metal_poor_unique=_feh_bins(per_star),
        )
        union_ids.append(linked.source_id.to_numpy())
    stars = star_table(obs)
    all_ids = np.unique(np.concatenate(union_ids)) if union_ids else np.array([], "i8")
    result["union"] = dict(
        unique_gaia=int(len(all_ids)),
        multi_source=int((stars.n_sources > 1).sum()),
        metal_poor_unique_any_source_median=_feh_bins(
            stars.fe_h_median_any.dropna().to_numpy()
        ),
        metal_poor_unique_highres=_feh_bins(
            stars.fe_h_median_highres_reference.dropna().to_numpy()
        ),
        edge_censored_only=int(stars.edge_censored_only.sum()),
    )
    exclusive = {}
    for name, part in obs[obs.source_id > 0].groupby("source"):
        ids = np.unique(part.source_id.to_numpy())
        others = obs[(obs.source_id > 0) & (obs.source != name)].source_id.to_numpy()
        exclusive[name] = int((~np.isin(ids, others)).sum())
    result["union"]["exclusive_to_source"] = exclusive
    if existing_ids:
        existing = np.unique(np.concatenate(list(existing_ids.values())))
        new = ~np.isin(all_ids, existing)
        mp = stars.set_index("source_id").fe_h_median_any
        mp_new = mp[~np.isin(mp.index.to_numpy(), existing)].dropna().to_numpy()
        result["against_existing"] = dict(
            existing_sources={
                k: int(len(np.unique(v))) for k, v in existing_ids.items()
            },
            existing_unique=int(len(existing)),
            new_unique=int(new.sum()),
            combined_unique=int(len(existing) + new.sum()),
            new_metal_poor_unique=_feh_bins(mp_new),
        )
    return result, stars
