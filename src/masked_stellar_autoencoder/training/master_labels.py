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
        for base, attempt in [(m, k) for m in mirrors for k in range(3)]:
            try:
                if attempt:
                    time.sleep(5 * attempt)
                with urllib.request.urlopen(base + "?" + query, timeout=timeout) as r:
                    payload = r.read()
                if b"#Column" not in payload:
                    raise ValueError("No table schema in response")
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
    return frame, json.loads(receipt.read_text())


_DESIGNATION = re.compile(
    r"J?(\d{2})(\d{2})(\d{2}(?:\.\d+)?)([+-])(\d{2})(\d{2})(\d{2}(?:\.\d+)?)"
)


def position_from_designation(names) -> tuple[np.ndarray, np.ndarray]:
    """RA/Dec (deg) from IAU-style ``JHHMMSS.ss+DDMMSS.s`` designations."""
    ra = np.full(len(names), np.nan)
    dec = np.full(len(names), np.nan)
    for i, name in enumerate(names):
        m = _DESIGNATION.search(str(name).replace(" ", ""))
        if not m:
            continue
        h, mi, se, sign, d, dm, ds = m.groups()
        ra[i] = 15 * (int(h) + int(mi) / 60 + float(se) / 3600)
        dec[i] = (1 if sign == "+" else -1) * (int(d) + int(dm) / 60 + float(ds) / 3600)
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
