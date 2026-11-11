"""Build the consolidated master spectroscopic label table (all records, traceable).

Output directory:

* ``records/<survey>.parquet``: one row per native catalogue row, no quality
  cut (``fit_ok`` holds each survey's failed-fit flags), with
  ``catalogue``/``native_hdu``/``native_row``/``native_id`` for trace-back,
  Gaia DR3 identity (``source_id``, ``id_method``, ``match_sep_arcsec``),
  ``star_id`` (Gaia ID or negative no-Gaia id), star-level ``v2_role``, ``label_groups`` (training,
  loose and chemistry groups holding this star for this survey) and ``has_xp``.
* ``stars.parquet``: one row per star (Gaia or not) with survey coverage and
  robust summaries; ``star_key`` is gaia:<id> or nogaia:<survey>:<native_id>.
* ``catalogues.json``: native file, HDU, release, URL and sha256 per
  catalogue; ``read_native_rows`` in ``training/master_spectroscopic.py``
  returns the full native rows (all abundances and flags) for any records.
* ``manifest.json`` and ``README.md``.

Records without a publisher Gaia ID but with a position are matched to Gaia
DR3 through ESA archive upload jobs (epoch-stable nearest neighbour, as in the
census). Records left without Gaia identity are kept and grouped across
surveys by position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from masked_stellar_autoencoder.training import master_labels as ml  # noqa: E402
from masked_stellar_autoencoder.training import (  # noqa: E402
    master_spectroscopic as ms,
)
from masked_stellar_autoencoder.training.master_label_sidecars import (  # noqa: E402
    TRAINING_SOURCES,
)

K = "/arc/projects/k-pop/"
RUN = K + "msa_runs/stellar-improvement-20261015/"
CENSUS = RUN + "master_label_census_20261110_v3/"
WORK = RUN + "master_label_sources_20261110_v1/"
TRAINING_SOURCE = "expanded_six_source_20261111_v7"
LOOSE = RUN + "loose_survey_labels_20261111_v1/loose-survey-sidecar.h5"
CHEM = RUN + "chemistry_labels_20261111_v1/chemistry-sidecar.h5"

CATALOGUES = {
    "lamost_dr12_lrs": ms.Catalogue(
        "lamost_dr12_lrs",
        RUN + "multisurvey_lamost_dr12_lrs_source_direct_v2/dr12_lrs_stellar.fits.gz",
        "fits",
        release="LAMOST DR12 v2.0 LRS stellar (LASP)",
        native_id="obsid",
    ),
    "lamost_dr12_mrs": ms.Catalogue(
        "lamost_dr12_mrs",
        RUN + "multisurvey_lamost_dr12_mrs_source_direct_v2/dr12_mrs_stellar.fits.gz",
        "fits",
        release="LAMOST DR12 v2.0 MRS stellar (LASP; CNN abundances)",
        native_id="mobsid",
    ),
    "desi_dr1": ms.Catalogue(
        "desi_dr1",
        RUN + "desi_collection_v1/mwsall-pix-iron.fits",
        "fits",
        hdu="RVTAB",
        release="DESI DR1 MWS iron mwsall-pix",
        native_id="TARGETID",
        extra=dict(aligned_hdus=["GAIA", "FIBERMAP", "SPTAB"]),
        notes="RVTAB/GAIA/FIBERMAP/SPTAB are row-aligned; SPTAB holds FERRE abundances",
    ),
    "desi_dr1_sp": ms.Catalogue(
        "desi_dr1_sp",
        RUN + "desi_collection_v1/mwsall-pix-iron.fits",
        "fits",
        hdu="SPTAB",
        release="DESI DR1 MWS iron mwsall-pix (SP: FERRE)",
        native_id="TARGETID",
        extra=dict(aligned_hdus=["RVTAB", "GAIA", "FIBERMAP"]),
        notes="Same rows as desi_dr1 (RVTAB); SPTAB is the FERRE SP pipeline",
    ),
    "lamost_dr12_mrs_cnn": ms.Catalogue(
        "lamost_dr12_mrs_cnn",
        RUN + "multisurvey_lamost_dr12_mrs_source_direct_v2/dr12_mrs_stellar.fits.gz",
        "fits",
        release="LAMOST DR12 v2.0 MRS stellar (CNN parameters)",
        native_id="mobsid",
        notes="Same rows as lamost_dr12_mrs; *_cnn columns",
    ),
    "apogee_dr17": ms.Catalogue(
        "apogee_dr17",
        RUN + "apogee_native_fetch_v1/allStar-dr17-synspec_rev1.fits",
        "fits",
        release="APOGEE DR17 allStar synspec_rev1",
        native_id="APOGEE_ID:TELESCOPE:FIELD",
    ),
    "galah_dr4": ms.Catalogue(
        "galah_dr4",
        K + "catalogues/galah_dr4_allstar_240705.fits",
        "fits",
        release="GALAH DR4 allstar 240705",
        native_id="sobject_id",
    ),
    "gaia_dr3_gspspec": ms.Catalogue(
        "gaia_dr3_gspspec",
        K + "catalogues/gaia_rvs_lite_flags.fits.gz",
        "fits",
        release="Gaia DR3 GSP-Spec (astrophysical_parameters subset)",
        native_id="source_id",
    ),
    "sdss_dr20_bossnet": ms.Catalogue(
        "sdss_dr20_bossnet",
        RUN + "decision_dr20_legacy_source_audit_20261106_v1/dr20_bossnet.fits.gz",
        "fits",
        release="SDSS DR20 Astra 0.8.1 BOSSNet",
        native_id="sdss_id:spectrum_pk",
    ),
    "rave_dr6": ms.Catalogue(
        "rave_dr6",
        RUN + "rave_native_admission_20261105_v1/madera.csv.gz",
        "csv",
        release="RAVE DR6 MADERA (+ GAUGUIN [Fe/H], EDR3 crossmatch)",
        url="https://www.rave-survey.org/files/csv/ravedr6_dr6_madera.csv.gz",
        native_id="rave_obs_id",
        extra=dict(
            iron=RUN + "rave_native_admission_20261105_v1/iron.csv.gz",
            crossmatch=RUN + "rave_native_admission_20261105_v1/crossmatch.xml",
        ),
        notes="native_row indexes madera.csv.gz; GAUGUIN and crossmatch join on rave_obs_id",
    ),
}
# Staged survey -> {groups holding this survey's stars} in the training source
# (v7, run master_label_six_nsf_fullfit_20261111_v2), loose and chemistry sidecars.
GROUPS = {
    "lamost_dr12_lrs": dict(
        v2=["lamost_dr12v2_lrs_lasp"],
        loose=["lamost_dr12v2_lrs_lasp_ext"],
        chem=["lamost_dr12v2_lrs_alpha_m"],
    ),
    "lamost_dr12_mrs": dict(
        v2=["lamost_dr12v2_mrs_lasp"],
        loose=["lamost_dr12v2_mrs_lasp_ext"],
        chem=["lamost_dr12v2_mrs_alpha_m"],
    ),
    "desi_dr1": dict(
        v2=["desi"], loose=["desi_rvs_ext_corrected", "desi_rvs_ext_uncorrected"]
    ),
    "apogee_dr17": dict(v2=["apogee"], chem=["apogee_dr17_alpha_m"]),
    "galah_dr4": dict(v2=["galah"], chem=["galah_dr4_mg_fe"]),
    "gaia_dr3_gspspec": dict(v2=["gaia_dr3_gspspec"], chem=["gaia_dr3_gspspec_chem"]),
    "sdss_dr20_bossnet": dict(
        v2=["sdss_dr20_bossnet_081"], loose=["sdss_dr20_bossnet_081_ext"]
    ),
    "rave_dr6": dict(v2=["rave_dr6_madera_gauguin"]),
}
for spec in TRAINING_SOURCES:
    for member in spec.members:
        GROUPS.setdefault(member, {}).setdefault("v2", []).append(spec.name)
GROUPS["sdss_dr12_sspp"]["chem"] = ["sdss_dr12_sspp_alpha_fe"]
GROUPS["sdss_dr19_astra_aspcap"]["chem"] = ["sdss_dr19_aspcap_sdss5_alpha_m"]
# Positional Gaia match for records without publisher IDs: (radius", epochs).
POSITIONAL = {
    "lamost_dr12_lrs": (1.5, (2015.5,)),
    "lamost_dr12_mrs": (1.5, (2015.5,)),
    "lamost_dr12_mrs_cnn": (1.5, (2015.5,)),
    "apogee_dr17": (1.5, (2000.0,)),
    "galah_dr4": (1.5, (2000.0,)),
    "sdss_dr20_bossnet": (1.0, (2015.5,)),
}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fits_columns(path, columns, hdu=1):
    from astropy.io import fits

    with fits.open(path, memmap=True) as f:
        table = f[hdu].data
        names = {n.lower(): n for n in f[hdu].columns.names}
        return {c: np.asarray(table[names[c.lower()]]) for c in columns}, len(table)


def text(values):
    return np.char.strip(np.asarray(values).astype("U"))


def lamost(key, cols, alpha, snr):
    d, n = fits_columns(
        CATALOGUES[key].path,
        [cols[0], "gaia_source_id", "ra", "dec", *cols[1:], alpha, alpha + "_err", snr],
    )
    out = ms.frame(n, survey=key, tier="survey", catalogue=key, native_hdu="1")
    out["release"] = CATALOGUES[key].release
    out["pipeline"] = "LASP"
    out["dependency_group"] = key.replace("_dr12_", "-dr12v2-") + "-lasp"
    out["native_row"] = np.arange(n)
    out["native_id"] = text(d[cols[0]])
    sid = ms.exact_ids(d["gaia_source_id"])
    out["source_id"] = sid
    out["id_method"] = np.where(sid > 0, "publisher_dr3", "")
    out["ra"], out["dec"] = d["ra"], d["dec"]
    for target, (v, e) in zip(
        ("teff", "logg", "fe_h"), zip(cols[1::2], cols[2::2], strict=True), strict=True
    ):
        out[target], out["e_" + target] = d[v], d[e]
    out["fe_h_kind"] = "fe_h"
    out["alpha"], out["e_alpha"], out["alpha_kind"] = (
        d[alpha],
        d[alpha + "_err"],
        "alpha_m",
    )
    out["snr"] = d[snr]
    out["fit_ok"] = np.isfinite(out.teff) & (out.teff != -9999)
    return out


def desi():
    from astropy.io import fits
    from desi_stellar_labels import corrected_feh, separation_arcsec

    c = CATALOGUES["desi_dr1"]
    with fits.open(c.path, memmap=True) as f:
        r, g = f["RVTAB"].data, f["GAIA"].data
        n = len(r)
        col = lambda t, k: np.asarray(t[k])  # noqa: E731
        out = ms.frame(n, survey="desi_dr1", tier="survey", catalogue="desi_dr1")
        out["native_hdu"] = "RVTAB"
        out["release"], out["pipeline"] = c.release, "RVSpecFit (RVTAB)"
        out["dependency_group"] = "desi-dr1-rvspecfit"
        out["native_row"] = np.arange(n)
        out["native_id"] = col(r, "TARGETID").astype("i8").astype(str)
        sid = ms.exact_ids(col(g, "SOURCE_ID"))
        out["source_id"] = sid
        out["id_method"] = np.where(sid > 0, "publisher_dr3", "")
        out["match_sep_arcsec"] = separation_arcsec(
            col(r, "TARGET_RA"), col(r, "TARGET_DEC"), col(g, "RA"), col(g, "DEC")
        )
        out["ra"], out["dec"] = col(r, "TARGET_RA"), col(r, "TARGET_DEC")
        for t, k in (("teff", "TEFF"), ("logg", "LOGG"), ("fe_h", "FEH")):
            out[t], out["e_" + t] = col(r, k), col(r, k + "_ERR")
        out["fe_h_kind"] = "fe_h"
        fixed, valid, _ = corrected_feh(col(r, "FEH"), col(r, "TEFF"), col(r, "LOGG"))
        out["fe_h_corrected"] = np.where(valid, fixed, np.nan)
        out["alpha"], out["e_alpha"] = col(r, "ALPHAFE"), col(r, "ALPHAFE_ERR")
        out["alpha_kind"] = "alpha_fe"
        out["snr"] = col(r, "SN_R")
        spectype = text(col(r, "RR_SPECTYPE"))
        out["flag_int"] = col(r, "RVS_WARN").astype("i8")
        out["flag_text"] = spectype
        out["fit_ok"] = (
            col(r, "SUCCESS").astype(bool)
            & col(r, "PRIMARY").astype(bool)
            & (col(r, "RVS_WARN") == 0)
            & (spectype == "STAR")
        )
    return out


def apogee():
    c = CATALOGUES["apogee_dr17"]
    cols = ["APOGEE_ID", "TELESCOPE", "FIELD", "GAIAEDR3_SOURCE_ID", "RA", "DEC"]
    cols += ["TEFF", "TEFF_ERR", "LOGG", "LOGG_ERR", "FE_H", "FE_H_ERR"]
    cols += ["ALPHA_M", "ALPHA_M_ERR", "SNR", "ASPCAPFLAG"]
    d, n = fits_columns(c.path, cols)
    out = ms.frame(n, survey="apogee_dr17", tier="survey", catalogue="apogee_dr17")
    out["native_hdu"], out["release"], out["pipeline"] = "1", c.release, "ASPCAP"
    out["dependency_group"] = "apogee-dr17-aspcap"
    out["native_row"] = np.arange(n)
    out["native_id"] = [
        f"{a}:{t}:{f}"
        for a, t, f in zip(
            text(d["APOGEE_ID"]), text(d["TELESCOPE"]), text(d["FIELD"]), strict=True
        )
    ]
    sid = ms.exact_ids(d["GAIAEDR3_SOURCE_ID"])
    out["source_id"] = sid
    out["id_method"] = np.where(sid > 0, "publisher_edr3", "")
    out["ra"], out["dec"] = d["RA"], d["DEC"]
    for t, k in (("teff", "TEFF"), ("logg", "LOGG"), ("fe_h", "FE_H")):
        out[t], out["e_" + t] = d[k], d[k + "_ERR"]
    out["fe_h_kind"] = "fe_h"
    out["alpha"], out["e_alpha"], out["alpha_kind"] = (
        d["ALPHA_M"],
        d["ALPHA_M_ERR"],
        "alpha_m",
    )
    out["snr"] = d["SNR"]
    flag = d["ASPCAPFLAG"].astype("i8")
    out["flag_int"] = flag
    out["fit_ok"] = (flag & ms.STAR_BAD) == 0
    return out


def galah():
    c = CATALOGUES["galah_dr4"]
    cols = ["sobject_id", "gaiadr3_source_id", "ra", "dec", "teff", "e_teff"]
    cols += ["logg", "e_logg", "fe_h", "e_fe_h", "mg_fe", "e_mg_fe"]
    cols += ["snr_px_ccd3", "flag_sp", "flag_sp_fit", "flag_fe_h"]
    d, n = fits_columns(c.path, cols)
    out = ms.frame(n, survey="galah_dr4", tier="survey", catalogue="galah_dr4")
    out["native_hdu"], out["release"], out["pipeline"] = "1", c.release, "GALAH DR4"
    out["dependency_group"] = "galah-dr4"
    out["native_row"] = np.arange(n)
    out["native_id"] = d["sobject_id"].astype("i8").astype(str)
    sid = ms.exact_ids(d["gaiadr3_source_id"])
    out["source_id"] = sid
    out["id_method"] = np.where(sid > 0, "publisher_dr3", "")
    out["ra"], out["dec"] = d["ra"], d["dec"]
    for t in ("teff", "logg", "fe_h"):
        out[t], out["e_" + t] = d[t], d["e_" + t]
    out["fe_h_kind"] = "fe_h"
    out["alpha"], out["e_alpha"], out["alpha_kind"] = d["mg_fe"], d["e_mg_fe"], "mg_fe"
    out["snr"] = d["snr_px_ccd3"]
    out["flag_int"] = d["flag_sp"].astype("i8")
    out["flag_text"] = [
        f"flag_sp_fit={a};flag_fe_h={b}"
        for a, b in zip(d["flag_sp_fit"], d["flag_fe_h"], strict=True)
    ]
    out["fit_ok"] = (d["flag_sp"] == 0) & (d["flag_sp_fit"] == 0)
    return out


def gspspec():
    c = CATALOGUES["gaia_dr3_gspspec"]
    q = ["teff_gspspec", "logg_gspspec", "mh_gspspec", "alphafe_gspspec"]
    cols = ["source_id", "flags_gspspec"]
    cols += [x + s for x in q for s in ("", "_lower", "_upper")]
    d, n = fits_columns(c.path, cols)
    out = ms.frame(
        n, survey="gaia_dr3_gspspec", tier="survey", catalogue="gaia_dr3_gspspec"
    )
    out["native_hdu"], out["release"], out["pipeline"] = (
        "1",
        c.release,
        "GSP-Spec MatisseGauguin",
    )
    out["dependency_group"] = "gaia-dr3-gspspec"
    out["native_row"] = np.arange(n)
    sid = ms.exact_ids(d["source_id"])
    out["native_id"], out["source_id"], out["id_method"] = (
        sid.astype(str),
        sid,
        "gaia_dr3",
    )
    half = lambda k: 0.5 * (d[k + "_upper"].astype("f8") - d[k + "_lower"].astype("f8"))  # noqa: E731
    for t, k in zip(("teff", "logg", "fe_h", "alpha"), q, strict=True):
        out[t], out["e_" + t] = d[k], half(k)
    out["fe_h_kind"], out["alpha_kind"] = "m_h", "alpha_fe"
    flags = text(d["flags_gspspec"])
    out["flag_text"] = flags
    out["fit_ok"] = np.isfinite(out.teff.to_numpy("f8"))
    return out


def bossnet():
    c = CATALOGUES["sdss_dr20_bossnet"]
    cols = ["sdss_id", "spectrum_pk", "gaia_dr3_source_id", "ra", "dec"]
    cols += ["teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h", "snr"]
    cols += ["result_flags", "flag_bad"]
    d, n = fits_columns(c.path, cols)
    out = ms.frame(
        n, survey="sdss_dr20_bossnet", tier="survey", catalogue="sdss_dr20_bossnet"
    )
    out["native_hdu"], out["release"], out["pipeline"] = "1", c.release, "BOSSNet"
    out["dependency_group"] = "sdss-dr20-bossnet-081"
    out["native_row"] = np.arange(n)
    out["native_id"] = [
        f"{int(a)}:{int(b)}"
        for a, b in zip(d["sdss_id"], d["spectrum_pk"], strict=True)
    ]
    sid = ms.exact_ids(d["gaia_dr3_source_id"])
    out["source_id"] = sid
    out["id_method"] = np.where(sid > 0, "publisher_dr3", "")
    out["ra"], out["dec"] = d["ra"], d["dec"]
    for t in ("teff", "logg", "fe_h"):
        out[t], out["e_" + t] = d[t], d["e_" + t]
    out["fe_h_kind"] = "fe_h"
    out["snr"] = d["snr"]
    out["flag_int"] = d["result_flags"].astype("i8")
    out["fit_ok"] = ~d["flag_bad"].astype(bool)
    return out


def rave():
    from astropy.table import Table

    c = CATALOGUES["rave_dr6"]
    madera = pd.read_csv(c.path)
    iron = pd.read_csv(c.extra["iron"])
    xm = Table.read(c.extra["crossmatch"], format="votable").to_pandas()
    a = madera.merge(
        iron[["rave_obs_id", "fe_h_gauguin", "fe_h_error_gauguin", "fe_h_nl_gauguin"]],
        on="rave_obs_id",
        how="left",
    ).merge(xm[["rave_obs_id", "source_id", "dist"]], on="rave_obs_id", how="left")
    if len(a) != len(madera):
        raise ValueError("RAVE joins must keep one row per MADERA row")
    n = len(a)
    out = ms.frame(n, survey="rave_dr6", tier="survey", catalogue="rave_dr6")
    out["native_hdu"], out["release"] = "csv", c.release
    out["pipeline"] = "MADERA (Teff, logg) + GAUGUIN ([Fe/H])"
    out["dependency_group"] = "rave-dr6-shared-spectra-madera-gauguin"
    out["native_row"] = np.arange(n)
    out["native_id"] = a.rave_obs_id.astype(str).to_numpy()
    sid = a.source_id.fillna(0).astype("i8").to_numpy()
    out["source_id"] = sid
    out["id_method"] = np.where(sid > 0, "publisher_edr3", "")
    out["match_sep_arcsec"] = a.dist.to_numpy("f8")
    out["teff"], out["e_teff"] = a.teff_cal_madera, a.teff_error_madera
    out["logg"], out["e_logg"] = a.logg_cal_madera, a.logg_error_madera
    out["fe_h"], out["e_fe_h"] = a.fe_h_gauguin, a.fe_h_error_gauguin
    out["fe_h_kind"] = "fe_h"
    out["snr"] = a.snr_madera
    out["flag_int"] = a.algo_conv_madera.fillna(-1).astype("i8")
    out["fit_ok"] = (a.algo_conv_madera == 0).to_numpy()
    return out


def desi_sp():
    """DESI SP (FERRE) parameters from SPTAB, row-aligned with RVTAB/GAIA."""
    from astropy.io import fits

    c = CATALOGUES["desi_dr1_sp"]
    with fits.open(c.path, memmap=True) as f:
        sp, r, g = f["SPTAB"].data, f["RVTAB"].data, f["GAIA"].data
        names = set(sp.columns.names)
        if not {"TEFF", "LOGG", "FEH"} <= names:
            raise ValueError(f"SPTAB lacks TEFF/LOGG/FEH: {sorted(names)}")
        n = len(sp)
        if n != len(r) or not np.array_equal(
            np.asarray(sp["TARGETID"])
            if "TARGETID" in names
            else np.asarray(r["TARGETID"]),
            np.asarray(r["TARGETID"]),
        ):
            raise ValueError("SPTAB is not row-aligned with RVTAB")
        col = lambda t, k: np.asarray(t[k])  # noqa: E731
        opt = lambda k: col(sp, k) if k in names else np.full(n, np.nan)  # noqa: E731
        out = ms.frame(n, survey="desi_dr1_sp", tier="survey", catalogue="desi_dr1_sp")
        out["native_hdu"], out["release"] = "SPTAB", c.release
        out["pipeline"] = "FERRE (SPTAB)"
        out["dependency_group"] = "desi-dr1-rvspecfit"
        out["native_row"] = np.arange(n)
        out["native_id"] = col(r, "TARGETID").astype("i8").astype(str)
        sid = ms.exact_ids(col(g, "SOURCE_ID"))
        out["source_id"] = sid
        out["id_method"] = np.where(sid > 0, "publisher_dr3", "")
        out["ra"], out["dec"] = col(r, "TARGET_RA"), col(r, "TARGET_DEC")
        for t, k in (("teff", "TEFF"), ("logg", "LOGG"), ("fe_h", "FEH")):
            out[t], out["e_" + t] = opt(k), opt(k + "_ERR")
        out["fe_h_kind"] = "fe_h"
        out["alpha"], out["e_alpha"] = opt("ALPHAFE"), opt("ALPHAFE_ERR")
        out["alpha_kind"] = "alpha_fe"
        out["snr"] = col(r, "SN_R")
        ok = np.isfinite(out.teff.to_numpy("f8"))
        if "SUCCESS" in names:
            ok &= col(sp, "SUCCESS").astype(bool)
        out["fit_ok"] = ok
    return out


def flag_desi_sp_edges(records):
    """DESI SP (FERRE) grid-edge and white-dwarf-grid fits are not valid FGKM fits.

    In DR1 SPTAB, failed or edge solutions pile up at [Fe/H] = -10 (white-dwarf
    grid, logg ~ 9), at the grid floor (-5.0/-4.99) and at -3.5 with logg ~ 6.
    They are kept as records but flagged and excluded from ``fit_ok``.
    """
    sp = (records.survey == "desi_dr1_sp").to_numpy()
    if not sp.any():
        return records
    feh = records.fe_h.to_numpy("f8")
    logg = records.logg.to_numpy("f8")
    wd = sp & (feh <= -9.9)
    floor = sp & ~wd & (feh <= -4.95)
    pile = sp & np.isclose(feh, -3.5, atol=0.005)
    high = sp & (logg >= 5.95)
    text = np.full(len(records), "", dtype=object)
    for mask, label in (
        (wd, "wd_grid"),
        (floor | pile, "fe_h_grid_edge"),
        (high, "logg_grid_edge"),
    ):
        text[mask] = [f"{t};{label}" if t else label for t in text[mask]]
    edge = wd | floor | pile | high
    records.loc[edge, "flag_text"] = text[edge]
    records.loc[edge, "fit_ok"] = False
    return records


def lamost_mrs_cnn():
    c = CATALOGUES["lamost_dr12_mrs_cnn"]
    cols = ["mobsid", "gaia_source_id", "ra", "dec", "snr"]
    cols += ["teff_cnn", "logg_cnn", "feh_cnn", "alpha_m_cnn", "m_h_cnn"]
    d, n = fits_columns(c.path, cols)
    out = ms.frame(n, survey="lamost_dr12_mrs_cnn", tier="survey")
    out["catalogue"], out["native_hdu"] = "lamost_dr12_mrs_cnn", "1"
    out["release"], out["pipeline"] = c.release, "LAMOST MRS CNN"
    out["dependency_group"] = "lamost-dr12v2-mrs-lasp"
    out["native_row"] = np.arange(n)
    out["native_id"] = text(d["mobsid"])
    sid = ms.exact_ids(d["gaia_source_id"])
    out["source_id"] = sid
    out["id_method"] = np.where(sid > 0, "publisher_dr3", "")
    out["ra"], out["dec"] = d["ra"], d["dec"]
    out["teff"], out["logg"], out["fe_h"] = d["teff_cnn"], d["logg_cnn"], d["feh_cnn"]
    out["fe_h_kind"] = "fe_h"
    out["alpha"], out["alpha_kind"] = d["alpha_m_cnn"], "alpha_m"
    out["snr"] = d["snr"]
    teff = np.asarray(d["teff_cnn"], "f8")
    out["fit_ok"] = np.isfinite(teff) & (teff != -9999)
    return out


STAGED = {
    "lamost_dr12_lrs": lambda: lamost(
        "lamost_dr12_lrs",
        ["obsid", "teff", "teff_err", "logg", "logg_err", "feh", "feh_err"],
        "alpha_m",
        "snrg",
    ),
    "lamost_dr12_mrs": lambda: lamost(
        "lamost_dr12_mrs",
        ["mobsid", "teff_lasp", "teff_lasp_err", "logg_lasp", "logg_lasp_err"]
        + ["feh_lasp", "feh_lasp_err"],
        "alpha_m_lasp",
        "snr",
    ),
    "desi_dr1": desi,
    "apogee_dr17": apogee,
    "galah_dr4": galah,
    "gaia_dr3_gspspec": gspspec,
    "sdss_dr20_bossnet": bossnet,
    "rave_dr6": rave,
    "desi_dr1_sp": desi_sp,
    "lamost_dr12_mrs_cnn": lamost_mrs_cnn,
}


# Census sources read from FITS: HDU ("auto" = first non-empty table).
FITS_CENSUS = {
    "sdss_dr19_astra_aspcap": "auto",
    "sdss_dr12_sspp": "1",
    "sdss_dr17_mastar_params": "1",
    **{s["source"]: "auto" for s in ml.ASTRA_SUMMARIES},
}


def census_records(path):
    obs = pd.read_parquet(path)
    acq = json.loads((Path(path).parent / "manifest.json").read_text())["acquisitions"]
    out = obs.rename(columns=dict(source="survey", quality="fit_ok"))
    out["catalogue"] = out.survey
    out["native_hdu"] = ""
    out["flag_text"] = np.where(obs.fe_h_grid_edge, "fe_h_grid_edge", "")
    out = out.assign(fe_h_corrected=np.nan, flag_int=0)
    catalogues = {}
    for name in out.survey.unique():
        a = acq.get(name, {})
        if name in FITS_CENSUS:
            hdu = FITS_CENSUS[name]
            cat = ms.Catalogue(name, a["path"], "fits", hdu=hdu, url=a.get("url", ""))
            out.loc[out.survey == name, "native_hdu"] = hdu
        elif name == "saga_mw":
            cat = ms.Catalogue(
                name,
                WORK + "saga",
                "census",
                url=a.get("url", ""),
                extra=dict(cache=WORK + "saga", query=a.get("query", "")),
            )
        elif name == "jinabase_2021":
            cat = ms.Catalogue(
                name,
                WORK + "jinabase/JINAbase_2021.csv",
                "census",
                url=a.get("url", ""),
                extra=dict(cache=WORK + "jinabase", commit=a.get("commit", "")),
            )
        else:
            cat = ms.Catalogue(
                name,
                WORK + f"vizier/{name}.tsv",
                "vizier_tsv",
                url=a.get("url", ""),
                extra=dict(cache=WORK + "vizier", vizier_table=a.get("table", "")),
            )
        cat.release = str(out.loc[out.survey == name, "release"].iloc[0])
        cat.native_id = "census native_id (see master_labels adapters)"
        catalogues[name] = cat
    return ms.finalize(out), catalogues


def group_ids(path, prefix, names):
    out = {}
    with h5py.File(path) as f:
        for name in names:
            g = f["native_observations/" + name]
            if "source_id" in g:
                obs = g["observed"][:].astype(bool).any(1)
                if "chem_observed" in g:
                    obs |= g["chem_observed"][:].astype(bool).any(1)
                out[f"{prefix}:{name}"] = np.unique(g["source_id"][:][obs].astype("i8"))
    return out


def training_groups(source_dir):
    """v2 training source: per-group star IDs, roles, XP availability."""
    pm = json.loads((Path(source_dir) / "manifest.json").read_text())
    with h5py.File(pm["prepared_path"]) as f:
        ids = f["source_id"][:].astype("i8")
        roles = f["role"][:].astype("U16")
        has_xp = f["input_has_xp"][:].astype(bool)
        groups = {
            "v2:" + name: np.unique(
                ids[f["native_observations/" + name]["star_row"][:]]
            )
            for name in f["native_observations"]
        }
    return ids, roles, has_xp, groups


def label_groups(records, survey, groups):
    sid = records.source_id.to_numpy("i8")
    names = []
    for kind, members in GROUPS.get(survey, {}).items():
        for name in members:
            ids = groups.get(f"{kind}:{name}")
            if ids is not None:
                names.append((f"{kind}:{name}", np.isin(sid, ids) & (sid > 0)))
    out = np.full(len(records), "", dtype=object)
    for label, hit in names:
        out[hit] = [x + ";" + label if x else label for x in out[hit]]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--census", default=CENSUS + "observations.parquet")
    p.add_argument("--training-source", default=TRAINING_SOURCE)
    p.add_argument("--gaia-batch", type=int, default=50000)
    p.add_argument("--only", nargs="*", help="staged surveys to build (smoke tests)")
    p.add_argument("--skip-positional", action="store_true")
    p.add_argument(
        "--reuse-stage",
        type=Path,
        help="earlier master run's work/stage to copy unchanged survey parts from",
    )
    a = p.parse_args()
    out = a.output
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "work").mkdir(exist_ok=True)
    m = dict(status="building_master_labels", training_performed=False, surveys={})
    mp = out / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text()) | dict(status="building_master_labels")

    def save():
        (out / "manifest.tmp").write_text(json.dumps(m, indent=2, default=str) + "\n")
        (out / "manifest.tmp").replace(mp)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    save()
    try:
        catalogues = dict(CATALOGUES)
        stage = out / "work" / "stage"
        stage.mkdir(exist_ok=True)
        if a.reuse_stage:
            import shutil

            for old in sorted(Path(a.reuse_stage).glob("*.parquet")):
                if old.stem in STAGED and not (stage / old.name).exists():
                    shutil.copyfile(old, stage / old.name)
                    m.setdefault("reused_stage", []).append(str(old))
        # 1. Native records per survey (resumable), with positional Gaia identity.
        for name, build in STAGED.items():
            if a.only and name not in a.only:
                continue
            part = stage / f"{name}.parquet"
            if part.exists():
                continue
            log(f"reading {name}")
            try:
                rec = ms.finalize(build())
            except Exception as error:  # one unreadable survey must not stop the rest
                m.setdefault("errors", {})[name] = f"{type(error).__name__}: {error}"
                log(f"FAILED {name}: {error}")
                save()
                continue
            need = (rec.source_id == 0).sum()
            if name in POSITIONAL and need and not a.skip_positional:
                radius, epochs = POSITIONAL[name]
                log(f"positional Gaia match {name}: {need} records")
                rec = ml.resolve_positional(
                    rec,
                    out / "work" / "gaia" / name,
                    radius_arcsec=radius,
                    epochs=epochs,
                    batch=a.gaia_batch,
                    search_arcsec=5.0,
                    log=log,
                )
                rec = ms.finalize(rec)
            rec.to_parquet(part.with_suffix(".tmp"), index=False)
            part.with_suffix(".tmp").replace(part)
            m["surveys"][name] = dict(
                records=len(rec),
                with_gaia=int((rec.source_id > 0).sum()),
                id_methods=rec.id_method.value_counts().to_dict(),
                fit_ok=int(rec.fit_ok.sum()),
            )
            save()
        census, census_cats = census_records(a.census)
        catalogues.update(census_cats)
        # 2. Star identity across all records (Gaia, else positional groups).
        frames = [pd.read_parquet(p) for p in sorted(stage.glob("*.parquet"))]
        frames.append(census)
        records = pd.concat(frames, ignore_index=True)
        del frames
        records = flag_desi_sp_edges(records)
        blank = (records.id_method == "") & (records.source_id > 0)
        records.loc[blank, "id_method"] = "publisher_dr3"  # LAMOST parts from v1
        log(f"{len(records)} records; assigning star ids")
        records["star_id"] = ms.assign_star_ids(records)
        # 3. Training/loose/chemistry membership, roles and XP availability.
        ids, roles, has_xp, groups = training_groups(RUN + a.training_source)
        groups |= group_ids(
            LOOSE, "loose", [g for v in GROUPS.values() for g in v.get("loose", [])]
        )
        groups |= group_ids(
            CHEM, "chem", [g for v in GROUPS.values() for g in v.get("chem", [])]
        )
        pos = np.searchsorted(ids, records.source_id.to_numpy("i8"))
        pos = pos.clip(max=len(ids) - 1)
        hit = (ids[pos] == records.source_id.to_numpy("i8")) & (records.source_id > 0)
        records["v2_role"] = np.where(hit, roles[pos], "")
        records["has_xp"] = pd.array(np.where(hit, has_xp[pos], False), dtype="boolean")
        records.loc[~hit, "has_xp"] = pd.NA
        records["has_xp_source"] = np.where(hit, "v2_training_inputs", "")
        records["label_groups"] = ""
        for name, idx in records.groupby("survey").indices.items():
            records.loc[records.index[idx], "label_groups"] = label_groups(
                records.iloc[idx], name, groups
            )
        # 4. Outputs.
        for name, block in records.groupby("survey"):
            block.to_parquet(out / "records" / f"{name}.parquet", index=False)
        stars = ms.star_table(records)
        stars.to_parquet(out / "stars.parquet", index=False)
        ms.save_catalogues(catalogues, out / "catalogues.json", sha=sha)
        g = records.source_id > 0
        m.update(
            status="complete",
            records=len(records),
            stars=len(stars),
            gaia_stars=int((stars.source_id > 0).sum()),
            nogaia_stars=int((stars.source_id <= 0).sum()),
            records_without_gaia=int((~g).sum()),
            per_survey={
                k: dict(
                    records=int(len(v)),
                    with_gaia=int((v.source_id > 0).sum()),
                    fit_ok=int(v.fit_ok.sum()),
                    stars=int(v.star_id.nunique()),
                )
                for k, v in records.groupby("survey")
            },
            stars_in_v2_training=int((stars.v2_role != "").sum()),
            stars_with_xp_known=int(stars.has_xp.notna().sum()),
            metal_poor_stars={
                f"lt_{e}": int((stars.fe_h_median < e).sum()) for e in ml.FEH_EDGES
            },
            metal_poor_stars_highres={
                f"lt_{e}": int((stars.fe_h_median_highres < e).sum())
                for e in ml.FEH_EDGES
            },
            metal_poor_note=(
                "fe_h_median is over all fit_ok [Fe/H] records, including low-S/N "
                "survey pipelines; fe_h_median_highres uses highres_reference only"
            ),
            outputs={
                p.name: sha(p) for p in [out / "stars.parquet", out / "catalogues.json"]
            },
        )
        save()
        log("complete")
    except Exception as error:
        m.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
