"""Metallicity/alpha label groups with declared quantities ([M/H], [alpha/M], [Mg/Fe]).

Each group carries its native metallicity-like and alpha-like values in
``chem_value/chem_error/chem_observed`` [N, 2] arrays with the registry key
``chem_quantities`` ({"fe_h": "fe_h"|"m_h", "alpha": "alpha_fe"|"alpha_m"|
"mg_fe"}). Teff/logg/[Fe/H] are unmapped in these groups, so the sidecar guard
against relabelling [M/H] as [Fe/H] still holds. The trainer converts the
common ([Fe/H], [alpha/Fe]) to each declared quantity before the survey map
(``training/quantity_kinds.py``).

Groups (failed-fit cuts only, one row per star):
* gaia_dr3_gspspec_chem: mh_gspspec [M/H], alphafe_gspspec [alpha/Fe];
  error = half the 16-84 interval; GSP-Spec extrapolation flag <= 1.
* apogee_dr17_alpha_m: ALPHA_M; ASPCAPFLAG STAR_BAD unset; best S/N per star.
* galah_dr4_mg_fe: mg_fe with flag_mg_fe == 0.
* lamost_dr12v2_lrs_alpha_m / lamost_dr12v2_mrs_alpha_m: LASP [alpha/M], S/N >= 10.
* master-label sources (from the census observations): SSPP [alpha/Fe],
  DR19 ASPCAP SDSS-V [alpha/M], Best & Brightest [alpha/Fe].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_loose_survey_sidecars import exact_ids, one_per_star  # noqa: E402

from masked_stellar_autoencoder.training.multisurvey_overlay import (  # noqa: E402
    write_observation_sidecar,
)

ROOT = "/arc/projects/k-pop/"
RUN = ROOT + "msa_runs/stellar-improvement-20261015/"
STAR_BAD = 1 << 23
GSP_EXTRAPOL_INDEX = 9  # 10th character of flags_gspspec (Recio-Blanco+2023)


def chem_block(ids, metal, e_metal, alpha, e_alpha, snr, rows):
    value = np.column_stack([metal, alpha]).astype("f8")
    error = np.column_stack([e_metal, e_alpha]).astype("f8")
    observed = np.isfinite(value) & (value != -9999) & np.isfinite(error) & (error > 0)
    observed[:, 0] &= (value[:, 0] > -5) & (value[:, 0] < 1)
    observed[:, 1] &= (value[:, 1] > -1) & (value[:, 1] < 1.5)
    keep = (ids > 0) & observed.any(1)
    return dict(
        source_id=ids[keep].astype("i8"),
        source_row=rows[keep].astype("i8"),
        chem_value=np.where(observed, value, np.nan)[keep],
        chem_error=np.where(observed, error, np.nan)[keep],
        chem_observed=observed[keep],
        snr=np.nan_to_num(np.asarray(snr, "f8"), nan=-1.0)[keep],
    )


def read_fits(path, columns):
    with fits.open(path, memmap=True) as f:
        table = f[1].data
        names = {n.lower(): n for n in f[1].columns.names}
        return {c: np.asarray(table[names[c.lower()]]) for c in columns}, len(table)


def gspspec(path):
    d, n = read_fits(
        path,
        [
            "source_id",
            "mh_gspspec",
            "mh_gspspec_lower",
            "mh_gspspec_upper",
            "alphafe_gspspec",
            "alphafe_gspspec_lower",
            "alphafe_gspspec_upper",
            "flags_gspspec",
        ],
    )
    flags = np.char.ljust(np.asarray(d["flags_gspspec"]).astype("U41"), 41, "9")
    extrapol = np.array([f[GSP_EXTRAPOL_INDEX] for f in flags])
    ok = np.isin(extrapol, ["0", "1"])
    half = lambda lo, hi: 0.5 * (np.asarray(hi, "f8") - np.asarray(lo, "f8"))  # noqa: E731
    metal = np.where(ok, d["mh_gspspec"], np.nan)
    alpha = np.where(ok, d["alphafe_gspspec"], np.nan)
    return chem_block(
        exact_ids(d["source_id"]),
        metal,
        half(d["mh_gspspec_lower"], d["mh_gspspec_upper"]),
        alpha,
        half(d["alphafe_gspspec_lower"], d["alphafe_gspspec_upper"]),
        np.zeros(n),
        np.arange(n),
    ), n


def apogee(path):
    d, n = read_fits(
        path, ["GAIAEDR3_SOURCE_ID", "ALPHA_M", "ALPHA_M_ERR", "ASPCAPFLAG", "SNR"]
    )
    good = (np.asarray(d["ASPCAPFLAG"]).astype("i8") & STAR_BAD) == 0
    alpha = np.where(good, d["ALPHA_M"], np.nan)
    nan = np.full(n, np.nan)
    return chem_block(
        exact_ids(d["GAIAEDR3_SOURCE_ID"]),
        nan,
        nan,
        alpha,
        d["ALPHA_M_ERR"],
        d["SNR"],
        np.arange(n),
    ), n


def galah(path):
    d, n = read_fits(path, ["gaiadr3_source_id", "mg_fe", "e_mg_fe", "flag_mg_fe"])
    alpha = np.where(np.asarray(d["flag_mg_fe"]) == 0, d["mg_fe"], np.nan)
    nan = np.full(n, np.nan)
    return chem_block(
        exact_ids(d["gaiadr3_source_id"]),
        nan,
        nan,
        alpha,
        d["e_mg_fe"],
        np.zeros(n),
        np.arange(n),
    ), n


def lamost(path, alpha_col, err_col, snr_col, snr_min):
    d, n = read_fits(path, ["gaia_source_id", alpha_col, err_col, snr_col])
    snr = np.asarray(d[snr_col], "f8")
    alpha = np.where(np.isfinite(snr) & (snr >= snr_min), d[alpha_col], np.nan)
    nan = np.full(n, np.nan)
    return chem_block(
        exact_ids(d["gaia_source_id"]), nan, nan, alpha, d[err_col], snr, np.arange(n)
    ), n


def master(obs, source):
    o = obs[(obs.source == source) & (obs.source_id > 0) & obs.quality].reset_index(
        drop=True
    )
    if source == "sdss_dr19_astra_aspcap":
        release = o.native_id.str.split(":").str[2]
        dr17 = o.source_id[release == "dr17"].unique()
        o = o[(release != "dr17") & ~o.source_id.isin(dr17)].reset_index(drop=True)
    # [M/H]-type metallicity (AMBRE) goes into the metallicity slot as m_h.
    m_h = (o.fe_h_kind == "m_h").to_numpy()
    metal = np.where(m_h, o.fe_h.to_numpy("f8"), np.nan)
    e_metal = np.where(m_h, o.e_fe_h.to_numpy("f8"), np.nan)
    return chem_block(
        o.source_id.to_numpy("i8"),
        metal,
        e_metal,
        o.alpha.to_numpy("f8"),
        o.e_alpha.to_numpy("f8"),
        o.snr.to_numpy("f8"),
        np.arange(len(o)),
    ), len(o)


GROUPS = {
    "gaia_dr3_gspspec_chem": dict(
        quantities={"fe_h": "m_h", "alpha": "alpha_fe"},
        release="Gaia DR3 GSP-Spec",
        pipeline="GSP-Spec MatisseGauguin",
        dependency="gaia-dr3-gspspec",
        fields=("mh_gspspec", "alphafe_gspspec"),
    ),
    "apogee_dr17_alpha_m": dict(
        quantities={"fe_h": "fe_h", "alpha": "alpha_m"},
        release="APOGEE DR17 synspec_rev1",
        pipeline="ASPCAP",
        dependency="apogee-dr17-aspcap",
        fields=(None, "ALPHA_M"),
    ),
    "galah_dr4_mg_fe": dict(
        quantities={"fe_h": "fe_h", "alpha": "mg_fe"},
        release="GALAH DR4 240705",
        pipeline="GALAH DR4 allstar",
        dependency="galah-dr4",
        fields=(None, "mg_fe"),
    ),
    "lamost_dr12v2_lrs_alpha_m": dict(
        quantities={"fe_h": "fe_h", "alpha": "alpha_m"},
        release="LAMOST DR12 v2.0 LRS",
        pipeline="LASP LRS",
        dependency="lamost-dr12v2-lrs-lasp",
        fields=(None, "alpha_m"),
    ),
    "lamost_dr12v2_mrs_alpha_m": dict(
        quantities={"fe_h": "fe_h", "alpha": "alpha_m"},
        release="LAMOST DR12 v2.0 MRS",
        pipeline="LASP MRS",
        dependency="lamost-dr12v2-mrs-lasp",
        fields=(None, "alpha_m_lasp"),
    ),
    "sdss_dr12_sspp_alpha_fe": dict(
        quantities={"fe_h": "fe_h", "alpha": "alpha_fe"},
        release="SDSS DR12 SSPP",
        pipeline="SSPP AFE",
        dependency="sdss_optical_spectra_sspp",
        fields=(None, "AFE"),
    ),
    "sdss_dr19_aspcap_sdss5_alpha_m": dict(
        quantities={"fe_h": "fe_h", "alpha": "alpha_m"},
        release="SDSS DR19 Astra 0.6.0",
        pipeline="ASPCAP",
        dependency="apogee_spectra_aspcap",
        fields=(None, "alpha_m_atm"),
    ),
    "ambre_feros_chem": dict(
        quantities={"fe_h": "m_h", "alpha": "alpha_fe"},
        release="AMBRE FEROS (Worley+2012)",
        pipeline="MATISSE (AMBRE)",
        dependency="ambre_feros",
        fields=("[M/H]", "[alpha/Fe]"),
    ),
    "bestbrightest_placco2019_alpha_fe": dict(
        quantities={"fe_h": "fe_h", "alpha": "alpha_fe"},
        release="Placco+2019",
        pipeline="n-SSPP",
        dependency="bb_medres",
        fields=(None, "[A/Fe]"),
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--observations",
        default=RUN + "master_label_census_20261110_v3/observations.parquet",
    )
    p.add_argument("--snr-min", type=float, default=10.0)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(status="building_chemistry_groups", groups={}, training_performed=False)

    def save():
        (a.output / "manifest.tmp").write_text(
            json.dumps(m, indent=2, default=str) + "\n"
        )
        (a.output / "manifest.tmp").replace(a.output / "manifest.json")

    save()
    try:
        obs = None
        builders = {
            "gaia_dr3_gspspec_chem": lambda: gspspec(
                ROOT + "catalogues/gaia_rvs_lite_flags.fits.gz"
            ),
            "apogee_dr17_alpha_m": lambda: apogee(
                RUN + "apogee_native_fetch_v1/allStar-dr17-synspec_rev1.fits"
            ),
            "galah_dr4_mg_fe": lambda: galah(
                ROOT + "catalogues/galah_dr4_allstar_240705.fits"
            ),
            "lamost_dr12v2_lrs_alpha_m": lambda: lamost(
                RUN
                + "multisurvey_lamost_dr12_lrs_source_direct_v2/dr12_lrs_stellar.fits.gz",
                "alpha_m",
                "alpha_m_err",
                "snrg",
                a.snr_min,
            ),
            "lamost_dr12v2_mrs_alpha_m": lambda: lamost(
                RUN
                + "multisurvey_lamost_dr12_mrs_source_direct_v2/dr12_mrs_stellar.fits.gz",
                "alpha_m_lasp",
                "alpha_m_lasp_err",
                "snr",
                a.snr_min,
            ),
        }
        census = {
            "sdss_dr12_sspp_alpha_fe": "sdss_dr12_sspp",
            "sdss_dr19_aspcap_sdss5_alpha_m": "sdss_dr19_astra_aspcap",
            "bestbrightest_placco2019_alpha_fe": "bestbrightest_placco2019",
            "ambre_feros_chem": "ambre_feros",
        }
        blocks, registry = {}, {}
        for name, spec in GROUPS.items():
            if a.only and name not in a.only:
                continue
            if name in census:
                if obs is None:
                    obs = pd.read_parquet(a.observations)
                block, rows = master(obs, census[name])
            else:
                block, rows = builders[name]()
            block = one_per_star(block)
            n = len(block["source_id"])
            block.update(
                value=np.full((n, 3), np.nan),
                error=np.full((n, 3), np.nan),
                observed=np.zeros((n, 3), bool),
            )
            blocks[name] = block
            registry[name] = dict(
                release=spec["release"],
                pipeline=spec["pipeline"],
                dependency_id=spec["dependency"],
                targets=dict(teff=None, logg=None, fe_h=None),
                chem_quantities=spec["quantities"],
                chem_fields=[f or "" for f in spec["fields"]],
                quality_policy="failed-fit cuts only; one row per star",
                training_enabled=False,
            )
            m["groups"][name] = dict(
                native_rows=rows,
                stars=n,
                quantities=spec["quantities"],
                observed=block["chem_observed"].sum(0).tolist(),
            )
            save()
        receipt = write_observation_sidecar(
            a.output / "chemistry-sidecar.h5",
            blocks,
            registry,
            provenance=dict(
                builder="scripts/build_chemistry_sidecars.py", snr_min=a.snr_min
            ),
        )
        m.update(
            status="complete",
            sidecar=receipt,
            additional_sources=[
                dict(name=n, path=receipt["path"], sha256=receipt["sha256"])
                for n in blocks
            ],
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
