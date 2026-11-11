import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from masked_stellar_autoencoder.training import master_spectroscopic as ms

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_master", ROOT / "scripts/build_master_spectroscopic_labels.py"
)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def records(**cols):
    n = len(next(iter(cols.values())))
    out = ms.frame(n, survey="s", catalogue="c", fe_h_kind="fe_h", fit_ok=True)
    for k, v in cols.items():
        out[k] = v
    return ms.finalize(out)


def test_finalize_keeps_schema_and_cleans_sentinels():
    r = records(teff=[5000.0, -9999.0], source_id=[5, -3])
    assert list(r.columns) == list(ms.RECORD_COLUMNS)
    assert np.isnan(r.teff[1]) and r.source_id.tolist() == [5, 0]


def test_exact_ids_never_round_through_float():
    big = 4185355646116022528
    assert ms.exact_ids(np.array([str(big), " 12", "x", ""])).tolist() == [
        big,
        12,
        0,
        0,
    ]
    with pytest.raises(ValueError):
        ms.exact_ids(np.array([1.0]))


def test_star_ids_group_nogaia_records_across_surveys_by_position():
    r = records(
        survey=["a", "b", "a", "c", "d"],
        native_id=["1", "2", "3", "4", "5"],
        source_id=[0, 0, 7, 0, 0],
        ra=[10.0, 10.0 + 0.5 / 3600, 10.0, 50.0, np.nan],
        dec=[0.0, 0.0, 0.0, 0.0, np.nan],
    )
    sid = ms.assign_star_ids(r)
    assert sid[2] == 7
    assert sid[0] == sid[1] == -1  # 0.5" apart: one star
    assert sid[3] == -2 and sid[4] == -3  # far / no position: own ids


def test_star_table_summaries_and_keys():
    r = records(
        survey=["gsp", "lamost", "lamost", "x"],
        native_id=["9", "a", "b", "z"],
        source_id=[9, 9, 0, 0],
        fe_h=[-1.0, -2.0, -3.0, 0.0],
        fe_h_kind=["m_h", "fe_h", "fe_h", "fe_h"],
        tier=["survey", "survey", "survey", "highres_reference"],
        ra=[1.0, 1.0, 2.0, np.nan],
        dec=[0.0, 0.0, 0.0, np.nan],
    )
    r["star_id"] = ms.assign_star_ids(r)
    r["label_groups"] = ["v2:g", "", "loose:l", ""]
    stars = ms.star_table(r).set_index("star_key")
    assert set(stars.index) == {"gaia:9", "nogaia:lamost:b", "nogaia:x:z"}
    gaia = stars.loc["gaia:9"]
    assert gaia.n_records == 2 and gaia.surveys == "gsp;lamost"
    assert gaia.fe_h_median == -2.0  # [M/H] record excluded
    assert gaia.label_groups == "v2:g"
    assert stars.loc["nogaia:x:z"].has_highres


def test_label_groups_are_survey_specific():
    r = records(source_id=[1, 2, 0])
    groups = {"v2:apogee": np.array([1]), "chem:apogee_dr17_alpha_m": np.array([1, 2])}
    out = build.label_groups(r, "apogee_dr17", groups)
    assert out.tolist() == [
        "v2:apogee;chem:apogee_dr17_alpha_m",
        "chem:apogee_dr17_alpha_m",
        "",
    ]
    assert build.label_groups(r, "galah_dr4", groups).tolist() == ["", "", ""]


def test_galah_adapter_and_trace_back(tmp_path, monkeypatch):
    n = 3
    cols = dict(
        sobject_id=np.array([1, 2, 3], "i8"),
        gaiadr3_source_id=np.array([11, 0, 13], "i8"),
        ra=np.array([1.0, 2.0, 3.0]),
        dec=np.zeros(n),
        teff=np.array([5000.0, 6000.0, 7000.0]),
        e_teff=np.full(n, 50.0),
        logg=np.full(n, 4.0),
        e_logg=np.full(n, 0.1),
        fe_h=np.array([-1.0, -2.0, -3.0]),
        e_fe_h=np.full(n, 0.1),
        mg_fe=np.full(n, 0.3),
        e_mg_fe=np.full(n, 0.05),
        snr_px_ccd3=np.full(n, 50.0),
        flag_sp=np.array([0, 1, 0], "i8"),
        flag_sp_fit=np.zeros(n, "i8"),
        flag_fe_h=np.zeros(n, "i8"),
        eu_fe=np.array([0.1, 0.2, 0.9]),
    )
    path = tmp_path / "galah.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name=k, array=v, format="K" if v.dtype.kind == "i" else "D")
            for k, v in cols.items()
        ]
    ).writeto(path)
    monkeypatch.setitem(
        build.CATALOGUES,
        "galah_dr4",
        ms.Catalogue("galah_dr4", str(path), "fits", native_id="sobject_id"),
    )
    r = ms.finalize(build.galah())
    assert r.source_id.tolist() == [11, 0, 13]
    assert r.fit_ok.tolist() == [True, False, True]
    assert r.alpha_kind.iloc[0] == "mg_fe" and r.native_id.tolist() == ["1", "2", "3"]
    ms.save_catalogues(build.CATALOGUES, tmp_path / "catalogues.json")
    native = ms.read_native_rows(r.iloc[[2]], tmp_path / "catalogues.json")["galah_dr4"]
    assert native.loc[2, "eu_fe"] == 0.9  # abundance not in the master table


def test_csv_trace_back(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame(dict(rave_obs_id=["a", "b"], mg=[1.5, 2.5])).to_csv(path, index=False)
    cats = {"rave_dr6": dict(path=str(path), format="csv", hdu="csv")}
    (tmp_path / "c.json").write_text(json.dumps(cats))
    r = records(native_row=[1], catalogue=["rave_dr6"])
    assert ms.read_native_rows(r, tmp_path / "c.json")["rave_dr6"].loc[1, "mg"] == 2.5
