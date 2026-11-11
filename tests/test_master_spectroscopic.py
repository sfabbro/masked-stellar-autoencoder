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
        fe_h_kind=["m_h", "fe_h", "fe_h", "fe_h_a09"],
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
    assert stars.loc["nogaia:x:z"].fe_h_median_highres == 0.0  # fe_h_a09 counts


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


def test_astra_summary_reads_first_nonempty_hdu_and_derives_mg_fe(tmp_path):
    from masked_stellar_autoencoder.training import master_labels as ml

    cols = dict(
        sdss_id=np.array([1, 2], "i8"),
        gaia_dr3_source_id=np.array([7, 0], "i8"),
        ra=np.array([1.0, 2.0]),
        dec=np.array([3.0, 4.0]),
        snr=np.array([50.0, 60.0]),
        teff=np.array([5000.0, 4000.0]),
        e_teff=np.array([10.0, 10.0]),
        logg=np.array([4.0, 2.0]),
        e_logg=np.array([0.1, 0.1]),
        fe_h=np.array([-1.0, -2.0]),
        e_fe_h=np.array([0.05, 0.05]),
        mg_h=np.array([-0.7, -1.6]),
        e_mg_h=np.array([0.05, 0.05]),
        flag_bad=np.array([False, True]),
    )
    fmt = {"i": "K", "f": "D", "b": "L"}
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=k, array=v, format=fmt[v.dtype.kind])
            for k, v in cols.items()
        ]
    )
    empty = fits.BinTableHDU.from_columns(table.columns, nrows=0)
    fits.HDUList([fits.PrimaryHDU(), empty, table]).writeto(tmp_path / "a.fits")
    spec = next(
        s for s in ml.ASTRA_SUMMARIES if s["source"] == "sdss_dr19_astra_astronn"
    )
    o = ml.astra_summary(tmp_path / "a.fits", spec)
    assert o.source_id.tolist() == [7, 0] and o.quality.tolist() == [True, False]
    np.testing.assert_allclose(o.alpha, [0.3, 0.4])
    assert set(o.alpha_kind) == {"mg_fe"}


def test_vizier_join_adds_positions_from_a_second_table(tmp_path):
    from dataclasses import replace

    from masked_stellar_autoencoder.training import master_labels as ml

    def cached(name, header, rows):
        body = "\n".join(["#Column x", header, "-", "-", *rows]) + "\n"
        (tmp_path / f"{name}.tsv").write_text(body)
        (tmp_path / f"{name}.json").write_text(json.dumps(dict(table=name)))

    spec = next(s for s in ml.VIZIER_SOURCES if s.source == "bensby2014")
    spec = replace(spec, expected_rows=None)
    cached(
        "bensby2014",
        "HIP\tTeff\te_Teff\tlogg\te_logg\tFe/H\te_Fe/H\tMg/Fe",
        [
            "12\t5800\t50\t4.4\t0.1\t-0.1\t0.05\t0.1",
            "99\t6000\t50\t4.2\t0.1\t-1\t0.05\t0.3",
        ],
    )
    cached("bensby2014__join", "HIP\t_RAJ2000\t_DEJ2000", ["12\t10.5\t-5.0", "7\t1\t1"])
    frame, meta = ml.fetch_vizier(spec, tmp_path)
    o = ml.vizier_observations(spec, frame)
    assert o.ra.iloc[0] == 10.5 and np.isnan(o.ra.iloc[1])
    assert o.native_id.tolist() == ["12", "99"] and "join" in meta


def test_target_names_are_cleaned_for_simbad():
    from masked_stellar_autoencoder.training import master_labels as ml

    assert ml.target_name("HD-153747") == "HD 153747"
    assert ml.target_name("HD_101189_2") == "HD 101189"
    assert ml.target_name("RS_Cha") == "RS Cha"
    assert ml.target_name("BD-12_345") == "BD-12 345"
    assert ml.target_name("94892") == ""


def test_name_resolution_uses_explicit_names_and_keeps_existing_ids():
    from masked_stellar_autoencoder.training import master_labels as ml

    obs = pd.DataFrame(
        dict(
            native_id=["a", "b", "c"],
            source_id=np.array([0, 0, 5], "i8"),
            id_method=["", "", "publisher_dr3"],
            ra=[1.0, np.nan, 2.0],
            dec=[1.0, np.nan, 2.0],
        )
    )
    resolved = pd.DataFrame(
        dict(name=["HD 1", "HD 2"], gaia_dr3=[11, 0], ra=[9.0, 8.0], dec=[7.0, 6.0])
    )
    out = ml.apply_name_resolution(obs, resolved, names=["HD 1", "HD 2", "HD 1"])
    assert out.source_id.tolist() == [11, 0, 5]
    assert out.id_method.tolist() == ["simbad_dr3", "", "publisher_dr3"]
    assert out.ra.tolist()[1] == 8.0


def test_desi_sp_grid_edges_are_flagged_not_dropped():
    r = records(
        survey=["desi_dr1_sp"] * 5 + ["desi_dr1"],
        fe_h=[-10.0, -4.99, -3.5, -1.0, -1.0, -10.0],
        logg=[9.0, 5.0, 6.0, 4.0, 6.0, 9.0],
    )
    out = build.flag_desi_sp_edges(r)
    assert out.fit_ok.tolist() == [False, False, False, True, False, True]
    assert out.flag_text.tolist()[:3] == [
        "wd_grid;logg_grid_edge",
        "fe_h_grid_edge",
        "fe_h_grid_edge;logg_grid_edge",
    ]
    assert len(out) == 6
