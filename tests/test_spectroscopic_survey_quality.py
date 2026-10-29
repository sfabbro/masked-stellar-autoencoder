import importlib.util
import sys

import numpy as np

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location(
    "quality", "scripts/spectroscopic_survey_quality.py"
)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def test_ges_flags_are_exact_and_label_specific():
    flags = np.array(
        [b"", b"NIA", b"SNR", b"SSP", b"BIN", b"NEWSNR", b"SSA"], dtype="S12"
    )
    n = len(flags)
    r = {"SFLAGS": flags}
    for k in ("TEFF", "LOGG", "FEH", "MG1", "SI1", "CA1", "TI1"):
        r[k] = np.full(
            n,
            5000.0
            if k == "TEFF"
            else (7.5 if k in ("MG1", "SI1", "CA1", "TI1") else 1.0),
        )
        r["E_" + k] = np.full(n, 0.1)
    x = q.quality_ges(r)
    assert x["FEH"]["quality_loose"].tolist() == [
        True,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert x["MG1"]["quality_loose"].tolist() == [
        True,
        False,
        False,
        True,
        True,
        False,
        True,
    ]
    assert x["FEH"]["quality_strict"].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    assert x["MG1"]["quality_strict"].tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert x["_unknown_flags"][5]


def test_metal_poor_preserved_in_loose_and_missing_error_explicit():
    r = {
        k: np.array(v)
        for k, v in {
            "algo_conv_madera": [4, 1, 0],
            "snr_madera": [12, 40, 50],
            "teff_cal_madera": [5000] * 3,
            "teff_error_madera": [100] * 3,
            "logg_cal_madera": [2] * 3,
            "logg_error_madera": [0.1] * 3,
            "m_h_cal_madera": [-3, -3, -3],
            "m_h_error_madera": [0.1, 0.1, np.nan],
        }.items()
    }
    m = q.quality_madera(r)["m_h_cal_madera"]
    assert m["quality_loose"].tolist() == [True, False, True]
    assert m["quality_strict"].tolist() == [False, False, False]
    assert m["error_known"].tolist() == [True, True, False]


def test_join_and_gauguin_sanity():
    idx = q.map_unique(np.array([b"a", b"b", b"c"]), np.array([b"a", b"b", b"b"]))
    assert idx.tolist() == [0, -1, -1]
    m = {
        "algo_conv_madera": np.array([0, 0, 0]),
        "snr_madera": np.array([30.0, 30.0, 30.0]),
    }
    a = {
        "fe_h": np.array([-3.0, -21.0, -0.2]),
        "fe_h_error": np.array([0.3, 0.2, np.nan]),
        "fe_h_chisq": np.array([1.0, 1.0, 1.0]),
        "fe_h_nlines": np.array([2, 2, 2]),
        "alpha_fe": np.array([0.4, 0.2, 0.2]),
        "alpha_fe_error": np.array([0.2, 0.2, 0.2]),
        "alpha_fe_chisq": np.array([1.0, 1.0, 1.0]),
        "n_elements": np.array([2, 2, 2]),
    }
    result = q.quality_gauguin(a, m, np.array([0, 1, 2]))
    assert result["fe_h"]["quality_loose"].tolist() == [True, False, True]
    assert result["fe_h"]["quality_strict"].tolist() == [True, False, False]


def test_unmatched_gaia_eso_is_unknown_not_false(tmp_path):
    import h5py

    ges, madera, gauguin, out = (
        tmp_path / name for name in ("ges.h5", "m.h5", "a.h5", "out.h5")
    )
    with h5py.File(ges, "w") as f:
        g = f.create_group("raw")
        g["SFLAGS"] = np.array([b""], dtype="S4")
        for k in ("TEFF", "LOGG", "FEH", "MG1", "SI1", "CA1", "TI1"):
            g[k] = np.array(
                [
                    5000.0
                    if k == "TEFF"
                    else (7.5 if k in ("MG1", "SI1", "CA1", "TI1") else 1.0)
                ]
            )
            g["E_" + k] = np.array([0.1])
    with h5py.File(madera, "w") as f:
        f["rave_obs_id"] = np.array([b"a"])
        f.create_group("published_gaia_edr3_crossmatch")["identity_candidate"] = (
            np.array([True])
        )
        g = f.create_group("madera_raw")
        for k, v in {
            "algo_conv_madera": [0],
            "snr_madera": [30.0],
            "teff_cal_madera": [5000.0],
            "teff_error_madera": [100.0],
            "logg_cal_madera": [2.0],
            "logg_error_madera": [0.2],
            "m_h_cal_madera": [-2.0],
            "m_h_error_madera": [0.2],
        }.items():
            g[k] = v
    with h5py.File(gauguin, "w") as f:
        f["rave_obs_id"] = np.array([b"a"])
        f.create_group("published_gaia_edr3_crossmatch")["identity_candidate"] = (
            np.array([True])
        )
        g = f.create_group("gauguin_raw")
        for k, v in {
            "fe_h": [-2.0],
            "fe_h_error": [0.2],
            "fe_h_chisq": [1.0],
            "fe_h_nlines": [2],
            "alpha_fe": [0.3],
            "alpha_fe_error": [0.2],
            "alpha_fe_chisq": [1.0],
            "n_elements": [2],
        }.items():
            g[k] = v
    result = q.produce(
        ges, madera, gauguin, out, [q.sha256_file(p) for p in (ges, madera, gauguin)]
    )
    assert result["counts"]["gaia_eso"]["FEH"]["quality_loose_with_identity"] is None
    with h5py.File(out) as f:
        assert not f["gaia_eso"].attrs["identity_evaluated"]
        assert "identity_candidate" not in f["gaia_eso"]
        assert f["rave_madera"].attrs["identity_evaluated"]
