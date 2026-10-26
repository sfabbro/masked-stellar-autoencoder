import numpy as np
import pytest

from masked_stellar_autoencoder.training.remediation_source import (
    development_roles,
    exact_ids,
    native_assignment,
    project_raw_native,
    role_counts,
)


def galah():
    a = np.zeros(
        4,
        dtype=[
            ("gaiadr3_source_id", "i8"),
            ("teff", "f8"),
            ("logg", "f8"),
            ("fe_h", "f8"),
            ("e_teff", "f8"),
            ("e_logg", "f8"),
            ("e_fe_h", "f8"),
            ("flag_sp", "i4"),
            ("flag_fe_h", "i4"),
            ("snr_px_ccd3", "f8"),
        ],
    )
    a["gaiadr3_source_id"] = [10, 20, 30, 40]
    for key, value in [
        ("teff", 5000),
        ("logg", 2.5),
        ("fe_h", -0.7),
        ("e_teff", 50),
        ("e_logg", 0.1),
        ("e_fe_h", 0.1),
        ("snr_px_ccd3", 50),
    ]:
        a[key] = value
    return a


def test_galah_raw_reselection_ignores_bad_fe_flag_and_is_sparse():
    a = galah()
    a["flag_fe_h"] = 1
    a["logg"][0] = np.nan
    a["snr_px_ccd3"][1] = 30
    a["flag_sp"][2] = 1
    a["snr_px_ccd3"][3] = np.nan
    _, value, error, mask = project_raw_native(a, "galah")
    np.testing.assert_array_equal(mask[0], [True, False, True])
    assert not mask[1:].any()
    assert value[0, 2] == -0.7 and error[0, 2] == 0.1


def test_development_roles_not_restricted_to_old_train_and_no_exposure_leak():
    ids = np.arange(1, 10001, dtype="i8")
    roles = development_roles(ids, np.ones(len(ids), bool), [1, 2], [2, 3, 4])
    assert roles[0] == b"external"
    assert roles[1] == b"excluded"
    assert np.all(roles[2:4] == b"fit")
    assert set(roles[4:]) == {b"fit", b"tune", b"cal", b"devtest"}
    np.testing.assert_array_equal(
        roles, development_roles(ids, np.ones(len(ids), bool), [1, 2], [2, 3, 4])
    )
    counts = role_counts(
        roles, np.tile([True, False, True, False, True, False], (len(ids), 1))
    )
    assert counts["fit"]["targets"]["logg"] == 0
    assert counts["fit"]["targets"]["teff"] > 6000


def test_native_duplicate_selection_per_target_and_exact_join():
    ids = np.array([9, 9, 15, 99], "i8")
    mask = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1]], bool)
    pairs = native_assignment(np.array([9, 15, 20], "i8"), ids, mask)
    np.testing.assert_array_equal(pairs[0][1], [0])
    np.testing.assert_array_equal(pairs[1][1], [1, 2])
    np.testing.assert_array_equal(pairs[1][0], [0, 1])


def test_float_ids_and_duplicate_features_rejected():
    with pytest.raises(ValueError, match="integer"):
        exact_ids(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Unique"):
        development_roles(np.array([1, 1]), np.ones(2, bool))
    with pytest.raises(ValueError, match="Sorted"):
        native_assignment(np.array([2, 1]), np.array([1]), np.ones((1, 3), bool))


def test_apogee_uses_metadata_not_fixed_parameter_positions():
    a = np.zeros(
        2,
        dtype=[
            ("GAIAEDR3_SOURCE_ID", "i8"),
            ("TEFF", "f8"),
            ("LOGG", "f8"),
            ("FE_H", "f8"),
            ("TEFF_ERR", "f8"),
            ("LOGG_ERR", "f8"),
            ("FE_H_ERR", "f8"),
            ("ASPCAPFLAG", "i8"),
            ("STARFLAG", "i8"),
            ("FE_H_FLAG", "i8"),
            ("PARAMFLAG", "i8", (3,)),
        ],
    )
    a["GAIAEDR3_SOURCE_ID"] = [1, 2]
    for key in ["TEFF", "LOGG", "FE_H", "TEFF_ERR", "LOGG_ERR", "FE_H_ERR"]:
        a[key] = 0.1
    a["PARAMFLAG"][0, 0] = 1
    a["PARAMFLAG"][1, 2] = 1 << 5  # informational, retained
    _, _, _, mask = project_raw_native(a, "apogee", ["LOGG", "OTHER", "TEFF"])
    np.testing.assert_array_equal(mask, [[True, False, True], [True, True, True]])
    with pytest.raises(ValueError, match="metadata"):
        project_raw_native(a, "apogee")


def test_feature_control_rejects_target_and_parallax_leakage():
    from masked_stellar_autoencoder.training.remediation_source import phot_xp_columns

    names = ["G", "BP", "RP", "W1", "W2", "PARALLAX", "LOGG"] + [
        f"{a}_{i}" for a in ("bp", "rp") for i in range(1, 56)
    ]
    mask = phot_xp_columns(names)
    assert mask.sum() == 115
    assert not mask[5:7].any()
    with pytest.raises(ValueError, match="110"):
        phot_xp_columns(["G", "LOGG"])


def test_materialize_synthetic_raw_reselection_streaming_and_roles(tmp_path):
    import hashlib
    import importlib.util
    from pathlib import Path

    import h5py
    from astropy.io import fits

    spec = importlib.util.spec_from_file_location(
        "remediation_prepare",
        Path(__file__).parents[1] / "scripts/prepare_remediation_source.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = np.arange(1, 61, dtype="i8")
    names = np.array(
        ["G", "BP", "RP", "W1", "W2", "PARALLAX", "LOGG"]
        + [f"{a}_{i}" for a in ("bp", "rp") for i in range(1, 56)],
        "S16",
    )
    width = len(names)
    expanded_path = tmp_path / "expanded.h5"
    with h5py.File(expanded_path, "w") as f:
        for key, array in [
            ("source_id", ids),
            ("feature_names", names),
            ("scaler_center", np.zeros(width)),
            ("scaler_scale", np.ones(width)),
            ("value", np.ones((60, 6))),
            ("error", np.ones((60, 6)) * 0.1),
            ("primary_score_eligible", np.ones((60, 6), bool)),
            ("error_observed", np.ones((60, 6), bool)),
            ("label_source", np.full((60, 6), b"teacher", "S64")),
            ("input_diagnostic_only", np.arange(60) == 59),
            ("input", np.ones((60, width), "f4")),
        ]:
            f.create_dataset(key, data=array)
    frozen_path = tmp_path / "frozen.npz"
    # Unsorted original rows exercise exact input restoration without H5 fancy-index errors.
    oldids = np.array([62, 61], "i8")
    np.savez(
        frozen_path,
        source_id=oldids,
        feature_names=names,
        scaler_center=np.zeros(width),
        scaler_scale=np.ones(width),
        value=np.ones((2, 6)),
        error=np.ones((2, 6)) * 0.1,
        observed=np.ones((2, 6), bool),
        quality=np.ones((2, 6), bool),
        error_observed=np.ones((2, 6), bool),
        label_source=np.full((2, 6), b"old", "S64"),
        role=np.array(["test", "fit"]),
        input=np.full((2, width), 7, "f4"),
    )
    g = np.tile(galah()[:1], 62)
    g["gaiadr3_source_id"] = np.arange(1, 63)
    g["flag_fe_h"] = 1
    g["logg"][0] = np.nan
    g["snr_px_ccd3"][1] = 10
    galah_path = tmp_path / "galah.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(g)]).writeto(galah_path)
    apo = np.zeros(
        1,
        dtype=[
            ("GAIAEDR3_SOURCE_ID", "i8"),
            ("TEFF", "f8"),
            ("LOGG", "f8"),
            ("FE_H", "f8"),
            ("TEFF_ERR", "f8"),
            ("LOGG_ERR", "f8"),
            ("FE_H_ERR", "f8"),
            ("ASPCAPFLAG", "i8"),
            ("STARFLAG", "i8"),
            ("FE_H_FLAG", "i8"),
            ("PARAMFLAG", "i8", (2,)),
        ],
    )
    apo["GAIAEDR3_SOURCE_ID"] = 2
    for key, val in [
        ("TEFF", 4000),
        ("LOGG", 3),
        ("FE_H", -0.2),
        ("TEFF_ERR", 100),
        ("LOGG_ERR", 0.2),
        ("FE_H_ERR", 0.2),
    ]:
        apo[key] = val
    metadata = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="PARAM_SYMBOL",
                format="8A",
                dim="(4,2)",
                array=np.array([["TEFF", "LOGG"]], "S4"),
            )
        ]
    )
    apo_path = tmp_path / "apogee.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(apo), metadata]).writeto(apo_path)
    ancestry_path = tmp_path / "ancestry.npz"
    np.savez(
        ancestry_path,
        historical_fit_source_id=np.array([3, 4]),
        historical_fit_sample_count=np.array([1, 0]),
    )
    refs_path = tmp_path / "refs.h5"
    with h5py.File(refs_path, "w") as f:
        f.create_dataset(
            "ALL_reference_holdout_source_ids", data=np.array([3, 4], "i8")
        )
    ast_path = tmp_path / "native.h5"
    with h5py.File(ast_path, "w") as f:
        f.create_dataset("source_id", data=ids)
        f.create_dataset("astrometry/parallax_mas", data=np.linspace(-1, 1, 60))
        f.create_dataset("astrometry/parallax_error_mas", data=np.full(60, 0.1))
        f.create_dataset("astrometry/observed", data=np.ones(60, bool))
    config = {
        "run_id": "synthetic",
        "library_commit": "synthetic",
        "chunk_rows": 7,
        "minimum_fit_per_target": 0,
    }
    for key, path in [
        ("expanded", expanded_path),
        ("frozen", frozen_path),
        ("galah", galah_path),
        ("apogee", apo_path),
        ("ancestry", ancestry_path),
        ("references", refs_path),
        ("native_astrometry", ast_path),
    ]:
        config[key] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report = module.materialize(config, tmp_path / "result")
    assert report["status"] == "remediation_source_complete"
    assert report["historical_supervised_heldout_count"] == 0
    assert (
        report["feature_profiles"]["phot_xp_measured_parallax"]["measured_rows"] == 60
    )
    with h5py.File(tmp_path / "result/prepared.h5", "r") as f:
        assert f["role"][2] == b"excluded" and f["role"][3] == b"external"
        assert f["role"][60] == b"fit"
        assert f["observed"][0, 0] and not f["observed"][0, 1] and f["observed"][0, 2]
        assert f["label_source"][1, 1] == b"apogee"
        assert not f["observed"][59].any()
        assert np.all(f["input"][:, 5:7] == -9999)
        assert np.all(f["input"][60:, 0] == 7)
        assert f["measured_parallax/input_value"][0] == -1
        assert f["measured_parallax/input_value"][61] == -9999
        assert f.attrs["consumed_inspected_test"]
