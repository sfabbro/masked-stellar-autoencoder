"""Independent source association, role leakage and native target-mask checks."""

import numpy as np
import pytest

from masked_stellar_autoencoder.training.joint_native_labels import (
    apogee_parameter_metadata,
    assign_roles,
    identity_hash,
    project_catalogue,
)


def test_apogee_metadata_discovery_rejects_missing_ambiguous_and_invalid_symbols():
    from astropy.io import fits

    def metadata(symbols):
        return fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="PARAM_SYMBOL",
                    format="15A",
                    dim="(5,3)",
                    array=np.asarray([symbols], dtype="S5"),
                )
            ]
        )

    empty = [fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros(3))]
    with pytest.raises(ValueError, match="exactly one PARAM_SYMBOL"):
        apogee_parameter_metadata(empty)
    good = metadata(["TEFF", "LOGG", "M_H"])
    with pytest.raises(ValueError, match="exactly one PARAM_SYMBOL"):
        apogee_parameter_metadata([*empty, good, good])
    with pytest.raises(ValueError, match="TEFF and LOGG uniquely"):
        apogee_parameter_metadata([*empty, metadata(["TEFF", "TEFF", "M_H"])])
    index, symbols = apogee_parameter_metadata([*empty, fits.ImageHDU(), good])
    assert index == 3
    assert symbols == ["TEFF", "LOGG", "M_H"]


def native_table(source="galah"):
    if source == "galah":
        dtype = [
            ("gaiadr3_source_id", "i8"),
            ("teff", "f4"),
            ("e_teff", "f4"),
            ("logg", "f4"),
            ("e_logg", "f4"),
            ("fe_h", "f4"),
            ("e_fe_h", "f4"),
            ("flag_sp", "i8"),
            ("flag_sp_fit", "i8"),
            ("flag_fe_h", "i8"),
        ]
    else:
        dtype = [
            ("GAIAEDR3_SOURCE_ID", "i8"),
            ("TEFF", "f4"),
            ("TEFF_ERR", "f4"),
            ("LOGG", "f4"),
            ("LOGG_ERR", "f4"),
            ("FE_H", "f4"),
            ("FE_H_ERR", "f4"),
            ("ASPCAPFLAG", "i8"),
            ("STARFLAG", "i8"),
            ("FE_H_FLAG", "i8"),
            ("PARAMFLAG", "i8", (3,)),
        ]
    table = np.zeros(3, dtype=dtype)
    names = {n.lower(): n for n in table.dtype.names}
    table[next(n for n in table.dtype.names if n.lower().endswith("source_id"))] = [
        10000000000000001,
        10000000000000002,
        10000000000000003,
    ]
    for low, value in [
        ("teff", 5000),
        ("logg", 2.5),
        ("fe_h", -0.5),
        ("e_teff", 50),
        ("e_logg", 0.1),
        ("e_fe_h", 0.02),
        ("teff_err", 50),
        ("logg_err", 0.1),
        ("fe_h_err", 0.02),
    ]:
        if low in names:
            table[names[low]] = value
    selected = np.zeros(
        3,
        dtype=[
            ("source_id", "i8"),
            ("source_row", "i8"),
            ("value", "f8"),
            ("uncertainty", "f8"),
        ],
    )
    selected["source_id"] = table[
        next(n for n in table.dtype.names if n.lower().endswith("source_id"))
    ]
    selected["source_row"] = np.arange(3)
    selected["value"] = table[names["fe_h"]]
    selected["uncertainty"] = table[
        names["e_fe_h"] if source == "galah" else names["fe_h_err"]
    ]
    return table, selected


def test_galah_preserves_rows_and_partial_missing_masks():
    table, selected = native_table()
    table["teff"][0] = np.nan
    table["e_logg"][1] = 0
    table["flag_fe_h"][2] = 1
    r = project_catalogue(table, "galah", selected)
    np.testing.assert_array_equal(
        r["observed"], [[False, True, True], [True, False, True], [True, True, False]]
    )
    assert np.isnan(r["value"][0, 0])
    assert r["uncertainty"][1, 1] == 0
    assert r["logg_parallax_conditioned"].all()
    assert r["raw"]["flag_fe_h"].tobytes() == table["flag_fe_h"].tobytes()


def test_source_rows_ids_feh_errors_must_agree():
    table, selected = native_table()
    for field, bad in [
        ("source_id", 9),
        ("value", -0.7),
        ("uncertainty", 0.3),
        ("source_row", -1),
    ]:
        wrong = selected.copy()
        wrong[field][0] = bad
        with pytest.raises(ValueError):
            project_catalogue(table, "galah", wrong)


def test_apogee_parameter_order_comes_from_metadata_and_info_bits_are_retained():
    table, selected = native_table("apogee")
    table["PARAMFLAG"][0, 2] = 1 << 16  # TEFF fixed -> unavailable
    table["PARAMFLAG"][1, 0] = 1 << 1  # LOGG bad calibration -> unavailable
    table["PARAMFLAG"][2, 0] = 1 << 25  # Informational RGB -> retained
    r = project_catalogue(
        table, "apogee", selected, parameter_symbols=np.array(["LOGG", "M_H", "TEFF"])
    )
    np.testing.assert_array_equal(
        r["observed"], [[False, True, True], [True, False, True], [True, True, True]]
    )
    assert not r["logg_parallax_conditioned"].any()


def test_flags_required_and_alpha_is_not_alias():
    table, selected = native_table("apogee")
    with pytest.raises(ValueError, match="PARAM_SYMBOL"):
        project_catalogue(table, "apogee", selected)
    with pytest.raises(ValueError, match="Only explicit"):
        project_catalogue(table, "Li2025", selected)
    with pytest.raises(ValueError, match="Missing native"):
        project_catalogue(table[["GAIAEDR3_SOURCE_ID", "TEFF"]], "apogee", selected)


def test_all_fit_tune_calibration_ban_original_validation_and_external_references():
    ids = np.arange(1, 10001, dtype="i8")
    partition = np.zeros(len(ids), dtype="u1")
    partition[::7] = 1
    refs = ids[::13]
    r = assign_roles(ids, partition, np.ones(len(ids), bool), refs)
    for k in ["fit", "tune", "calibration"]:
        assert not (r[k] & (partition == 1)).any()
        assert not np.isin(ids[r[k]], refs).any()
    assert not (r["tune"] & r["calibration"]).any()
    assert r["tune"].sum() > 100 and r["calibration"].sum() > 100
    assert np.array_equal(
        r["identity_split"],
        np.where(
            identity_hash(ids) % 1000 < 700,
            0,
            np.where(identity_hash(ids) % 1000 < 850, 1, 2),
        ),
    )


def test_roles_reject_duplicate_and_float_ids_and_keep_unavailable_out():
    with pytest.raises(ValueError, match="Duplicate"):
        assign_roles(
            np.array([1, 1]),
            np.array([0, 0]),
            np.array([True, True]),
            np.array([], dtype="i8"),
        )
    with pytest.raises(ValueError, match="positive integers"):
        assign_roles(
            np.array([1.0]), np.array([0]), np.array([True]), np.array([], dtype="i8")
        )
    r = assign_roles(
        np.arange(1, 10), np.zeros(9), np.zeros(9, bool), np.array([], dtype="i8")
    )
    assert not any(r[k].any() for k in ["fit", "tune", "test", "calibration"])


def test_fits_to_hdf5_materialization_integration(tmp_path, monkeypatch):
    """Exercise actual FITS readers/HDF5 publication, not mocked source records."""
    import hashlib
    import importlib.util
    import json
    from pathlib import Path

    import h5py
    from astropy.io import fits

    module_path = (
        Path(__file__).parents[1] / "scripts/materialize_joint_native_labels.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_joint_native_labels", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sources = {}
    records = []
    dtype = [
        ("source_id", "i8"),
        ("source_row", "i8"),
        ("value", "f8"),
        ("uncertainty", "f8"),
        ("survey", "S32"),
        ("input_file", "S512"),
    ]
    for offset, source in enumerate(["galah", "apogee"]):
        table, selected = native_table(source)
        idcol = "gaiadr3_source_id" if source == "galah" else "GAIAEDR3_SOURCE_ID"
        table[idcol] += offset * 10
        selected["source_id"] = table[idcol]
        p = tmp_path / (source + ".fits")
        hdus = [fits.PrimaryHDU(), fits.BinTableHDU(data=table)]
        if source == "apogee":
            # Match the actual production layout that rejected the original hardcoded HDU2.
            hdus.append(fits.ImageHDU(data=np.zeros(2, dtype="f4")))
            hdus.append(
                fits.BinTableHDU.from_columns(
                    [
                        fits.Column(
                            name="PARAM_SYMBOL",
                            format="15A",
                            dim="(5,3)",
                            array=np.array([["TEFF", "LOGG", "M_H"]], dtype="S5"),
                        )
                    ]
                )
            )
        fits.HDUList(hdus).writeto(p)
        sources[source] = p
        rows = np.zeros(3, dtype=dtype)
        for k in selected.dtype.names:
            rows[k] = selected[k]
        rows["survey"] = source.encode()
        rows["input_file"] = str(p).encode()
        records.append(rows)
    records = np.concatenate(records)
    ids = records["source_id"]
    cohort = tmp_path / "cohort.h5"
    with h5py.File(cohort, "w") as f:
        f.attrs.update(
            complete=True,
            content_id="fixture-not-production",
            feature_columns_json=json.dumps(["PARALLAX", "EBV"]),
        )
        for name, data in {
            "source_id": ids,
            "benchmark_partition": np.array([0, 1, 0, 0, 1, 0], dtype="u1"),
            "feature_available": np.ones(6, bool),
            "label_split": assign_roles(
                ids, np.zeros(6), np.ones(6, bool), np.array([], dtype="i8")
            )["identity_split"],
            "values": np.array(
                [[-0.5, 0.1], [0, 0.2], [1, 0.3], [2, 0.4], [3, 0.5], [4, 0.6]],
                dtype="f4",
            ),
            "reported_feature_errors": np.full((6, 2), 0.1, dtype="f4"),
            "feature_observed": np.ones((6, 2), bool),
            "error_observed": np.ones((6, 2), bool),
        }.items():
            f.create_dataset("features/" + name, data=data)
        f.create_dataset("raw_layers/native/measurements", data=records)
        f.create_dataset("layer_index/native/feature_row", data=np.arange(6))
    output = tmp_path / "joint.h5"
    fixture_refs = np.arange(1, 99575, dtype="i8")
    report = module.materialize(
        cohort,
        sources,
        {s: module.sha(p) for s, p in sources.items()},
        fixture_refs,
        output,
        cohort_sha256=module.sha(cohort),
        reference_sha256=hashlib.sha256(
            fixture_refs.astype("<i8").tobytes()
        ).hexdigest(),
    )
    assert not report["training_launch_ready"]
    with h5py.File(output) as f:
        assert f.attrs["complete"]
        np.testing.assert_array_equal(
            f["targets/value"][:], np.tile([5000, 2.5, -0.5], (6, 1))
        )
        assert f["targets/observed"][:].all()
        assert f["astrometry/parallax_mas"][0] == -0.5
        assert f["astrometry/observed"][:].all()
        assert not f["astrometry/distance_fit_eligible"][:].any()
        np.testing.assert_array_equal(
            f["targets/logg_parallax_conditioned"][:], [1, 1, 1, 0, 0, 0]
        )
    verify_path = Path(__file__).parents[1] / "scripts/verify_joint_native_sidecar.py"
    verify_spec = importlib.util.spec_from_file_location(
        "verify_joint_native_sidecar", verify_path
    )
    verifier = importlib.util.module_from_spec(verify_spec)
    verify_spec.loader.exec_module(verifier)
    verified = verifier.verify(
        output,
        cohort,
        sources,
        {s: module.sha(p) for s, p in sources.items()},
        tmp_path / "verified.json",
    )
    assert verified["native_labels_verified"]
    assert not verified["training_launch_ready"]
    # Repeated CPU preparation must reverify, not overwrite or refit.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    prepare_path = Path(__file__).parents[1] / "scripts/prepare_joint_native_labels.py"
    prepare_spec = importlib.util.spec_from_file_location(
        "prepare_joint_native_labels", prepare_path
    )
    preparation = importlib.util.module_from_spec(prepare_spec)
    prepare_spec.loader.exec_module(preparation)
    reference_h5 = tmp_path / "reference.h5"
    with h5py.File(reference_h5, "w") as f:
        f.create_dataset("ALL_reference_holdout_source_ids", data=fixture_refs)
    config = {
        "cohort": {"path": str(cohort), "sha256": module.sha(cohort)},
        "references": {
            "path": str(reference_h5),
            "sha256": module.sha(reference_h5),
            "count": 99574,
            "dataset": "ALL_reference_holdout_source_ids",
            "ordered_id_sha256": hashlib.sha256(
                fixture_refs.astype("<i8").tobytes()
            ).hexdigest(),
        },
        "sources": {
            s: {
                "path": str(p),
                "bytes": p.stat().st_size,
                "rows": 3,
                "sha256": None if s == "galah" else module.sha(p),
            }
            for s, p in sources.items()
        },
        "sidecar_filename": output.name,
        "requested_resources": {"gpus": 0},
        "notes": [],
    }
    assert preparation.preflight(config)["cpu_only"]
    pins = preparation.record_source_pins(config, tmp_path)
    assert preparation.record_source_pins(config, tmp_path) == pins
    before = module.sha(output)
    assert preparation.execute(config, tmp_path)["native_labels_verified"]
    report = preparation.execute(config, tmp_path)
    assert report["native_labels_verified"]
    preparation.publish_manifest(tmp_path, report)
    preparation.publish_manifest(tmp_path, report)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["training_launch_ready"] is False
    assert module.sha(output) == before
    with pytest.raises(FileExistsError, match="conflicts"):
        preparation.immutable_json(tmp_path / "source-pins.json", {"bad": "pin"})
    badconfig = dict(config)
    badconfig["cohort"] = {"path": str(tmp_path / "missing-cohort.h5")}
    with pytest.raises(FileNotFoundError, match="Existing source mounts"):
        preparation.preflight(badconfig)
    quality_path = Path(__file__).parents[1] / "scripts/run_joint_native_quality.py"
    quality_spec = importlib.util.spec_from_file_location(
        "run_joint_native_quality", quality_path
    )
    quality_runner = importlib.util.module_from_spec(quality_spec)
    quality_spec.loader.exec_module(quality_runner)
    quality_directory = tmp_path / "distinct-quality-qa"
    quality_manifest = quality_runner.verify_existing(
        config, tmp_path, quality_directory, source_sha="fixture-not-production"
    )
    assert quality_manifest["status"] == "complete"
    assert quality_manifest["all_target_quality_masks_verified"]
    assert quality_manifest["sidecar_sha256"] == module.sha(output)
    assert quality_manifest["source_pins"] == pins
    assert quality_manifest["role_target_counts"]["fit"]["complete_targets"] >= 1
    assert quality_manifest["training_launch_ready"] is False
    assert (
        quality_runner.verify_existing(
            config, tmp_path, quality_directory, source_sha="fixture-not-production"
        )
        == quality_manifest
    )
    with pytest.raises(ValueError, match="must differ"):
        quality_runner.verify_existing(
            config, tmp_path, tmp_path, source_sha="fixture-not-production"
        )
    # A stale/corrupted quality mask must not escape verification merely because
    # the independently checked observed mask still has its original value.
    with h5py.File(output, "r+") as f:
        f["targets/quality_pass"][0, 0] = False
    with pytest.raises(AssertionError):
        verifier.verify(
            output,
            cohort,
            sources,
            {s: module.sha(p) for s, p in sources.items()},
            tmp_path / "bad-quality-verified.json",
        )
    assert not (tmp_path / "bad-quality-verified.json").exists()
    with h5py.File(output, "r+") as f:
        f["targets/quality_pass"][0, 0] = True
    with h5py.File(output, "r+") as f:
        f["targets/value"][0, 0] += 1
    with pytest.raises(AssertionError):
        verifier.verify(
            output,
            cohort,
            sources,
            {s: module.sha(p) for s, p in sources.items()},
            tmp_path / "bad-verified.json",
        )
    assert not (tmp_path / "bad-verified.json").exists()
    with pytest.raises(FileExistsError):
        module.materialize(cohort, sources, {}, np.arange(1, 99575, dtype="i8"), output)
