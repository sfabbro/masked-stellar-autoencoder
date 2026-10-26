"""Synthetic counterexamples for the model-free input identity audit."""

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "remediation_input_audit",
    Path(__file__).parents[1] / "scripts/audit_remediation_inputs.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_shuffled_exact_ids_and_missing():
    ids = np.array([4185355646116022528, 7, 8], dtype="i8")
    rows, missing = audit.exact_rows(
        ids, np.array([8, ids[0], 9], dtype="i8"), chunk_rows=1
    )
    np.testing.assert_array_equal(rows, [2, 0, -1])
    assert missing == {9}


def test_duplicate_and_float_ids_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        audit.exact_rows(np.array([5, 5]), np.array([5]))
    with pytest.raises(ValueError, match="integers"):
        audit.exact_rows(np.array([5.0]), np.array([5]))


def test_scaler_identity_and_withheld_sentinel():
    names = ["G", "bp_1", "LOGG", "rp_1"]
    actual, visible = audit.normalize(
        [12, 4, 3, -9999], names, [10, 2, 1, 0], [2, 2, 1, 1]
    )
    np.testing.assert_array_equal(actual, [1, 1, -9999, -9999])
    np.testing.assert_array_equal(visible, [True, True, False, False])
    actual, _ = audit.normalize([np.nan, np.inf, 3, 4], names, [0] * 4, [1] * 4)
    np.testing.assert_array_equal(actual, [-9999, -9999, -9999, 4])


def test_cache_feature_order_id_and_bytes(tmp_path):
    names = ["G", "bp_1"]
    center = [10, 2]
    scale = [2, 2]
    request = dict(features=names, center=center, scale=scale)
    sha = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(status="complete", request=request))
    )
    np.save(tmp_path / "source_id.npy", np.array([9, 7], dtype="i8"))
    np.save(tmp_path / "features.npy", np.array([[3, 5], [1, 2]], dtype="f4"))
    np.save(
        tmp_path / "provenance.npy",
        np.array([(0, 2), (0, 3)], dtype=[("shard", "u4"), ("row", "i8")]),
    )
    values, provenance, _ = audit.cache_row_proof(
        tmp_path, sha, 7, 1, names, center, scale
    )
    np.testing.assert_array_equal(values, [1, 2])
    assert provenance["row"] == 3
    with pytest.raises(ValueError, match="ID/row"):
        audit.cache_row_proof(tmp_path, sha, 7, 0, names, center, scale)
    with pytest.raises(ValueError, match="order"):
        audit.cache_row_proof(tmp_path, sha, 7, 1, names[::-1], center, scale)
    with pytest.raises(ValueError, match="bytes"):
        audit.same(values, [2, 1], "Changed cache bytes")


def test_raw_shard_id_association(tmp_path):
    path = tmp_path / "original.h5"
    with h5py.File(path, "w") as f:
        f["train"] = np.array(
            [(7, 12, 6), (9, 14, 8)],
            dtype=[("source_id", "i8"), ("G", "f4"), ("bp_1", "f4")],
        )
    request = dict(train_keys=["train"], validation_keys=[])
    np.testing.assert_array_equal(
        audit.raw_record(path, request, 0, 0, 7, ["G", "bp_1"]), [12, 6]
    )
    with pytest.raises(ValueError, match="ID/row"):
        audit.raw_record(path, request, 0, 1, 7, ["G", "bp_1"])


def test_unavailable_source_is_incomplete_and_no_overwrite(tmp_path):
    config = dict(prepared=dict(path=str(tmp_path / "absent"), sha256="missing"))
    output = tmp_path / "out"
    result = audit.audit(config, output)
    assert result["status"] == "incomplete"
    assert result["model_execution"] is False
    assert json.loads((output / "manifest.json").read_text())["unresolved"]
    with pytest.raises(FileExistsError):
        audit.audit(config, output)


def test_end_to_end_selected_rows_fixture(tmp_path):
    names = ["G"] + [f"{arm}_{j}" for arm in ("bp", "rp") for j in range(1, 56)]
    width = len(names)
    paths = {n: tmp_path / (n + ".h5") for n in ("prepared", "legacy", "native")}
    raw = np.arange(width, dtype="f4")
    with h5py.File(paths["prepared"], "w") as f:
        for key, value in dict(
            source_id=[9],
            corner_rows=[0],
            feature_names=np.array(names, dtype="S12"),
            scaler_center=np.zeros(width),
            scaler_scale=np.ones(width),
            raw=raw[None],
            input=raw[None],
            visible=np.ones((1, width), bool),
            feature_origin=np.array(["legacy_cohort"], dtype="S32"),
            primary_score_eligible=np.zeros((1, 6), bool),
            value=np.full((1, 6), np.nan),
        ).items():
            f[key] = value
    with h5py.File(paths["legacy"], "w") as f:
        for key, value in {
            "features/source_id": [9],
            "features/values": raw[None],
            "features/input_source": [2],
            "features/input_original_row": [4],
            "reference_scaler/center": np.zeros(width),
            "reference_scaler/scale": np.ones(width),
        }.items():
            f[key] = value
    with h5py.File(paths["native"], "w") as f:
        f["source_id"] = np.array([], dtype="i8")
    frozen = tmp_path / "frozen.npz"
    np.savez(frozen, source_id=np.array([], dtype="i8"))
    config = dict(
        source_ids=[9],
        verify_source_sha256=False,
        frozen=dict(path=str(frozen), sha256="unused"),
    )
    config.update({n: dict(path=str(p), sha256="unused") for n, p in paths.items()})
    result = audit.audit(config, tmp_path / "audit")
    assert result["status"] == "incomplete"
    assert result["stars"][0]["xp_visible_count"] == 110
    assert (
        "legacy cohort exact Gaia ID/110 raw XP values" in result["stars"][0]["checks"]
    )
    assert (tmp_path / "audit" / "selected-arrays.npz").is_file()


def test_real_fits_record_uses_table_schema_not_numpy_dtype(tmp_path):
    from astropy.io import fits

    path = tmp_path / "original.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="source_id",
                format="K",
                array=np.array([4185355646116022528], dtype="i8"),
            ),
            fits.Column(name="G", format="E", array=np.array([12.5], dtype="f4")),
            fits.Column(name="bp_1", format="E", array=np.array([7.25], dtype="f4")),
        ]
    ).writeto(path)
    with fits.open(path, memmap=True) as hdul:
        row = hdul[1].data[0]
        assert not hasattr(row, "dtype")
        values = audit.raw_record_values(
            row, ["G", "bp_1", "rp_1"], hdul[1].columns.names
        )
        np.testing.assert_array_equal(values, [12.5, 7.25, np.nan])
