"""Tiny source-only fixtures; no model/checkpoint/data acquisition."""

import json

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.multisurvey_overlay import (
    build_overlay,
    file_sha256,
    open_verified_overlay,
    sparse_dataset_from_manifest,
    validate_overlay,
    write_observation_sidecar,
)


def metadata():
    return {
        "release": "synthetic-r1",
        "pipeline": "synthetic-iron-only",
        "dependency_id": "common-spectrum",
        "targets": {
            "teff": None,
            "logg": None,
            "fe_h": {"quantity": "fe_h", "unit": "dex", "field": "FEH"},
        },
    }


def records():
    return {
        "source_id": np.array([22, 11, 99], dtype="i8"),
        "source_row": np.array([8, 4, 7], dtype="i8"),
        "value": np.array(
            [[np.nan, np.nan, -2.0], [np.nan, np.nan, -0.5], [np.nan, np.nan, -3.1]]
        ),
        "error": np.array(
            [[np.nan, np.nan, 0.1], [np.nan, np.nan, np.nan], [np.nan, np.nan, 0.2]]
        ),
        "observed": np.array([[False, False, True]] * 3),
        "quality": np.array([1, 2, 3], dtype="i4"),
    }


def parent(path):
    with h5py.File(path, "x") as h5:
        h5["source_id"] = np.array([11, 22, 33], dtype="i8")
        h5["input"] = np.arange(12, dtype="f4").reshape(3, 4)
        h5["role"] = np.array([b"fit", b"tune", b"fit"])
        h5["primary_input_eligible"] = np.ones(3, dtype=bool)
        h5["scaler_scale"] = np.ones(4)


def fixture(tmp_path):
    source = tmp_path / "parent.h5"
    parent(source)
    side = write_observation_sidecar(
        tmp_path / "side.h5",
        {"extra": records()},
        {"extra": metadata()},
        provenance={"origin": "synthetic fixture", "independence": "not claimed"},
    )
    manifest = build_overlay(
        tmp_path / "overlay.h5",
        parent_path=source,
        parent_sha256=file_sha256(source),
        parent_registry={},
        sidecars=[side],
    )
    return source, side, manifest


def test_links_preserve_feature_rows_and_partial_observations(tmp_path):
    source, side, manifest_path = fixture(tmp_path)
    manifest = validate_overlay(manifest_path)
    with open_verified_overlay(manifest_path) as overlay, h5py.File(source) as original:
        assert isinstance(overlay.get("input", getlink=True), h5py.ExternalLink)
        np.testing.assert_array_equal(overlay["input"][:], original["input"][:])
        np.testing.assert_array_equal(overlay["source_id"][:], [11, 22, 33])
        group = overlay["native_observations/extra"]
        np.testing.assert_array_equal(group["source_id"][:], [22, 11, 99])
        np.testing.assert_array_equal(group["source_row"][:], [8, 4, 7])
        assert list(group.attrs["source_fields"]) == ["", "", "FEH"]
        assert list(group.attrs["target_units"]) == ["K", "dex(cm/s2)", "dex"]
        assert np.isnan(group["error"][1, 2])
        assert "synthetic fixture" in group.attrs["origin_json"]
    assert manifest["registry"]["extra"]["dependency_id"] == "common-spectrum"
    assert manifest["sidecars"][0] == side


def test_overlay_consumed_by_sparse_reader_without_copy(tmp_path):
    from masked_stellar_autoencoder.training.multisurvey_data import (
        collate_multisurvey,
    )

    _, _, path = fixture(tmp_path)
    ds = sparse_dataset_from_manifest(
        path, expected_manifest_sha256=file_sha256(path), role="fit"
    )
    assert ds.unmatched_counts == {"extra": 1}
    assert ds[0]["observations"][0]["source_row"] == 4
    batch = collate_multisurvey([ds[0], ds[1]])
    assert batch["observed"].sum().item() == 1
    assert not batch["error_known"].any()
    ds.close()


@pytest.mark.parametrize("which", ["parent", "sidecar", "overlay"])
def test_tampered_file_is_rejected(tmp_path, which):
    source, side, path = fixture(tmp_path)
    target = {
        "parent": source,
        "sidecar": side["path"],
        "overlay": json.loads(path.read_text())["overlay"]["path"],
    }[which]
    with open(target, "ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_overlay(path)


def test_missing_attachment_and_broken_link_are_rejected(tmp_path):
    source, side, path = fixture(tmp_path)
    side_path = tmp_path / "side.h5"
    renamed = tmp_path / "missing.h5"
    side_path.rename(renamed)
    with pytest.raises(FileNotFoundError):
        validate_overlay(path)
    renamed.rename(side_path)
    manifest = json.loads(path.read_text())
    overlay_path = manifest["overlay"]["path"]
    with h5py.File(overlay_path, "r+") as h5:
        del h5["input"]
        h5["input"] = h5py.ExternalLink(str(source), "/nonexistent")
    # Even if an overlay digest was refreshed, its link must match the contract.
    manifest["overlay"]["sha256"] = file_sha256(overlay_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="external link differs"):
        validate_overlay(path)


def test_no_overwrite_or_source_collision(tmp_path):
    source, side, path = fixture(tmp_path)
    with pytest.raises(FileExistsError):
        write_observation_sidecar(
            side["path"],
            {"extra": records()},
            {"extra": metadata()},
            provenance={"test": True},
        )
    with pytest.raises(FileExistsError):
        build_overlay(
            tmp_path / "overlay.h5",
            parent_path=source,
            parent_sha256=file_sha256(source),
            parent_registry={},
            sidecars=[side],
        )
    another = write_observation_sidecar(
        tmp_path / "another.h5",
        {"extra": records()},
        {"extra": metadata()},
        provenance={"test": True},
    )
    with pytest.raises(ValueError, match="collision"):
        build_overlay(
            tmp_path / "collision.h5",
            parent_path=source,
            parent_sha256=file_sha256(source),
            parent_registry={},
            sidecars=[side, another],
        )
    assert not (tmp_path / "collision.h5").exists()


def test_reject_imprecise_ids_bad_mapping_and_unmapped_observation(tmp_path):
    for kind in ("float_id", "bad_unit", "unmapped", "zero_error"):
        data = records()
        meta = metadata()
        if kind == "float_id":
            data["source_id"] = data["source_id"].astype(float)
        elif kind == "bad_unit":
            meta["targets"]["fe_h"]["unit"] = "M/H"
        elif kind == "unmapped":
            data["observed"][0, 0] = True
            data["value"][0, 0] = 4000
        else:
            data["error"][0, 2] = 0
        with pytest.raises(ValueError):
            write_observation_sidecar(
                tmp_path / f"{kind}.h5",
                {"extra": data},
                {"extra": meta},
                provenance={"test": True},
            )


def test_legacy_parent_order_receipt_preserves_alternatives(tmp_path):
    source = tmp_path / "legacy.h5"
    parent(source)
    meta = metadata()
    meta["targets"] = {
        name: {"quantity": name, "unit": unit, "field": name}
        for name, unit in zip(
            ("teff", "logg", "fe_h"), ("K", "dex(cm/s2)", "dex"), strict=True
        )
    }
    with h5py.File(source, "r+") as h5:
        h5.attrs["schema"] = "remediation-sparse-six-target-source-v1"
        group = h5.create_group("native_observations/galah")
        for key, array in records().items():
            group[key] = array
    manifest = build_overlay(
        tmp_path / "legacy-overlay.h5",
        parent_path=source,
        parent_sha256=file_sha256(source),
        parent_registry={"galah": meta},
        sidecars=[],
    )
    with open_verified_overlay(manifest) as h5:
        group = h5["native_observations/galah"]
        assert list(group.attrs["source_fields"]) == ["teff", "logg", "fe_h"]
        assert isinstance(group.get("value", getlink=True), h5py.ExternalLink)
        assert "original_attrs" in json.loads(group.attrs["origin_json"])


def test_manifest_registry_cannot_relabel_observations(tmp_path):
    _, _, path = fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["registry"]["extra"]["targets"]["fe_h"]["field"] = "M_H"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="registry differs"):
        validate_overlay(path)


def test_bad_parent_digest_and_hidden_dependencies_rejected(tmp_path):
    source = tmp_path / "parent.h5"
    parent(source)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_overlay(
            tmp_path / "bad.h5",
            parent_path=source,
            parent_sha256="0" * 64,
            parent_registry={},
            sidecars=[],
        )
    with h5py.File(source, "r+") as h5:
        h5["hidden"] = h5py.ExternalLink("undeclared.h5", "/input")
    with pytest.raises(ValueError, match="undeclared"):
        build_overlay(
            tmp_path / "hidden.h5",
            parent_path=source,
            parent_sha256=file_sha256(source),
            parent_registry={},
            sidecars=[],
        )


def test_manifest_reader_rechecks_linked_contents(tmp_path):
    _, side, path = fixture(tmp_path)
    ds = sparse_dataset_from_manifest(
        path, expected_manifest_sha256=file_sha256(path), role="fit"
    )
    assert len(ds) == 2
    ds.close()
    with h5py.File(side["path"], "r+") as h5:
        h5["native_observations/extra/value"][0, 2] = -9.0
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        sparse_dataset_from_manifest(
            path, expected_manifest_sha256=file_sha256(path), role="fit"
        )


def test_changed_sidecar_cannot_use_direct_or_pinned_manifest_access(tmp_path):
    from masked_stellar_autoencoder.training.multisurvey_data import SparseSurveyDataset

    _, side, path = fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest_pin = file_sha256(path)
    with h5py.File(side["path"], "r+") as h5:
        h5["native_observations/extra/value"][0, 2] = -8.0
    # The overlay digest is unchanged, but never authorizes its dependencies.
    assert file_sha256(manifest["overlay"]["path"]) == manifest["overlay"]["sha256"]
    with pytest.raises(ValueError, match="pinned manifest"):
        SparseSurveyDataset(
            manifest["overlay"]["path"],
            manifest["registry"],
            role="fit",
            expected_source_sha256=manifest["overlay"]["sha256"],
        )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        sparse_dataset_from_manifest(
            path, expected_manifest_sha256=manifest_pin, role="fit"
        )
    # Updating the manifest to match changed labels must not defeat its old pin.
    manifest["sidecars"][0]["sha256"] = file_sha256(side["path"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Manifest SHA256 mismatch"):
        sparse_dataset_from_manifest(
            path, expected_manifest_sha256=manifest_pin, role="fit"
        )


def test_pin_is_required_and_late_mutation_invalidates_verified_reader(tmp_path):
    _, side, path = fixture(tmp_path)
    with pytest.raises(TypeError, match="expected_manifest_sha256"):
        sparse_dataset_from_manifest(path, role="fit")
    ds = sparse_dataset_from_manifest(
        path, expected_manifest_sha256=file_sha256(path), role="fit"
    )
    with h5py.File(side["path"], "r+") as h5:
        h5["native_observations/extra/value"][0, 2] = -8.0
    with pytest.raises(ValueError, match="attachment changed"):
        ds[0]
    ds.close()


def test_overlay_is_hashed_once_and_pickled_receipt_still_checks(tmp_path, monkeypatch):
    import pickle

    from masked_stellar_autoencoder.training import multisurvey_overlay as utility

    _, _, path = fixture(tmp_path)
    manifest = json.loads(path.read_text())
    calls = []
    original = utility.file_sha256

    def counted(file):
        calls.append(str(file))
        return original(file)

    monkeypatch.setattr(utility, "file_sha256", counted)
    ds = utility.sparse_dataset_from_manifest(
        path, expected_manifest_sha256=original(path), role="fit"
    )
    assert calls.count(manifest["overlay"]["path"]) == 1
    assert calls.count(manifest["parent"]["path"]) == 1
    assert calls.count(manifest["sidecars"][0]["path"]) == 1
    restored = pickle.loads(pickle.dumps(ds))
    assert restored[0]["source_id"] == 11
    restored.close()
    ds.close()


def test_removing_schema_does_not_allow_direct_external_link_reader(tmp_path):
    from masked_stellar_autoencoder.training.multisurvey_data import SparseSurveyDataset

    _, _, path = fixture(tmp_path)
    manifest = json.loads(path.read_text())
    overlay = manifest["overlay"]["path"]
    with h5py.File(overlay, "r+") as h5:
        del h5.attrs["schema"]
    with pytest.raises(ValueError, match="pinned manifest"):
        SparseSurveyDataset(
            overlay,
            manifest["registry"],
            role="fit",
            expected_source_sha256=file_sha256(overlay),
        )
