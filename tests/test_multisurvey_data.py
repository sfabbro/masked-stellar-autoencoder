"""Synthetic full-union, partial-label and provenance checks."""

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.multisurvey_data import (
    SparseSurveyDataset,
    collate_multisurvey,
)


def make_source(path, duplicate=False):
    with h5py.File(path, "w") as h:
        h["source_id"] = np.array([11, 22, 33, 44], dtype="i8")
        h["input"] = np.array([[1.0, 2.0], [3.0, -9999.0], [4.0, 5.0], [6.0, 7.0]])
        h["role"] = np.array([b"fit", b"fit", b"fit", b"tune"])
        h["primary_input_eligible"] = np.ones(4, dtype=bool)
        g = h.create_group("native_observations")
        for name, ids, masks in [
            (
                "a",
                [11, 22] if not duplicate else [11, 11, 22],
                [[1, 1, 0], [0, 1, 1]]
                if not duplicate
                else [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
            ),
            ("b", [22, 33, 99], [[1, 0, 1], [0, 0, 1], [1, 1, 1]]),
        ]:
            grp = g.create_group(name)
            grp.attrs["source_fields"] = ["teff", "logg", "fe_h"]
            grp["source_id"] = np.asarray(ids, dtype="i8")
            grp["source_row"] = np.arange(len(ids), dtype="i8") + 10
            grp["value"] = np.full((len(ids), 3), 0.5)
            error = np.full((len(ids), 3), 0.1)
            error[-1, -1] = np.nan
            grp["error"] = error
            grp["observed"] = np.array(masks, dtype=bool)


def registry():
    fields = {
        name: {"field": name, "quantity": name, "unit": unit}
        for name, unit in (("teff", "K"), ("logg", "dex(cm/s2)"), ("fe_h", "dex"))
    }
    return {
        s: {"release": "r1", "pipeline": "p1", "dependency_id": s, "targets": fields}
        for s in ("a", "b")
    }


def test_full_union_and_overlap_without_intersection(tmp_path):
    p = tmp_path / "source.h5"
    make_source(p)
    ds = SparseSurveyDataset(p, registry(), role="fit")
    assert len(ds) == 3
    stars = [ds[i] for i in range(3)]
    assert [len(s["observations"]) for s in stars] == [1, 2, 1]
    assert ds.unmatched_counts["b"] == 1
    batch = collate_multisurvey(stars)
    assert batch["source_id"].tolist() == [11, 22, 33]
    assert batch["observed"].sum().item() == 7
    assert not batch["input_available"][1, 1]
    assert batch["source_row"][0, 0] == 10
    assert batch["provenance"][1][0]["release"] == "r1"
    ds.close()


def test_duplicate_dependencies_remain_visible_but_cannot_be_silently_combined(
    tmp_path,
):
    p = tmp_path / "source.h5"
    make_source(p, duplicate=True)
    ds = SparseSurveyDataset(p, registry(), role="fit")
    assert len(ds[0]["observations"]) == 2
    with pytest.raises(ValueError, match="dependent or repeated"):
        collate_multisurvey([ds[0]])
    ds.close()


def test_unknown_error_is_not_declared_known_and_shared_dependencies_rejected(tmp_path):
    p = tmp_path / "source.h5"
    make_source(p)
    ds = SparseSurveyDataset(p, registry(), role="fit")
    record = ds[1]["observations"][0]
    assert record["observed"][2] and not record["error_known"][2]
    assert np.isnan(record["error"][2])
    bad = registry()
    bad["b"]["dependency_id"] = "a"
    ds2 = SparseSurveyDataset(p, bad, role="fit")
    with pytest.raises(ValueError, match="dependent or repeated"):
        collate_multisurvey([ds2[1]])
    ds.close()
    ds2.close()


def test_duplicate_star_ids_and_missing_registered_source_are_rejected(tmp_path):
    p = tmp_path / "source.h5"
    make_source(p)
    with pytest.raises(ValueError, match="not present"):
        SparseSurveyDataset(
            p, {"missing": {**registry()["a"], "dependency_id": "d"}}, role="fit"
        )
    with h5py.File(p, "r+") as h:
        h["source_id"][1] = h["source_id"][0]
    with pytest.raises(ValueError, match="unique"):
        SparseSurveyDataset(p, registry(), role="fit")


def test_registry_and_role_cannot_be_mixed_and_anchor_index_is_negative(tmp_path):
    p = tmp_path / "source.h5"
    make_source(p)
    a = SparseSurveyDataset(p, registry(), role="fit")
    different = registry()
    different["a"]["release"] = "other"
    b = SparseSurveyDataset(p, different, role="fit")
    with pytest.raises(ValueError, match="registry_sha256"):
        collate_multisurvey([a[0], b[1]])
    anchor_reg = registry()
    anchor_reg["b"]["is_anchor"] = True
    anchored = SparseSurveyDataset(p, anchor_reg, role="fit")
    record = anchored[2]["observations"][0]
    assert record["is_anchor"] and record["source_index"] == -1
    assert collate_multisurvey([anchored[2]])["source_index"][0, 0] == -1
    a.close()
    b.close()
    anchored.close()


def test_field_order_and_global_metallicity_cannot_be_mislabeled(tmp_path):
    p = tmp_path / "source.h5"
    make_source(p)
    bad = registry()
    bad["a"]["targets"] = {
        **bad["a"]["targets"],
        "fe_h": {"field": "M_H", "quantity": "fe_h", "unit": "dex"},
    }
    with pytest.raises(ValueError, match="not direct Fe/H"):
        SparseSurveyDataset(p, bad, role="fit")
    with h5py.File(p, "r+") as h:
        h["native_observations/a"].attrs["source_fields"] = ["logg", "teff", "fe_h"]
    with pytest.raises(ValueError, match="field order"):
        SparseSurveyDataset(p, registry(), role="fit")


def test_digest_bound_source_changed_after_init_is_rejected(tmp_path):
    import hashlib

    path = tmp_path / "source.h5"
    make_source(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ds = SparseSurveyDataset(
        path, registry(), role="fit", expected_source_sha256=digest
    )
    with h5py.File(path, "r+") as h:
        h["input"][0, 0] = 42.0
    with pytest.raises(ValueError, match="changed after SHA256 verification"):
        ds[0]
    ds.close()


def test_digest_bound_source_changed_during_hash_is_rejected(tmp_path, monkeypatch):
    import hashlib
    import os

    path = tmp_path / "source.h5"
    make_source(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = hashlib.file_digest

    def mutate_during_digest(stream, algorithm):
        result = original(stream, algorithm)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(hashlib, "file_digest", mutate_during_digest)
    with pytest.raises(ValueError, match="changed during SHA256 verification"):
        SparseSurveyDataset(path, registry(), role="fit", expected_source_sha256=digest)


def test_registry_is_isolated_and_mutation_is_detected(tmp_path):
    path = tmp_path / "source.h5"
    make_source(path)
    supplied = registry()
    ds = SparseSurveyDataset(path, supplied, role="fit")
    supplied["a"]["targets"]["logg"]["field"] = "wrong"
    returned = ds[0]
    assert returned["observations"][0]["release"] == "r1"
    returned["observations"][0]["target_definitions"]["logg"]["field"] = "wrong"
    assert ds[0]["observations"][0]["target_definitions"]["logg"]["field"] == "logg"
    ds.registry["a"]["targets"]["logg"]["field"] = "wrong"
    with pytest.raises(ValueError, match="registry changed"):
        ds[0]
    ds.close()
