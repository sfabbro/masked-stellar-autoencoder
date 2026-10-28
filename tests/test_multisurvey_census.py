"""Independent synthetic tests of ID union, roles and cache membership."""

import hashlib

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.multisurvey_census import (
    join_base,
    scan_cache,
    source_union,
    summarize,
    verify_cache_associations,
)


def test_union_retains_duplicate_rows_and_partial_labels():
    ids, counts, observed = source_union(
        [5, 3, 5, 7], np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=bool)
    )
    np.testing.assert_array_equal(ids, [3, 5, 7])
    np.testing.assert_array_equal(counts, [1, 2, 1])
    np.testing.assert_array_equal(observed, [[0, 1, 0], [1, 0, 1], [1, 1, 1]])
    with pytest.raises(ValueError, match="exact integer"):
        source_union([3.0], np.array([[True, False, False]]))


def test_base_roles_and_unknown_cache_membership_are_separate():
    ids = np.array([3, 5, 7, 9], dtype="i8")
    base = np.array([3, 7, 10], dtype="i8")
    b, role, primary, exposed, reserved = join_base(
        ids,
        base,
        ["fit", "devtest", "cal"],
        np.array([1, 1, 0], bool),
        np.array([0, 1, 0], bool),
        np.array([0, 1, 0], bool),
    )
    assert role.tolist() == ["fit", "not_in_base", "devtest", "not_in_base"]
    cache_ids = np.array([4, 3, 5, 7, 12], dtype="i8")
    features = np.array([[1, 1], [1, 2], [1, -9999], [2, 3], [1, 1]], dtype="f4")
    rows, xp, digest, invalid = scan_cache(
        ids, cache_ids, features, [0, 1], chunk_rows=2, hash_prefix=b"header"
    )
    np.testing.assert_array_equal(rows, [1, 2, 3, -1])
    np.testing.assert_array_equal(xp, [1, 0, 1, 0])
    assert invalid == 0
    assert digest == hashlib.sha256(b"header" + cache_ids.tobytes()).hexdigest()
    summary = summarize(
        ids,
        np.ones(4, int),
        np.ones((4, 3), bool),
        b,
        role,
        primary,
        exposed,
        reserved,
        rows,
        xp,
    )
    assert summary["base_role_counts"]["devtest"] == 1
    assert summary["cache_matched_ids"] == 3
    assert summary["cache_complete_xp_not_in_base"] == 0
    assert summary["base_known_supervised_ancestor_ids"] == 1
    _, _, _, invalid = scan_cache(
        ids, np.array([0, -1, 3], dtype="i8"), np.ones((3, 2)), [0]
    )
    assert invalid == 2
    with pytest.raises(ValueError, match="Repeated candidate"):
        scan_cache(
            ids, np.array([3, 3], dtype="i8"), np.ones((2, 2)), [0], chunk_rows=1
        )


def test_cached_sample_association_against_raw_source(tmp_path):
    path = tmp_path / "original.h5"
    rows = np.array(
        [(3, 1.0, 2.0), (5, 3.0, 4.0)],
        dtype=[("source_id", "i8"), ("bp_1", "f4"), ("rp_1", "f4")],
    )
    with h5py.File(path, "w") as h:
        h["train"] = rows
    stat = path.stat()
    request = {
        "source": {
            "path": str(path.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "train_keys": ["train"],
        "validation_keys": [],
        "schema": {"train": {"shape": [2]}},
    }
    features = np.array([[1, 2], [3, 4]], dtype="f4")
    prov = np.array([(0, 0), (0, 1)], dtype=[("shard", "<u4"), ("row", "<i8")])
    assert (
        verify_cache_associations(
            path,
            np.array([3, 5]),
            np.array([0, 1]),
            features,
            prov,
            ["bp_1", "rp_1"],
            np.zeros(2),
            np.ones(2),
            request,
        )
        == 2
    )
    features[1, 0] += 1
    with pytest.raises(AssertionError):
        verify_cache_associations(
            path,
            np.array([3, 5]),
            np.array([0, 1]),
            features,
            prov,
            ["bp_1", "rp_1"],
            np.zeros(2),
            np.ones(2),
            request,
        )


def test_end_to_end_census_preserves_membership_and_roles(tmp_path, monkeypatch):
    import json

    import masked_stellar_autoencoder.training.multisurvey_census as module

    sidecar, base = tmp_path / "side.h5", tmp_path / "base.h5"
    cache = tmp_path / "cache"
    cache.mkdir()
    names = [f"{arm}_{j}" for arm in ("bp", "rp") for j in range(1, 56)]
    ids = np.array([1, 2, 4], dtype="i8")
    features = np.zeros((3, 110), dtype="f4")
    provenance = np.array(
        [(0, 0), (0, 1), (0, 2)], dtype=[("shard", "<u4"), ("row", "<i8")]
    )
    np.save(cache / "source_id.npy", ids)
    np.save(cache / "features.npy", features)
    np.save(cache / "provenance.npy", provenance)
    (cache / "manifest.json").write_text("{}")
    original = tmp_path / "original.h5"
    raw = np.zeros(3, dtype=[("source_id", "i8"), *[(name, "f4") for name in names]])
    raw["source_id"] = ids
    with h5py.File(original, "w") as h:
        h["train"] = raw
    stat = original.stat()
    manifest = {
        "files": {
            "source_id.npy": {
                "sha256": hashlib.sha256(
                    (cache / "source_id.npy").read_bytes()
                ).hexdigest()
            }
        },
        "request": {
            "source": {
                "path": str(original.resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            "train_keys": ["train"],
            "validation_keys": [],
            "schema": {"train": {"shape": [3]}},
        },
    }

    def mapped(*args):
        return (
            np.load(cache / "features.npy", mmap_mode="r"),
            np.load(cache / "source_id.npy", mmap_mode="r"),
            np.load(cache / "provenance.npy", mmap_mode="r"),
            manifest,
        )

    monkeypatch.setattr(module, "_cache_maps", mapped)
    with h5py.File(sidecar, "w") as h:
        g = h.create_group("native_observations/desi_dr1_rvs_raw")
        g["source_id"] = np.array([1, 2, 2], dtype="i8")
        g["observed"] = np.ones((3, 3), dtype=bool)
    with h5py.File(base, "w") as h:
        h["source_id"] = np.array([1, 3], dtype="i8")
        h["role"] = np.array([b"fit", b"test"])
        h["primary_input_eligible"] = np.array([True, True])
        h["known_supervised_ancestor_exposed"] = np.array([True, False])
        h["reserved_reference"] = np.array([False, True])
        h["feature_names"] = np.array(names, dtype="S8")
        h["scaler_center"] = np.zeros(110)
        h["scaler_scale"] = np.ones(110)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = module.census(
        sidecar,
        base,
        cache,
        original_path=original,
        sidecar_sha=sha(sidecar),
        base_sha=sha(base),
        output=tmp_path / "output",
        expected_cache_rows=3,
        chunk_rows=2,
    )
    assert result["unique_desi_ids"] == 2 and result["quality_rows"] == 3
    assert result["cache_complete_xp_not_in_base"] == 1
    assert result["cache_raw_association_rows_verified"] == 2
    with np.load(tmp_path / "output/identities.npz") as data:
        assert data["base_role"].tolist() == ["fit", "not_in_base"]
    assert json.loads((tmp_path / "output/summary.json").read_text()) == result
    with pytest.raises(FileExistsError):
        module.census(
            sidecar,
            base,
            cache,
            original_path=original,
            sidecar_sha=sha(sidecar),
            base_sha=sha(base),
            output=tmp_path / "output",
        )
