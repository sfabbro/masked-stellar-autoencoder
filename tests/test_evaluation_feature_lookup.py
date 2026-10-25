"""Synthetic source-only lookup contracts; no neural model imports or weights."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
    lookup_evaluation_features,
    match_ids,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def files(tmp_path):
    names = (
        ["W1", "W2", "G", "BP", "RP"]
        + [f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)]
        + ["PARALLAX"]
    )
    n = len(names)
    center, scale = np.linspace(-1, 1, n), np.linspace(1, 3, n)

    def record(ids):
        result = np.zeros(
            len(ids), dtype=[("source_id", "i8"), *[(x, "f8") for x in names]]
        )
        result["source_id"] = ids
        for i, name in enumerate(names):
            result[name] = 10 + np.asarray(ids) / 100 + i / 200
        return result

    original = tmp_path / "original.h5"
    train, valid = record([20, 30, 70]), record([40, 50])
    for name in names[5:-1]:
        valid[name][0] = np.nan
    with h5py.File(original, "w") as h:
        h["train"] = train
        h["valid"] = valid
    legacy = tmp_path / "legacy.h5"
    raw = np.column_stack([record([10, 20])[name].astype("f4") for name in names])
    raw[1, 5:-1] = np.nan
    with h5py.File(legacy, "w") as h:
        h.attrs["feature_columns_json"] = json.dumps(names)
        h["features/source_id"] = [10, 20]
        h["features/values"] = raw
        h["features/reported_feature_errors"] = np.full_like(raw, 0.01)
    cache = tmp_path / "cache"
    cache.mkdir()
    values = np.column_stack([train[name][:2].astype("f4") for name in names])
    np.save(cache / "features.npy", ((values - center) / scale).astype("f4"))
    np.save(cache / "source_id.npy", np.array([20, 30], dtype="i8"))
    np.save(
        cache / "provenance.npy",
        np.array([(0, 0), (0, 1)], dtype=[("shard", "<u4"), ("row", "<i8")]),
    )
    request = dict(
        rows=2,
        features=names,
        center=center.tolist(),
        scale=scale.tolist(),
        train_keys=["train"],
        validation_keys=["valid"],
        source=dict(
            path=str(original.resolve()),
            bytes=original.stat().st_size,
            mtime_ns=original.stat().st_mtime_ns,
        ),
        schema={"train": {"shape": [3]}, "valid": {"shape": [2]}},
    )
    request_sha = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    manifest = dict(
        status="complete",
        format_version=1,
        sample_source_ids_unique=True,
        original_validation_ids_excluded=True,
        request=request,
        request_sha256=request_sha,
        files={},
    )
    for name in ["features.npy", "source_id.npy", "provenance.npy"]:
        path = cache / name
        manifest["files"][name] = dict(
            bytes=path.stat().st_size,
            sha256=digest(path),
            shape=list(np.load(path, allow_pickle=False).shape),
        )
    (cache / "manifest.json").write_text(json.dumps(manifest))
    return dict(
        ids=np.array([50, 20, 10, 30, 40, 60], dtype="i8"),
        names=names,
        center=center,
        scale=scale,
        legacy_path=legacy,
        cache_path=cache,
        original_path=original,
        expected_cache_request_sha256=request_sha,
        expected_cache_rows=2,
        chunk_rows=2,
    )


def test_id_join_exact_integer_and_original_order():
    ids = np.array([10, 20, 30], dtype="i8")
    q, dest = match_ids(
        ids, np.array([2, 0, 1]), np.array([30, 0, 10, 999], dtype="i8")
    )
    np.testing.assert_array_equal(q, [0, 2])
    np.testing.assert_array_equal(dest, [1, 2])
    with pytest.raises(ValueError, match="Integer"):
        match_ids(ids, np.arange(3), np.array([10.0]))


def test_legacy_cache_raw_fallback_and_no_astrometry(files):
    arrays, proof = lookup_evaluation_features(**files)
    assert arrays["feature_origin"].tolist() == [
        "original_raw_hdf5",
        "original150m_cache",
        "legacy_cohort",
        "original150m_cache",
        "original_raw_hdf5",
        "unavailable",
    ]
    assert arrays["cache_row"].tolist() == [-1, 0, -1, 1, -1, -1]
    assert arrays["feature_recoverable"].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert np.all(arrays["input"][:, -1] == -9999)
    assert not arrays["visible"][:, -1].any()
    assert arrays["raw_reconstructed_from_normalized_cache"].tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
    ]
    assert proof["cache_raw_association_rows_verified"] == 2
    assert not proof["full_cache_feature_sha_recomputed"]
    cached = np.load(files["cache_path"] / "features.npy")
    np.testing.assert_array_equal(arrays["input"][[1, 3], :-1], cached[:, :-1])


def test_missing_xp_is_not_official_absence(files, monkeypatch):
    import masked_stellar_autoencoder.training.evaluation_feature_lookup as module

    def vac(spec, ids, raw, errors, names):
        del spec, errors, names
        published, known = np.zeros(len(ids), bool), ids == 40
        raw[ids == 40, :5] = 12
        return published, known, np.where(known, 0, -1), {"synthetic": True}

    monkeypatch.setattr(module, "_vac_projection", vac)
    arrays, _ = lookup_evaluation_features(**files, gaia_vac={"test": True})
    assert arrays["xp_status"][4] == "official_no_xp"
    assert arrays["feature_recoverable"][4]
    assert not arrays["real_no_xp"][5]
    assert arrays["xp_status"][5] == "unknown"


def test_cache_tamper_fails_id_sha(files):
    path = files["cache_path"] / "source_id.npy"
    ids = np.load(path)
    ids[1] = 99
    np.save(path, ids)
    with pytest.raises(ValueError, match="content SHA"):
        lookup_evaluation_features(**files)


def test_bound_memmap_resume_and_changed_scaler_rejected(files, tmp_path):
    directory = tmp_path / "output"
    a, first = lookup_evaluation_features(**files, output_directory=directory)
    saved = np.array(a["input"])
    b, resumed = lookup_evaluation_features(**files, output_directory=directory)
    np.testing.assert_array_equal(saved, b["input"])
    assert first["original_id_rows_scanned"] == resumed["original_id_rows_scanned"]
    changed = dict(files, center=files["center"] + 0.1)
    with pytest.raises(ValueError, match="binding"):
        lookup_evaluation_features(**changed, output_directory=directory)


def test_real_photometry_unknown_publication_and_release_domain(files):
    fallback = dict(
        source_id=files["ids"].copy(),
        identity_verified=np.ones(6, bool),
        names=["G", "BP", "RP"],
        values=np.full((6, 3), 12.0),
        release="Gaia DR2",
        provenance={"source": "synthetic verified-ID catalogue"},
    )
    arrays, proof = lookup_evaluation_features(
        **files, photometry_fallback=fallback, allow_unknown_photometry=True
    )
    assert arrays["feature_recoverable"][5]
    assert arrays["xp_status"][5] == "photometry_only_publication_unknown"
    assert not arrays["real_no_xp"][5]
    assert arrays["photometry_is_domain_transfer"][5]
    assert arrays["photometry_release_code"][5] == 2
    assert np.isnan(arrays["feature_errors"][5]).all()
    assert proof["photometry_fallback"]["filled_star_count"] == 1
    # Genuine source photometry in the original HDF5 remains observable even
    # without publication metadata, but not declared genuinely no-published-XP.
    assert arrays["feature_recoverable"][4]
    assert not arrays["real_no_xp"][4]


def test_unverified_float_ids_and_bad_magnitudes_are_not_promoted(files):
    fallback = dict(
        source_id=files["ids"].astype(float),
        identity_verified=np.ones(6, bool),
        names=["G", "BP", "RP"],
        values=np.full((6, 3), 12.0),
    )
    with pytest.raises(ValueError, match="exact integers"):
        lookup_evaluation_features(
            **files, photometry_fallback=fallback, allow_unknown_photometry=True
        )
    fallback["source_id"] = files["ids"]
    fallback["values"][-1, 0] = 999
    arrays, _ = lookup_evaluation_features(
        **files, photometry_fallback=fallback, allow_unknown_photometry=True
    )
    assert not arrays["feature_recoverable"][-1]
    fallback["values"][-1, 0] = 12
    fallback["identity_verified"][-1] = False
    arrays, _ = lookup_evaluation_features(
        **files, photometry_fallback=fallback, allow_unknown_photometry=True
    )
    assert not arrays["feature_recoverable"][-1]
