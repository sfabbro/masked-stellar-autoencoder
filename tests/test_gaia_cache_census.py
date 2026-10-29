"""Synthetic source, exact-ID, quality-mask and cache receipt checks."""

import hashlib
import json

import h5py
import numpy as np
import pytest
from astropy.io import fits

from masked_stellar_autoencoder.training import multisurvey_gaia_cache_census as module

BIG1 = 9_007_199_254_740_993
BIG2 = 9_007_199_254_740_995


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_data(tmp_path, monkeypatch):
    source = tmp_path / "gaia.fits.gz"
    base = tmp_path / "base.h5"
    original = tmp_path / "original.h5"
    cache = tmp_path / "cache"
    cache.mkdir()

    columns = [
        fits.Column(
            name="source_id", format="K", array=np.array([BIG1, 17, BIG2, 30], "i8")
        ),
        fits.Column(name="teff_gspspec", format="D", array=[5000, np.nan, 5000, 909]),
        fits.Column(name="logg_gspspec", format="D", array=[4, 3, np.nan, 4]),
        fits.Column(name="mh_gspspec", format="D", array=[-1, -0.5, np.nan, 0]),
        fits.Column(name="alphafe_gspspec", format="D", array=[0.1, np.nan, 0.3, 0.2]),
        fits.Column(name="flags_gspspec", format="41A", array=["0" * 41] * 4),
    ]
    # Explicit file object forces uncompressed FITS bytes despite the .gz suffix.
    with source.open("wb") as stream:
        fits.BinTableHDU.from_columns(columns).writeto(stream)

    names = [f"{arm}_{j}" for arm in ("bp", "rp") for j in range(1, 56)]
    with h5py.File(base, "w") as h:
        h["source_id"] = np.array([17, 42, BIG1], dtype="i8")
        h["role"] = np.array([b"fit", b"excluded", b"devtest"])
        h["primary_input_eligible"] = np.array([True, False, True])
        h["known_supervised_ancestor_exposed"] = np.array([False, True, True])
        h["reserved_reference"] = np.array([False, True, True])
        h["feature_names"] = np.array(names, dtype="S8")
        h["scaler_center"] = np.zeros(110)
        h["scaler_scale"] = np.ones(110)

    cache_ids = np.array([17, BIG2, 30, 88], dtype="<i8")
    raw = np.zeros(4, dtype=[("source_id", "<i8"), *[(name, "<f4") for name in names]])
    raw["source_id"] = cache_ids
    raw["bp_1"][2] = -9999
    with h5py.File(original, "w") as h:
        h["train"] = raw
    features = np.column_stack([raw[name] for name in names]).astype("<f4")
    provenance = np.array(
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        dtype=[("shard", "<u4"), ("row", "<i8")],
    )
    np.save(cache / "source_id.npy", cache_ids)
    np.save(cache / "features.npy", features)
    np.save(cache / "provenance.npy", provenance)
    (cache / "manifest.json").write_text("{}")
    stat = original.stat()
    manifest = {
        "files": {"source_id.npy": {"sha256": sha(cache / "source_id.npy")}},
        "request": {
            "source": {
                "path": str(original.resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            "train_keys": ["train"],
            "validation_keys": [],
            "schema": {"train": {"shape": [4]}},
        },
    }

    def mapped(directory, actual_names, center, scale, request_sha, expected_rows):
        assert directory == cache and actual_names == names and expected_rows == 4
        np.testing.assert_array_equal(center, np.zeros(110))
        np.testing.assert_array_equal(scale, np.ones(110))
        assert request_sha == module.CACHE_REQUEST_SHA
        return (
            np.load(cache / "features.npy", mmap_mode="r"),
            np.load(cache / "source_id.npy", mmap_mode="r"),
            np.load(cache / "provenance.npy", mmap_mode="r"),
            manifest,
        )

    monkeypatch.setattr(module, "_cache_maps", mapped)
    return {
        "source": source,
        "base": base,
        "cache": cache,
        "original_path": original,
        "source_sha": sha(source),
        "base_sha": sha(base),
        "expected_source_rows": 4,
        "expected_cache_rows": 4,
        "chunk_rows": 2,
        "manifest": manifest,
    }


def run_census(d, output):
    args = {k: v for k, v in d.items() if k != "manifest"}
    return module.gaia_cache_census(**args, output=output)


def test_source_union_roles_unknown_cache_states_and_hashes(source_data, tmp_path):
    output = tmp_path / "out"
    summary = run_census(source_data, output)
    assert summary["source_rows"] == 4 and summary["source_only"]
    assert summary["target_order"] == list(module.TARGETS)
    assert summary["counts"]["all_source_rows"]["selected"] == {
        "teff": 2,
        "logg": 2,
        "mh_gspspec": 2,
        "alphafe_gspspec": 2,
    }
    assert summary["counts"]["outside_base"]["stars"] == 2
    assert summary["counts"]["complete_xp_in_cache_outside_base"]["stars"] == 1
    assert summary["matched_base_roles"]["fit"]["selected_primary_input"]["logg"] == 1
    assert summary["cache_raw_association_rows_verified"] == 3
    assert summary["identities_sha256"] == sha(output / "identities.npz")
    assert summary["identities_crc"] == summary["identities_roundtrip"] == "pass"
    assert json.loads((output / "summary.json").read_text()) == summary
    with np.load(output / "identities.npz", allow_pickle=False) as z:
        assert z["source_id"].dtype == np.dtype("<i8")
        assert z["source_id"].tolist() == [17, 30, BIG1, BIG2]
        assert z["source_row"].tolist() == [1, 3, 0, 2]
        assert z["raw_available"].tolist() == [
            [False, True, True, False],
            [True, True, True, True],
            [True, True, True, True],
            [True, False, False, True],
        ]
        assert z["selected_usable"].tolist() == [
            [False, True, True, False],
            [False, False, False, False],
            [True, True, True, True],
            [True, False, False, True],
        ]
        assert z["base_match"].tolist() == [True, False, True, False]
        assert z["base_role"].tolist() == [
            "fit",
            "not_in_base",
            "devtest",
            "not_in_base",
        ]
        assert z["base_ancestor_listed_state"].tolist() == [0, -1, 1, -1]
        assert z["base_reserved_state"].tolist() == [0, -1, 1, -1]
        assert z["base_primary_state"].tolist() == [1, -1, 1, -1]
        assert z["cache_row"].tolist() == [0, 2, -1, 1]
        assert z["cache_xp_state"].tolist() == [1, 0, -1, 1]
    with pytest.raises(FileExistsError):
        run_census(source_data, output)


def test_bad_source_base_or_cache_digests_fail_closed(source_data, tmp_path):
    data = dict(source_data, source_sha="0" * 64)
    with pytest.raises(ValueError, match="content"):
        run_census(data, tmp_path / "bad-source")
    data = dict(source_data, base_sha="0" * 64)
    with pytest.raises(ValueError, match="content"):
        run_census(data, tmp_path / "bad-base")
    source_data["manifest"]["files"]["source_id.npy"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Cache source-ID"):
        run_census(source_data, tmp_path / "bad-cache")


def test_duplicate_source_and_duplicate_cache_ids_fail_closed(source_data, tmp_path):
    source = source_data["source"]
    with fits.open(source, mode="update", memmap=False) as h:
        h[1].data["source_id"][0] = 17
        h.flush()
    source_data["source_sha"] = sha(source)
    with pytest.raises(ValueError, match="unique"):
        run_census(source_data, tmp_path / "bad-source-duplicate")

    # Restore the source and make a separate duplicated cache candidate.
    with fits.open(source, mode="update", memmap=False) as h:
        h[1].data["source_id"][0] = BIG1
        h.flush()
    source_data["source_sha"] = sha(source)
    ids = np.load(source_data["cache"] / "source_id.npy").copy()
    ids[3] = 17
    np.save(source_data["cache"] / "source_id.npy", ids)
    source_data["manifest"]["files"]["source_id.npy"]["sha256"] = sha(
        source_data["cache"] / "source_id.npy"
    )
    with pytest.raises(ValueError, match="Repeated candidate"):
        run_census(source_data, tmp_path / "bad-cache-duplicate")


def test_cache_feature_must_match_original_source(source_data, tmp_path):
    file = source_data["cache"] / "features.npy"
    features = np.load(file).copy()
    features[0, 0] = 2.0
    np.save(file, features)
    with pytest.raises(AssertionError):
        run_census(source_data, tmp_path / "bad-association")
