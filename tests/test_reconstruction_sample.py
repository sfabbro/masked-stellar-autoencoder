"""Exact original-shard staging gates using small real HDF5 compound fixtures."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training import reconstruction_sample as sample


@pytest.mark.parametrize(
    "capacities,budget,expected",
    [
        ([1, 99], 50, [1, 49]),
        ([1, 3, 11], 14, [1, 3, 10]),
        ([0, 2, 8], 10, [0, 2, 8]),
        ([2, 2, 2], 4, [2, 1, 1]),
        ([7, 1, 2], 1, [1, 0, 0]),
    ],
)
def test_exact_capacity_proportional_allocation(capacities, budget, expected):
    assert sample.proportional_allocation(capacities, budget) == expected


@pytest.mark.parametrize(
    "capacities,budget",
    [
        ([], 1),
        ([0, 0], 1),
        ([1, 2], 4),
        ([1, 2], 0),
        ([1, -1], 1),
        ([1, True], 1),
        ([1, 2], 1.1),
    ],
)
def test_invalid_allocation_fails_closed(capacities, budget):
    with pytest.raises(ValueError):
        sample.proportional_allocation(capacities, budget)


@pytest.mark.parametrize(
    "capacity,count,chunk",
    [(100, 37, 7), (12, 12, 3), (10, 1, 4), (0, 0, 8), (200003, 155001, 257)],
)
def test_streamed_sampler_exact_unique_sorted_and_chunk_bounded(capacity, count, chunk):
    parts = list(
        sample.selected_row_chunks(
            capacity, count, seed=42, shard="train/a", chunk_rows=chunk
        )
    )
    selected = (
        np.concatenate([p[2] for p in parts]) if parts else np.empty(0, dtype="i8")
    )
    assert len(selected) == count
    assert np.all(selected[1:] > selected[:-1])
    assert np.all((selected >= 0) & (selected < capacity))
    assert all(
        stop - start <= chunk and len(index) <= chunk for start, stop, index in parts
    )
    repeated = list(
        sample.selected_row_chunks(
            capacity, count, seed=42, shard="train/a", chunk_rows=chunk
        )
    )
    for expected, actual in zip(parts, repeated, strict=True):
        np.testing.assert_array_equal(expected[2], actual[2])


def test_shard_seed_separation_and_no_prefix_sampling():
    def draw(seed, shard):
        return np.concatenate(
            [
                p[2]
                for p in sample.selected_row_chunks(
                    100, 20, seed=seed, shard=shard, chunk_rows=13
                )
            ]
        )

    assert not np.array_equal(draw(42, "a"), draw(43, "a"))
    assert not np.array_equal(draw(42, "a"), draw(42, "b"))
    assert draw(42, "a").max() > 80
    # All source rows can be selected; no deterministic prefix/tail exclusion.
    counts = np.zeros(10, dtype="i8")
    for seed in range(500):
        ids = np.concatenate(
            [
                p[2]
                for p in sample.selected_row_chunks(
                    10, 3, seed=seed, shard="a", chunk_rows=4
                )
            ]
        )
        counts[ids] += 1
    assert np.all((counts > 100) & (counts < 200))


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "original.h5"
    keys = ["train/a", "train/b", "train/c"]
    validation = ["holdout/records"]
    dtype = np.dtype(
        [
            ("source_id", "<i8"),
            ("G", "<f8"),
            ("bp_1", "<f8"),
            ("rp_1", "<f8"),
            ("unused", "<f8"),
        ]
    )
    rows = {}
    next_id = 6000000000000000000
    with h5py.File(path, "w") as f:
        for key, length in zip([*keys, *validation], [1, 3, 11, 4], strict=True):
            values = np.zeros(length, dtype=dtype)
            values["source_id"] = np.arange(next_id, next_id + length, dtype="i8")
            next_id += length
            for j, name in enumerate(["G", "bp_1", "rp_1"]):
                values[name] = np.arange(length) * 0.3 + j + 0.123456789
            values["unused"] = 999
            if key == "train/c":
                values["rp_1"][3] = np.nan
            f.create_dataset(key, data=values)
            rows[key] = values
    center = np.array([0.1, 0.2, 0.3])
    scale = np.array([0.5, 1.5, 2.5])
    features = ["G", "bp_1", "rp_1"]
    signature = dict(
        train_keys=keys,
        feature_cols=features,
        scaler_center=center.tolist(),
        scaler_scale=scale.tolist(),
    )
    return dict(
        source_path=path,
        train_keys=keys,
        validation_keys=validation,
        features=features,
        center=center,
        scale=scale,
        output_dir=tmp_path / "sample",
        rows=14,
        seed=42,
        chunk_rows=3,
        parent_signature=signature,
    ), rows


def test_compound_staging_exact_scaling_id_provenance_checksums_readonly(
    source, monkeypatch
):
    cfg, raw = source
    observed_slices = []
    original = h5py.Dataset.__getitem__

    def record_reads(self, args, *a, **kw):
        if isinstance(args, slice):
            observed_slices.append((args.start, args.stop))
        return original(self, args, *a, **kw)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_reads)
    manifest = sample.stage_reconstruction_sample(**cfg)
    x, ids, provenance, saved = sample.open_reconstruction_sample(
        cfg["output_dir"], expected_request_sha256=manifest["request_sha256"]
    )
    assert len(x) == len(ids) == len(provenance) == 14
    assert (
        not x.flags.writeable
        and not ids.flags.writeable
        and not provenance.flags.writeable
    )
    assert (
        saved["sample_source_ids_unique"] and saved["original_validation_ids_excluded"]
    )
    assert saved["validation_ids_checked"] == 4
    assert saved["request"]["allocation"] == [1, 3, 10]
    assert (
        saved["ordered_source_id_sha256"]
        == hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    )
    for i, p in enumerate(provenance):
        r = raw[cfg["train_keys"][int(p["shard"])]][int(p["row"])]
        expected = np.array([r[name] for name in cfg["features"]], dtype="f4")
        expected = ((expected - cfg["center"]) / cfg["scale"]).astype("f4")
        np.testing.assert_array_equal(x[i], expected)
        assert ids[i] == r["source_id"]
    assert observed_slices and all(stop - start <= 3 for start, stop in observed_slices)
    assert "full raw-source SHA256 not computed" in saved["raw_source_identity"]
    for name, info in saved["files"].items():
        p = cfg["output_dir"] / name
        assert info["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        sample.stage_reconstruction_sample(**cfg)


def test_duplicate_sample_ids_rejected_and_incomplete_artifacts_retained(source):
    cfg, _ = source
    cfg["rows"] = 15
    with h5py.File(cfg["source_path"], "r+") as f:
        values = f["train/c"][:]
        values["source_id"][0] = f["train/a"]["source_id"][0]
        f["train/c"][:] = values
    with pytest.raises(ValueError, match="duplicate source IDs"):
        sample.stage_reconstruction_sample(**cfg)
    assert not cfg["output_dir"].exists()
    building = cfg["output_dir"].with_name("sample.building")
    assert building.exists() and not (building / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        sample.stage_reconstruction_sample(**cfg)


def test_validation_source_id_overlap_rejected_even_with_distinct_dataset_keys(source):
    cfg, _ = source
    cfg["rows"] = 15
    with h5py.File(cfg["source_path"], "r+") as f:
        values = f["holdout/records"][:]
        values["source_id"][2] = f["train/c"]["source_id"][3]
        f["holdout/records"][:] = values
    with pytest.raises(ValueError, match="overlaps original validation"):
        sample.stage_reconstruction_sample(**cfg)


def test_validation_dataset_overlap_rejected_before_staging(source):
    cfg, _ = source
    cfg["validation_keys"] = ["train/a"]
    with pytest.raises(ValueError, match="paths overlap"):
        sample.stage_reconstruction_sample(**cfg)
    assert not cfg["output_dir"].exists()


@pytest.mark.parametrize(
    "change", ["signature", "scaler", "schema", "float_id", "metadata", "memory"]
)
def test_wrong_source_signature_schema_scaler_or_memory_rejected(source, change):
    cfg, _ = source
    if change == "signature":
        cfg["parent_signature"] = {**cfg["parent_signature"], "train_keys": ["wrong"]}
    if change == "scaler":
        cfg["scale"][0] = 0
    if change == "schema":
        cfg["features"] = ["absent"]
        cfg["center"] = np.array([0])
        cfg["scale"] = np.array([1])
        cfg["parent_signature"] = None
    if change == "float_id":
        with h5py.File(cfg["source_path"], "r+") as f:
            f.create_dataset(
                "bad",
                data=np.ones(
                    3,
                    dtype=[
                        ("source_id", "f8"),
                        ("G", "f8"),
                        ("bp_1", "f8"),
                        ("rp_1", "f8"),
                    ],
                ),
            )
        cfg["train_keys"] = ["bad"]
        cfg["parent_signature"] = None
        cfg["rows"] = 2
    if change == "metadata":
        cfg["expected_source"] = {"bytes": 1}
    if change == "memory":
        cfg["max_id_sort_bytes"] = 1
    with pytest.raises(ValueError):
        sample.stage_reconstruction_sample(**cfg)


def test_cache_content_and_request_tampering_fail_closed(source):
    cfg, _ = source
    sample.stage_reconstruction_sample(**cfg)
    with pytest.raises(ValueError, match="identity differs"):
        sample.open_reconstruction_sample(
            cfg["output_dir"], expected_request_sha256="wrong"
        )
    sample.open_reconstruction_sample(
        cfg["output_dir"],
        expected_identity={
            "source": {"path": str(cfg["source_path"].resolve())},
            "features": cfg["features"],
            "center": cfg["center"],
            "scale": cfg["scale"],
            "train_keys": cfg["train_keys"],
            "validation_keys": cfg["validation_keys"],
        },
    )
    for mismatch in [
        {"features": list(reversed(cfg["features"]))},
        {"source": {"path": "/different-source.h5"}},
        {"scale": [1, 1, 1]},
        {"parent_sha256": "a" * 64},
    ]:
        with pytest.raises(ValueError, match="expected identity differs"):
            sample.open_reconstruction_sample(
                cfg["output_dir"], expected_identity=mismatch
            )
    file = cfg["output_dir"] / "features.npy"
    with file.open("r+b") as f:
        f.seek(-1, 2)
        value = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        sample.open_reconstruction_sample(cfg["output_dir"])


def test_raw_source_metadata_mutation_during_stage_rejected(source, monkeypatch):
    cfg, _ = source
    calls = 0
    original = sample._source_stat

    def mutated(path):
        nonlocal calls
        calls += 1
        actual = original(path)
        return actual if calls == 1 else {**actual, "mtime_ns": actual["mtime_ns"] + 1}

    monkeypatch.setattr(sample, "_source_stat", mutated)
    with pytest.raises(ValueError, match="changed while"):
        sample.stage_reconstruction_sample(**cfg)
    assert not cfg["output_dir"].exists()


def test_parent_hash_bound_and_incomplete_manifest_rejected(source):
    cfg, _ = source
    cfg["parent_sha256"] = "a" * 64
    manifest = sample.stage_reconstruction_sample(**cfg)
    assert manifest["request"]["parent_sha256"] == "a" * 64
    sample.open_reconstruction_sample(
        cfg["output_dir"], expected_identity={"parent_sha256": "a" * 64}
    )
    path = cfg["output_dir"] / "manifest.json"
    manifest["sample_source_ids_unique"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source checks are incomplete"):
        sample.open_reconstruction_sample(cfg["output_dir"])


@pytest.mark.parametrize("kind", ["external", "virtual"])
def test_unpinned_external_dataset_dependencies_are_rejected(source, kind):
    cfg, raw = source
    cfg.update(train_keys=["dependency"], parent_signature=None, rows=2)
    if kind == "external":
        other = cfg["source_path"].with_name("other.h5")
        with h5py.File(other, "w") as f:
            f.create_dataset("table", data=raw["train/c"])
        with h5py.File(cfg["source_path"], "r+") as f:
            f["dependency"] = h5py.ExternalLink(str(other), "table")
    else:
        with h5py.File(cfg["source_path"], "r+") as f:
            layout = h5py.VirtualLayout(shape=(11,), dtype=raw["train/c"].dtype)
            layout[:] = h5py.VirtualSource(
                str(cfg["source_path"]), "train/c", shape=(11,)
            )
            f.create_virtual_dataset("dependency", layout)
    with pytest.raises(ValueError, match="separately pinned dependencies"):
        sample.stage_reconstruction_sample(**cfg)
