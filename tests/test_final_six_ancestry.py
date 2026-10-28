"""Synthetic sets/HDF fixtures and historical source invariants, never models."""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_final_six_ancestry import (
    TARGETS,
    ancestry_masks,
    function_from_archived_source,
    label_sampler_exposure,
)


def test_ancestry_preserves_sparse_roles_and_distinguishes_target_exposure():
    data = dict(
        source_id=np.array([11, 12, 13, 14]),
        role=np.array(["fit", "tune", "cal", "test"]),
        target_names=np.asarray(TARGETS),
        observed=np.ones((4, 6), bool),
        quality=np.ones((4, 6), bool),
        value=np.ones((4, 6)),
        label_source_role=np.tile(
            np.array(["fit", "tune", "cal", "test"])[:, None], (1, 6)
        ),
    )
    data["observed"][2, 5] = False
    data["quality"][2, 5] = False
    data["value"][2, 5] = np.nan
    arrays, counts = ancestry_masks(data, [11, 13])
    assert arrays["feh_aux_fit_exposed"].tolist() == [True, False, True, False]
    assert counts["fe_h"]["cal"]["feh_aux_fit_exposed_count"] == 1
    assert (
        counts["fe_h"]["cal"]["exposure_kind"] == "same_target_native_FeH_supervision"
    )
    assert (
        counts["alpha"]["cal"]["exposure_kind"] == "cross_target_native_FeH_supervision"
    )
    assert counts["distance"]["cal"]["eligible_count"] == 0
    assert arrays["direct_feh_target_exposed"].sum() == 2
    assert not arrays["direct_feh_target_exposed"][:, [0, 1, 3, 4, 5]].any()
    assert not arrays["cross_target_feh_aux_exposed"][:, 2].any()
    data["label_source_role"][1, 0] = "fit"
    with pytest.raises(ValueError, match="promoted"):
        ancestry_masks(data, [])


def test_sampler_stream_and_occurrences_match_independent_bulk_draws():
    counts, stream = label_sampler_exposure(
        19, seed=42, warmup_updates=5, updates=7, groups=3, batch=16
    )
    independent_rows = np.random.default_rng(42).integers(19, size=(5 + 7 * 3, 16))
    np.testing.assert_array_equal(
        counts, np.bincount(independent_rows.ravel(), minlength=19)
    )
    assert counts.sum() == independent_rows.size
    assert (
        stream == hashlib.sha256(independent_rows.astype("<i8").tobytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        label_sampler_exposure(
            0, seed=42, warmup_updates=5, updates=7, groups=3, batch=16
        )


def test_frozen_historical_archive_and_split_function_are_exact():
    config = yaml.safe_load(
        Path("configs/final-six-ancestry-20261024.yaml").read_text()
    )
    archive_path = Path(config["historical_archive"])
    assert (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        == config["historical_archive_sha256"]
    )
    archive = json.loads(archive_path.read_text())
    assert archive["historical_commit"] == config["historical_source_commit"]
    for source in archive["files"].values():
        assert hashlib.sha256(source["text"].encode()).hexdigest() == source["sha256"]
    text = archive["files"]["scripts/stellar_catalog_jobs.py"]["text"]
    split = function_from_archived_source(text, "identity_split", {"np": np})
    expected = []
    mask64 = (1 << 64) - 1
    for source_id in range(100):
        x = source_id ^ 42
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask64
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask64
        x ^= x >> 31
        expected.append(0 if x % 1000 < 700 else 1 if x % 1000 < 850 else 2)
    np.testing.assert_array_equal(split(np.arange(100)), expected)


def test_historical_selector_uses_original_native_first_source_rules(tmp_path):
    archive = json.loads(
        Path("configs/final-six-ancestry-historical-source-e525ae7.json").read_text()
    )
    split = function_from_archived_source(
        archive["files"]["scripts/stellar_catalog_jobs.py"]["text"],
        "identity_split",
        {"np": np},
    )
    selector = function_from_archived_source(
        archive["files"]["scripts/run_enriched_value.py"]["text"],
        "prepare",
        {
            "np": np,
            "h5py": h5py,
            "json": json,
            "identity_split": split,
            "LAYERS": ("native", "PIGS", "Li2025_CaT_silver"),
        },
    )
    candidates = np.arange(1, 100)
    ids = np.array(
        [
            candidates[split(candidates) == 0][0],
            candidates[split(candidates) == 2][0],
            candidates[split(candidates) == 0][1],
        ]
    )
    measurement_dtype = np.dtype([("source_id", "i8"), ("value", "f8")])
    path = tmp_path / "synthetic-cohort.h5"
    with h5py.File(path, "w") as f:
        f.attrs["feature_columns_json"] = json.dumps(["G", "bp_1"])
        for key, value in dict(
            source_id=ids,
            benchmark_partition=np.zeros(3, "i1"),
            label_split=split(ids),
            feature_available=np.ones(3, bool),
            values=np.zeros((3, 2)),
        ).items():
            f.create_dataset("features/" + key, data=value)
        for key, value in dict(
            feature_row=np.arange(3),
            source_id=ids,
            strict_fit=split(ids) == 0,
            layer=np.array([0, 0, 1]),
            layer_row=np.array([0, 1, 0]),
        ).items():
            f.create_dataset("canonical/" + key, data=value)
        layers = dict(
            native=(
                np.array([(ids[0], -0.4), (ids[1], -2.4)], dtype=measurement_dtype),
                np.array([0, 1]),
            ),
            PIGS=(np.array([(ids[2], -2.8)], dtype=measurement_dtype), np.array([2])),
            Li2025_CaT_silver=(
                np.array([], dtype=measurement_dtype),
                np.array([], dtype="i8"),
            ),
        )
        for layer, (measurements, rows) in layers.items():
            f.create_dataset("raw_layers/" + layer + "/measurements", data=measurements)
            f.create_dataset("layer_index/" + layer + "/feature_row", data=rows)
            f.create_dataset(
                "layer_index/" + layer + "/identity_test", data=split(ids[rows]) == 2
            )
        f.create_dataset("reference_scaler/center", data=np.zeros(2))
        f.create_dataset("reference_scaler/scale", data=np.ones(2))
    selected = selector(path)
    assert selected["source_id"][selected["native_fit"]].tolist() == [int(ids[0])]
    assert selected["fit_truth"][selected["native_fit"]].tolist() == [-0.4]
    assert selected["tests"]["native"]["measurements"]["source_id"].tolist() == [
        int(ids[1])
    ]
