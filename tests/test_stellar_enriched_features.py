import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stellar_age_cohorts import file_digest
from stellar_enriched_features import (
    LAYERS,
    materialize,
    record_projection,
)
from stellar_enriched_views import fit_masks


def test_real_nested_record_projection_order_identity_and_masks(tmp_path):
    path = tmp_path / "benchmark.h5"
    data = np.array(
        [
            (2**60 + 3, 1.0, 0.1),
            (2**60 + 1, -9999.0, 0.2),
            (2**60 + 7, np.nan, -9999.0),
        ],
        dtype=[("source_id", "<i8"), ("bp_1", "<f4"), ("bpe_1", "<f4")],
    )
    with h5py.File(path, "w") as f:
        records = f.create_dataset("train/records", data=data, chunks=(2,))
        v, e = record_projection(
            records,
            np.array([2, 0, 1]),
            data["source_id"][[2, 0, 1]],
            ["bp_1", "missing"],
            ["bpe_1", None],
            block=2,
        )
        assert np.isnan(v[[0, 2], 0]).all() and v[1, 0] == 1 and np.isnan(v[:, 1]).all()
        assert np.isnan(e[0, 0]) and e[1, 0] == pytest.approx(0.1)
        with pytest.raises(AssertionError):
            record_projection(
                records, np.array([0]), data["source_id"][[1]], ["bp_1"], [None]
            )


def test_full_materialization_original_benchmark_priority_and_fixed_label_tiers(
    tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ids = np.arange(2**60, 2**60 + 320, dtype="i8")
    train, val = ids[:200], ids[200:250]
    expected = hashlib.sha256(np.r_[train, val].astype("<i8").tobytes()).hexdigest()
    xp = [f"{band}_{j}" for band in ["bp", "rp"] for j in range(1, 56)]
    features = ["G", *xp, "RA", "DEC", "pmra", "pmdec"]
    errors = [None] * len(features)
    errors[1] = "bpe_1"
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(dict(data=dict(feature_cols=features, error_cols=errors)))
    )
    rng = np.random.default_rng(42)
    values = rng.normal(size=(len(ids), len(features))).astype("f4")
    values[:, 0] = 15
    dtype = [("source_id", "<i8")] + [(n, "<f4") for n in ["G", *xp, "bpe_1"]]
    rows = np.zeros(250, dtype=dtype)
    rows["source_id"] = ids[:250]
    for j, n in enumerate(["G", *xp]):
        rows[n] = values[:250, j]
    rows["bpe_1"] = 0.1
    rows["bp_1"][0] = np.nan
    benchmark = tmp_path / "benchmark.h5"
    center = np.zeros(len(features))
    scale = np.ones(len(features))
    scaler_sha = hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest()
    with h5py.File(benchmark, "w") as f:
        f.attrs["provenance"] = json.dumps(dict(source_id_sha256=expected))
        f.create_dataset("train/records", data=rows[:200], chunks=(17,))
        f.create_dataset("validation/records", data=rows[200:], chunks=(17,))
        f.create_dataset("reference_scaler/center", data=center)
        f.create_dataset("reference_scaler/scale", data=scale)
    (root / "dataset_view_v2").mkdir()
    with h5py.File(root / "dataset_view_v2/dataset-view.h5", "w") as f:
        f.create_dataset("train/source_id", data=train)
        f.create_dataset("validation/source_id", data=val)
    # Outside-benchmark ID250 has an incomplete FIRST row and a later complete duplicate.
    source_ids = np.r_[ids, ids[250]]
    fv = np.r_[values, values[[250]]]
    fv[250, 1] = np.nan
    legacy = tmp_path / "features.fits"
    fits.BinTableHDU.from_columns(
        [fits.Column(name="source_id", format="K", array=source_ids)]
        + [
            fits.Column(name=n, format="E", array=fv[:, j])
            for j, n in enumerate(features)
        ]
    ).writeto(legacy)
    label_dir = root / "enriched_label_view_v1"
    label_dir.mkdir()
    labels = label_dir / "enriched-label-view.h5"
    measurement_dtype = [
        ("source_id", "<i8"),
        ("value", "<f8"),
        ("quantity", "S8"),
        ("uncertainty", "<f8"),
    ]
    source_rows = {
        "native": ids[:270],
        "PIGS": np.r_[ids[:4], ids[270:290]],
        "Li2025_CaT_silver": np.r_[ids[:4], ids[290:]],
    }
    with h5py.File(labels, "w") as f:
        f.attrs["complete"] = True
        f.attrs["input_provenance_json"] = "fixture original raw sources"
        for n in LAYERS:
            measurement = np.zeros(len(source_rows[n]), dtype=measurement_dtype)
            measurement["source_id"] = source_rows[n]
            measurement["value"] = -2
            measurement["quantity"] = b"FeH"
            measurement["uncertainty"] = 0.1
            g = f.create_group(n)
            g.create_dataset("measurements", data=measurement)
            sp, part, broad, strict = fit_masks(source_rows[n], train, val)
            for key, v in dict(
                label_split=sp,
                benchmark_partition=part,
                broader_fit_eligible=broad,
                strict_fit_eligible=strict,
            ).items():
                g.create_dataset(key, data=v)
    report = materialize(
        out,
        scratch,
        root=root,
        benchmark=benchmark,
        legacy=legacy,
        expected_ids=expected,
        expected_counts=(200, 50),
        expected_label_sha=file_digest(labels),
        expected_legacy_sha=file_digest(legacy),
        expected_scaler_sha=scaler_sha,
        config_path=config,
    )
    assert (
        report["unit_audit"]["matched_stars"] == 128
        and not report["training_performed"]
    )
    with (
        h5py.File(out / "enriched-feature-cohort.h5") as f,
        h5py.File(labels) as original,
    ):
        np.testing.assert_array_equal(f["features/source_id"][:], ids)
        assert (
            f["features/input_source"][0] == 0
            and not f["features/feature_available"][0]
        )  # no legacy rescue
        assert (
            f["features/input_source"][250] == 2
            and not f["features/feature_available"][250]
        )  # no duplicate rescue
        assert (
            f["features/input_source"][251] == 2
            and f["features/feature_available"][251]
        )
        chosen_ids = f["canonical/source_id"][:]
        layer = f["canonical/layer"][:]
        assert not np.isin(
            chosen_ids[f["canonical/strict_fit"][:] | f["canonical/broader_fit"][:]],
            val,
        ).any()
        assert (
            layer[np.isin(chosen_ids, ids[1:4])] == 0
        ).all()  # native before lower-error/other-layer values
        for n in LAYERS:
            np.testing.assert_array_equal(
                f["raw_layers/" + n + "/measurements"][:],
                original[n + "/measurements"][:],
            )

    from types import SimpleNamespace

    from stellar_enriched_audit import verify

    args = SimpleNamespace(
        cohort=out / "enriched-feature-cohort.h5",
        labels=labels,
        index=root / "dataset_view_v2/dataset-view.h5",
        legacy=legacy,
        benchmark=benchmark,
        report=out / "independent.json",
    )
    pins = dict(
        expected_label_sha=file_digest(labels),
        expected_legacy_sha=file_digest(legacy),
        expected_counts=(200, 50),
        expected_ids=expected,
        expected_scaler_sha=scaler_sha,
    )
    checked = verify(args, **pins)
    assert (
        checked["publication_gate_complete"]
        and checked["all_original_benchmark_raw_records_verified"]
    )
    args.benchmark = None
    assert not verify(args, **pins)["publication_gate_complete"]
    args.benchmark = benchmark
    with h5py.File(benchmark, "r+") as changed:
        row = changed["train/records"][1]
        row["G"] = 999
        changed["train/records"][1] = row
    with pytest.raises(AssertionError):
        verify(args, **pins)
