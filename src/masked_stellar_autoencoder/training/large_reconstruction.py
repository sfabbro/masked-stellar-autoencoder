"""Exact original-source reconstruction cache for residual-Huber continuation.

The parent split and scaler are immutable. Staging checks unique integer IDs and
excludes all original validation IDs before publishing a reusable read-only cache.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

from .hdf5_io import _clean_column
from .reconstruction_sample import (
    open_reconstruction_sample,
    stage_reconstruction_sample,
)


def prepare_original_reconstruction(suite, checkpoint, features, center, scale):
    """Stage once, or checksum-verify an existing exactly matching cache.

    Returns ``(features_memmap, source_counts, manifest)``. No training starts
    until staging, source-ID disjointness and checksum verification complete.
    A retained incomplete .building is never silently replaced.
    """
    if suite.get("reconstruction_source") != "original_sample":
        raise ValueError("Expected explicit original_sample policy")
    rows = suite["reconstruction_training_rows"]
    if type(rows) is not int or rows < 1:
        raise ValueError("reconstruction_training_rows must be a positive integer")
    if suite["reconstruction_presentations"] % rows:
        raise ValueError("Large continuation requires complete declared sample epochs")
    signature = checkpoint["run_signature"]
    features = list(features)
    train = list(signature["train_keys"])
    valid = list(suite["reconstruction_validation_keys"])
    if list(signature["feature_cols"]) != features:
        raise ValueError("Parent feature order differs")
    for key, values in (("scaler_center", center), ("scaler_scale", scale)):
        np.testing.assert_array_equal(signature[key], values)
    source_path = Path(suite["reconstruction_original"])
    source_stat = source_path.stat()
    identity = dict(
        source=dict(
            path=str(source_path.resolve()),
            bytes=source_stat.st_size,
            mtime_ns=source_stat.st_mtime_ns,
        ),
        rows=rows,
        features=features,
        center=np.asarray(center, dtype="f8").tolist(),
        scale=np.asarray(scale, dtype="f8").tolist(),
        train_keys=train,
        validation_keys=valid,
        parent_sha256=suite["checkpoint_sha256"],
        seed=suite.get("reconstruction_sample_seed", 42),
        chunk_rows=suite.get("reconstruction_chunk_rows", 65536),
        id_field=suite.get("reconstruction_id_field", "source_id"),
    )
    with h5py.File(source_path, "r") as source:
        if set(source) != set(train) | set(valid) or set(train) & set(valid):
            raise ValueError("Original source shard split differs from parent/policy")
        for key in train:
            if len(source[key]) != signature["train_rows_by_key"][key]:
                raise ValueError(f"Original source training row count differs: {key}")
    cache_path = Path(suite["reconstruction_cache"])
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        required = rows * (4 * len(features) + 20) + 1024**3
        if shutil.disk_usage(cache_path.parent).free < required:
            raise ValueError(
                f"Insufficient cache disk space; need at least {required} bytes"
            )
        stage_reconstruction_sample(
            source_path,
            train,
            valid,
            features,
            center,
            scale,
            cache_path,
            rows,
            seed=identity["seed"],
            chunk_rows=identity["chunk_rows"],
            id_field=identity["id_field"],
            parent_signature=signature,
            parent_sha256=suite["checkpoint_sha256"],
            expected_source=identity["source"],
        )
    cache, ids, provenance, manifest = open_reconstruction_sample(
        cache_path,
        expected_request_sha256=suite.get("reconstruction_request_sha256"),
        expected_identity=identity,
    )
    # Original association check is independent of the staged file hashes.
    checked = np.linspace(0, rows - 1, min(128, rows), dtype=np.int64)
    with h5py.File(source_path, "r") as source:
        for index in checked:
            location = provenance[index]
            raw = source[train[int(location["shard"])]][int(location["row"])]
            if int(raw[identity["id_field"]]) != int(ids[index]):
                raise ValueError("Selected sample source-ID association differs")
            values = np.array(
                [_clean_column(name, np.asarray([raw[name]]))[0] for name in features],
                dtype="f8",
            )
            values[~np.isfinite(values)] = np.nan
            expected = ((values - center) / scale).astype("f4")
            np.testing.assert_array_equal(cache[index], expected)
    manifest = {
        **manifest,
        "selected_raw_source_audit": dict(
            status="passed",
            rows=len(checked),
            all_features=len(features),
            max_abs_standardized_difference=0.0,
            method="evenly spaced staged indices; original row read by shard/row provenance",
        ),
    }
    return (
        cache,
        dict(zip(train, manifest["request"]["allocation"], strict=True)),
        manifest,
    )
