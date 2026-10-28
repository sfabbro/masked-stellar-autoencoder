"""Exact, bounded-memory reconstruction sampling from original HDF5 shards.

This stages data only. It does not authorize training or certify a 10M pilot.
Sampling is without replacement within each original shard, proportionally
stratified by shard capacity. Selected rows retain original shard/row and exact
integer source IDs. Validation source IDs are checked globally before commit.

The raw source is pinned by path/size/mtime/schema, NOT an invented full-file
SHA. Actual staged NPY files and the canonical request receive SHA256 hashes.
An incomplete .building directory is deliberately retained after failure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .hdf5_io import _clean_column

PROVENANCE_DTYPE = np.dtype([("shard", "<u4"), ("row", "<i8")])
FORMAT_VERSION = 1


def _positive_integer(name, value, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return int(value)


def proportional_allocation(capacities, budget):
    """Largest-remainder allocation with exact integer arithmetic and no underfill."""
    capacities = [_positive_integer("capacity", n, zero=True) for n in capacities]
    budget = _positive_integer("budget", budget)
    total = sum(capacities)
    if not capacities or budget > total:
        raise ValueError("sample budget exceeds total training capacity")
    quotient_remainders = [divmod(n * budget, total) for n in capacities]
    allocation = [q for q, _ in quotient_remainders]
    order = sorted(
        (i for i, n in enumerate(capacities) if allocation[i] < n),
        key=lambda i: (-quotient_remainders[i][1], i),
    )
    for i in order[: budget - sum(allocation)]:
        allocation[i] += 1
    if sum(allocation) != budget or any(
        a > n for a, n in zip(allocation, capacities, strict=True)
    ):
        raise RuntimeError("internal allocation invariant failed")
    return allocation


def selected_row_chunks(capacity, count, *, seed, shard, chunk_rows=65536):
    """Yield sorted int64 row indices; memory bounded by one source chunk.

    Hypergeometric chunk counts followed by uniform subsets give a uniform
    fixed-size subset of the shard. The seed, shard name AND chunk size are part
    of the cache request: changing chunking may change the selected subset.
    """
    capacity = _positive_integer("capacity", capacity, zero=True)
    count = _positive_integer("count", count, zero=True)
    seed = _positive_integer("seed", seed, zero=True)
    chunk_rows = _positive_integer("chunk_rows", chunk_rows)
    if count > capacity or not isinstance(shard, str) or not shard:
        raise ValueError("invalid shard or sample count")
    digest = hashlib.sha256(f"{seed}\0{shard}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))
    remaining = count
    for start in range(0, capacity, chunk_rows):
        if remaining == 0:
            break
        size = min(chunk_rows, capacity - start)
        available = capacity - start
        if remaining == available:
            chosen = size
        elif size == available:
            chosen = remaining
        else:
            chosen = int(rng.hypergeometric(remaining, available - remaining, size))
        if chosen:
            local = (
                np.arange(size, dtype=np.int64)
                if chosen == size
                else np.sort(rng.choice(size, chosen, replace=False))
            )
            yield start, start + size, local + start
        remaining -= chosen
    if remaining:
        raise RuntimeError("internal sampler count invariant failed")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value):
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _ordered_id_sha(ids, chunk_rows):
    digest = hashlib.sha256()
    for start in range(0, len(ids), chunk_rows):
        digest.update(
            np.asarray(ids[start : start + chunk_rows], dtype="<i8").tobytes()
        )
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        f.write(
            json.dumps(_jsonable(value), indent=2, allow_nan=False).encode() + b"\n"
        )
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _keys(keys, name):
    result = list(keys)
    if (
        not result
        or any(not isinstance(k, str) or not k for k in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must be nonempty unique dataset paths")
    return result


def _source_stat(path):
    s = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def _integer_ids(values):
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("source IDs must be scalar integers, never float-cast IDs")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError("source ID exceeds signed int64 range")
    result = values.astype("<i8", copy=False)
    if np.any(result <= 0):
        raise ValueError("source IDs must be positive Gaia identifiers")
    return result


def _has_overlap(sorted_ids, query):
    indices = np.searchsorted(sorted_ids, query)
    in_range = indices < len(sorted_ids)
    return bool(np.any(sorted_ids[indices[in_range]] == query[in_range]))


def _matches_subset(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches_subset(actual[key], value)
            for key, value in expected.items()
        )
    return _jsonable(actual) == _jsonable(expected)


def stage_reconstruction_sample(
    source_path,
    train_keys,
    validation_keys,
    features,
    center,
    scale,
    output_dir,
    rows,
    *,
    seed=42,
    id_field="source_id",
    chunk_rows=65536,
    parent_signature=None,
    parent_sha256=None,
    expected_source=None,
    max_id_sort_bytes=4 * 1024**3,
):
    """Stage an exact sample in a NEW output directory and commit its manifest.

    Existing/incomplete outputs are never overwritten or implicitly resumed.
    Reader semantics match pretraining _clean_column -> fit scaler -> float32.
    Raw missing/nonfinite values stay NaN. The full source is read-only.
    expected_source may pin path, bytes and/or mtime_ns from an earlier audit.
    A launcher, not this generic CPU staging helper, enforces pilot promotion.
    """
    rows = _positive_integer("rows", rows)
    seed = _positive_integer("seed", seed, zero=True)
    chunk_rows = _positive_integer("chunk_rows", chunk_rows)
    max_id_sort_bytes = _positive_integer("max_id_sort_bytes", max_id_sort_bytes)
    if parent_sha256 is not None and (
        not isinstance(parent_sha256, str)
        or len(parent_sha256) != 64
        or any(c not in "0123456789abcdef" for c in parent_sha256)
    ):
        raise ValueError("parent SHA256 must be lowercase hexadecimal")
    train_keys, validation_keys = (
        _keys(train_keys, "train_keys"),
        _keys(validation_keys, "validation_keys"),
    )
    if set(train_keys) & set(validation_keys):
        raise ValueError("training and validation dataset paths overlap")
    features = list(features)
    if (
        not features
        or len(set(features)) != len(features)
        or any(not isinstance(f, str) or not f for f in features)
    ):
        raise ValueError("features must be unique nonempty field names")
    center, scale = np.asarray(center, dtype="f8"), np.asarray(scale, dtype="f8")
    if (
        center.shape != (len(features),)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid scaler shape or finite positive scale")
    if rows * np.dtype("i8").itemsize > max_id_sort_bytes:
        raise ValueError("ID uniqueness sort exceeds the declared memory budget")
    if parent_signature is not None:
        for field, expected in (("train_keys", train_keys), ("feature_cols", features)):
            if list(parent_signature[field]) != expected:
                raise ValueError(f"parent signature {field} differs")
        for field, expected in (("scaler_center", center), ("scaler_scale", scale)):
            if not np.array_equal(parent_signature[field], expected):
                raise ValueError(f"parent signature {field} differs")
    source_path, output_dir = Path(source_path), Path(output_dir)
    before = _source_stat(source_path)
    if expected_source is not None and any(
        before.get(k) != v for k, v in expected_source.items()
    ):
        raise ValueError("raw source metadata differs from the audited source")
    building = output_dir.with_name(output_dir.name + ".building")
    if output_dir.exists() or building.exists():
        raise FileExistsError(
            "sample output or retained incomplete .building already exists"
        )
    with h5py.File(source_path, "r") as source:
        schema = {}
        for key in [*train_keys, *validation_keys]:
            if key not in source or not isinstance(source[key], h5py.Dataset):
                raise ValueError(f"source dataset is absent: {key}")
            data = source[key]
            if (
                Path(data.file.filename).resolve() != source_path.resolve()
                or data.is_virtual
                or data.external
            ):
                raise ValueError(
                    f"external/virtual datasets need separately pinned dependencies: {key}"
                )
            if data.ndim != 1 or data.dtype.names is None:
                raise ValueError(f"expected a one-dimensional compound table: {key}")
            required = [id_field, *features] if key in train_keys else [id_field]
            if set(required) - set(data.dtype.names):
                raise ValueError(
                    f"missing fields in {key}: {sorted(set(required) - set(data.dtype.names))}"
                )
            if data.dtype[id_field].kind not in "iu" or data.dtype[id_field].shape:
                raise ValueError(f"{key} source ID field is not a scalar integer")
            if key in train_keys and any(data.dtype[f].shape for f in features):
                raise ValueError(f"model features must be scalar fields: {key}")
            schema[key] = dict(
                shape=list(data.shape), dtype=_jsonable(data.dtype.descr)
            )
        allocation = proportional_allocation([len(source[k]) for k in train_keys], rows)
        request = dict(
            format_version=FORMAT_VERSION,
            source=before,
            schema=schema,
            train_keys=train_keys,
            validation_keys=validation_keys,
            features=features,
            center=center.tolist(),
            scale=scale.tolist(),
            rows=rows,
            seed=seed,
            chunk_rows=chunk_rows,
            id_field=id_field,
            allocation=allocation,
            algorithm="proportional-largest-remainder/streamed-hypergeometric-PCG64-v1",
            numpy_version=np.__version__,
            h5py_version=h5py.__version__,
            parent_sha256=parent_sha256,
            parent_signature_sha256=hashlib.sha256(
                _canonical(parent_signature)
            ).hexdigest()
            if parent_signature is not None
            else None,
        )
        request_hash = hashlib.sha256(_canonical(request)).hexdigest()
        building.mkdir(parents=True, exist_ok=False)
        _atomic_json(
            building / "request.json",
            {"request_sha256": request_hash, "request": request, "status": "building"},
        )
        arrays = {
            "features.npy": np.lib.format.open_memmap(
                building / "features.npy",
                mode="w+",
                dtype="<f4",
                shape=(rows, len(features)),
            ),
            "source_id.npy": np.lib.format.open_memmap(
                building / "source_id.npy", mode="w+", dtype="<i8", shape=(rows,)
            ),
            "provenance.npy": np.lib.format.open_memmap(
                building / "provenance.npy",
                mode="w+",
                dtype=PROVENANCE_DTYPE,
                shape=(rows,),
            ),
        }
        offset = 0
        for shard_number, (key, count) in enumerate(
            zip(train_keys, allocation, strict=True)
        ):
            data = source[key]
            fields = list(dict.fromkeys([id_field, *features]))
            for start, stop, selected in selected_row_chunks(
                len(data), count, seed=seed, shard=key, chunk_rows=chunk_rows
            ):
                # Contiguous projected reads avoid h5py's huge fancy-index list
                # overhead while keeping the full compound table out of memory.
                records = data.fields(fields)[start:stop][selected - start]
                n = len(records)
                raw = np.column_stack([_clean_column(f, records[f]) for f in features])
                raw[~np.isfinite(raw)] = np.nan
                standardized = ((raw - center) / scale).astype("<f4")
                if np.isinf(standardized).any():
                    raise ValueError("float32 scaler transformation overflow")
                arrays["features.npy"][offset : offset + n] = standardized
                arrays["source_id.npy"][offset : offset + n] = _integer_ids(
                    records[id_field]
                )
                arrays["provenance.npy"]["shard"][offset : offset + n] = shard_number
                arrays["provenance.npy"]["row"][offset : offset + n] = selected
                offset += n
        if offset != rows:
            raise RuntimeError("staged row count differs from exact requested budget")
        for value in arrays.values():
            value.flush()
        # Only the compact int64 IDs are copied for sorting: 1.2GB at150M.
        sorted_ids = np.array(arrays["source_id.npy"], copy=True)
        sorted_ids.sort(kind="quicksort")
        if np.any(sorted_ids[1:] == sorted_ids[:-1]):
            raise ValueError("duplicate source IDs in selected reconstruction sample")
        validation_rows = 0
        for key in validation_keys:
            data = source[key]
            for start in range(0, len(data), chunk_rows):
                query = _integer_ids(data.fields(id_field)[start : start + chunk_rows])
                if _has_overlap(sorted_ids, query):
                    raise ValueError("selected source ID overlaps original validation")
                validation_rows += len(query)
        del sorted_ids
        if _source_stat(source_path) != before:
            raise ValueError("raw source changed while sample was being staged")
        files = {}
        for name, value in arrays.items():
            path = building / name
            with path.open("rb") as f:
                os.fsync(f.fileno())
            files[name] = dict(
                bytes=path.stat().st_size,
                sha256=_sha(path),
                shape=list(value.shape),
                dtype=_jsonable(
                    value.dtype.descr if value.dtype.names else value.dtype.str
                ),
            )
        manifest = dict(
            status="complete",
            format_version=FORMAT_VERSION,
            request=request,
            request_sha256=request_hash,
            files=files,
            ordered_source_id_sha256=_ordered_id_sha(
                arrays["source_id.npy"], chunk_rows
            ),
            sample_source_ids_unique=True,
            original_validation_ids_excluded=True,
            validation_ids_checked=validation_rows,
            raw_source_identity="path/bytes/mtime/schema only; full raw-source SHA256 not computed",
        )
        _atomic_json(building / "manifest.json", manifest)
        arrays.clear()
        os.rename(building, output_dir)
    return manifest


def open_reconstruction_sample(
    output_dir, *, expected_request_sha256=None, expected_identity=None
):
    """Verify a complete cache once and return read-only memory maps + manifest."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("sample manifest is not a supported complete publication")
    if (
        manifest.get("sample_source_ids_unique") is not True
        or manifest.get("original_validation_ids_excluded") is not True
    ):
        raise ValueError("sample uniqueness/validation source checks are incomplete")
    request_hash = hashlib.sha256(_canonical(manifest["request"])).hexdigest()
    if manifest["request_sha256"] != request_hash or (
        expected_request_sha256 is not None and request_hash != expected_request_sha256
    ):
        raise ValueError("sample request identity differs")
    if expected_identity is not None and not _matches_subset(
        manifest["request"], expected_identity
    ):
        raise ValueError("sample expected identity differs")
    if set(manifest["files"]) != {"features.npy", "source_id.npy", "provenance.npy"}:
        raise ValueError("sample file inventory differs")
    maps = []
    for name in ("features.npy", "source_id.npy", "provenance.npy"):
        path, info = output_dir / name, manifest["files"][name]
        if path.stat().st_size != info["bytes"] or _sha(path) != info["sha256"]:
            raise ValueError(f"sample checksum/size mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != info["shape"]:
            raise ValueError(f"sample shape mismatch: {name}")
        maps.append(array)
    x, ids, provenance = maps
    if (
        x.shape != (manifest["request"]["rows"], len(manifest["request"]["features"]))
        or x.dtype != np.dtype("<f4")
        or ids.shape != (len(x),)
        or ids.dtype != np.dtype("<i8")
        or provenance.shape != ids.shape
        or provenance.dtype != PROVENANCE_DTYPE
    ):
        raise ValueError("sample array contract differs")
    return x, ids, provenance, manifest
