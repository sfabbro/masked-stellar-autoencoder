"""Source-only lookup beyond the legacy feature cohort; no model construction.

The immutable original-cache proof is transferred with explicit scope. Its
87GB feature file is not rehashed: IDs are streamed and hashed, file metadata
is checked and recovered features receive bounded raw-source association tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

from .hdf5_io import _clean_column

CACHE_REQUEST_SHA = "8d14aa3f741946e8fa220fd0d9f8d74bb44f5949e2f9d2ff0449d21cfa5e5b9c"


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def match_ids(sorted_ids, order, query):
    """Return query offsets and original candidate offsets, never float-cast IDs."""
    query = np.asarray(query)
    if query.ndim != 1 or query.dtype.kind not in "iu":
        raise ValueError("Integer one-dimensional source IDs required")
    pos = np.searchsorted(sorted_ids, query)
    valid = (pos < len(sorted_ids)) & (query > 0)
    valid[valid] &= sorted_ids[pos[valid]] == query[valid]
    q = np.flatnonzero(valid)
    return q, order[pos[q]]


def _array(directory, key, shape, dtype, fill):
    if directory is None:
        return np.full(shape, fill, dtype=dtype)
    file = directory / (key + ".npy")
    if file.exists():
        result = np.load(file, mmap_mode="r+", allow_pickle=False)
        if result.shape != shape or result.dtype != np.dtype(dtype):
            raise ValueError("Changed persisted feature matrix layout: " + key)
        return result
    result = np.lib.format.open_memmap(file, mode="w+", dtype=dtype, shape=shape)
    result[:] = fill
    return result


def _raw_values(records, names):
    result = np.column_stack([_clean_column(n, records[n]) for n in names])
    result[~np.isfinite(result)] = np.nan
    return result


def _safe_input(raw, center, scale, allowed):
    raw = np.asarray(raw, dtype="f4")
    scaled = ((raw.astype("f8") - center) / scale).astype("f4")
    visible = np.isfinite(raw) & (raw != -9999) & np.isfinite(scaled)
    visible &= allowed[None]
    return np.where(visible, scaled, -9999).astype("f4"), visible


def _cache_maps(directory, names, center, scale, request_sha, expected_rows):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    request = manifest["request"]
    canonical = json.dumps(
        request, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    actual_request = hashlib.sha256(canonical).hexdigest()
    if (
        manifest.get("status") != "complete"
        or manifest.get("format_version") != 1
        or manifest.get("sample_source_ids_unique") is not True
        or manifest.get("original_validation_ids_excluded") is not True
        or manifest.get("request_sha256") != actual_request
        or actual_request != request_sha
        or request["rows"] != expected_rows
        or request["features"] != list(names)
    ):
        raise ValueError("Original cache request, shape or source gate differs")
    np.testing.assert_array_equal(request["center"], center)
    np.testing.assert_array_equal(request["scale"], scale)
    arrays = []
    for filename in ("features.npy", "source_id.npy", "provenance.npy"):
        spec = manifest["files"][filename]
        file = directory / filename
        if file.stat().st_size != spec["bytes"]:
            raise ValueError("Original cache file size differs: " + filename)
        array = np.load(file, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != spec["shape"]:
            raise ValueError("Original cache array shape differs: " + filename)
        arrays.append(array)
    features, ids, provenance = arrays
    if (
        features.shape != (expected_rows, len(names))
        or features.dtype != np.dtype("<f4")
        or ids.shape != (expected_rows,)
        or ids.dtype != np.dtype("<i8")
        or provenance.dtype != np.dtype([("shard", "<u4"), ("row", "<i8")])
    ):
        raise ValueError("Original cache dtype differs")
    return features, ids, provenance, manifest


def _vac_projection(spec, ids, raw, errors, names):
    # Reuse the tested staged-Gaia duplicate/flag/flux-error policy, without
    # importing any model runner. This helper lives in the same committed repo.
    file = Path(__file__).resolve().parents[3] / "scripts/prepare_five_job_source.py"
    module_spec = importlib.util.spec_from_file_location("_evaluation_vac_source", file)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    evidence = {}
    published, known, rows, digest = module.vac_projection(
        spec, ids, raw, errors, list(names), evidence=evidence
    )
    evidence.update(source_sha256=digest, source_path=spec["path"])
    return published, known, rows, evidence


def aligned_photometry_fallback(fallback, ids, names, raw, errors):
    """Fill absent measured magnitudes only, after an exact verified-ID join.

    Publication-unknown photometry remains a separate diagnostic population.
    DR2/unknown-release magnitudes are marked domain transfer, never DR3 proof.
    """
    used = np.zeros(len(ids), bool)
    release_code = np.zeros(len(ids), dtype="u1")
    if fallback is None:
        return used, release_code, {}
    fallback_ids = np.asarray(fallback["source_id"])
    if fallback_ids.dtype.kind not in "iu" or fallback_ids.ndim != 1:
        raise ValueError("Fallback source IDs must remain exact integers")
    np.testing.assert_array_equal(fallback_ids, ids)
    verified = np.asarray(fallback["identity_verified"], bool)
    columns = list(fallback["names"])
    values = np.asarray(fallback["values"], dtype="f8")
    uncertainties = np.asarray(
        fallback.get("errors", np.full_like(values, np.nan)), dtype="f8"
    )
    if (
        verified.shape != ids.shape
        or values.shape != (len(ids), len(columns))
        or uncertainties.shape != values.shape
    ):
        raise ValueError("Aligned photometry shape differs")
    if len(set(columns)) != len(columns) or not set(columns).issubset(
        {"W1", "W2", "G", "BP", "RP"}
    ):
        raise ValueError("Only explicit native magnitude columns supported")
    for index, column in enumerate(columns):
        if column not in names:
            continue
        j = names.index(column)
        measured = values[:, index]
        valid = verified & np.isfinite(measured) & (measured > -20) & (measured < 40)
        fill = valid & (~np.isfinite(raw[:, j]) | (raw[:, j] == -9999))
        raw[fill, j] = measured[fill]
        error = uncertainties[:, index]
        errors[fill, j] = np.where(
            np.isfinite(error[fill]) & (error[fill] > 0) & (error[fill] <= 5),
            error[fill],
            np.nan,
        )
        used |= fill
    release = np.asarray(fallback.get("release", "unknown")).astype(str)
    if release.ndim == 0:
        release = np.full(len(ids), str(release))
    if release.shape != ids.shape:
        raise ValueError("Photometry release must be scalar or aligned per star")
    lower = np.char.lower(release)
    release_code[used & np.isin(lower, ["gaia dr3", "dr3"])] = 3
    release_code[used & np.isin(lower, ["gaia edr3", "edr3"])] = 1
    release_code[used & np.isin(lower, ["gaia dr2", "dr2"])] = 2
    proof = dict(
        columns=columns,
        source_provenance=fallback.get("provenance", {}),
        values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        errors_sha256=hashlib.sha256(uncertainties.tobytes()).hexdigest(),
        identity_verified_sha256=hashlib.sha256(verified.tobytes()).hexdigest(),
        release_sha256=hashlib.sha256(release.astype("U64").tobytes()).hexdigest(),
        filled_star_count=int(used.sum()),
        unverified_identity_rows_rejected=int((~verified).sum()),
        domain_transfer_star_count=int((used & (release_code != 3)).sum()),
    )
    return used, release_code, proof


def lookup_evaluation_features(
    ids,
    names,
    center,
    scale,
    *,
    legacy_path,
    cache_path,
    original_path,
    gaia_vac=None,
    photometry_fallback=None,
    allow_unknown_photometry=False,
    chunk_rows=65536,
    output_directory=None,
    expected_cache_request_sha256=CACHE_REQUEST_SHA,
    expected_legacy_sha256=None,
    expected_cache_rows=150_000_000,
    association_rows=128,
    scan_original=True,
):
    """Preserve candidate order; unavailable rows remain explicit, never dropped.

    Returns ``(arrays, proof)``. Output matrices may be memmaps in a new directory.
    Cached normalized inputs are copied EXACTLY; ``raw`` for those rows is the
    inverse-scaled approximation and flagged, not asserted original raw data.
    All non-photometry/non-XP encoder fields remain masked.
    """
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids <= 0).any():
        raise ValueError("Positive integer Gaia candidate IDs required")
    if ids.dtype.kind == "u" and (ids > np.iinfo("i8").max).any():
        raise ValueError("Gaia ID outside signed integer range")
    if len(np.unique(ids)) != len(ids) or chunk_rows < 1:
        raise ValueError("Unique candidates and positive chunk size required")
    ids = ids.astype("<i8", copy=False)
    names = list(names)
    center, scale = np.asarray(center, dtype="f8"), np.asarray(scale, dtype="f8")
    if (
        center.shape != (len(names),)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Finite parent scaler with positive scales required")
    if len(set(names)) != len(names):
        raise ValueError("Unique feature names required")
    xp = [
        i
        for i, name in enumerate(names)
        if name in {f"{arm}_{j}" for arm in ("bp", "rp") for j in range(1, 56)}
    ]
    if len(xp) != 110:
        raise ValueError("Complete 110-column parent XP layout required")
    photo = [names.index(n) for n in ("W1", "W2", "G", "BP", "RP") if n in names]
    if "G" not in names or len(photo) < 2:
        raise ValueError("Real G and other photometry required")
    allowed = np.array([i in xp or i in photo for i in range(len(names))])
    directory = None if output_directory is None else Path(output_directory)
    if not isinstance(allow_unknown_photometry, bool):
        raise ValueError("Unknown-publication eligibility must be explicit boolean")
    if photometry_fallback is not None:
        fallback_ids = np.asarray(photometry_fallback["source_id"])
        if fallback_ids.dtype.kind not in "iu" or fallback_ids.ndim != 1:
            raise ValueError("Fallback source IDs must remain exact integers")
        np.testing.assert_array_equal(fallback_ids, ids)
        fallback_names = list(photometry_fallback["names"])
        fallback_values = np.asarray(photometry_fallback["values"])
        if (
            fallback_values.shape != (len(ids), len(fallback_names))
            or np.asarray(photometry_fallback["identity_verified"]).shape != ids.shape
        ):
            raise ValueError("Aligned photometry shape differs")
        if len(set(fallback_names)) != len(fallback_names) or not set(
            fallback_names
        ).issubset({"W1", "W2", "G", "BP", "RP"}):
            raise ValueError("Only explicit native magnitude columns supported")
    legacy_digest = _sha(legacy_path)
    if expected_legacy_sha256 is not None and legacy_digest != expected_legacy_sha256:
        raise ValueError("Legacy source content SHA differs")
    binding = dict(
        legacy_source_sha256=legacy_digest,
        candidate_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        names=names,
        center=center.tolist(),
        scale=scale.tolist(),
        chunk_rows=chunk_rows,
        cache_request_sha256=expected_cache_request_sha256,
        legacy_path=str(legacy_path),
        original_path=str(original_path),
        gaia_vac=gaia_vac,
        photometry_fallback_identity=None
        if photometry_fallback is None
        else dict(
            names=list(photometry_fallback["names"]),
            ids_sha256=hashlib.sha256(
                np.asarray(photometry_fallback["source_id"]).tobytes()
            ).hexdigest(),
            values_sha256=hashlib.sha256(
                np.asarray(photometry_fallback["values"], dtype="f8").tobytes()
            ).hexdigest(),
            errors_sha256=hashlib.sha256(
                np.asarray(
                    photometry_fallback.get(
                        "errors",
                        np.full(
                            np.asarray(photometry_fallback["values"]).shape,
                            np.nan,
                            dtype="f8",
                        ),
                    ),
                    dtype="f8",
                ).tobytes()
            ).hexdigest(),
            identity_verified_sha256=hashlib.sha256(
                np.asarray(photometry_fallback["identity_verified"], bool).tobytes()
            ).hexdigest(),
            provenance=photometry_fallback.get("provenance", {}),
            release_sha256=hashlib.sha256(
                np.asarray(photometry_fallback.get("release", "unknown"))
                .astype("U64")
                .tobytes()
            ).hexdigest(),
        ),
        allow_unknown_photometry=allow_unknown_photometry,
        helper_sha256=_sha(__file__),
        scan_original=scan_original,
    )
    progress = {}
    if directory is not None:
        if directory.exists():
            file = directory / "bindings.json"
            if not file.exists() or json.loads(file.read_text()) != binding:
                raise ValueError(
                    "Changed feature lookup binding; retained output not reusable"
                )
            if (directory / "progress.json").exists():
                progress = json.loads((directory / "progress.json").read_text())
        else:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "bindings.json").write_text(
                json.dumps(binding, indent=2) + "\n"
            )
    n, width = len(ids), len(names)
    raw = _array(directory, "raw", (n, width), "<f4", np.nan)
    safe = _array(directory, "input", (n, width), "<f4", -9999)
    visible = _array(directory, "visible", (n, width), "?", False)
    errors = _array(directory, "feature_errors", (n, width), "<f4", np.nan)
    origin = _array(directory, "feature_origin", (n,), "U24", "unavailable")
    cache_row = _array(directory, "cache_row", (n,), "i8", -1)
    original_row = _array(directory, "original_row", (n,), "i8", -1)
    original_shard = _array(directory, "original_shard", (n,), "i4", -1)
    inverse_raw = _array(directory, "inverse_raw", (n,), "?", False)

    def checkpoint(**updates):
        progress.update(updates)
        if directory is not None:
            for array in (
                raw,
                safe,
                visible,
                errors,
                origin,
                cache_row,
                original_row,
                original_shard,
                inverse_raw,
            ):
                array.flush()
            temporary = directory / "progress.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(directory / "progress.json")
        print(json.dumps({"feature_lookup_progress": progress}), flush=True)

    order = np.argsort(ids)
    sorted_ids = ids[order]
    proof = dict(
        scope="Source-only ID and feature lookup; no encoder/model/checkpoint loaded",
        ordered_candidate_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        feature_order=names,
        chunk_rows=chunk_rows,
        full_cache_feature_sha_recomputed=False,
        no_reconstruction_restaging=True,
    )
    with h5py.File(legacy_path, "r") as legacy:
        stored_names = json.loads(legacy.attrs["feature_columns_json"])
        if stored_names != names:
            raise ValueError("Legacy feature order differs")
        dataset = legacy["features/values"]
        fids = legacy["features/source_id"]
        for start in range(0, len(fids), chunk_rows):
            q, dest = match_ids(sorted_ids, order, fids[start : start + chunk_rows])
            keep = origin[dest] == "unavailable"
            q, dest = q[keep], dest[keep]
            if not len(dest):
                continue
            values = np.asarray(dataset[start : start + chunk_rows])[q]
            raw[dest] = values
            safe[dest], visible[dest] = _safe_input(values, center, scale, allowed)
            if "reported_feature_errors" in legacy["features"]:
                errors[dest] = legacy["features/reported_feature_errors"][
                    start : start + chunk_rows
                ][q]
            origin[dest] = "legacy_cohort"
        proof["legacy_path"] = str(legacy_path)
        proof["legacy_source_sha256"] = legacy_digest
    checkpoint(legacy_complete=True)
    features, cache_ids, provenance, manifest = _cache_maps(
        cache_path,
        names,
        center,
        scale,
        expected_cache_request_sha256,
        expected_cache_rows,
    )
    need = ~visible[:, xp].all(1)
    digest = hashlib.sha256()
    with (Path(cache_path) / "source_id.npy").open("rb") as stream:
        digest.update(stream.read(cache_ids.offset))
    recovered_cache = []
    for start in range(0, len(cache_ids), chunk_rows):
        block = np.asarray(cache_ids[start : start + chunk_rows])
        digest.update(block.tobytes())
        q, dest = match_ids(sorted_ids, order, block)
        wanted = need[dest]
        q, dest = q[wanted], dest[wanted]
        if not len(dest):
            continue
        locations = start + q
        values = np.asarray(features[locations])
        values[:, ~allowed] = -9999
        present = np.isfinite(values) & (values != -9999) & allowed[None]
        good = present[:, xp].all(1)
        locations, dest, values, present = (
            locations[good],
            dest[good],
            values[good],
            present[good],
        )
        safe[dest], visible[dest] = np.where(present, values, -9999), present
        raw[dest] = np.where(present, values.astype("f8") * scale + center, np.nan)
        errors[dest] = np.nan
        cache_row[dest] = locations
        original_row[dest] = provenance[locations]["row"]
        original_shard[dest] = provenance[locations]["shard"]
        origin[dest] = "original150m_cache"
        inverse_raw[dest] = True
        need[dest] = False
        recovered_cache.extend(dest.tolist())
    if digest.hexdigest() != manifest["files"]["source_id.npy"]["sha256"]:
        raise ValueError("Original150M source-ID content SHA differs")
    recovered_cache = np.flatnonzero(cache_row >= 0).tolist()
    checkpoint(cache_id_scan_complete=True, cache_recovered_count=len(recovered_cache))
    proof.update(
        cache_request_sha256=manifest["request_sha256"],
        cache_manifest_sha256=_sha(Path(cache_path) / "manifest.json"),
        cache_source_id_sha256=digest.hexdigest(),
        cache_files_transferred=manifest["files"],
        cache_recovered_count=len(recovered_cache),
    )
    source_identity = manifest["request"]["source"]
    stat = Path(original_path).stat()
    if (
        str(Path(original_path).resolve()) != source_identity["path"]
        or stat.st_size != source_identity["bytes"]
        or stat.st_mtime_ns != source_identity["mtime_ns"]
    ):
        raise ValueError("Original raw source path/size/mtime differs from cache proof")
    train_keys = manifest["request"]["train_keys"]
    valid_keys = manifest["request"]["validation_keys"]
    with h5py.File(original_path, "r") as original:
        if set(original) != set(train_keys) | set(valid_keys):
            raise ValueError("Original shard inventory differs")
        for key in [*train_keys, *valid_keys]:
            dataset = original[key]
            schema = manifest["request"]["schema"][key]
            if list(dataset.shape) != schema["shape"] or not {
                "source_id",
                *names,
            }.issubset(dataset.dtype.names or ()):
                raise ValueError("Original shard schema differs: " + key)
        sampled = np.asarray(recovered_cache, dtype="i8")
        if len(sampled) > association_rows:
            sampled = sampled[
                np.linspace(0, len(sampled) - 1, association_rows, dtype=int)
            ]
        for row in sampled if not progress.get("cache_association_passed") else []:
            location = int(cache_row[row])
            pr = provenance[location]
            shard, source_row = int(pr["shard"]), int(pr["row"])
            record = original[train_keys[shard]][source_row : source_row + 1]
            if record["source_id"][0] != ids[row]:
                raise ValueError("Recovered cache/source ID association differs")
            actual = (
                (_raw_values(record, names).astype("f8") - center) / scale
            ).astype("f4")
            cached = np.asarray(features[location])
            np.testing.assert_array_equal(actual, cached[None])
            original_row[row], original_shard[row] = source_row, shard
        proof["cache_raw_association_rows_verified"] = len(sampled)
        checkpoint(cache_association_passed=True)
        scanned = int(progress.get("original_id_rows_scanned", 0))
        if scan_original:
            for shard, key in enumerate([*train_keys, *valid_keys]):
                if not need.any():
                    break
                dataset = original[key]
                last_shard = int(progress.get("original_completed_shard", -1))
                last_stop = int(progress.get("original_completed_stop", 0))
                if shard < last_shard:
                    continue
                for start in range(0, len(dataset), chunk_rows):
                    if shard == last_shard and start < last_stop:
                        continue
                    query = dataset.fields("source_id")[start : start + chunk_rows]
                    q, dest = match_ids(sorted_ids, order, query)
                    wanted = need[dest]
                    q, dest = q[wanted], dest[wanted]
                    scanned += len(query)
                    if not len(dest):
                        checkpoint(
                            original_completed_shard=shard,
                            original_completed_stop=start + len(query),
                            original_id_rows_scanned=scanned,
                        )
                        continue
                    records = dataset[start : start + len(query)][q]
                    values = _raw_values(records, names)
                    inputs, present = _safe_input(values, center, scale, allowed)
                    good = present.any(1)
                    q, dest, values, inputs, present = (
                        q[good],
                        dest[good],
                        values[good],
                        inputs[good],
                        present[good],
                    )
                    raw[dest], safe[dest], visible[dest] = values, inputs, present
                    errors[dest] = np.nan
                    original_row[dest], original_shard[dest] = start + q, shard
                    origin[dest] = "original_raw_hdf5"
                    inverse_raw[dest] = False
                    need[dest] = False
                    checkpoint(
                        original_completed_shard=shard,
                        original_completed_stop=start + len(query),
                        original_id_rows_scanned=scanned,
                    )
        proof["original_id_rows_scanned"] = scanned
    published, known, vac_rows = (
        np.zeros(n, bool),
        np.zeros(n, bool),
        np.full(n, -1, dtype="i8"),
    )
    if gaia_vac is not None:
        published, known, vac_rows, proof["gaia_vac"] = _vac_projection(
            gaia_vac, ids, raw, errors, names
        )
        changed = (~visible[:, xp].any(1)) & known & ~published
        safe[changed], visible[changed] = _safe_input(
            raw[changed], center, scale, allowed
        )
        origin[changed] = "gaia_vac_photometry"
    fallback_used, release_code, fallback_proof = aligned_photometry_fallback(
        photometry_fallback, ids, names, raw, errors
    )
    if fallback_used.any():
        new_input, new_visible = _safe_input(raw[fallback_used], center, scale, allowed)
        old_visible = np.asarray(visible[fallback_used])
        add = new_visible & ~old_visible
        existing_input = np.asarray(safe[fallback_used])
        existing_input[add] = new_input[add]
        safe[fallback_used] = existing_input
        visible[fallback_used] = old_visible | new_visible
        origin[(origin == "unavailable") & fallback_used] = "catalogue_photometry"
    if fallback_proof:
        proof["photometry_fallback"] = fallback_proof
    actual = visible[:, xp]
    complete, partial, missing = (
        actual.all(1),
        actual.any(1) & ~actual.all(1),
        ~actual.any(1),
    )
    no_xp = missing & known & ~published
    conflict = actual.any(1) & known & ~published
    status = np.full(n, "unknown", dtype="U48")
    status[missing & known & published] = "xp_published_not_staged"
    status[complete] = "complete_xp"
    status[partial] = "partial_xp"
    status[no_xp] = "official_no_xp"
    status[conflict] = "availability_conflict"
    photo_ok = visible[:, photo].sum(1) >= 2
    photo_ok &= visible[:, names.index("G")]
    unknown_photo = missing & ~known & photo_ok
    status[unknown_photo] = "photometry_only_publication_unknown"
    photometry_only = missing & photo_ok
    recoverable = (
        photo_ok
        & (actual.any(1) | no_xp | (photometry_only & allow_unknown_photometry))
        & ~conflict
    )
    proof["counts"] = {str(k): int((status == k).sum()) for k in np.unique(status)}
    proof["recoverable_count"] = int(recoverable.sum())
    checkpoint(feature_lookup_complete=True, recoverable_count=int(recoverable.sum()))
    arrays = dict(
        raw=raw,
        input=safe,
        visible=visible,
        feature_errors=errors,
        feature_origin=origin,
        feature_recoverable=recoverable,
        cache_row=cache_row,
        original_row=original_row,
        original_shard=original_shard,
        raw_reconstructed_from_normalized_cache=inverse_raw,
        xp_status=status,
        real_xp_present=complete,
        input_has_xp=actual.any(1),
        real_no_xp=no_xp,
        official_has_xp_known=known,
        official_has_xp_continuous=published,
        gaia_vac_source_row=vac_rows,
        photometry_fallback_used=fallback_used,
        photometry_release_code=release_code,
        photometry_is_domain_transfer=fallback_used & (release_code != 3),
        input_xp_missing_publication_unknown=unknown_photo,
    )
    for value in arrays.values():
        if isinstance(value, np.memmap):
            value.flush()
    return arrays, proof
