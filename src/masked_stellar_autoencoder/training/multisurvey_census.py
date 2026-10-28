"""Bounded exact-ID DESI source and feature-cache census; no model execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .evaluation_feature_lookup import CACHE_REQUEST_SHA, _cache_maps, _raw_values


def _stat(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _verify(path, expected):
    path = Path(path)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Expected content SHA256 required")
    before = _stat(path)
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if _stat(path) != before or actual != expected:
        raise ValueError("Bound source changed or content SHA256 differs")
    return before


def _sorted_ids(values):
    ids = np.asarray(values)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids <= 0).any():
        raise ValueError("Positive exact integer IDs required")
    if ids.dtype.kind == "u" and (ids > np.iinfo("i8").max).any():
        raise ValueError("Source identity exceeds int64")
    return ids.astype("<i8", copy=False)


def source_union(source_ids, observed):
    ids = _sorted_ids(source_ids)
    mask = np.asarray(observed)
    if mask.dtype.kind != "b" or mask.shape != (len(ids), 3):
        raise ValueError("Native observed labels must be boolean [rows,3]")
    unique, inverse, count = np.unique(ids, return_inverse=True, return_counts=True)
    per_star = np.zeros((len(unique), 3), dtype=bool)
    np.logical_or.at(per_star, inverse, mask)
    return unique, count, per_star


def join_base(candidate_ids, base_ids, roles, primary, exposed, reserved):
    candidates, base = _sorted_ids(candidate_ids), _sorted_ids(base_ids)
    if len(np.unique(candidates)) != len(candidates) or np.any(
        np.diff(candidates) <= 0
    ):
        raise ValueError("Candidates must be sorted and unique")
    if len(np.unique(base)) != len(base) or np.any(np.diff(base) <= 0):
        raise ValueError("Base IDs must be sorted and unique")
    role = np.asarray(roles).astype("U16")
    fields = [np.asarray(a) for a in (primary, exposed, reserved)]
    if role.shape != base.shape or any(
        v.shape != base.shape or v.dtype.kind != "b" for v in fields
    ):
        raise ValueError("Base roles and masks must align")
    pos = np.searchsorted(base, candidates)
    match = pos < len(base)
    ix = np.flatnonzero(match)
    match[ix] &= base[pos[ix]] == candidates[ix]
    out_role = np.full(len(candidates), "not_in_base", dtype="U16")
    out_role[match] = role[pos[match]]
    flags = []
    for src in fields:
        dest = np.zeros(len(candidates), dtype=bool)
        dest[match] = src[pos[match]]
        flags.append(dest)
    return match, out_role, *flags


def scan_cache(
    candidates,
    cache_ids,
    features,
    xp_indices,
    *,
    chunk_rows=65536,
    hash_prefix=b"",
    progress=None,
):
    candidates = _sorted_ids(candidates)
    if np.any(np.diff(candidates) <= 0):
        raise ValueError("Candidates must be sorted and unique")
    if chunk_rows < 1 or features.shape[0] != len(cache_ids):
        raise ValueError("Cache feature and identity rows must align")
    xp_indices = np.asarray(xp_indices, dtype=int)
    if (
        xp_indices.ndim != 1
        or not len(xp_indices)
        or np.any(xp_indices < 0)
        or np.any(xp_indices >= features.shape[1])
    ):
        raise ValueError("Valid XP indices required")
    cache_row = np.full(len(candidates), -1, dtype="i8")
    xp_complete = np.zeros(len(candidates), dtype=bool)
    digest = hashlib.sha256(hash_prefix)
    invalid_ids = 0
    for start in range(0, len(cache_ids), chunk_rows):
        block = np.asarray(cache_ids[start : start + chunk_rows])
        if block.ndim != 1 or block.dtype != np.dtype("<i8"):
            raise ValueError("Cache IDs must retain their exact int64 representation")
        digest.update(block.tobytes())
        invalid_ids += int((block <= 0).sum())
        pos = np.searchsorted(candidates, block)
        match = (pos < len(candidates)) & (block > 0)
        valid = np.flatnonzero(match)
        match[valid] &= candidates[pos[valid]] == block[valid]
        selected = np.flatnonzero(match)
        dst = pos[selected]
        if len(np.unique(dst)) != len(dst) or (cache_row[dst] >= 0).any():
            raise ValueError("Repeated candidate identity in cache")
        locations = start + selected
        cache_row[dst] = locations
        x = np.asarray(features[locations])[:, xp_indices]
        xp_complete[dst] = (np.isfinite(x) & (x != -9999)).all(axis=1)
        if progress is not None and (start // chunk_rows) % 64 == 0:
            progress(min(start + len(block), len(cache_ids)), len(cache_ids))
    if progress is not None:
        progress(len(cache_ids), len(cache_ids))
    return cache_row, xp_complete, digest.hexdigest(), invalid_ids


def verify_cache_associations(
    original_path,
    ids,
    cache_row,
    features,
    provenance,
    names,
    center,
    scale,
    request,
    *,
    sample_size=128,
):
    original_path = Path(original_path)
    info = request["source"]
    stat = original_path.stat()
    if (
        str(original_path.resolve()) != info["path"]
        or stat.st_size != info["bytes"]
        or stat.st_mtime_ns != info["mtime_ns"]
    ):
        raise ValueError("Original cache source identity differs")
    matched = np.flatnonzero(cache_row >= 0)
    if len(matched) > sample_size:
        matched = matched[np.linspace(0, len(matched) - 1, sample_size, dtype=int)]
    train_keys, valid_keys = request["train_keys"], request["validation_keys"]
    with h5py.File(original_path, "r") as raw:
        if set(raw) != set(train_keys) | set(valid_keys):
            raise ValueError("Original cache shard inventory differs")
        for key in [*train_keys, *valid_keys]:
            if list(raw[key].shape) != request["schema"][key]["shape"] or not {
                "source_id",
                *names,
            }.issubset(raw[key].dtype.names or ()):
                raise ValueError("Original cache shard schema differs")
        for row in matched:
            loc = int(cache_row[row])
            reference = provenance[loc]
            shard, index = int(reference["shard"]), int(reference["row"])
            if not 0 <= shard < len(train_keys):
                raise ValueError("Cache provenance references a non-training shard")
            record = raw[train_keys[shard]][index : index + 1]
            if len(record) != 1 or record["source_id"][0] != ids[row]:
                raise ValueError("Cache/source identity association differs")
            normalized = (
                (_raw_values(record, names).astype("f8") - center) / scale
            ).astype("f4")
            np.testing.assert_array_equal(normalized, np.asarray(features[loc])[None])
    return int(len(matched))


def summarize(
    ids,
    repeats,
    observed,
    base_match,
    roles,
    primary,
    exposed,
    reserved,
    cache_row,
    cache_xp,
):
    n = len(ids)
    if any(
        len(x) != n
        for x in (
            repeats,
            observed,
            base_match,
            roles,
            primary,
            exposed,
            reserved,
            cache_row,
            cache_xp,
        )
    ):
        raise ValueError("Census arrays do not align")
    role_counts = {
        str(k): int(v)
        for k, v in zip(*np.unique(roles, return_counts=True), strict=True)
    }
    return {
        "unique_desi_ids": n,
        "quality_rows": int(repeats.sum()),
        "ids_with_multiple_rows": int((repeats > 1).sum()),
        "observed_per_target_distinct_ids": observed.sum(axis=0).tolist(),
        "base_matched_ids": int(base_match.sum()),
        "base_role_counts": role_counts,
        "base_primary_input_ids": int((base_match & primary).sum()),
        "base_known_supervised_ancestor_ids": int((base_match & exposed).sum()),
        "base_reserved_reference_ids": int((base_match & reserved).sum()),
        "cache_matched_ids": int((cache_row >= 0).sum()),
        "cache_complete_xp_ids": int(cache_xp.sum()),
        "cache_complete_xp_not_in_base": int((cache_xp & ~base_match).sum()),
        "scope": "Distinct DESI identities and cache membership only. Nonmatches are not proof of absent features or absent Gaia XP. Roles/exposure outside base remain unknown; repeat spectra and label dependencies are unresolved. No FIT or independent TEST claim.",
    }


def census(
    sidecar,
    base,
    cache,
    *,
    original_path,
    sidecar_sha,
    base_sha,
    output,
    expected_cache_rows=150_000_000,
    chunk_rows=65536,
    progress=None,
):
    """Verify immutable sources, stream cache IDs and retain exact join arrays."""
    sidecar, base, cache, output = map(Path, (sidecar, base, cache, output))
    if output.exists():
        raise FileExistsError(output)
    sidecar_stat, base_stat = _verify(sidecar, sidecar_sha), _verify(base, base_sha)
    with h5py.File(sidecar, "r") as h:
        g = h["native_observations/desi_dr1_rvs_raw"]
        ids, repeats, observed = source_union(g["source_id"][:], g["observed"][:])
    with h5py.File(base, "r") as h:
        base_ids = h["source_id"][:]
        base_match, roles, primary, exposed, reserved = join_base(
            ids,
            base_ids,
            h["role"][:],
            h["primary_input_eligible"][:],
            h["known_supervised_ancestor_exposed"][:],
            h["reserved_reference"][:],
        )
        names = h["feature_names"][:].astype("U").tolist()
        center, scale = h["scaler_center"][:], h["scaler_scale"][:]
    watched_cache_files = [
        cache / name
        for name in ("manifest.json", "features.npy", "source_id.npy", "provenance.npy")
    ]
    cache_stats = [_stat(path) for path in watched_cache_files]
    original_stat = _stat(original_path)
    features, cache_ids, provenance, manifest = _cache_maps(
        cache, names, center, scale, CACHE_REQUEST_SHA, expected_cache_rows
    )
    xp = [names.index(f"{arm}_{j}") for arm in ("bp", "rp") for j in range(1, 56)]
    with (cache / "source_id.npy").open("rb") as stream:
        prefix = stream.read(cache_ids.offset)
    rows, xp_complete, cache_sha, invalid_cache_ids = scan_cache(
        ids,
        cache_ids,
        features,
        xp,
        chunk_rows=chunk_rows,
        hash_prefix=prefix,
        progress=progress,
    )
    if cache_sha != manifest["files"]["source_id.npy"]["sha256"]:
        raise ValueError("Cache identity content SHA256 differs")
    association_count = verify_cache_associations(
        original_path,
        ids,
        rows,
        features,
        provenance,
        names,
        center,
        scale,
        manifest["request"],
    )
    if (
        _stat(sidecar) != sidecar_stat
        or _stat(base) != base_stat
        or _stat(original_path) != original_stat
        or [_stat(path) for path in watched_cache_files] != cache_stats
    ):
        raise ValueError("Source or cache changed during census")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "identities.npz",
        source_id=ids,
        quality_rows=repeats,
        observed=observed,
        base_match=base_match,
        base_role=roles,
        base_primary_input=primary,
        base_known_ancestor=exposed,
        base_reserved=reserved,
        cache_row=rows,
        cache_complete_xp=xp_complete,
    )
    with (output / "identities.npz").open("rb") as f:
        result_sha = hashlib.file_digest(f, "sha256").hexdigest()
    summary = summarize(
        ids,
        repeats,
        observed,
        base_match,
        roles,
        primary,
        exposed,
        reserved,
        rows,
        xp_complete,
    )
    summary.update(
        sidecar_sha256=sidecar_sha,
        base_sha256=base_sha,
        cache_request_sha256=CACHE_REQUEST_SHA,
        cache_source_id_sha256=cache_sha,
        invalid_cache_ids=invalid_cache_ids,
        identities_sha256=result_sha,
        cache_feature_bytes_only_not_full_sha=True,
        cache_raw_association_rows_verified=association_count,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary
