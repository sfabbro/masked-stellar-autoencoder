"""Source-only Gaia RVS to existing feature-cache census.

All source rows are retained. Gaia M/H and alpha are source-specific observed
coordinates, not direct Fe/H. A cache absence or outside-base identity says
nothing about a star's XP publication status or unrecorded ancestral exposure.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
    CACHE_REQUEST_SHA,
    _cache_maps,
)
from masked_stellar_autoencoder.training.multisurvey_census import (
    _stat,
    _verify,
    join_base,
    scan_cache,
    verify_cache_associations,
)
from masked_stellar_autoencoder.training.multisurvey_gaia_gspspec import (
    REGISTRY,
    quality_masks,
)

TARGETS = ("teff", "logg", "mh_gspspec", "alphafe_gspspec")
REQUIRED = {
    "source_id",
    "teff_gspspec",
    "logg_gspspec",
    "mh_gspspec",
    "alphafe_gspspec",
    "flags_gspspec",
}


def _read_gaia(source, expected_rows):
    """Return sorted exact IDs, native row numbers and separate target masks."""
    with fits.open(source, memmap=True) as hdus:
        if len(hdus) < 2:
            raise ValueError("Gaia FITS binary table is missing")
        data = hdus[1].data
        if len(data) != expected_rows:
            raise ValueError("Gaia source row count differs")
        if not REQUIRED.issubset(data.dtype.names or ()):
            raise ValueError("Gaia source lacks the required native fields")
        ids = np.asarray(data["source_id"])
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids <= 0).any():
            raise ValueError("Positive exact integer Gaia source IDs required")
        if ids.dtype.kind == "u" and (ids > np.iinfo("i8").max).any():
            raise ValueError("Gaia source ID exceeds int64")
        ids = ids.astype("<i8", copy=False)
        order = np.argsort(ids, kind="stable")
        ids = np.array(ids[order], dtype="<i8", copy=True)
        if np.any(ids[1:] <= ids[:-1]):
            raise ValueError("Gaia source IDs must be unique")
        masks = quality_masks(
            data["teff_gspspec"],
            data["logg_gspspec"],
            data["flags_gspspec"],
            mh=data["mh_gspspec"],
            alpha=data["alphafe_gspspec"],
        )
        raw = np.column_stack(
            (
                masks["raw"][:, 0],
                masks["raw"][:, 1],
                masks["raw_mh"],
                masks["raw_alpha"],
            )
        )[order]
        selected = np.column_stack(
            (
                masks["selected"][:, 0],
                masks["selected"][:, 1],
                masks["selected_mh"],
                masks["selected_alpha"],
            )
        )[order]
    if raw.dtype.kind != "b" or selected.dtype.kind != "b" or (selected & ~raw).any():
        raise ValueError("Native quality masks are invalid")
    return ids, order.astype("<i8", copy=False), raw, selected


def _tri_state(known, values):
    """-1 means not known in the base, 0 unlisted/false, and 1 listed/true."""
    return np.where(known, np.asarray(values, dtype=bool).astype("i1"), -1).astype("i1")


def _mask_counts(condition, observed):
    return {
        name: int(value)
        for name, value in zip(TARGETS, observed[condition].sum(axis=0), strict=True)
    }


def _write_arrays(path, arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Gaia cache census NPZ CRC failed")
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("Gaia cache census fields changed")
        for name, value in arrays.items():
            if not np.array_equal(saved[name], value):
                raise ValueError("Gaia cache census values changed: " + name)
    temporary.replace(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest


def gaia_cache_census(
    source,
    base,
    cache,
    *,
    original_path,
    source_sha,
    base_sha,
    output,
    expected_source_rows=5_591_594,
    expected_cache_rows=150_000_000,
    chunk_rows=65_536,
    progress=None,
):
    """Verify and census full Gaia RVS identities against base/cache, no model.

    All Gaia source IDs survive whether or not any individual label or feature
    is present. Base-state and cache-XP arrays explicitly encode unknown as -1.
    """
    source, base, cache, original_path, output = map(
        Path, (source, base, cache, original_path, output)
    )
    if output.exists():
        raise FileExistsError(output)
    for name, value in (
        ("expected_source_rows", expected_source_rows),
        ("expected_cache_rows", expected_cache_rows),
        ("chunk_rows", chunk_rows),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + " must be a positive integer")
    source_stat = _verify(source, source_sha)
    base_stat = _verify(base, base_sha)
    ids, source_rows, raw, selected = _read_gaia(source, expected_source_rows)
    with h5py.File(base, "r") as h:
        base_match, role, primary, exposed, reserved = join_base(
            ids,
            h["source_id"][:],
            h["role"][:],
            h["primary_input_eligible"][:],
            h["known_supervised_ancestor_exposed"][:],
            h["reserved_reference"][:],
        )
        names = h["feature_names"][:].astype("U").tolist()
        center, scale = h["scaler_center"][:], h["scaler_scale"][:]

    watched_cache = [
        cache / name
        for name in ("manifest.json", "features.npy", "source_id.npy", "provenance.npy")
    ]
    cache_stats = [_stat(path) for path in watched_cache]
    original_stat = _stat(original_path)
    features, cache_ids, provenance, cache_manifest = _cache_maps(
        cache, names, center, scale, CACHE_REQUEST_SHA, expected_cache_rows
    )
    xp = [names.index(f"{arm}_{j}") for arm in ("bp", "rp") for j in range(1, 56)]
    with (cache / "source_id.npy").open("rb") as stream:
        prefix = stream.read(cache_ids.offset)
    cache_row, cache_complete_xp, cache_sha, invalid = scan_cache(
        ids,
        cache_ids,
        features,
        xp,
        chunk_rows=chunk_rows,
        hash_prefix=prefix,
        progress=progress,
    )
    if cache_sha != cache_manifest["files"]["source_id.npy"]["sha256"]:
        raise ValueError("Cache source-ID content SHA256 differs")
    verified_associations = verify_cache_associations(
        original_path,
        ids,
        cache_row,
        features,
        provenance,
        names,
        center,
        scale,
        cache_manifest["request"],
    )
    if (
        _stat(source) != source_stat
        or _stat(base) != base_stat
        or _stat(original_path) != original_stat
        or [_stat(path) for path in watched_cache] != cache_stats
    ):
        raise ValueError("Source, base, original data or cache changed during census")

    cache_match = cache_row >= 0
    if (cache_complete_xp & ~cache_match).any():
        raise ValueError("Complete cached XP requires a cache identity")
    arrays = {
        "source_id": ids,
        "source_row": source_rows,
        "raw_available": raw,
        "selected_usable": selected,
        "base_match": base_match,
        "base_role": role,
        "base_primary_state": _tri_state(base_match, primary),
        "base_ancestor_listed_state": _tri_state(base_match, exposed),
        "base_reserved_state": _tri_state(base_match, reserved),
        "cache_row": cache_row,
        "cache_xp_state": _tri_state(cache_match, cache_complete_xp),
    }
    conditions = {
        "all_source_rows": np.ones(len(ids), dtype=bool),
        "matched_base": base_match,
        "outside_base": ~base_match,
        "matched_cache": cache_match,
        "complete_xp_in_cache": cache_match & cache_complete_xp,
        "complete_xp_in_cache_outside_base": cache_match
        & cache_complete_xp
        & ~base_match,
    }
    groups = {
        name: {
            "stars": int(mask.sum()),
            "raw": _mask_counts(mask, raw),
            "selected": _mask_counts(mask, selected),
        }
        for name, mask in conditions.items()
    }
    role_counts = {}
    for name in np.unique(role[base_match]):
        mask = base_match & (role == name)
        eligible = mask & primary
        role_counts[str(name)] = {
            "matched_stars": int(mask.sum()),
            "primary_input_stars": int(eligible.sum()),
            "selected_any_input": _mask_counts(mask, selected),
            "selected_primary_input": _mask_counts(eligible, selected),
        }

    output.mkdir(parents=True, exist_ok=False)
    identities_path = output / "identities.npz"
    identities_sha = _write_arrays(identities_path, arrays)
    summary = {
        "schema": "multisurvey-gaia-cache-census-v1",
        "source_only": True,
        "model_loaded": False,
        "training_performed": False,
        "target_order": list(TARGETS),
        "source_path": str(source.resolve()),
        "source_sha256": source_sha,
        "source_rows": len(ids),
        "base_path": str(base.resolve()),
        "base_sha256": base_sha,
        "cache_request_sha256": CACHE_REQUEST_SHA,
        "cache_source_id_sha256": cache_sha,
        "cache_invalid_ids": invalid,
        "cache_raw_association_rows_verified": verified_associations,
        "cache_feature_bytes_only_not_full_sha": True,
        "identities_path": str(identities_path.resolve()),
        "identities_sha256": identities_sha,
        "identities_bytes": identities_path.stat().st_size,
        "identities_crc": "pass",
        "identities_roundtrip": "pass",
        "counts": groups,
        "matched_base_roles": role_counts,
        "quality_policy": REGISTRY["quality_policy"],
        "tri_state_semantics": {
            "base_primary_state": "-1 not in base, 0 false, 1 true",
            "base_ancestor_listed_state": "-1 not in base, 0 not listed, 1 listed; a zero does not certify complete ancestry",
            "base_reserved_state": "-1 not in base, 0 false, 1 true",
            "cache_xp_state": "-1 absent from this cache, 0 cache row incomplete, 1 complete finite 110 coefficients in this cache",
        },
        "scope": "Exact Gaia DR3 source IDs, source-specific quality masks and cache membership only; no Fe/H conversion, new split or ancestry certification. An absent cache row is not evidence of absent Gaia XP. Counts overlap by target and cannot be added.",
    }
    summary_path = output / "summary.json"
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(summary_path)
    return summary
