"""Build a unique-spectrum DESI sidecar and immutable base-source overlay."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .multisurvey_overlay import (
    build_overlay,
    file_sha256,
    validate_overlay,
    write_observation_sidecar,
)


def select_one_per_star(source_id, source_row, snr):
    """Highest native S/N per exact Gaia ID; ties use smallest original row.

    Selection never examines the target values or model outputs. The original
    duplicate observations remain in the original SHA-bound source file.
    """
    ids, rows, score = np.asarray(source_id), np.asarray(source_row), np.asarray(snr)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or rows.shape != ids.shape
        or rows.dtype.kind not in "iu"
        or score.shape != ids.shape
    ):
        raise ValueError("Aligned exact integer IDs/rows and S/N required")
    if (ids <= 0).any() or (rows < 0).any() or len(np.unique(rows)) != len(rows):
        raise ValueError(
            "Positive star IDs and unique nonnegative source rows required"
        )
    if (ids.dtype.kind == "u" and (ids > np.iinfo("i8").max).any()) or (
        rows.dtype.kind == "u" and (rows > np.iinfo("i8").max).any()
    ):
        raise ValueError("Identity exceeds signed int64")
    if (
        not np.issubdtype(score.dtype, np.number)
        or not np.isfinite(score).all()
        or (score < 0).any()
    ):
        raise ValueError("S/N must be finite and nonnegative")
    order = np.lexsort((rows.astype("i8"), -score.astype("f8"), ids.astype("i8")))
    sorted_ids = ids[order]
    chosen = order[np.r_[True, np.diff(sorted_ids) != 0]] if len(order) else order
    return np.sort(chosen)


def create_selected_sidecar(raw_sidecar, expected_sha, output):
    """Retain all columns for one selected native row per star in a new file."""
    raw_sidecar, output = Path(raw_sidecar), Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = raw_sidecar.stat()
    if file_sha256(raw_sidecar) != expected_sha:
        raise ValueError("Raw DESI sidecar SHA differs")
    with h5py.File(raw_sidecar, "r") as h:
        group = h["native_observations/desi_dr1_rvs_raw"]
        registry = json.loads(group.attrs["registry_json"])
        ids, rows, snr = (
            group["source_id"][:],
            group["source_row"][:],
            group["snr_r"][:],
        )
        chosen = select_one_per_star(ids, rows, snr)
        data = {name: ds[:][chosen] for name, ds in group.items()}
        original_provenance = json.loads(h.attrs["provenance_json"])
    after = raw_sidecar.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Raw DESI sidecar changed during selection")
    if len(np.unique(data["source_id"])) != len(data["source_id"]):
        raise ValueError("Selection retained duplicate identities")
    provenance = {
        "raw_sidecar_path": str(raw_sidecar.resolve()),
        "raw_sidecar_sha256": expected_sha,
        "selection": "highest SN_R; tie-break smallest original source_row; target values and model outputs not used",
        "raw_rows": len(ids),
        "selected_rows": len(chosen),
        "original_provenance": original_provenance,
        "scope": "repeat-resolution view only; no physical calibration or target-quality certification",
    }
    receipt = write_observation_sidecar(
        output,
        {"desi_dr1_rvs_raw": data},
        {"desi_dr1_rvs_raw": registry},
        provenance=provenance,
    )
    return {
        **receipt,
        "raw_rows": len(ids),
        "selected_rows": len(chosen),
        "discarded_from_view": len(ids) - len(chosen),
    }


def summarize_existing_roles(parent, selected):
    """Exact overlap against existing old-source logg masks by base role."""
    with h5py.File(parent, "r") as base, h5py.File(selected, "r") as selected_file:
        ids = base["source_id"][:]
        if ids.dtype.kind not in "iu" or np.any(np.diff(ids) <= 0):
            raise ValueError("Base IDs must be sorted unique exact integers")
        group = selected_file["native_observations/desi_dr1_rvs_raw"]
        stars = group["source_id"][:]
        observed = group["observed"][:]
        if len(np.unique(stars)) != len(stars):
            raise ValueError("Selected DESI stars must be unique")
        index = np.searchsorted(ids, stars)
        match = index < len(ids)
        valid = np.flatnonzero(match)
        match[valid] &= ids[index[valid]] == stars[valid]
        matched_index = index[match]
        has_desi = np.zeros(len(ids), bool)
        has_desi[matched_index] = observed[match, 1]
        roles = base["role"][:].astype("U16")
        primary = base["primary_input_eligible"][:].astype(bool)
        current = base["observed"][:, 1].astype(bool) & base["quality"][:, 1].astype(
            bool
        )
        rows = {}
        for role in ("fit", "tune", "cal", "devtest", "excluded", "external"):
            mask = roles == role
            rows[role] = {
                "base_stars": int(mask.sum()),
                "primary_input_desi_logg": int((mask & primary & has_desi).sum()),
                "previous_logg": int((mask & current).sum()),
                "overlap_desi_and_previous_logg": int(
                    (mask & primary & has_desi & current).sum()
                ),
                "desi_logg_not_previously_observed": int(
                    (mask & primary & has_desi & ~current).sum()
                ),
            }
    return {
        "roles": rows,
        "selected_desi_ids": len(stars),
        "selected_desi_not_in_base": int((~match).sum()),
        "scope": "Exact source-role/mask overlap only; labels, errors, source dependencies and physical scale still require scientific validation. Original FIT/held-out roles unchanged.",
    }


def create_overlay(
    parent, parent_sha, parent_registry, raw_sidecar, sidecar_sha, output
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = create_selected_sidecar(
        raw_sidecar, sidecar_sha, output / "desi-selected.h5"
    )
    manifest = build_overlay(
        output / "overlay.h5",
        parent_path=parent,
        parent_sha256=parent_sha,
        parent_registry=parent_registry,
        sidecars=[selected],
        manifest_path=output / "overlay-manifest.json",
    )
    proof = validate_overlay(manifest)
    counts = summarize_existing_roles(parent, selected["path"])
    return {
        "selected": selected,
        "overlay_manifest_path": str(manifest.resolve()),
        "overlay_manifest_sha256": file_sha256(manifest),
        "overlay_sha256": proof["overlay"]["sha256"],
        "counts": counts,
    }
