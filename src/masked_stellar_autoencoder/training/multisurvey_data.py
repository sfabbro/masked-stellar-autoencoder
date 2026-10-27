"""Star-indexed sparse survey observations; never require a survey intersection.

This reader consumes the preserved ``native_observations`` records of the
remediation source. It retains original row identities and all observations.
Survey comparability, covariance dependencies, and feature provenance remain
explicit responsibilities of the supplied registry and caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _has_linked_objects(group: h5py.Group, seen: set[int] | None = None) -> bool:
    """Detect undeclared linked attachments even if an overlay schema was removed."""
    seen = set() if seen is None else seen
    address = h5py.h5o.get_info(group.id).addr
    if address in seen:
        return False
    seen.add(address)
    for key in group:
        if not isinstance(group.get(key, getlink=True), h5py.HardLink):
            return True
        value = group[key]
        if isinstance(value, h5py.Group) and _has_linked_objects(value, seen):
            return True
    return False


class SparseSurveyDataset(Dataset):
    """One feature row per star, with a sparse list of native survey records.

    The index joins exact integer Gaia source IDs, retaining every matching
    native row. It does not read merged/preference-selected label arrays.
    ``registry`` maps each HDF5 source group to release, pipeline and dependency
    identifiers. Only explicitly registered groups are included. An unknown
    measurement error is represented by NaN and a false ``error_known`` flag.
    """

    def __init__(
        self,
        path: str | Path,
        registry: dict[str, dict[str, Any]],
        *,
        role: str,
        require_primary_input: bool = True,
        expected_source_sha256: str | None = None,
        _verified_overlay_access: Any = None,
    ) -> None:
        self.path = str(path)
        if role not in {"fit", "tune", "cal", "devtest", "external", "excluded"}:
            raise ValueError("An explicit known star role is required")
        self.registry = {str(k): dict(v) for k, v in registry.items()}
        self.registry_sha256 = hashlib.sha256(
            json.dumps(self.registry, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.source_binding = os.path.realpath(self.path)
        self._verified_overlay_access = _verified_overlay_access
        from .multisurvey_overlay import OVERLAY_SCHEMA, _VerifiedOverlayAccess

        with h5py.File(self.path, "r") as source:
            linked = source.attrs.get(
                "schema"
            ) == OVERLAY_SCHEMA or _has_linked_objects(source)
        if linked:
            if not isinstance(_verified_overlay_access, _VerifiedOverlayAccess):
                raise ValueError(
                    "External-link overlays require sparse_dataset_from_manifest "
                    "with a pinned manifest SHA256"
                )
            _verified_overlay_access.check(self.path, self.registry_sha256)
            if expected_source_sha256 != _verified_overlay_access.overlay_sha256:
                raise ValueError("Overlay digest differs from verified receipt")
            self.source_binding = (
                _verified_overlay_access.overlay_sha256
                + ":"
                + _verified_overlay_access.manifest_sha256
            )
        elif _verified_overlay_access is not None:
            raise ValueError(
                "Verified overlay receipt supplied for an unlinked dataset"
            )
        elif expected_source_sha256 is not None:
            if len(expected_source_sha256) != 64:
                raise ValueError("Expected source SHA must be a SHA256 digest")
            with open(self.path, "rb") as stream:
                actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_sha != expected_source_sha256:
                raise ValueError("Source SHA does not match its manifest")
            self.source_binding = actual_sha
        self.names = tuple(sorted(self.registry))
        if not self.names:
            raise ValueError("At least one source must be registered")
        for name, meta in self.registry.items():
            if not all(meta.get(k) for k in ("release", "pipeline", "dependency_id")):
                raise ValueError(
                    f"Source {name} lacks release, pipeline, or dependency provenance"
                )
            targets = meta.get("targets")
            if not isinstance(targets, dict) or set(targets) != {
                "teff",
                "logg",
                "fe_h",
            }:
                raise ValueError(f"Source {name} needs explicit target definitions")
            for target, unit in (
                ("teff", "K"),
                ("logg", "dex(cm/s2)"),
                ("fe_h", "dex"),
            ):
                definition = targets[target]
                if definition is not None and (
                    not isinstance(definition, dict)
                    or definition.get("quantity") != target
                    or definition.get("unit") != unit
                    or not definition.get("field")
                ):
                    raise ValueError(
                        f"Source {name} has an incompatible {target} definition"
                    )
                if (
                    target == "fe_h"
                    and definition is not None
                    and str(definition["field"]).strip().upper()
                    in {"M_H", "MH", "MH_GSPSPEC", "M_H_BDASP"}
                ):
                    raise ValueError(
                        f"Source {name} global metallicity is not direct Fe/H"
                    )
        self._file: h5py.File | None = None
        self._file_pid: int | None = None
        self._indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.unmatched_counts: dict[str, int] = {}
        with h5py.File(self.path, "r") as h5:
            ids = np.asarray(h5["source_id"][:])
            if ids.ndim != 1 or ids.dtype.kind not in "iu":
                raise ValueError("source_id must be integer IDs")
            if ids.dtype.kind == "u" and np.any(ids > np.iinfo("i8").max):
                raise ValueError("source_id exceeds int64")
            self.ids = ids.astype(np.int64, copy=False)
            if np.any(self.ids <= 0) or np.any(np.diff(self.ids) <= 0):
                raise ValueError(
                    "source_id must be sorted, unique, positive integer IDs"
                )
            if h5["input"].shape[0] != len(ids) or h5["role"].shape != (len(ids),):
                raise ValueError("Feature, role and identity rows must align")
            roles = h5["role"][:].astype("U")
            keep = np.ones(len(ids), dtype=bool)
            keep &= roles == role
            if require_primary_input:
                if "primary_input_eligible" not in h5:
                    raise ValueError("Explicit primary input eligibility is required")
                keep &= h5["primary_input_eligible"][:].astype(bool)
            self.star_rows = np.flatnonzero(keep)
            if "native_observations" not in h5:
                raise ValueError("Source contains no native observation groups")
            for name in self.names:
                if name not in h5["native_observations"]:
                    raise ValueError(f"Registered source {name} is not present")
                group = h5["native_observations"][name]
                expected_fields = [
                    self.registry[name]["targets"][key]["field"]
                    if self.registry[name]["targets"][key] is not None
                    else ""
                    for key in ("teff", "logg", "fe_h")
                ]
                if "source_fields" in group.attrs:
                    actual_fields = [
                        v.decode() if isinstance(v, bytes) else str(v)
                        for v in group.attrs["source_fields"]
                    ]
                else:
                    # This known legacy schema lacked group field-name attrs. Its
                    # published code fixes the column order; bind that assertion to
                    # the complete source-file digest, not to a filename alone.
                    from .joint_native_labels import SOURCE_COLUMNS

                    if (
                        expected_source_sha256 is None
                        or h5.attrs.get("schema")
                        != "remediation-sparse-six-target-source-v1"
                        or name not in SOURCE_COLUMNS
                    ):
                        raise ValueError(
                            f"{name} lacks a verified source field-order receipt"
                        )
                    actual_fields = list(SOURCE_COLUMNS[name]["value"])
                if actual_fields != expected_fields:
                    raise ValueError(
                        f"{name} source field order differs from its registry"
                    )
                sid = np.asarray(group["source_id"][:])
                if sid.ndim != 1 or sid.dtype.kind not in "iu":
                    raise ValueError(f"{name} IDs must be integer")
                if sid.dtype.kind == "u" and np.any(sid > np.iinfo("i8").max):
                    raise ValueError(f"{name} IDs exceed int64")
                sid = sid.astype(np.int64, copy=False)
                n = len(sid)
                if (
                    group["source_row"].shape != (n,)
                    or group["value"].shape != (n, 3)
                    or group["error"].shape != (n, 3)
                    or group["observed"].shape != (n, 3)
                ):
                    raise ValueError(f"{name} native arrays have incompatible shapes")
                pos = np.searchsorted(self.ids, sid)
                matched = (sid > 0) & (pos < len(self.ids))
                valid = np.flatnonzero(matched)
                matched[valid] &= self.ids[pos[valid]] == sid[valid]
                self.unmatched_counts[name] = int((~matched).sum())
                rows = np.flatnonzero(matched)
                order = np.argsort(pos[rows], kind="stable")
                self._indices[name] = (pos[rows][order], rows[order])

    def __len__(self) -> int:
        return len(self.star_rows)

    def _handle(self) -> h5py.File:
        if self._verified_overlay_access is not None:
            self._verified_overlay_access.check(self.path, self.registry_sha256)
        pid = os.getpid()
        if self._file is not None and self._file_pid != pid:
            self._file.close()
            self._file = None
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._file_pid = pid
        return self._file

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_file_pid"] = None
        return state

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._file_pid = None

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = int(self.star_rows[item])
        h5 = self._handle()
        raw = np.asarray(h5["input"][row], dtype=np.float32)
        available = np.isfinite(raw) & (raw != -9999)
        records = []
        for source_index, name in enumerate(self.names):
            stars, native_rows = self._indices[name]
            lo, hi = np.searchsorted(stars, [row, row + 1])
            group = h5["native_observations"][name]
            for native_row in native_rows[lo:hi]:
                i = int(native_row)
                value = np.asarray(group["value"][i], dtype=np.float64)
                error = np.asarray(group["error"][i], dtype=np.float64)
                observed = np.asarray(group["observed"][i], dtype=bool)
                if not np.isfinite(value[observed]).all():
                    raise ValueError(f"{name} has nonfinite observed labels")
                known = observed & np.isfinite(error) & (error > 0)
                meta = self.registry[name]
                comparable = np.array(
                    [
                        meta["targets"][key] is not None
                        for key in ("teff", "logg", "fe_h")
                    ]
                )
                if np.any(observed & ~comparable):
                    raise ValueError(
                        f"{name} contains observed labels without a comparable target mapping"
                    )
                records.append(
                    {
                        "source": name,
                        "source_index": -1
                        if meta.get("is_anchor", False)
                        else source_index,
                        "source_row": int(group["source_row"][i]),
                        "value": value,
                        "error": error,
                        "observed": observed,
                        "error_known": known,
                        "release": meta["release"],
                        "pipeline": meta["pipeline"],
                        "dependency_id": meta["dependency_id"],
                        "target_definitions": meta["targets"],
                        "is_anchor": bool(meta.get("is_anchor", False)),
                    }
                )
        return {
            "source_id": int(self.ids[row]),
            "source_registry": self.names,
            "registry_sha256": self.registry_sha256,
            "source_binding": self.source_binding,
            "input": raw,
            "input_available": available,
            "role": h5["role"][row].decode()
            if isinstance(h5["role"][row], bytes)
            else str(h5["role"][row]),
            "observations": records,
        }


def collate_multisurvey(stars: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad records while keeping provenance and rejecting unresolved dependence.

    Repeated dependency IDs for a star are not assumed independent. Supply an
    explicit joint error model or resolve the dependence upstream before using
    this baseline collator; the underlying dataset never discards the records.
    """
    if not stars:
        raise ValueError("Cannot collate an empty batch")
    for key in ("source_registry", "registry_sha256", "source_binding", "role"):
        if any(star[key] != stars[0][key] for star in stars[1:]):
            raise ValueError(f"Mixed {key} values are not permitted in one batch")
    ids = [int(star["source_id"]) for star in stars]
    if len(ids) != len(set(ids)):
        raise ValueError("A batch must contain at most one row per source_id")
    b = len(stars)
    m = max(len(star["observations"]) for star in stars)
    value = torch.full((b, m, 3), float("nan"), dtype=torch.float64)
    error = torch.full_like(value, float("nan"))
    mask = torch.zeros((b, m, 3), dtype=torch.bool)
    error_known = torch.zeros_like(mask)
    source_index = torch.full((b, m), -1, dtype=torch.long)
    source_row = torch.full((b, m), -1, dtype=torch.long)
    is_anchor = torch.zeros((b, m), dtype=torch.bool)
    records = []
    for i, star in enumerate(stars):
        deps = [
            r["dependency_id"] for r in star["observations"] if np.any(r["observed"])
        ]
        if len(deps) != len(set(deps)):
            raise ValueError(
                f"Unresolved dependent or repeated measurements for source_id {ids[i]}"
            )
        provenance = []
        for j, record in enumerate(star["observations"]):
            value[i, j] = torch.as_tensor(record["value"])
            error[i, j] = torch.as_tensor(record["error"])
            mask[i, j] = torch.as_tensor(record["observed"])
            error_known[i, j] = torch.as_tensor(record["error_known"])
            source_index[i, j] = record["source_index"]
            source_row[i, j] = record["source_row"]
            is_anchor[i, j] = record["is_anchor"]
            provenance.append(
                {
                    key: record[key]
                    for key in (
                        "source",
                        "release",
                        "pipeline",
                        "dependency_id",
                        "target_definitions",
                    )
                }
            )
        records.append(provenance)
    return {
        "source_id": torch.tensor(ids, dtype=torch.int64),
        "source_registry": stars[0]["source_registry"],
        "registry_sha256": stars[0]["registry_sha256"],
        "source_binding": stars[0]["source_binding"],
        "input": torch.as_tensor(np.stack([s["input"] for s in stars])),
        "input_available": torch.as_tensor(
            np.stack([s["input_available"] for s in stars])
        ),
        "role": [s["role"] for s in stars],
        "value": value,
        "error": error,
        "observed": mask,
        "error_known": error_known,
        "source_index": source_index,
        "source_row": source_row,
        "is_anchor": is_anchor,
        "provenance": records,
        "target_names": ("teff", "logg", "fe_h"),
    }
