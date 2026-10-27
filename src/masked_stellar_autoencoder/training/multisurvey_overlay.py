"""Digest-bound sparse-label sidecars and feature-cache external-link overlays.

No feature arrays are copied or new survey data acquired. Files referenced by a
validated manifest must remain immutable for the lifetime of every reader;
validation is a content check at open, not a lock against concurrent mutation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

TARGET_NAMES = ("teff", "logg", "fe_h")
TARGET_UNITS = ("K", "dex(cm/s2)", "dex")
SIDECAR_SCHEMA = "msa-native-observation-sidecar-v1"
OVERLAY_SCHEMA = "msa-native-observation-overlay-v1"


def file_sha256(path: str | Path) -> str:
    """Hash full file contents, including linked files separately in manifests."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strings(value: Any) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in value]


def _checked_file(path: str | Path, expected: str) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("A full expected SHA256 digest is required")
    if file_sha256(resolved) != expected.lower():
        raise ValueError(f"SHA256 mismatch: {resolved}")
    return {"path": str(resolved), "sha256": expected.lower()}


def _registry_fields(meta: dict[str, Any]) -> list[str]:
    for key in ("release", "pipeline", "dependency_id"):
        if not isinstance(meta.get(key), str) or not meta[key]:
            raise ValueError(f"Nonempty {key} provenance is required")
    targets = meta.get("targets", {})
    if set(targets) != set(TARGET_NAMES):
        raise ValueError("Exactly the three named target definitions are required")
    fields = []
    for target, unit in zip(TARGET_NAMES, TARGET_UNITS, strict=True):
        definition = targets[target]
        if definition is None:
            fields.append("")
            continue
        if (definition.get("quantity"), definition.get("unit")) != (target, unit):
            raise ValueError(f"Incompatible target quantity/unit: {target}")
        field = definition.get("field")
        if not isinstance(field, str) or not field:
            raise ValueError("Explicit native field names are required")
        if target == "fe_h" and field.upper() in {
            "M_H",
            "MH",
            "MH_GSPSPEC",
            "M_H_BDASP",
            "FEM_GSPSPEC",
        }:
            raise ValueError("Global metallicity or Fe/M is not direct Fe/H")
        fields.append(field)
    _json(meta)  # Require portable, finite JSON metadata.
    return fields


def _name(name: str) -> None:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise ValueError("Observation source names must be single HDF5 components")


def _ids(values: Any, *, positive: bool) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("IDs/row identities must be exact one-dimensional integers")
    if array.dtype.kind == "u" and np.any(array > np.iinfo("i8").max):
        raise ValueError("Integer identity exceeds int64")
    array = array.astype("i8", copy=False)
    if np.any(array < (1 if positive else 0)):
        raise ValueError("Invalid negative/zero identity")
    return array


def write_observation_sidecar(
    path: str | Path,
    observations: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    *,
    provenance: dict[str, Any],
) -> dict[str, str]:
    """Write explicitly supplied arrays; return its path/full SHA receipt.

    Each observation dictionary requires source_id[N], source_row[N], and
    value/error/observed[N,3] in TARGET_NAMES order. Missing target definitions
    are None and their mask must be false. Unknown errors remain NaN, not zero.
    Additional row-aligned numeric arrays are retained without reinterpretation.
    Registry metadata defines native fields, units, release, pipeline and shared
    dependency IDs; provenance must describe the caller's actual source.
    """
    path = Path(path).absolute()
    if path.exists():
        raise FileExistsError(path)
    if not observations or set(observations) != set(registry):
        raise ValueError("Observation and registry sources must match exactly")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Explicit nonempty source provenance is required")
    provenance_json = _json(provenance)
    arrays = {}
    for name, data in observations.items():
        _name(name)
        fields = _registry_fields(registry[name])
        required = {"source_id", "source_row", "value", "error", "observed"}
        if not required <= set(data):
            raise ValueError("Missing required observation arrays")
        sid = _ids(data["source_id"], positive=True)
        rows = _ids(data["source_row"], positive=False)
        n = len(sid)
        if rows.shape != (n,) or len(np.unique(rows)) != n:
            raise ValueError("Source rows must be unique within a source/reduction")
        block = {key: np.asarray(value) for key, value in data.items()}
        block.update(source_id=sid, source_row=rows)
        for key, array in block.items():
            _name(key)
            if array.ndim == 0 or array.shape[0] != n or array.dtype.kind not in "biuf":
                raise ValueError(f"{key} must be a row-aligned numeric array")
        for key in ("value", "error", "observed"):
            if block[key].shape != (n, 3):
                raise ValueError(f"{key} must have shape [N,3]")
        if block["observed"].dtype.kind != "b":
            raise ValueError("observed must be an explicit boolean mask")
        mask = block["observed"]
        if np.any(mask & ~np.isfinite(block["value"])):
            raise ValueError("Observed values must be finite")
        if np.any(mask[:, np.array(fields) == ""]):
            raise ValueError("An unmapped target cannot be observed")
        errors = block["error"]
        if np.any(mask & ~np.isnan(errors) & (~np.isfinite(errors) | (errors <= 0))):
            raise ValueError("Observed errors must be positive finite or unknown NaN")
        if np.any(mask[:, 0] & (block["value"][:, 0] <= 0)):
            raise ValueError("Observed effective temperatures must be positive")
        arrays[name] = block
    with h5py.File(path, "x") as h5:
        h5.attrs["schema"] = SIDECAR_SCHEMA
        h5.attrs["provenance_json"] = provenance_json
        for name, data in arrays.items():
            group = h5.create_group("native_observations/" + name)
            group.attrs["source_fields"] = _registry_fields(registry[name])
            group.attrs["target_names"] = TARGET_NAMES
            group.attrs["target_units"] = TARGET_UNITS
            group.attrs["registry_json"] = _json(registry[name])
            group.attrs["provenance_json"] = provenance_json
            for key, array in data.items():
                group.create_dataset(key, data=array)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _group_metadata(
    h5: h5py.File, name: str, meta: dict[str, Any], *, legacy_parent: bool
) -> dict[str, Any]:
    group = h5["native_observations"][name]
    fields = _registry_fields(meta)
    if "source_fields" in group.attrs:
        actual_fields = _strings(group.attrs["source_fields"])
    elif (
        legacy_parent
        and h5.attrs.get("schema") == "remediation-sparse-six-target-source-v1"
    ):
        from .joint_native_labels import SOURCE_COLUMNS

        if name not in SOURCE_COLUMNS:
            raise ValueError("Unknown legacy observation field order")
        actual_fields = list(SOURCE_COLUMNS[name]["value"])
    else:
        raise ValueError("Observation group lacks a field-order receipt")
    if actual_fields != fields:
        raise ValueError("Registry differs from stored native field order")
    for key, expected in (
        ("target_names", TARGET_NAMES),
        ("target_units", TARGET_UNITS),
    ):
        if key in group.attrs and _strings(group.attrs[key]) != list(expected):
            raise ValueError(f"Incompatible stored {key}")
    if (
        "registry_json" in group.attrs
        and json.loads(group.attrs["registry_json"]) != meta
    ):
        raise ValueError("Stored registry differs from caller registry")
    # Parent digest is the legacy field-order receipt; no original attrs invented.
    return {
        "registry": meta,
        "fields": fields,
        "original_attrs": {k: _attribute_json(v) for k, v in group.attrs.items()},
    }


def _attribute_json(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [_attribute_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_overlay(
    path: str | Path,
    *,
    parent_path: str | Path,
    parent_sha256: str,
    parent_registry: dict[str, dict[str, Any]],
    sidecars: list[dict[str, str]],
    manifest_path: str | Path | None = None,
) -> Path:
    """Create only links/small receipts, preserving the parent's exact star rows.

    Sidecar receipts are returned by write_observation_sidecar. Every parent
    native group must have a registry entry. Colliding source names are rejected.
    Returns the separate JSON manifest path; validate_overlay rehashes all files.
    """
    path = Path(path).absolute()
    manifest_path = (
        Path(manifest_path).absolute()
        if manifest_path
        else path.with_suffix(".manifest.json")
    )
    if path == manifest_path or path.exists() or manifest_path.exists():
        raise FileExistsError("Overlay and manifest outputs must be distinct and new")
    parent = _checked_file(parent_path, parent_sha256)
    receipts = [_checked_file(s["path"], s["sha256"]) for s in sidecars]
    inputs = [parent["path"], *(s["path"] for s in receipts)]
    if (
        len(set(inputs)) != len(inputs)
        or str(path) in inputs
        or str(manifest_path) in inputs
    ):
        raise ValueError("Input paths must be distinct from each other and outputs")
    groups = {}
    with h5py.File(parent["path"], "r") as h5:
        for required in ("source_id", "input", "role", "primary_input_eligible"):
            if required not in h5:
                raise ValueError(f"Missing parent feature metadata: {required}")
        # Refuse hidden external dependencies not included in the file receipts.
        _require_local_links(h5)
        ids = _ids(h5["source_id"][:], positive=True)
        if np.any(np.diff(ids) <= 0):
            raise ValueError("Parent star IDs must be sorted and unique")
        if h5["input"].shape[0] != len(ids) or h5["role"].shape != (len(ids),):
            raise ValueError("Parent feature/role rows do not align")
        names = set(h5["native_observations"]) if "native_observations" in h5 else set()
        if names != set(parent_registry):
            raise ValueError(
                "Parent registry must cover every native observation group"
            )
        for name in sorted(names):
            _name(name)
            groups[name] = dict(
                file=parent["path"],
                group="/native_observations/" + name,
                **_group_metadata(h5, name, parent_registry[name], legacy_parent=True),
            )
        root_keys = [key for key in h5 if key != "native_observations"]
    for receipt in receipts:
        with h5py.File(receipt["path"], "r") as h5:
            _require_local_links(h5)
            if h5.attrs.get("schema") != SIDECAR_SCHEMA:
                raise ValueError("Unsupported observation sidecar schema")
            if not json.loads(h5.attrs.get("provenance_json", "{}")):
                raise ValueError("Sidecar lacks source provenance")
            for name in h5["native_observations"]:
                if name in groups:
                    raise ValueError(f"Observation source collision: {name}")
                group = h5["native_observations"][name]
                meta = json.loads(group.attrs["registry_json"])
                groups[name] = dict(
                    file=receipt["path"],
                    group=group.name,
                    **_group_metadata(h5, name, meta, legacy_parent=False),
                )
    with h5py.File(path, "x") as out:
        with h5py.File(parent["path"], "r") as source:
            for key, value in source.attrs.items():
                out.attrs[key] = value
            out.attrs["parent_schema"] = source.attrs.get("schema", "unspecified")
        out.attrs["schema"] = OVERLAY_SCHEMA
        out.attrs["manifest_path"] = str(manifest_path)
        for key in root_keys:
            out[key] = h5py.ExternalLink(parent["path"], "/" + key)
        for name, receipt in groups.items():
            # A small wrapper supplies explicit field order to the existing reader.
            group = out.create_group("native_observations/" + name)
            group.attrs["source_fields"] = receipt["fields"]
            group.attrs["target_names"] = TARGET_NAMES
            group.attrs["target_units"] = TARGET_UNITS
            group.attrs["registry_json"] = _json(receipt["registry"])
            group.attrs["origin_json"] = _json(receipt)
            with h5py.File(receipt["file"], "r") as source:
                for key in source[receipt["group"]]:
                    group[key] = h5py.ExternalLink(
                        receipt["file"], receipt["group"] + "/" + key
                    )
    manifest = dict(
        schema=OVERLAY_SCHEMA,
        parent=parent,
        sidecars=receipts,
        overlay={"path": str(path.resolve()), "sha256": file_sha256(path)},
        root_links=root_keys,
        observation_groups=groups,
        registry={name: receipt["registry"] for name, receipt in groups.items()},
        claim="Supplied observations only; no new survey ingestion or calibration certified",
    )
    with manifest_path.open("x") as stream:
        stream.write(_json(manifest) + "\n")
    validate_overlay(manifest_path)
    return manifest_path


def _require_local_links(group: h5py.Group) -> None:
    for key in group:
        if not isinstance(group.get(key, getlink=True), h5py.HardLink):
            raise ValueError(
                "Input files must not contain undeclared external/soft links"
            )
        value = group[key]
        if isinstance(value, h5py.Group):
            _require_local_links(value)


def validate_overlay(
    manifest_path: str | Path, *, expected_manifest_sha256: str | None = None
) -> dict[str, Any]:
    """Rehash attachments and links; optionally require a pinned manifest digest.

    Training uses sparse_dataset_from_manifest, which requires the manifest pin.
    Unpinned validation remains available for inspecting a newly built artifact.
    """
    raw_manifest = Path(manifest_path).read_bytes()
    if expected_manifest_sha256 is not None:
        if hashlib.sha256(raw_manifest).hexdigest() != expected_manifest_sha256:
            raise ValueError("Manifest SHA256 mismatch")
    manifest = json.loads(raw_manifest)
    if manifest.get("schema") != OVERLAY_SCHEMA:
        raise ValueError("Unsupported overlay manifest schema")
    for receipt in [manifest["parent"], *manifest["sidecars"], manifest["overlay"]]:
        _checked_file(receipt["path"], receipt["sha256"])
    files = {receipt["path"] for receipt in [manifest["parent"], *manifest["sidecars"]]}
    if set(manifest["registry"]) != set(manifest["observation_groups"]):
        raise ValueError("Manifest registry sources differ")
    with h5py.File(manifest["parent"]["path"], "r") as parent:
        if set(manifest["root_links"]) != set(parent) - {"native_observations"}:
            raise ValueError("Manifest omits parent feature metadata")
    with h5py.File(manifest["overlay"]["path"], "r") as h5:
        if h5.attrs.get("schema") != OVERLAY_SCHEMA:
            raise ValueError("Unsupported overlay HDF5 schema")
        expected = set(manifest["root_links"]) | {"native_observations"}
        if set(h5) != expected:
            raise ValueError("Unexpected overlay root links")
        for key in manifest["root_links"]:
            _check_link(h5, key, manifest["parent"]["path"], "/" + key)
        if set(h5["native_observations"]) != set(manifest["observation_groups"]):
            raise ValueError("Unexpected overlay source groups")
        for name, receipt in manifest["observation_groups"].items():
            if (
                receipt["file"] not in files
                or receipt["group"] != "/native_observations/" + name
            ):
                raise ValueError("Observation source is not bound to hashed inputs")
            if manifest["registry"][name] != receipt["registry"]:
                raise ValueError("Manifest registry differs from source receipt")
            group = h5["native_observations"][name]
            if json.loads(group.attrs["registry_json"]) != receipt["registry"]:
                raise ValueError("Overlay registry differs from source receipt")
            if _strings(group.attrs["source_fields"]) != receipt["fields"]:
                raise ValueError("Overlay field order receipt differs")
            with h5py.File(receipt["file"], "r") as source:
                actual = _group_metadata(
                    source,
                    name,
                    receipt["registry"],
                    legacy_parent=receipt["file"] == manifest["parent"]["path"],
                )
                if (
                    actual["fields"] != receipt["fields"]
                    or actual["original_attrs"] != receipt["original_attrs"]
                ):
                    raise ValueError("Original source metadata differs from receipt")
                if set(group) != set(source[receipt["group"]]):
                    raise ValueError("Observation links differ from original source")
                for key in group:
                    _check_link(
                        group, key, receipt["file"], receipt["group"] + "/" + key
                    )
    return manifest


def _check_link(group: h5py.Group, key: str, file: str, target: str) -> None:
    link = group.get(key, getlink=True)
    if not isinstance(link, h5py.ExternalLink) or (link.filename, link.path) != (
        file,
        target,
    ):
        raise ValueError("Overlay external link differs from manifest")
    try:
        group[key]  # Force link resolution: existence of the link is insufficient.
    except (KeyError, OSError) as error:
        raise ValueError("Missing external link target") from error


@contextmanager
def open_verified_overlay(manifest_path: str | Path) -> Iterator[h5py.File]:
    """Validate attachments before opening; caller must keep inputs immutable."""
    manifest = validate_overlay(manifest_path)
    with h5py.File(manifest["overlay"]["path"], "r") as h5:
        yield h5


def _file_stat(path: str | Path) -> tuple[int, int, int, int, int]:
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class _VerifiedOverlayAccess:
    """Internal content-verification receipt, not a public validation bypass.

    All attachment bytes were hashed once against the pinned manifest. Stable
    inode/size/mtime/ctime guards detect later mutation before each star read,
    avoiding repeated multi-GB content hashing. This is not a hostile-writer lock.
    """

    overlay_path: str
    overlay_sha256: str
    manifest_sha256: str
    registry_sha256: str
    file_stats: tuple[tuple[str, tuple[int, int, int, int, int]], ...]

    def check(self, path: str | Path, registry_sha256: str) -> None:
        if (
            str(Path(path).resolve()) != self.overlay_path
            or registry_sha256 != self.registry_sha256
        ):
            raise ValueError("Verified overlay receipt does not match dataset")
        for file, expected in self.file_stats:
            if _file_stat(file) != expected:
                raise ValueError(f"Verified overlay attachment changed: {file}")


def sparse_dataset_from_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
    role: str,
    require_primary_input: bool = True,
):
    """Require a pinned manifest, rehash all attachments, and return a reader.

    Plain SparseSurveyDataset opening of external-link overlays fails closed.
    The internal receipt avoids hashing the overlay twice; file stat guards are
    checked before every subsequent star read. Keep attachments immutable while
    reading: these guards are not locks against a concurrent hostile writer.
    """
    from .multisurvey_data import SparseSurveyDataset

    # Snapshot first, then hash. Check snapshots again after all validation so
    # no receipt can certify metadata captured only after a concurrent mutation.
    manifest_path = Path(manifest_path).resolve(strict=True)
    raw_manifest = manifest_path.read_bytes()
    if (
        not isinstance(expected_manifest_sha256, str)
        or hashlib.sha256(raw_manifest).hexdigest() != expected_manifest_sha256
    ):
        raise ValueError("Manifest SHA256 mismatch")
    initial = json.loads(raw_manifest)
    paths = [
        str(manifest_path),
        initial["parent"]["path"],
        *(r["path"] for r in initial["sidecars"]),
        initial["overlay"]["path"],
    ]
    stats = tuple(
        (str(Path(path).resolve(strict=True)), _file_stat(path)) for path in paths
    )
    manifest = validate_overlay(
        manifest_path, expected_manifest_sha256=expected_manifest_sha256
    )
    registry_sha = hashlib.sha256(_json(manifest["registry"]).encode()).hexdigest()
    receipt = _VerifiedOverlayAccess(
        overlay_path=manifest["overlay"]["path"],
        overlay_sha256=manifest["overlay"]["sha256"],
        manifest_sha256=expected_manifest_sha256,
        registry_sha256=registry_sha,
        file_stats=stats,
    )
    receipt.check(receipt.overlay_path, registry_sha)
    return SparseSurveyDataset(
        receipt.overlay_path,
        manifest["registry"],
        role=role,
        require_primary_input=require_primary_input,
        expected_source_sha256=receipt.overlay_sha256,
        _verified_overlay_access=receipt,
    )
