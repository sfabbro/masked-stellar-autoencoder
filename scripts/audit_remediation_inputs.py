"""Read-only, model-free exact-ID forensic audit for fixed remediation stars.

Only NumPy, HDF5 and FITS source data are read. No checkpoint deserialization,
model import, fitting, source mutation or job submission is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def exact_rows(dataset, requested, chunk_rows=65536):
    """Return rows in requested-ID order; never use float Gaia IDs or row order."""
    requested = np.asarray(requested)
    if requested.dtype.kind not in "iu" or requested.ndim != 1:
        raise ValueError("Requested source IDs must be exact integer vectors")
    if np.any(requested <= 0) or len(np.unique(requested)) != len(requested):
        raise ValueError("Requested IDs must be distinct and positive")
    found = {}
    wanted = set(map(int, requested))
    for start in range(0, len(dataset), chunk_rows):
        values = np.asarray(dataset[start : start + chunk_rows])
        if values.dtype.kind not in "iu":
            raise ValueError("Source Gaia IDs must be stored as integers")
        for offset in np.flatnonzero(np.isin(values, requested)):
            sid = int(values[offset])
            if sid in found:
                raise ValueError(f"Duplicate source ID {sid}")
            found[sid] = start + int(offset)
    return np.asarray(
        [found.get(int(sid), -1) for sid in requested]
    ), wanted - found.keys()


def normalize(raw, names, center, scale):
    """Independent reproduction of parent f4 raw → f8 scaler → f4 input."""
    raw = np.asarray(raw, dtype="f4")
    center, scale = np.asarray(center), np.asarray(scale)
    if (
        raw.shape[-1] != len(names)
        or center.shape != scale.shape
        or center.shape != (len(names),)
    ):
        raise ValueError("Feature names/scaler dimensions differ")
    if (
        len(set(names)) != len(names)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid feature order/scaler")
    allowed = np.asarray(
        [
            n in {"W1", "W2", "G", "BP", "RP"} or n.startswith(("bp_", "rp_"))
            for n in names
        ]
    )
    scaled = ((raw.astype("f8") - center) / scale).astype("f4")
    visible = np.isfinite(raw) & (raw != -9999) & np.isfinite(scaled) & allowed
    return np.where(visible, scaled, -9999).astype("f4"), visible


def same(actual, expected, message):
    if not np.array_equal(actual, expected, equal_nan=True):
        raise ValueError(message)


def cache_row_proof(directory, request_sha, sid, row, names, center, scale):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    request = manifest["request"]
    canonical = json.dumps(
        request, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if (
        manifest.get("status") != "complete"
        or hashlib.sha256(canonical).hexdigest() != request_sha
    ):
        raise ValueError("Cache request/status binding differs")
    if request["features"] != names:
        raise ValueError("Cache feature order differs from prepared inputs")
    same(request["center"], center, "Cache center differs")
    same(request["scale"], scale, "Cache scale differs")
    ids = np.load(directory / "source_id.npy", mmap_mode="r", allow_pickle=False)
    values = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
    provenance = np.load(
        directory / "provenance.npy", mmap_mode="r", allow_pickle=False
    )
    if row < 0 or row >= len(ids) or int(ids[row]) != sid:
        raise ValueError("Cache ID/row association differs")
    if (
        ids.dtype.kind not in "iu"
        or values.shape != (len(ids), len(names))
        or len(provenance) != len(ids)
    ):
        raise ValueError("Cache schema differs")
    return np.asarray(values[row]), provenance[row], request


def raw_record(path, request, shard, row, sid, names):
    if "source" in request:
        expected = request["source"]
        stat = Path(path).stat()
        if (
            str(Path(path).resolve()) != expected["path"]
            or stat.st_size != expected["bytes"]
            or stat.st_mtime_ns != expected["mtime_ns"]
        ):
            raise ValueError(
                "Original raw path/size/mtime differs from cache source binding"
            )
    keys = request["train_keys"] + request["validation_keys"]
    with h5py.File(path, "r") as handle:
        record = handle[keys[shard]][row]
        if (
            np.asarray(record["source_id"]).dtype.kind not in "iu"
            or int(record["source_id"]) != sid
        ):
            raise ValueError("Raw original Gaia ID/row association differs")
        return np.asarray(
            [float(record[n]) if record[n] not in (b"", "") else np.nan for n in names],
            dtype="f4",
        )


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [serial(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, bytes):
        return value.decode()
    return value


def write_json(path, data):
    path.write_text(json.dumps(serial(data), indent=2, allow_nan=False) + "\n")


def raw_record_values(record, requested_names, available_names):
    """FITS rows expose fields through their table, unlike NumPy records."""
    available = set(available_names or ())
    return np.asarray(
        [record[name] if name in available else np.nan for name in requested_names],
        dtype="f4",
    )


def audit(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "run_id": config.get("run_id", "remediation_inputs_v1"),
        "training_performed": False,
        "model_execution": False,
        "script_sha256": digest(__file__),
        "audit_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_receipts": {},
        "stars": [],
        "unresolved": [],
        "readiness": {
            "phot_xp_input_pass": False,
            "native_label_association_pass": False,
            "astrometry_physical_likelihood_ready": False,
        },
        "scope": "Exact selected source rows; no whole-cache content hash, physical truth or model certification",
        "source_commit": config.get("prepared_source_commit"),
    }
    write_json(output / "manifest.json", result)
    try:
        for key in ("prepared", "native", "legacy", "frozen"):
            spec = config[key]
            path = Path(spec["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Required {key} source unavailable: {path}")
            receipt = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            if config.get("verify_source_sha256", True):
                receipt["sha256"] = digest(path)
                if receipt["sha256"] != spec["sha256"]:
                    raise ValueError(f"{key} source SHA differs")
            else:
                result["unresolved"].append(
                    f"{key} full content hash not independently verified"
                )
            result["source_receipts"][key] = receipt
        ids = np.asarray(config["source_ids"], dtype="i8")
        saved = []
        with (
            h5py.File(config["prepared"]["path"], "r") as prepared,
            h5py.File(config["native"]["path"], "r") as native,
            h5py.File(config["legacy"]["path"], "r") as legacy,
        ):
            if (
                config.get("prepared_source_commit")
                and text(prepared.attrs.get("source_commit", ""))
                != config["prepared_source_commit"]
            ):
                raise ValueError("Prepared source commit differs from pin")
            rows, missing = exact_rows(prepared["source_id"], ids)
            if missing:
                raise ValueError(
                    f"Fixed corner IDs absent from prepared inputs: {sorted(missing)}"
                )
            corner_ids = set(
                map(int, prepared["source_id"][np.sort(prepared["corner_rows"][:])])
            )
            if set(map(int, ids)) != corner_ids:
                raise ValueError(
                    "Configured fixed stars differ from model-blind corner selection"
                )
            native_rows, _ = exact_rows(native["source_id"], ids)
            legacy_rows, _ = exact_rows(legacy["features/source_id"], ids)
            names = [text(v) for v in prepared["feature_names"][:]]
            center, scale = prepared["scaler_center"][:], prepared["scaler_scale"][:]
            same(
                center,
                legacy["reference_scaler/center"][:],
                "Legacy parent center differs",
            )
            same(
                scale,
                legacy["reference_scaler/scale"][:],
                "Legacy parent scale differs",
            )
            xp = [j for j, n in enumerate(names) if n.startswith(("bp_", "rp_"))]
            if len(xp) != 110:
                raise ValueError("Expected all 110 XP coefficients")
            frozen = np.load(config["frozen"]["path"], allow_pickle=False)
            frozen_rows, _ = exact_rows(frozen["source_id"], ids)
            for k, sid in enumerate(ids):
                row = int(rows[k])
                sid = int(sid)
                star = {
                    "source_id": sid,
                    "prepared_row": row,
                    "checks": [],
                    "unresolved": [],
                }
                result["stars"].append(star)
                for key in (
                    "feature_origin",
                    "xp_status",
                    "official_has_xp_known",
                    "official_has_xp_continuous",
                    "real_no_xp",
                    "photometry_release_code",
                    "photometry_fallback_used",
                    "raw_reconstructed_from_normalized_cache",
                ):
                    if key in prepared:
                        star[key] = serial(prepared[key][row])
                raw, actual, visible = (
                    prepared["raw"][row],
                    prepared["input"][row],
                    prepared["visible"][row],
                )
                expected, expected_visible = normalize(raw, names, center, scale)
                origin = text(prepared["feature_origin"][row])
                if frozen_rows[k] >= 0:
                    base_row = int(frozen_rows[k])
                    same(
                        actual,
                        frozen["input"][base_row],
                        "Frozen benchmark input bytes differ",
                    )
                    same(
                        raw,
                        frozen["raw"][base_row],
                        "Frozen benchmark raw bytes differ",
                    )
                    star["checks"].append(
                        "original benchmark override exact input/raw bytes"
                    )
                elif origin == "original150m_cache":
                    cached, provenance, request = cache_row_proof(
                        config["cache_path"],
                        config["cache_request_sha256"],
                        sid,
                        int(prepared["cache_row"][row]),
                        names,
                        center,
                        scale,
                    )
                    source_raw = raw_record(
                        config["original_path"],
                        request,
                        int(provenance["shard"]),
                        int(provenance["row"]),
                        sid,
                        names,
                    )
                    # Cache itself holds all140 scaled fields, including withheld labels.
                    full_scaled = ((source_raw.astype("f8") - center) / scale).astype(
                        "f4"
                    )
                    same(
                        cached,
                        full_scaled,
                        "Raw original source/cache normalized bytes differ",
                    )
                    same(
                        actual[xp], cached[xp], "All110 prepared XP/cache values differ"
                    )
                    photo = [
                        j
                        for j, name in enumerate(names)
                        if name in {"W1", "W2", "G", "BP", "RP"}
                    ]
                    same(
                        actual[photo],
                        cached[photo],
                        "Prepared photometry/cache values differ",
                    )
                    star["checks"].append(
                        "raw original ID → cache row → all110 XP values exact"
                    )
                    star["raw_original_shard"] = int(provenance["shard"])
                    star["raw_original_row"] = int(provenance["row"])
                elif origin == "original_raw_hdf5":
                    cache_manifest = json.loads(
                        (Path(config["cache_path"]) / "manifest.json").read_text()
                    )
                    source_raw = raw_record(
                        config["original_path"],
                        cache_manifest["request"],
                        int(prepared["original_shard"][row]),
                        int(prepared["original_row"][row]),
                        sid,
                        names,
                    )
                    same(source_raw[xp], raw[xp], "Original raw XP source bytes differ")
                    same(actual, expected, "Prepared raw/scaler input differs")
                    star["checks"].append("direct raw original ID/row and scaler exact")
                else:
                    same(actual, expected, "Prepared raw/scaler input differs")
                    star["checks"].append("raw/scaler input exact")
                    if origin != "legacy_cohort":
                        star["unresolved"].append(
                            "Nonlegacy photometry source branch not independently certified"
                        )
                if origin == "legacy_cohort" and legacy_rows[k] >= 0:
                    lr = int(legacy_rows[k])
                    same(
                        raw[xp],
                        legacy["features/values"][lr][xp],
                        "Legacy XP input ID alignment differs",
                    )
                    star["legacy_row"] = lr
                    star["legacy_input_source"] = int(
                        legacy["features/input_source"][lr]
                    )
                    star["legacy_original_row"] = int(
                        legacy["features/input_original_row"][lr]
                    )
                    same(
                        raw,
                        legacy["features/values"][lr],
                        "Legacy full raw feature association differs",
                    )
                    star["checks"].append(
                        "legacy cohort exact Gaia ID/110 raw XP values"
                    )
                    source_code = star["legacy_input_source"]
                    source_features = [
                        j
                        for j, name in enumerate(names)
                        if j in xp or name in {"W1", "W2", "G", "BP", "RP"}
                    ]
                    source_row = star["legacy_original_row"]
                    original_path = text(
                        legacy.attrs.get(
                            "benchmark_input" if source_code < 2 else "legacy_input", ""
                        )
                    )
                    if (
                        original_path
                        and Path(original_path).is_file()
                        and source_code in (0, 1, 2)
                    ):
                        if source_code < 2:
                            with h5py.File(original_path, "r") as original:
                                record = original[
                                    ("train" if source_code == 0 else "validation")
                                    + "/records"
                                ][source_row]
                                if int(record["source_id"]) != sid:
                                    raise ValueError(
                                        "Legacy benchmark raw ID/row differs"
                                    )
                                original_xp = raw_record_values(
                                    record,
                                    [names[j] for j in source_features],
                                    record.dtype.names,
                                )
                        else:
                            from astropy.io import fits

                            with fits.open(original_path, memmap=True) as hdul:
                                table = next(
                                    h for h in hdul if isinstance(h, fits.BinTableHDU)
                                )
                                record = table.data[source_row]
                                if int(record["source_id"]) != sid:
                                    raise ValueError("Legacy FITS raw ID/row differs")
                                original_xp = raw_record_values(
                                    record,
                                    [names[j] for j in source_features],
                                    table.columns.names,
                                )
                        original_xp[
                            ~np.isfinite(original_xp) | (original_xp == -9999)
                        ] = np.nan
                        same(
                            original_xp,
                            raw[source_features],
                            "Legacy original all110 XP/photometry raw bytes differ",
                        )
                        star["checks"].append(
                            "legacy original ID/row → all110 XP exact"
                        )
                        star["legacy_original_file"] = original_path
                    else:
                        star["unresolved"].append(
                            "Legacy pre-cohort raw source unavailable"
                        )
                if origin == "legacy_cohort" and legacy_rows[k] < 0:
                    star["unresolved"].append(
                        "Legacy source unavailable for selected Gaia ID"
                    )
                same(
                    visible,
                    np.isfinite(actual) & (actual != -9999),
                    "Visible/sentinel semantics differ",
                )
                if bool(star.get("real_no_xp", False)) and np.any(visible[xp]):
                    raise ValueError("Official noXP star exposes XP coefficients")
                if bool(star.get("official_has_xp_continuous", False)) and not bool(
                    star.get("official_has_xp_known", False)
                ):
                    raise ValueError("Official XP flag has no known publication status")
                star["xp_visible_count"] = int(visible[xp].sum())
                nr = int(native_rows[k])
                if nr >= 0:
                    native_values = native["targets/value"][nr]
                    primary = prepared["primary_score_eligible"][row][:3].astype(bool)
                    same(
                        prepared["value"][row][:3][primary],
                        native_values[primary],
                        "Prepared/native primary labels differ",
                    )
                    observed = native["targets/observed"][nr]
                    survey = text(native["targets/survey"][nr])
                    path = text(native["targets/source_file"][nr])
                    source_row = int(native["targets/source_row"][nr])
                    star["native"] = {
                        "row": nr,
                        "survey": survey,
                        "source_file": path,
                        "source_row": source_row,
                        "value": native_values,
                        "observed": observed,
                        "quality_pass": native["targets/quality_pass"][nr],
                        "uncertainty": native["targets/uncertainty"][nr],
                    }
                    if path and Path(path).is_file() and survey in ("galah", "apogee"):
                        from astropy.io import fits

                        from masked_stellar_autoencoder.training.joint_native_labels import (
                            SOURCE_COLUMNS,
                        )

                        spec = SOURCE_COLUMNS[survey]
                        with fits.open(path, memmap=True) as fits_file:
                            source = fits_file[1].data[source_row]
                            source_id = source[spec["id"]]
                            if (
                                isinstance(source_id, float | np.floating)
                                or int(source_id) != sid
                            ):
                                raise ValueError("Raw survey Gaia ID/row differs")
                            values = np.asarray(
                                [source[n] for n in spec["value"]], dtype="f8"
                            )
                            same(
                                values[observed],
                                native_values[observed],
                                "Raw survey/native observed labels differ",
                            )
                            star["native"]["raw_quality_flags"] = {
                                n: source[n] for n in spec["flags"]
                            }
                            star["native"]["raw_column_units"] = {
                                n: fits_file[1].columns[n].unit for n in spec["value"]
                            }
                            star["checks"].append(
                                "raw survey exact integer ID/row/observed labels"
                            )
                    else:
                        star["unresolved"].append("Raw survey file unavailable")
                    star["astrometry"] = {
                        n: native["astrometry/" + n][nr] for n in native["astrometry"]
                    }
                else:
                    star["unresolved"].append("No native three-target source row")
                star["unresolved"].append(
                    "Gaia astrometric quality and regime-specific parallax zero-point not certified; no correction invented"
                )
                star["astrometry_ready"] = False
                if star.get("photometry_fallback_used", False):
                    star["unresolved"].append(
                        "Fallback photometry raw catalogue row/release not independently certified"
                    )
                saved.append(np.stack([raw, actual, expected, visible.astype("f4")]))
            frozen.close()
            result["feature_names"] = names
            result["normalization_sha256"] = hashlib.sha256(
                np.asarray([center, scale], dtype="<f8").tobytes()
            ).hexdigest()
        np.savez_compressed(
            output / "selected-arrays.npz",
            source_id=ids,
            raw_input_recomputed_visible=np.asarray(saved),
            feature_names=np.asarray(names),
            center=center,
            scale=scale,
        )
        for star in result["stars"]:
            star["phot_xp_input_pass"] = not any(
                "source unavailable" in issue.lower()
                or "source branch" in issue.lower()
                or "photometry" in issue.lower()
                for issue in star["unresolved"]
            )
        source_hash_pass = not result["unresolved"]
        phot_xp_pass = source_hash_pass and all(
            s["phot_xp_input_pass"] for s in result["stars"]
        )
        label_pass = source_hash_pass and all(
            any("raw survey exact" in check for check in s["checks"])
            for s in result["stars"]
        )
        result["readiness"] = {
            "phot_xp_input_pass": phot_xp_pass,
            "native_label_association_pass": label_pass,
            "astrometry_physical_likelihood_ready": False,
        }
        result["status"] = (
            "remediation_input_audit_complete"
            if phot_xp_pass and label_pass
            else "incomplete"
        )
    except FileNotFoundError as exc:
        result["status"] = "incomplete"
        result["unresolved"].append(str(exc))
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(output / "manifest.json", result)
    return result


def independent_vac_photometry(table, source_id, raw, names):
    """Independently bind the no-XP photometry fallback to all raw VAC rows."""
    ids = np.asarray(table["source_id"])
    if ids.dtype.kind not in "iu":
        raise ValueError("VAC source identities must be exact integers")
    rows = np.flatnonzero(ids == source_id)
    if not len(rows):
        raise ValueError("VAC source identity missing")

    def flag(value):
        if isinstance(value, bytes):
            value = value.decode()
        token = str(value).strip().lower()
        return (
            False
            if token in ("false", "f", "0")
            else True
            if token in ("true", "t", "1")
            else None
        )

    if any(flag(table["has_xp_continuous"][r]) is not False for r in rows):
        raise ValueError("No-XP fallback requires unanimous official false flags")
    mapping = {
        "W1": "W1mag",
        "W2": "W2mag",
        "G": "phot_g_mean_mag",
        "BP": "phot_bp_mean_mag",
        "RP": "phot_rp_mean_mag",
    }
    fields = list(mapping.values()) + [
        band + suffix
        for band in ("phot_g", "phot_bp", "phot_rp")
        for suffix in ("_mean_flux", "_mean_flux_error")
    ]
    for field in fields:
        values = np.asarray(table[field][rows], dtype="f8").copy()
        values[~np.isfinite(values) | (values == -9999)] = np.nan
        if field in mapping.values():
            values[(values <= -20) | (values >= 40)] = np.nan
        if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError("VAC repeated-ID photometry/flux conflict: " + field)
    actual = []
    for name, field in mapping.items():
        value = float(table[field][rows[0]])
        value = value if np.isfinite(value) and -20 < value < 40 else np.nan
        actual.append(value)
        same(
            np.asarray(raw[names.index(name)], dtype="f4"),
            np.asarray(value, dtype="f4"),
            "VAC raw photometry differs: " + name,
        )
    xp = [j for j, name in enumerate(names) if name.startswith(("bp_", "rp_"))]
    if len(xp) != 110 or np.any(np.isfinite(raw[xp]) & (raw[xp] != -9999)):
        raise ValueError("VAC no-XP branch contains actual XP coefficients")
    return dict(
        source_rows=rows.tolist(),
        canonical_row=int(rows[0]),
        raw_photometry=dict(zip(mapping, actual, strict=True)),
        official_has_xp_continuous=False,
        duplicate_consistency_checked=True,
    )


def augment_vac_audit(config, output):
    """Reuse hash-bound passed v3 checks; independently add only missing VAC proof."""
    import copy

    from astropy.io import fits

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reuse = config["reuse_prior_audit"]
    prior_path = Path(reuse["path"])
    if digest(prior_path) != reuse["sha256"]:
        raise ValueError("Passed prior audit receipt content differs")
    prior = json.loads(prior_path.read_text())
    if (
        prior["audit_code_commit"] != reuse["code_commit"]
        or not prior["readiness"]["native_label_association_pass"]
        or prior["unresolved"]
    ):
        raise ValueError("Prior native/source proof cannot be reused")
    if [a["source_id"] for a in prior["stars"]] != config["source_ids"]:
        raise ValueError("Diagnostic identities differ from passed prior checks")
    for key in ("prepared", "native", "legacy", "frozen"):
        if prior["source_receipts"][key].get("sha256") != config[key]["sha256"]:
            raise ValueError("Reused source binding differs: " + key)
    result = copy.deepcopy(prior)
    result.update(
        run_id=config["run_id"],
        status="running",
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        audit_code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        script_sha256=digest(__file__),
        reuse_prior_audit=reuse,
        scope="Prior hash-bound native/legacy/scaler checks reused; independent raw Gaia-VAC no-XP photometry proof added; no neural execution",
    )
    write_json(output / "manifest.json", result)
    try:
        vac = config["gaia_vac"]
        if (
            Path(vac["path"]).stat().st_size != vac["bytes"]
            or digest(vac["path"]) != vac["sha256"]
        ):
            raise ValueError("Raw Gaia VAC content differs from mounted-source pin")
        result["gaia_vac_receipt"] = dict(vac, publisher_sha_known=False)
        target = [a for a in result["stars"] if not a["phot_xp_input_pass"]]
        if any(a["feature_origin"] != "gaia_vac_photometry" for a in target):
            raise ValueError("Unresolved non-VAC branch must not be silently cleared")
        saved = []
        with (
            h5py.File(config["prepared"]["path"], "r") as data,
            fits.open(vac["path"], memmap=True) as hdus,
        ):
            names = [text(v) for v in data["feature_names"][:]]
            for star in target:
                row = star["prepared_row"]
                sid = star["source_id"]
                if int(data["source_id"][row]) != sid:
                    raise ValueError("Prepared selected row identity changed")
                if (
                    not star.get("official_has_xp_known")
                    or star.get("official_has_xp_continuous")
                    or not star.get("real_no_xp")
                ):
                    raise ValueError("Prepared official-no-XP classification differs")
                raw = data["raw"][row]
                expected, visible = normalize(
                    raw, names, data["scaler_center"][:], data["scaler_scale"][:]
                )
                same(
                    expected,
                    data["input"][row],
                    "Selected VAC parent-normalized input differs",
                )
                same(visible, data["visible"][row], "Selected VAC visibility differs")
                star["gaia_vac"] = independent_vac_photometry(
                    hdus[1].data, sid, raw, names
                )
                star["checks"].append(
                    "independent raw Gaia VAC exact ID/duplicate flags/flux/magnitude/withheld XP proof"
                )
                star["unresolved"] = [
                    v
                    for v in star["unresolved"]
                    if v
                    != "Nonlegacy photometry source branch not independently certified"
                ]
                star["phot_xp_input_pass"] = True
                saved.append(raw)
        result["readiness"]["phot_xp_input_pass"] = all(
            a["phot_xp_input_pass"] for a in result["stars"]
        )
        result["status"] = (
            "remediation_input_audit_complete"
            if result["readiness"]["phot_xp_input_pass"]
            else "incomplete"
        )
        np.savez_compressed(
            output / "vac-selected-raw.npz",
            source_id=np.array([a["source_id"] for a in target], dtype="i8"),
            raw=np.asarray(saved),
            feature_names=np.asarray(names),
        )
        result["vac_selected_arrays_sha256"] = digest(output / "vac-selected-raw.npz")
    except Exception as exc:
        result.update(
            status="failed", error=dict(type=type(exc).__name__, message=str(exc))
        )
    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    result = (
        augment_vac_audit(config, args.output)
        if config.get("reuse_prior_audit")
        else audit(config, args.output)
    )
    print(json.dumps({"status": result["status"], "output": args.output}))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
