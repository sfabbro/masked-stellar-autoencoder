"""CANFAR CPU saved-array reporting recovery: no neural imports or model execution."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml
from expanded_saved_array_reporting import atomic, external_report


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_ready(manifest, binding, cfg, arm):
    """A running or partially scored producer is never eligible for recovery."""
    if (
        manifest.get("status") != "failed"
        or manifest.get("error_type") != "ValueError"
        or manifest.get("error") != "output array is read-only"
        or manifest.get("processed") != cfg["expected_rows"]
        or manifest.get("total") != cfg["expected_rows"]
        or manifest.get("source_sha") != cfg["producer_source_sha"]
        or manifest.get("run_id") != arm["producer_run_id"]
        or manifest.get("zero_training") is not True
        or manifest.get("optimizer_constructed") is not False
        or manifest.get("reconstruction_presentations") != 0
    ):
        raise ValueError(
            "Actual producer must have completed scoring and failed only at the known reporting boundary"
        )
    if (
        binding.get("source_sha") != cfg["producer_source_sha"]
        or binding.get("prepared_sha256") != cfg["prepared_sha256"]
        or binding.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        or manifest.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        or binding.get("target_names")
        != ["teff", "logg", "fe_h", "alpha", "age", "distance"]
        or binding.get("library", {}).get("vcs_info", {}).get("commit_id")
        != cfg["library_revision"]
    ):
        raise ValueError(
            "Saved-array source, input, checkpoint or library binding differs"
        )


def load_arrays(directory, expected_rows):
    arrays, schemas = {}, {}
    for path in sorted(Path(directory).glob("*.npy")):
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if not value.ndim or len(value) != expected_rows or value.dtype.kind == "O":
            raise ValueError("Saved-array row count/layout differs: " + path.name)
        stat = path.stat()
        arrays[path.stem] = value
        schemas[path.name] = dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            content_sha256_verified=False,
        )
    for key in [
        "source_id",
        "point_native",
        "quantiles_native",
        "target_native",
        "observed",
        "diagnostic_eligible",
        "evaluation_mode",
        "xp_status",
        "label_source",
    ]:
        if key not in arrays:
            raise ValueError("Missing saved reporting array: " + key)
    ids = arrays["source_id"]
    if (
        ids.dtype.kind not in "iu"
        or (ids <= 0).any()
        or len(np.unique(ids)) != expected_rows
    ):
        raise ValueError("Unique exact positive saved source IDs required")
    for key in [
        "point_native",
        "target_native",
        "observed",
        "diagnostic_eligible",
        "label_source",
    ]:
        if arrays[key].shape != (expected_rows, 6):
            raise ValueError("Saved six-coordinate layout differs: " + key)
    if arrays["quantiles_native"].shape != (expected_rows, 3, 6):
        raise ValueError("Saved three-quantile layout differs")
    if "ancestor_feh_supervised_fit_exposed" in arrays:
        exposed = np.asarray(arrays["ancestor_feh_supervised_fit_exposed"], bool)
        if arrays["observed"][exposed].any():
            raise ValueError(
                "Known ancestor-exposed row entered primary reporting mask"
            )
    score_keys = [
        key
        for key in (
            "intrinsic_log_prob_native",
            "observed_log_prob_native",
            "intrinsic_pit",
            "observed_pit",
        )
        if key in arrays
    ]
    for key in score_keys:
        if arrays[key].shape != (expected_rows, 6):
            raise ValueError("Saved density/CDF layout differs: " + key)
    for start in range(0, expected_rows, 65536):
        stop = min(start + 65536, expected_rows)
        point = arrays["point_native"][start:stop]
        quantiles = arrays["quantiles_native"][start:stop]
        if not np.isfinite(point).all() or (point[:, 4:] <= 0).any():
            raise FloatingPointError(
                "Invalid saved point predictions at row block " + str(start)
            )
        if (
            not np.isfinite(quantiles).all()
            or (np.diff(quantiles, axis=1) < 0).any()
            or (quantiles[:, :, 4:] <= 0).any()
        ):
            raise FloatingPointError(
                "Invalid saved quantiles at row block " + str(start)
            )
        union = np.asarray(arrays["observed"][start:stop], bool) | np.asarray(
            arrays["diagnostic_eligible"][start:stop], bool
        )
        for key in score_keys:
            values = arrays[key][start:stop]
            invalid = ~np.isfinite(values)
            if key.endswith("_pit"):
                invalid |= (values < 0) | (values > 1)
            if (invalid & union).any():
                row, column = np.argwhere(invalid & union)[0]
                kind = "CDF" if key.endswith("_pit") else "density"
                raise FloatingPointError(
                    f"Invalid saved {kind} {key} at row {start + int(row)}, coordinate {int(column)}"
                )
    arrays["quantile_levels"] = np.array([0.05, 0.5, 0.95])
    return arrays, schemas


def diagnostic_only_mask(predictions):
    return np.asarray(predictions["diagnostic_eligible"], bool) & ~np.asarray(
        predictions["observed"], bool
    )


def primary_corner_references(corners, predictions):
    """Reuse exact draws while rendering only source-qualified primary references."""
    ids = np.asarray(predictions["source_id"])
    wanted = np.asarray(corners["source_id"])
    if (
        wanted.ndim != 1
        or wanted.dtype.kind not in "iu"
        or len(np.unique(wanted)) != len(wanted)
    ):
        raise ValueError("Corner source IDs must be unique exact integer identities")
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], wanted)
    if (positions >= len(ids)).any() or not np.array_equal(
        ids[order[positions]], wanted
    ):
        raise ValueError("Saved corner source ID absent from primary predictions")
    rows = order[positions]
    original = np.asarray(corners["observed"], bool)
    primary = np.array(predictions["observed"][rows], dtype=bool, copy=True)
    if original.shape != primary.shape or (primary & ~original).any():
        raise ValueError("Corner references contradict primary eligibility")
    if not np.array_equal(
        np.asarray(corners["target_native"]),
        predictions["target_native"][rows],
        equal_nan=True,
    ):
        raise ValueError(
            "Saved corner reference values differ from source-bound predictions"
        )
    result = dict(corners)
    result["observed"] = primary
    if "error_observed" in result:
        result["error_observed"] = np.asarray(result["error_observed"], bool) & primary
    proof = dict(
        reference_policy="Primary prediction mask joined by exact source ID; diagnostics excluded",
        source_ids=wanted.tolist(),
        original_reference_count=original.sum(0).tolist(),
        rendered_primary_reference_count=primary.sum(0).tolist(),
        excluded_diagnostic_reference_count=(original & ~primary).sum(0).tolist(),
        native_draws_changed=False,
        original_corner_file_modified=False,
    )
    return result, proof


def run(args):
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError("Run reporting recovery on CANFAR headless CPU only")
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("optimizer_steps") != 0 or cfg.get("model_forward_rows") != 0:
        raise ValueError("Recovery forbids model execution and optimization")
    arm = next(a for a in cfg["arms"] if a["run_id"] == args.run_id)
    producer = Path(cfg["output_root"]) / arm["producer_run_id"]
    producer_manifest = producer / "manifest.json"
    bindings_file = producer / "bindings.json"
    original_sha = sha(producer_manifest)
    manifest = json.loads(producer_manifest.read_text())
    binding = json.loads(bindings_file.read_text())
    validate_ready(manifest, binding, cfg, arm)
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output / "recovery.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = dict(
        status="running",
        run_id=args.run_id,
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        producer_run_id=arm["producer_run_id"],
        producer_manifest_sha256=original_sha,
        producer_bindings_sha256=sha(bindings_file),
        source_input_sha256=cfg["prepared_sha256"],
        checkpoint_sha256=arm["checkpoint_sha256"],
        native_model_execution=False,
        checkpoint_loaded=False,
        resampling_performed=False,
        optimizer_constructed=False,
        original_failed_manifest_preserved=True,
    )
    report_path = args.output / "manifest.json"
    atomic(report_path, report)
    try:
        predictions, schemas = load_arrays(
            producer / "prediction-arrays", cfg["expected_rows"]
        )
        atomic(args.output / "array-schema-receipts.json", schemas)
        corners = None
        if arm["head_kind"] != "quantile3":
            corner_file = producer / "representative-corner-samples.npz"
            corner_receipt = json.loads(
                corner_file.with_suffix(".sha256.json").read_text()
            )
            if sha(corner_file) != corner_receipt["sha256"]:
                raise ValueError("Existing saved corner-draw SHA differs")
            with np.load(corner_file, allow_pickle=False) as archive:
                corners = {key: archive[key] for key in archive.files}
            report["saved_corner_sha256"] = corner_receipt["sha256"]
            corners, report["corner_reference_policy"] = primary_corner_references(
                corners, predictions
            )
        external_report(predictions, args.output / "external-apokasc3-age")
        from five_wave_report import make_report

        metadata = dict(
            point_functional="intrinsic conditional median",
            physical_truth_claim=False,
            reporting_recovery_only=True,
            producer_source_sha=cfg["producer_source_sha"],
            source_input_sha256=cfg["prepared_sha256"],
            checkpoint_sha256=arm["checkpoint_sha256"],
            original_failed_report_manifest_preserved=True,
            source_expanded_evaluation=True,
            array_content_scope="Read-only producer memmaps; schema/size/mtime checked; source and checkpoint bindings transferred. Not an independent full array/model certificate.",
            corner_scope="Existing native draws only, original source axes retained and clipping reported; no resampling or invented copula",
            andrae_projection_scope="Cached partial source-ID projection only; unmatched new IDs unqueried, not absent from175M catalogue",
            parent86_full_training_history_certified=False,
        )
        make_report(
            predictions,
            args.output / "report",
            run_id=args.run_id,
            head_kind=arm["head_kind"],
            corners=corners,
            metadata=metadata,
        )
        diagnostic = diagnostic_only_mask(predictions)
        if diagnostic.any():
            secondary = dict(predictions)
            secondary["observed"] = diagnostic
            make_report(
                secondary,
                args.output / "diagnostic-reference-report",
                run_id=args.run_id + "-diagnostic-only",
                head_kind=arm["head_kind"],
                metadata=dict(
                    metadata,
                    diagnostic_only=True,
                    scope="Disjoint diagnostic-only coordinates: producer diagnostic mask minus primary coordinates; no physical-truth claim",
                ),
            )
        if sha(producer_manifest) != original_sha:
            raise ValueError("Producer manifest changed during recovery")
        for name, schema in schemas.items():
            stat = (producer / "prediction-arrays" / name).stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                schema["bytes"],
                schema["mtime_ns"],
            ):
                raise ValueError(
                    "Producer saved array changed during reporting: " + name
                )
        report.update(
            status="saved_array_reporting_complete",
            processed=cfg["expected_rows"],
            report_status="Figures require independent visual review",
            diagnostic_mask_policy="diagnostic AND NOT primary",
            reporter_sha256=sha(__file__),
            lightweight_external_reporter_sha256=sha(
                Path(__file__).with_name("expanded_saved_array_reporting.py")
            ),
        )
        atomic(report_path, report)
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic(report_path, report)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
