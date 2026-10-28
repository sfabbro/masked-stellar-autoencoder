"""Materialize uncapped development supervision without loading a native model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy.io import fits

from masked_stellar_autoencoder.training.joint_native_labels import (
    apogee_parameter_metadata,
)
from masked_stellar_autoencoder.training.remediation_source import (
    TARGET_NAMES,
    development_roles,
    exact_ids,
    native_assignment,
    phot_xp_columns,
    project_raw_native,
    role_counts,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(spec):
    path = Path(spec["path"])
    if sha(path) != spec["sha256"]:
        raise ValueError(f"Pinned source differs: {path}")
    return path


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def observation_support(observed, role):
    """Count actual FIT joint-label support without implying identified covariance."""
    mask = np.asarray(observed, dtype=bool)[np.asarray(role) == b"fit"]
    if mask.ndim != 2 or mask.shape[1] != 6:
        raise ValueError("Six-target FIT mask required")
    integer = mask.astype("i8")
    pattern = integer @ (1 << np.arange(6, dtype="i8"))
    counts = np.bincount(pattern, minlength=64)
    return {
        "target_names": list(TARGET_NAMES),
        "fit_rows": len(mask),
        "mask_pattern_bit_order": "teff=1,logg=2,fe_h=4,alpha=8,age=16,distance=32",
        "mask_pattern_counts": counts.tolist(),
        "pairwise_coobserved_counts": (integer.T @ integer).tolist(),
        "all_six_coobserved_count": int(counts[63]),
        "scope": "Observed source-qualified catalogue-reference support, not identified physical covariance or joint truth",
    }


def materialize(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": config["run_id"],
        "status": "verifying_inputs",
        "physical_truth_claim": False,
    }
    write_json(output / "manifest.json", manifest)
    paths = {}
    for key in (
        "expanded",
        "frozen",
        "galah",
        "apogee",
        "ancestry",
        "references",
        "native_astrometry",
    ):
        manifest.update(status="verifying_input", current_input=key)
        write_json(output / "manifest.json", manifest)
        paths[key] = checked(config[key])
        manifest.setdefault("verified_inputs", {})[key] = config[key]
        write_json(output / "manifest.json", manifest)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest.update(
        source_commit=source_commit,
        library_commit=config["library_commit"],
        inputs=config,
    )
    write_json(output / "manifest.json", manifest)
    with (
        h5py.File(paths["expanded"], "r") as expanded,
        np.load(paths["frozen"], allow_pickle=False) as frozen,
    ):
        old_ids = exact_ids(frozen["source_id"])
        expanded_ids = exact_ids(expanded["source_id"][:])
        ids = np.union1d(expanded_ids, old_ids)
        n = len(ids)
        names = expanded["feature_names"][:]
        allowed = phot_xp_columns(names)
        if not np.array_equal(names.astype("U"), frozen["feature_names"].astype("U")):
            raise ValueError("Feature order differs")
        for key in ("scaler_center", "scaler_scale"):
            if not np.array_equal(expanded[key][:], frozen[key], equal_nan=True):
                raise ValueError("Parent normalization differs")
        value = np.full((n, 6), np.nan)
        error = np.full_like(value, np.nan)
        observed = np.zeros((n, 6), bool)
        error_observed = np.zeros_like(observed)
        label_source = np.full((n, 6), b"", "S64")
        primary_input = np.zeros(n, bool)
        original_role = np.full(n, b"absent", "S8")
        original_ref = np.zeros((n, 6), bool)
        modality = {
            key: np.zeros(n, bool)
            for key in (
                "real_xp_present",
                "real_no_xp",
                "official_has_xp_known",
                "official_has_xp_continuous",
                "input_has_xp",
            )
        }
        xp_status = np.full(n, b"unknown", "S40")
        ds = np.searchsorted(ids, expanded_ids)
        # Only source-qualified primary targets enter supervision; unknown Gaia
        # publication status remains a diagnostic input population.
        for start in range(0, len(ds), config.get("chunk_rows", 32768)):
            sl = slice(start, min(start + config.get("chunk_rows", 32768), len(ds)))
            d = ds[sl]
            value[d] = expanded["value"][sl]
            error[d] = expanded["error"][sl]
            observed[d] = expanded["primary_score_eligible"][sl].astype(bool)
            error_observed[d] = (
                expanded["error_observed"][sl].astype(bool) & observed[d]
            )
            label_source[d] = expanded["label_source"][sl]
            primary_input[d] = ~expanded["input_diagnostic_only"][sl].astype(bool)
            for key in modality:
                if key in expanded:
                    modality[key][d] = expanded[key][sl]
            if "xp_status" in expanded:
                xp_status[d] = expanded["xp_status"][sl]
        fs = np.searchsorted(ids, old_ids)
        for key, target in (
            ("value", value),
            ("error", error),
            ("observed", observed),
            ("error_observed", error_observed),
            ("label_source", label_source),
        ):
            target[fs] = frozen[key]
        if "quality" in frozen:
            observed[fs] &= frozen["quality"]
        primary_input[fs] = True
        for key in modality:
            if key in frozen:
                modality[key][fs] = frozen[key]
        if "xp_status" in frozen:
            xp_status[fs] = frozen["xp_status"].astype("S40")
        original_role[fs] = frozen["role"].astype("S8")
        original_ref[fs] = frozen["observed"] & frozen["quality"]
        # Rebuild ALL native targets from raw full catalogues, never old masks.
        observed[:, :3] = False
        error_observed[:, :3] = False
        label_source[:, :3] = b""
        value[:, :3] = np.nan
        error[:, :3] = np.nan
        native_counts = {}
        destination = output / "prepared.h5"
        with h5py.File(destination, "w") as out:
            for source in ("apogee", "galah"):
                manifest.update(status="materializing_native", current_survey=source)
                write_json(output / "manifest.json", manifest)
                with fits.open(paths[source], memmap=True) as hdul:
                    table = hdul[1].data
                    symbols = (
                        apogee_parameter_metadata(hdul)[1]
                        if source == "apogee"
                        else None
                    )
                    sid, val, err, mask = project_raw_native(table, source, symbols)
                    group = out.create_group("native_observations/" + source)
                    # Retain original eligible observations and their raw row identities,
                    # including source overlaps; canonical precedence is explicit GALAH.
                    keep = np.flatnonzero(mask.any(1))
                    for key, array in [
                        ("source_id", sid[keep]),
                        ("source_row", keep),
                        ("value", val[keep]),
                        ("error", err[keep]),
                        ("observed", mask[keep]),
                    ]:
                        group.create_dataset(key, data=array, compression="lzf")
                    native_counts[source] = {
                        "raw_rows": len(sid),
                        "eligible_per_target": mask.sum(0).tolist(),
                        "feature_matched_per_target": [],
                    }
                    for j, (d, rows) in enumerate(native_assignment(ids, sid, mask)):
                        value[d, j], error[d, j] = val[rows, j], err[rows, j]
                        observed[d, j] = True
                        error_observed[d, j] = True
                        label_source[d, j] = source.encode()
                        native_counts[source]["feature_matched_per_target"].append(
                            len(d)
                        )
            observed &= primary_input[:, None]
            error_observed &= observed & np.isfinite(error) & (error > 0)
            with np.load(paths["ancestry"], allow_pickle=False) as ancestry:
                historical = ancestry["historical_fit_source_id"][
                    ancestry["historical_fit_sample_count"] > 0
                ]
            historical = np.union1d(
                historical, old_ids[frozen["role"].astype("U") == "fit"]
            )
            with h5py.File(paths["references"], "r") as refs:
                reserved = exact_ids(refs["ALL_reference_holdout_source_ids"][:])
            reserved = np.union1d(
                reserved,
                np.asarray(config.get("diagnostic_holdout_ids", []), dtype="i8"),
            )
            role = development_roles(
                ids, observed.any(1) & primary_input, reserved, historical
            )
            counts = role_counts(role, observed)
            out.create_dataset(
                "reserved_reference_source_id",
                data=np.unique(reserved),
                compression="lzf",
            )
            out.create_dataset(
                "historical_supervised_source_id",
                data=np.unique(historical),
                compression="lzf",
            )
            fit_targets = np.array(list(counts["fit"]["targets"].values()))
            if np.any(fit_targets < config.get("minimum_fit_per_target", 1000)):
                raise ValueError(f"Insufficient sparse FIT counts: {fit_targets}")
            input_ds = out.create_dataset(
                "input",
                shape=(n, len(names)),
                dtype=expanded["input"].dtype,
                chunks=(min(8192, n), len(names)),
                compression="lzf",
                fillvalue=-9999,
            )
            # Streaming source writes: sorted destination identity order only.
            manifest.update(status="materializing_features")
            write_json(output / "manifest.json", manifest)
            for start in range(0, len(ds), config.get("chunk_rows", 32768)):
                sl = slice(start, min(start + config.get("chunk_rows", 32768), len(ds)))
                batch = expanded["input"][sl]
                batch[:, ~allowed] = -9999
                input_ds[ds[sl]] = batch
                write_json(
                    output / "progress.json",
                    dict(
                        phase="materializing_features",
                        completed_rows=sl.stop,
                        total_rows=len(ds),
                    ),
                )
            ordering = np.argsort(fs)
            for key, array in modality.items():
                out.create_dataset(key, data=array, compression="lzf")
            out.create_dataset("xp_status", data=xp_status, compression="lzf")
            for key in ("value", "error"):
                original = out.create_dataset(
                    "original_benchmark_" + key,
                    shape=(n, 6),
                    dtype="f8",
                    fillvalue=np.nan,
                    chunks=(min(n, 8192), 6),
                    compression="lzf",
                )
                original[fs[ordering]] = frozen[key][ordering]
            old_input = frozen["input"][ordering].copy()
            old_input[:, ~allowed] = -9999
            input_ds[fs[ordering]] = old_input
            for key, array in [
                ("source_id", ids),
                ("value", value),
                ("error", error),
                ("observed", observed),
                ("quality", observed),
                ("error_observed", error_observed),
                ("label_source", label_source),
                ("role", role),
                ("original_benchmark_role", original_role),
                ("original_benchmark_reference_mask", original_ref),
                ("known_supervised_ancestor_exposed", np.isin(ids, historical)),
                ("reserved_reference", np.isin(ids, reserved)),
                ("primary_input_eligible", primary_input),
                ("feature_names", names),
                ("target_names", np.array(TARGET_NAMES, "S16")),
                (
                    "target_units",
                    np.array(["K", "dex(cm/s2)", "dex", "dex", "Gyr", "kpc"], "S16"),
                ),
                ("scaler_center", expanded["scaler_center"][:]),
                ("scaler_scale", expanded["scaler_scale"][:]),
            ]:
                out.create_dataset(
                    key, data=array, compression="lzf" if np.ndim(array) > 0 else None
                )
            with h5py.File(paths["native_astrometry"], "r") as native:
                ni = exact_ids(native["source_id"][:])
                p = np.searchsorted(ids, ni)
                match = p < n
                match[match] &= ids[p[match]] == ni[match]
                for key in ("parallax_mas", "parallax_error_mas", "observed"):
                    array = (
                        np.zeros(n, bool)
                        if key == "observed"
                        else np.full(n, np.nan, "f4")
                    )
                    array[p[match]] = native["astrometry/" + key][:][match]
                    out.create_dataset(
                        "measured_parallax/" + key, data=array, compression="lzf"
                    )
                parallax_names = names.astype("U").tolist()
                if parallax_names.count("PARALLAX") != 1:
                    raise ValueError("Parent PARALLAX feature identity must be unique")
                parallax_index = parallax_names.index("PARALLAX")
                center = expanded["scaler_center"][parallax_index]
                scale = expanded["scaler_scale"][parallax_index]
                if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Parent parallax normalization invalid")
                raw_p = out["measured_parallax/parallax_mas"][:]
                raw_e = out["measured_parallax/parallax_error_mas"][:]
                p_observed = out["measured_parallax/observed"][:]
                good_p = (
                    p_observed
                    & np.isfinite(raw_p)
                    & (raw_p != -9999)
                    & np.isfinite(raw_e)
                    & (raw_e > 0)
                )
                standardized = np.full(n, -9999.0, "f4")
                standardized[good_p] = (raw_p[good_p] - center) / scale
                if not np.isfinite(standardized[good_p]).all():
                    raise ValueError("Nonfinite standardized measured parallax")
                out["measured_parallax/observed"][:] = good_p
                out.create_dataset(
                    "measured_parallax/input_value",
                    data=standardized,
                    compression="lzf",
                )
            out.attrs.update(
                schema="remediation-sparse-six-target-source-v1",
                source_commit=source_commit,
                library_commit=config["library_commit"],
                consumed_inspected_test=True,
                fresh_unseen_test_claim=False,
                physical_truth_claim=False,
                native_quality_rebuilt_from_raw=True,
                canonical_native_precedence="galah_then_apogee",
                astrometry_profile_ready=True,
                parallax_feature_index=parallax_index,
                parallax_input_scope="uncorrected_raw_measurement_not_physical_distance_likelihood",
            )
        manifest.update(
            status="remediation_source_complete",
            training_source_ready=True,
            prepared_path=str(destination),
            prepared_sha256=sha(destination),
            rows=n,
            identity_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            identity_min=int(ids.min()),
            identity_max=int(ids.max()),
            reserved_reference_id_count=int(len(np.unique(reserved))),
            historical_supervised_id_count=int(len(np.unique(historical))),
            role_counts=counts,
            fit_observation_support=observation_support(observed, role),
            native_counts=native_counts,
            source_target_counts={
                (source.decode() if isinstance(source, bytes) else str(source)): {
                    name: int(np.sum(observed[:, j] & (label_source[:, j] == source)))
                    for j, name in enumerate(TARGET_NAMES)
                }
                for source in np.unique(label_source[observed])
            },
            modality_counts={key: int(array.sum()) for key, array in modality.items()},
            domain_coverage={
                name: {
                    "edges": edges,
                    "fit_counts": np.histogram(
                        value[(role == b"fit") & observed[:, j], j], bins=edges
                    )[0].tolist(),
                    "devtest_counts": np.histogram(
                        value[(role == b"devtest") & observed[:, j], j], bins=edges
                    )[0].tolist(),
                }
                for j, name, edges in [
                    (1, "logg", [-5, 0, 1, 2, 3, 4, 5, 6, 10]),
                    (2, "fe_h", [-10, -3, -2, -1, 0, 1, 3]),
                ]
            },
            feature_names=names.astype("U").tolist(),
            target_names=list(TARGET_NAMES),
            raw_native_quality_reselected=True,
            development_test_scope="Previously inspected test identities consumed as development; no fresh-unseen claim",
            historical_supervised_heldout_count=int(
                np.sum(
                    np.isin(ids, historical)
                    & np.isin(role, [b"tune", b"cal", b"devtest", b"external"])
                )
            ),
            feature_profiles={
                "phot_xp": {"ready": True},
                "phot_xp_measured_parallax": {
                    "ready": True,
                    "feature_index": parallax_index,
                    "override_dataset": "measured_parallax/input_value",
                    "availability_dataset": "measured_parallax/observed",
                    "measured_rows": int(good_p.sum()),
                    "scope": "Raw signed measured mas normalized with parent scaler, formal errors available; no zero-point correction, inverse parallax or physical-distance likelihood",
                },
            },
            source_limitations=[
                "GALAH gravity is Gaia-parallax conditioned; label and astrometric likelihood errors are not independent",
                "Age/distance are independent sparse StarHorse posterior-reference masks, not physical truth or quoted measurement errors",
                "DESI alpha source-qualified FeH>-2 masks inherited; no new abundance-scale homogenization",
                "Known ancestry protected, full parent86 historical exposure is not certified",
                "Raw native eligible rows without staged features are counted but not fabricated",
            ],
        )
        write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    try:
        materialize(config, args.output)
    except Exception as exc:
        path = Path(args.output) / "manifest.json"
        if path.exists():
            state = json.loads(path.read_text())
            state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            write_json(path, state)
        raise


if __name__ == "__main__":
    main()
