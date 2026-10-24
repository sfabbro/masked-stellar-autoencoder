"""Source-bound sparse six-target assembly; CPU headless CANFAR only.

No acquisitions, model forwards, imputed labels, or dense six-dimensional density
claim. Native three-label measurements, DESI alpha measurements and StarHorse
age/distance posterior medians retain separate source/error semantics. Existing
role conflicts resolve toward held-out use, never toward training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

TARGET_NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")
TARGET_UNITS = ("K", "dex", "dex", "dex", "Gyr", "kpc")
ROLE_PRIORITY = {"fit": 0, "tune": 1, "cal": 2, "test": 3}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_npz(spec):
    path = Path(spec["path"])
    if sha(path) != spec["sha256"]:
        raise ValueError(f"Source NPZ hash changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def assemble_sparse(native, alpha, posterior, *, excluded_ids=()):
    """Assemble exact ID union, preserving NaNs and global held-out priority.

    `native` is the SHA-bound original paired cohort, alpha/posterior are the
    original feature-prepared artifacts. This is array assembly, not inference.
    """
    streams = []
    for name, data in (("native", native), ("alpha", alpha), ("posterior", posterior)):
        ids = np.asarray(data["source_id"])
        if (
            ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)
            or np.any(ids <= 0)
        ):
            raise ValueError(f"{name}: exact positive unique integer IDs required")
        if name == "native":
            roles = np.full(len(ids), "", dtype="U5")
            for role, key in (
                ("fit", "fit_rows"),
                ("tune", "tune_rows"),
                ("cal", "calibration_rows"),
                ("test", "test_rows"),
            ):
                rows = np.asarray(data[key], dtype="i8")
                if (
                    np.any(rows < 0)
                    or np.any(rows >= len(ids))
                    or len(np.unique(rows)) != len(rows)
                ):
                    raise ValueError("Invalid native source role rows")
                if np.any(roles[rows] != ""):
                    raise ValueError("Native source role overlap")
                roles[rows] = role
            selected = roles != ""
            values = np.asarray(data["value"], dtype="f8")
            observed = np.asarray(data["observed"], bool) & np.asarray(
                data["quality"], bool
            )
            observed &= np.isfinite(values)
            errors = np.asarray(data["error"], dtype="f8")
            sources = np.asarray(data["survey"]).astype("U64")
            if (
                values.shape != (len(ids), 3)
                or observed.shape != values.shape
                or errors.shape != values.shape
                or sources.shape != values.shape
            ):
                raise ValueError("Native three-label array widths differ")
            streams.append(
                (
                    name,
                    ids[selected],
                    roles[selected],
                    values[selected],
                    observed[selected],
                    errors[selected],
                    sources[selected],
                    (0, 1, 2),
                    None,
                )
            )
        else:
            roles = np.asarray(data["role"]).astype("U5")
            if (
                roles.shape != ids.shape
                or not np.isin(roles, list(ROLE_PRIORITY)).all()
            ):
                raise ValueError(f"{name}: invalid source roles")
            values = np.asarray(data["target"], dtype="f8")
            dims = 1 if name == "alpha" else 2
            if values.shape != (len(ids), dims) or not np.isfinite(values).all():
                raise ValueError(f"{name}: target shape/nonfinite values")
            if np.any(
                np.asarray(data["benchmark_partition"])[np.isin(roles, ["fit", "tune"])]
                == 1
            ):
                raise ValueError(f"{name}: benchmark validation in training/tuning")
            if name == "alpha":
                if np.any(
                    np.asarray(data["raw_feh"])[np.isin(roles, ["fit", "tune"])] <= -2
                ):
                    raise ValueError("Unrecommended DESI alpha in FIT/TUNE")
                error = np.asarray(data["quoted_error"], dtype="f8")
                if error.shape != values.shape:
                    raise ValueError("DESI alpha quoted-error shape changed")
                quantiles = None
                source = np.full(values.shape, "DESI_RVSpecFit_alpha_fe", dtype="U64")
                columns = (3,)
            else:
                values = np.exp(values)
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError("Posterior natural-log conversion invalid")
                quantiles = np.asarray(data["posterior_quantiles"], dtype="f8")
                if (
                    quantiles.shape != (len(ids), 2, 3)
                    or not np.isfinite(quantiles).all()
                    or np.any(quantiles <= 0)
                    or np.any(np.diff(quantiles, axis=-1) <= 0)
                ):
                    raise ValueError("Posterior native 16/50/84 quantiles invalid")
                if not np.allclose(quantiles[:, :, 1], values, atol=1e-12, rtol=1e-12):
                    raise ValueError("Posterior native median/log target mismatch")
                error = np.full_like(values, np.nan)
                source = np.full(
                    values.shape, "StarHorse_catalogue_posterior_median", dtype="U64"
                )
                columns = (4, 5)
            streams.append(
                (
                    name,
                    ids,
                    roles,
                    values,
                    np.ones_like(values, bool),
                    error,
                    source,
                    columns,
                    quantiles,
                )
            )
    ids = np.unique(np.concatenate([stream[1] for stream in streams]))
    ids = ids[~np.isin(ids, np.asarray(list(excluded_ids), dtype="i8"))]
    value = np.full((len(ids), 6), np.nan)
    error = value.copy()
    observed = np.zeros(value.shape, bool)
    source = np.full(value.shape, "", dtype="U64")
    source_role = np.full(value.shape, "", dtype="U5")
    quantiles = np.full((*value.shape, 3), np.nan)
    role_number = np.full(len(ids), -1, dtype="i1")
    conflicts = 0
    for _, sids, roles, values, mask, errors, sources, columns, bounds in streams:
        keep = np.isin(sids, ids)
        rows = np.searchsorted(ids, sids[keep])
        numbers = np.array([ROLE_PRIORITY[role] for role in roles[keep]], dtype="i1")
        conflicts += int(
            ((role_number[rows] >= 0) & (role_number[rows] != numbers)).sum()
        )
        role_number[rows] = np.maximum(role_number[rows], numbers)
        for index, column in enumerate(columns):
            selected_rows = rows[mask[keep, index]]
            if observed[selected_rows, column].any():
                raise ValueError("Duplicate source labels would silently overwrite")
            value[selected_rows, column] = values[keep, index][mask[keep, index]]
            error[selected_rows, column] = errors[keep, index][mask[keep, index]]
            source[selected_rows, column] = sources[keep, index][mask[keep, index]]
            observed[selected_rows, column] = True
            source_role[selected_rows, column] = roles[keep][mask[keep, index]]
            if bounds is not None:
                quantiles[selected_rows, column] = bounds[keep, index][
                    mask[keep, index]
                ]
    if np.any(role_number < 0) or not observed.any(1).all():
        raise ValueError("Union has unassigned roles or no observed labels")
    roles = np.array(list(ROLE_PRIORITY))[role_number]
    return dict(
        source_id=ids,
        role=roles,
        value=value,
        observed=observed,
        quality=observed & (source_role == roles[:, None]),
        label_source_role=source_role,
        original_role_eligible=observed & (source_role == roles[:, None]),
        error=error,
        error_observed=observed & np.isfinite(error) & (error > 0),
        label_source=source,
        posterior_quantiles=quantiles,
    ), conflicts


def prepare(config, output):
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError("Six-target preparation must execute on CANFAR CPU headless")
    if output.exists():
        raise FileExistsError(output)
    paths = {
        key: Path(config[key]["path"]) for key in ("features", "references", "parent")
    }
    for key, path in paths.items():
        if sha(path) != config[key]["sha256"]:
            raise ValueError(f"{key} pinned source changed")
    native_quality = json.loads(Path(config["native"]["source_qa"]).read_text())
    if (
        native_quality.get("status") != "passed"
        or not all(
            native_quality.get(key) is True
            for key in (
                "native_labels_verified",
                "all_original_source_target_values_errors_flags_rows_verified",
                "all_source_id_roles_and_reference_bans_verified",
                "all_target_quality_masks_verified",
            )
        )
        or native_quality.get("sidecar_sha256")
        != config["native"]["original_sidecar_sha256"]
        or native_quality.get("cohort_sha256") != config["features"]["sha256"]
    ):
        raise ValueError("Original native source-quality proof is not passed/pinned")
    native = checked_npz(config["native"])
    alpha = checked_npz(config["alpha"])
    posterior = checked_npz(config["posterior"])
    with h5py.File(paths["references"]) as f:
        excluded = f["ALL_reference_holdout_source_ids"][:]
    data, conflicts = assemble_sparse(native, alpha, posterior)
    reference_holdout = np.isin(data["source_id"], excluded)
    if np.any(reference_holdout & np.isin(data["role"], ["fit", "tune"])):
        raise ValueError("Reference identities entered global FIT/TUNE")
    data["reference_holdout"] = reference_holdout
    parent = torch_load_trusted(paths["parent"], map_location="cpu", weights_only=False)
    signature = parent["run_signature"]
    with h5py.File(paths["features"]) as f:
        names = json.loads(f.attrs["feature_columns_json"])
        if names != signature["feature_cols"]:
            raise ValueError("Feature names differ from frozen parent signature")
        fids = f["features/source_id"][:]
        if len(np.unique(fids)) != len(fids):
            raise ValueError("Feature IDs duplicated")
        lookup = {int(s): i for i, s in enumerate(fids)}
        if not np.isin(data["source_id"], fids).all():
            raise ValueError("Prepared source lacks actual original feature row")
        rows = np.array([lookup[int(s)] for s in data["source_id"]])
        order = np.argsort(rows)
        raw = np.empty((len(rows), len(names)), dtype="f8")
        errors = np.empty_like(raw)
        raw[order] = f["features/values"][rows[order]]
        errors[order] = f["features/reported_feature_errors"][rows[order]]
        partitions = f["features/benchmark_partition"][:][rows]
    if np.any(partitions[np.isin(data["role"], ["fit", "tune"])] == 1):
        raise ValueError("Global FIT/TUNE still contain benchmark validation")
    center, scale = [
        np.asarray(signature[key], dtype="f8")
        for key in ("scaler_center", "scaler_scale")
    ]
    if (
        center.shape != (len(names),)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Parent feature scaler invalid")
    allowed = np.array(
        [
            name in ("W1", "W2", "G", "BP", "RP")
            or name in {f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)}
            for name in names
        ]
    )
    visible = allowed[None] & np.isfinite(raw) & (raw != -9999)
    scaled = (raw - center) / scale
    visible &= np.isfinite(scaled)
    data.update(
        input=np.where(visible, scaled, -9999).astype("f4"),
        visible=visible,
        raw=raw,
        feature_errors=errors,
        benchmark_partition=partitions,
        feature_row=rows,
        feature_names=np.asarray(names),
        scaler_center=center,
        scaler_scale=scale,
        target_names=np.asarray(TARGET_NAMES),
        target_units=np.asarray(TARGET_UNITS),
        parent_checkpoint_sha256=np.array(config["parent"]["sha256"]),
    )
    counts = {
        role: data["quality"][data["role"] == role].sum(0).tolist()
        for role in ROLE_PRIORITY
    }
    if any(
        min(values) < config.get("minimum_observed_per_role_target", 16)
        for values in counts.values()
    ):
        raise ValueError(
            f"Insufficient observed target counts after role reconciliation: {counts}"
        )
    output.mkdir(parents=True)
    np.savez_compressed(output / "prepared.npz", **data)
    report = dict(
        status="sparse_six_target_source_assembly_complete",
        input_sha256=sha(output / "prepared.npz"),
        schema="sparse-six-target-v1",
        target_names=TARGET_NAMES,
        target_units=TARGET_UNITS,
        role_counts={r: int((data["role"] == r).sum()) for r in ROLE_PRIORITY},
        observed_per_role_target=counts,
        role_promoted_labels_excluded_from_scoring=True,
        observed_pattern_counts={
            str(int(pattern)): int(count)
            for pattern, count in zip(
                *np.unique(
                    (data["observed"] * (1 << np.arange(6))).sum(1), return_counts=True
                ),
                strict=True,
            )
        },
        role_conflict_events=conflicts,
        role_conflict_policy="test > cal > tune > fit; never promote held-out source IDs into training",
        reference_exclusions="Original stream restrictions retained: no reference union in global FIT/TUNE; canonical native TEST remains held out rather than deleted",
        source_provenance=config,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        created_at=dt.datetime.now(dt.UTC).isoformat(),
        frozen_encoder_forward_performed=False,
        target_error_semantics="Native3+DESI quoted symmetric errors only; StarHorse posterior widths are NOT measurement errors",
        primary_age_source="StarHorse posterior median only; APOKASC remains separate source-specific auxiliary evidence",
        covariance_semantics="Unknown target-error offdiagonals and cross-source covariance are not imputed; no empirical full6D density/copula certified",
        physical_truth_claim=False,
        full_six_dimensional_observation_count=int(data["observed"].all(1).sum()),
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
