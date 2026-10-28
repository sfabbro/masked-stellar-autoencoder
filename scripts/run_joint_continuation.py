"""Checkpointed 30-epoch native continuation. Preparation is fail-closed.

Gaussian native Teff/logg/FeH with measurement-error convolution; optional
separate raw-parallax point loss. This is NOT a trained joint distance PDF.
All scientific readiness gates precede any long-running production training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import run_pretrain_experiments as pt
import run_xp_representation_probes as probes
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.models.model import (
    _capture_rng_state,
    _restore_rng_state,
)
from masked_stellar_autoencoder.training.continuation_state import (
    atomic_json,
    exclusive_run,
    load_checkpoint,
    save_checkpoint,
    sha256,
    write_qa_snapshot,
)
from masked_stellar_autoencoder.training.joint_prediction import (
    ResidualJointHead,
    masked_gaussian_measurement_nll,
    signed_log_distance_nll,
)
from masked_stellar_autoencoder.training.reconstruction_sample import (
    open_reconstruction_sample,
)
from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)

TARGETS = ("teff", "logg", "fe_h")
UNITS = ("K", "dex", "dex")


def validate_policy(cfg, *, smoke=False):
    if smoke and (cfg["epochs"] > 2 or cfg["reconstruction_rows_per_epoch"] > 65536):
        raise ValueError("Smoke mode is bounded to two epochs and65536rows per epoch")
    for key in (
        "epochs",
        "qa_every_epochs",
        "checkpoint_every_steps",
        "effective_batch",
        "micro_batch",
        "supervised_batch",
        "supervised_micro_batch",
        "reconstruction_rows_per_epoch",
        "validation_rows",
        "qa_prediction_rows_per_target",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        cfg["micro_batch"] > cfg["effective_batch"]
        or cfg["supervised_micro_batch"] > cfg["supervised_batch"]
    ):
        raise ValueError("Microbatch exceeds effective batch")
    if cfg["include_positions"] or cfg["include_proper_motions"]:
        raise ValueError(
            "Astrometric input extension requires its own tested policy, not a silent config switch"
        )
    if cfg["age_enabled"] or cfg["alpha_enabled"]:
        raise ValueError("Age/alpha labels and source definitions are not certified")
    for key in ("encoder_lr", "head_lr", "gradient_clip", "stellar_weight"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("reconstruction_weight", "spectral_weight", "parallax_weight"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    if bool(cfg["parallax_weight"] > 0) != cfg["distance_enabled"]:
        raise ValueError("Distance enabled/weight differ")
    if not smoke:
        report = json.loads(Path(cfg["readiness_report"]).read_text())
        required = [
            "native_labels_verified",
            "original_xp_units_verified",
            "source_roles_verified",
            "resume_verified",
            "gpu_preflight_verified",
            "vos_publication_verified",
            "gradient_balance_verified",
        ]
        if cfg["distance_enabled"]:
            required += [
                "gaia_quality_and_zero_point_verified",
                "distance_input_no_copy_verified",
                "correlated_teacher_policy_verified",
            ]
        if cfg["include_reddening"]:
            required += ["reddening_provenance_verified"]
        if report.get("status") != "passed" or not all(
            report.get(k) is True for k in required
        ):
            raise ValueError("Production readiness gates are incomplete")
        # Readiness is bound to this policy, not an old report for a different run.
        if report.get("policy_sha256") != policy_digest(cfg):
            raise ValueError("Readiness report certifies a different policy")


def policy_digest(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_promotion(cfg, identity):
    """A staged large sample requires an actual completed, matched pilot."""
    if cfg.get("reconstruction_source", "benchmark") == "benchmark":
        if cfg["reconstruction_rows_per_epoch"] > 9000000:
            raise ValueError("Benchmark training has only9M rows; no silent repeats")
        return
    if cfg["reconstruction_source"] != "staged_sample":
        raise ValueError("Unknown reconstruction source")
    if not cfg.get("sample_request_sha256") or not cfg.get("pilot_policy_sha256"):
        raise ValueError("Large sample needs pinned sample and pilot policies")
    report = json.loads(Path(cfg["promotion_report"]).read_text())
    required = (
        "native_labels_verified",
        "original_xp_units_verified",
        "source_roles_verified",
        "resume_verified",
        "gpu_preflight_verified",
        "vos_publication_verified",
        "gradient_balance_verified",
        "declared_tuning_checks_passed",
    )
    if report.get("status") != "passed" or not all(
        report.get(k) is True for k in required
    ):
        raise ValueError("10M pilot promotion gates are incomplete")
    if (
        report.get("training_rows_per_epoch") != 9000000
        or report.get("completed_epochs", 0) < 3
        or report.get("policy_sha256") != cfg["pilot_policy_sha256"]
    ):
        raise ValueError("Promotion report is not the completed matched10M pilot")
    for key in ("source_sha", "parent_sha256", "cohort_sha256", "joint_labels_sha256"):
        if report.get(key) != identity[key]:
            raise ValueError(f"Promotion report differs in {key}")


def load_data(cfg, features, center, scale):
    """Load aligned sidecar; raw fields stay available for independent verification."""
    with h5py.File(cfg["cohort"]) as f, h5py.File(cfg["joint_labels"]) as labels:
        ids = f["features/source_id"][:]
        np.testing.assert_array_equal(ids, labels["source_id"][:])
        values = f["features/values"][:]
        observed = f["features/feature_observed"][:]
        safe, _ = spectrophotometric_inputs(
            torch.from_numpy(((values - center) / scale).astype("f4")),
            torch.from_numpy(observed),
            features,
            include_reddening=cfg["include_reddening"],
        )
        if not labels.attrs.get("complete", False):
            raise ValueError("Joint sidecar is incomplete")
        result = dict(
            source_id=ids,
            input=safe.numpy(),
            value=labels["targets/value"][:],
            error=labels["targets/uncertainty"][:],
            observed=labels["targets/observed"][:],
            survey=labels["targets/survey"][:],
        )
        result.update(
            {
                k: labels["roles/" + k][:]
                for k in (
                    "fit",
                    "tune",
                    "calibration",
                    "test",
                    "reference_holdout",
                    "benchmark_partition",
                )
            }
        )
        result["parallax"] = labels["astrometry/parallax_mas"][:]
        result["parallax_error"] = labels["astrometry/parallax_error_mas"][:]
        result["parallax_observed"] = labels["astrometry/observed"][:]
        if cfg["distance_enabled"]:
            # Explicit independent source audit supplies eligibility and corrections.
            result["distance_fit"] = (
                labels["astrometry/distance_fit_eligible"][:] & result["fit"]
            )
            result["zero_point"] = labels["astrometry/zero_point_mas"][:]
        else:
            result["distance_fit"] = np.zeros(len(ids), bool)
            result["zero_point"] = np.zeros(len(ids))
    if np.any(
        result["fit"]
        & (
            result["tune"]
            | result["calibration"]
            | result["test"]
            | result["reference_holdout"]
        )
    ):
        raise ValueError("Supervised source roles overlap")
    for role in ("fit", "tune", "calibration"):
        if np.any(
            result[role]
            & ((result["benchmark_partition"] != 0) | result["reference_holdout"])
        ):
            raise ValueError(
                "Benchmark validation/reference entered fit/tune/calibration"
            )
    for j in range(3):
        selected = result["fit"] & result["observed"][:, j]
        if selected.sum() < cfg["minimum_fit_labels_per_target"]:
            raise ValueError(f"Insufficient audited fit labels for {TARGETS[j]}")
    fit = np.flatnonzero(result["fit"] & result["observed"].any(1))
    yc = np.array(
        [
            result["value"][result["fit"] & result["observed"][:, j], j].mean()
            for j in range(3)
        ]
    )
    ys = np.array(
        [
            result["value"][result["fit"] & result["observed"][:, j], j].std()
            for j in range(3)
        ]
    )
    if not np.isfinite(yc).all() or not np.isfinite(ys).all() or np.any(ys <= 0):
        raise ValueError("Degenerate target scaling")
    result.update(fit_rows=fit, label_center=yc, label_scale=ys)
    return result


def supervised_sum(model, head, data, rows, device):
    output = head(model.encoder(torch.as_tensor(data["input"][rows], device=device)))
    y = torch.as_tensor(
        (data["value"][rows] - data["label_center"]) / data["label_scale"],
        device=device,
        dtype=torch.float32,
    )
    err = torch.as_tensor(
        data["error"][rows] / data["label_scale"], device=device, dtype=torch.float32
    )
    valid = torch.as_tensor(data["observed"][rows], device=device)
    core = masked_gaussian_measurement_nll(
        output["mean"], output["scale_tril"], y, err, eligible=valid
    )
    distance = signed_log_distance_nll(
        output["log_distance"],
        torch.as_tensor(data["parallax"][rows], device=device, dtype=torch.float32),
        torch.as_tensor(
            data["parallax_error"][rows], device=device, dtype=torch.float32
        ),
        zero_point_mas=torch.as_tensor(
            data["zero_point"][rows], device=device, dtype=torch.float32
        ),
        eligible=torch.as_tensor(data["distance_fit"][rows], device=device),
    )
    return core, distance


@torch.no_grad()
def evaluate(model, head, data, rows, device):
    model.eval()
    head.eval()
    means = []
    covariances = []
    distances = []
    for start in range(0, len(rows), 512):
        out = head(
            model.encoder(
                torch.as_tensor(data["input"][rows[start : start + 512]], device=device)
            )
        )
        means.append(out["mean"].cpu().numpy())
        tril = out["scale_tril"]
        covariances.append((tril @ tril.transpose(-1, -2)).cpu().numpy())
        distances.append(out["log_distance"].cpu().numpy())
    if not means:
        raise ValueError("Empty QA evaluation role")
    return dict(
        mean=np.concatenate(means) * data["label_scale"] + data["label_center"],
        covariance=np.concatenate(covariances)
        * data["label_scale"][None, :, None]
        * data["label_scale"][None, None, :],
        log_distance=np.concatenate(distances),
    )


def qa_tables(
    model, head, data, device, epoch, history, predictions, resources, *, final=False
):
    role = "test" if final else "tune"
    rows = np.flatnonzero(data[role] & data["observed"].any(1))
    out = evaluate(model, head, data, rows, device)
    if data.get("archive_directory"):
        archive = Path(data["archive_directory"])
        archive.mkdir(parents=True, exist_ok=True)
        path = archive / f"{role}-epoch{epoch:03d}.npz"
        arrays = dict(
            source_id=data["source_id"][rows],
            truth=data["value"][rows],
            formal_error=data["error"][rows],
            observed=data["observed"][rows],
            survey=data["survey"][rows],
            prediction=out["mean"],
            conditional_covariance=out["covariance"],
        )
        if data.get("distance_enabled"):
            arrays["log_distance_kpc_point"] = out["log_distance"]
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                for name, value in arrays.items():
                    if (
                        saved[name].dtype != value.dtype
                        or saved[name].tobytes() != value.tobytes()
                    ):
                        raise ValueError(
                            "An existing QA archive differs at the same committed epoch"
                        )
        else:
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, path)
    metrics = []
    for j, (name, unit) in enumerate(zip(TARGETS, UNITS, strict=True)):
        valid = data["observed"][rows, j]
        truth = data["value"][rows, j]
        point = out["mean"][:, j]
        error = point - truth
        observed_sigma = np.sqrt(
            out["covariance"][:, j, j] + data["error"][rows, j] ** 2
        )
        for subset, mask in [
            ("all", valid),
            *[
                (str(s.decode()), valid & (data["survey"][rows] == s))
                for s in np.unique(data["survey"][rows])
            ],
            *([("fe_h_below_-2", valid & (truth < -2))] if name == "fe_h" else []),
        ]:
            if not mask.any():
                continue
            for metric, value in [
                ("mae", np.abs(error[mask]).mean()),
                ("bias", error[mask].mean()),
                ("rmse", np.sqrt(np.mean(error[mask] ** 2))),
                (
                    "observed_label_coverage68",
                    np.mean(
                        np.abs(error[mask]) <= 0.994457883209753 * observed_sigma[mask]
                    ),
                ),
                (
                    "observed_label_coverage95",
                    np.mean(
                        np.abs(error[mask]) <= 1.959963984540054 * observed_sigma[mask]
                    ),
                ),
            ]:
                metrics.append(
                    dict(
                        epoch=epoch,
                        split=role,
                        target=name,
                        subset=subset,
                        metric=metric,
                        value=float(value),
                        unit="fraction"
                        if metric.startswith("observed_label_coverage")
                        else unit,
                        count=int(mask.sum()),
                    )
                )
        # Fixed ID sample, no float conversion of Gaia IDs. Intervals cover future
        # reported measurements under the quoted-noise model, not certified truth.
        eligible = np.flatnonzero(valid)[
            : data.get("qa_prediction_rows_per_target", 3000)
        ]
        sigma = observed_sigma
        for k in eligible:
            predictions.append(
                dict(
                    epoch=epoch,
                    split=role,
                    target=name,
                    source_id=str(data["source_id"][rows[k]]),
                    truth=float(truth[k]),
                    prediction=float(point[k]),
                    unit=unit,
                    lower68=float(point[k] - 0.994457883209753 * sigma[k]),
                    upper68=float(point[k] + 0.994457883209753 * sigma[k]),
                    lower95=float(point[k] - 1.959963984540054 * sigma[k]),
                    upper95=float(point[k] + 1.959963984540054 * sigma[k]),
                    survey=data["survey"][rows[k]].decode(),
                )
            )
    return metrics


@torch.no_grad()
def reconstruction_qa(model, validation, features, width, device, center, scale, basis):
    """Fixed missing-modality diagnostics; never consume training RNG streams."""
    masks = pt.fixed_masks(validation, features, 42, 0.5, 0.2)["common"]
    allowed = np.array(
        [
            name not in {"PARALLAX", "EBV", "RA", "DEC", "pmra", "pmdec"}
            for name in features
        ]
    )
    predictions = []
    scored = masks[:, :width] & np.isfinite(validation[:, :width]) & allowed[:width]
    model.eval()
    for start in range(0, len(validation), 512):
        raw = torch.as_tensor(
            np.array(validation[start : start + 512], copy=True), device=device
        )
        clean, _ = spectrophotometric_inputs(
            raw, torch.isfinite(raw), features, include_reddening=False
        )
        hidden = clean.masked_fill(
            torch.as_tensor(masks[start : start + 512], device=device), -9999
        )
        predictions.append(model(hidden)[0].cpu().numpy())
    prediction = np.concatenate(predictions)
    truth = validation[:, :width]
    value = float(np.abs(prediction.astype("f8") - truth)[scored].mean())
    spectra = pt.spectral_summary(
        truth, prediction, scored, scale[:width], center[:width], basis
    )
    return value, spectra["integrated_absolute_fraction_mean"]


def run(cfg, *, resume=False, smoke=False, stop_after_steps=None):
    validate_policy(cfg, smoke=smoke)
    output = Path(cfg["output"])
    scratch = Path(cfg["scratch"])
    scratch.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if not smoke:
        readiness = json.loads(Path(cfg["readiness_report"]).read_text())
        if readiness.get("source_sha") != source:
            raise ValueError("Readiness source SHA differs from this checkout")
    identity = dict(
        policy_sha256=policy_digest(cfg),
        source_sha=source,
        parent_sha256=sha256(cfg["parent"]),
        cohort_sha256=sha256(cfg["cohort"]),
        joint_labels_sha256=sha256(cfg["joint_labels"]),
    )
    for key, expected in [
        ("parent_sha256", cfg["parent_sha256"]),
        ("cohort_sha256", cfg["cohort_sha256"]),
        ("joint_labels_sha256", cfg["joint_labels_sha256"]),
    ]:
        if not smoke and identity[key] != expected:
            raise ValueError(f"Pinned {key} differs")
    validate_promotion(cfg, identity)
    if cfg["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("Production requires CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (
        device.type == "cuda"
        and torch.cuda.get_device_properties(device).total_memory
        < cfg["minimum_gpu_memory_gib"] * 1024**3
    ):
        raise RuntimeError("GPU memory below preflight requirement")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    checkpoint = torch_load_trusted(
        cfg["parent"], map_location="cpu", weights_only=False
    )
    model_cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
    model_cfg["training"]["micro_batch_size"] = cfg["micro_batch"]
    features = model_cfg["data"]["feature_cols"]
    width = len(model_cfg["data"]["recon_cols"])
    with (
        exclusive_run(output),
        exclusive_run(scratch),
        h5py.File(cfg["benchmark"]) as bench,
    ):
        if (output / "checkpoints/latest.json").exists() and not resume:
            raise ValueError("Existing state requires explicit --resume")
        keys, validation_keys, center, scale = pt.validate_source(
            checkpoint, model_cfg, bench
        )
        if not smoke:
            provenance = json.loads(bench.attrs["provenance"])
            if (
                provenance["source_id_sha256"] != cfg["benchmark_ordered_id_sha256"]
                or len(bench["train/records"]) != 9000000
            ):
                raise ValueError("Original benchmark differs")
        data = load_data(cfg, features, center, scale)
        data["archive_directory"] = str(output / "qa-arrays")
        data["distance_enabled"] = cfg["distance_enabled"]
        data["qa_prediction_rows_per_target"] = cfg["qa_prediction_rows_per_target"]
        if not smoke:
            with h5py.File(cfg["cohort"]) as cohort:
                unit_audit = probes.audit_fits_xp_units(
                    cfg["benchmark"],
                    features,
                    cohort["features/values"][:],
                    data["source_id"],
                    "g_normalized",
                )
        else:
            unit_audit = {
                "schema_fixture_only": True,
                "fresh_original_128_certificate": False,
            }
        atomic_json(output / "unit-audit.json", unit_audit)
        if cfg.get("reconstruction_source", "benchmark") == "staged_sample":
            cache, _, _, sample_manifest = open_reconstruction_sample(
                cfg["reconstruction_cache"],
                expected_request_sha256=cfg["sample_request_sha256"],
                expected_identity=dict(
                    features=features,
                    center=center.tolist(),
                    scale=scale.tolist(),
                    rows=cfg["reconstruction_rows_per_epoch"],
                    train_keys=checkpoint["run_signature"]["train_keys"],
                    validation_keys=model_cfg["data"]["valid_keys"],
                    parent_sha256=identity["parent_sha256"],
                ),
            )
            atomic_json(output / "reconstruction-sample-manifest.json", sample_manifest)
        else:
            # Original benchmark is authority for the bounded pilot.
            cache, _, _ = pt.read_prefixes(
                bench,
                keys,
                features,
                None,
                cfg["reconstruction_rows_per_epoch"],
                center,
                scale,
                scratch / "replay.npy",
            )
        if len(cache) != cfg["reconstruction_rows_per_epoch"]:
            raise ValueError("Reconstruction sample underfilled the declared epoch")
        validation, _, _ = pt.read_prefixes(
            bench,
            validation_keys,
            features,
            None,
            cfg["validation_rows"],
            center,
            scale,
            scratch / "validation.npy",
        )
        wrapper, optimizer, _ = pt.build_wrapper(
            model_cfg, checkpoint, bench, center, scale, {}, output
        )
        model = wrapper.model
        for group in optimizer.param_groups:
            group["lr"] = cfg["encoder_lr"]
        torch.manual_seed(cfg["seed"])
        head = ResidualJointHead().to(device)
        head_optimizer = torch.optim.AdamW(
            head.parameters(), lr=cfg["head_lr"], weight_decay=cfg["weight_decay"]
        )
        rng = np.random.default_rng(cfg["seed"])
        mask_rng = torch.Generator(device=device).manual_seed(cfg["seed"])
        epoch = 0
        next_row = 0
        step = 0
        history = []
        metrics = []
        predictions = []
        resources = []
        if resume:
            state = load_checkpoint(output / "checkpoints", identity)
            model.load_state_dict(state["model"])
            head.load_state_dict(state["head"])
            optimizer.load_state_dict(state["optimizer"])
            head_optimizer.load_state_dict(state["head_optimizer"])
            rng.bit_generator.state = state["label_rng"]
            mask_rng.set_state(state["mask_rng"].cpu())
            _restore_rng_state(state["rng"])
            epoch = state["epoch"]
            next_row = state["next_row"]
            step = state["step"]
            history = state["history"]
            metrics = state["metrics"]
            predictions = state["predictions"]
            resources = state["resources"]
        xp = [i for i, n in enumerate(features) if n.startswith(("bp_", "rp_"))]
        basis = pt.spectral_basis(features[:width])
        spectral = {
            "scale": torch.as_tensor(scale[:width], device=device, dtype=torch.float32),
            "center": torch.as_tensor(
                center[:width], device=device, dtype=torch.float32
            ),
            "basis": torch.as_tensor(
                basis["basis"], device=device, dtype=torch.float32
            ),
            "wavelength": torch.as_tensor(
                basis["wavelength_nm"], device=device, dtype=torch.float32
            ),
        }
        stopped = False

        def interrupt(*_):
            nonlocal stopped
            stopped = True

        previous = {
            sig: signal.signal(sig, interrupt)
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        started = time.monotonic()
        publisher = None
        publisher_log = None

        def save():
            state = dict(
                model=model.state_dict(),
                head=head.state_dict(),
                optimizer=optimizer.state_dict(),
                head_optimizer=head_optimizer.state_dict(),
                rng=_capture_rng_state(),
                label_rng=rng.bit_generator.state,
                mask_rng=mask_rng.get_state(),
                epoch=epoch,
                next_row=next_row,
                step=step,
                history=history,
                metrics=metrics,
                predictions=predictions,
                resources=resources,
                label_center=data["label_center"],
                label_scale=data["label_scale"],
            )
            save_checkpoint(
                output / "checkpoints", state, epoch=epoch, step=step, identity=identity
            )
            if not smoke and next_row == 0:
                restored = load_checkpoint(output / "checkpoints", identity)
                for name, module in [("model", model), ("head", head)]:
                    for key, tensor in module.state_dict().items():
                        if not torch.equal(tensor.detach().cpu(), restored[name][key]):
                            raise RuntimeError(
                                "Saved full model state failed byte-exact reload"
                            )

        def export(status):
            nonlocal publisher, publisher_log
            publication = f"e{epoch:03d}-s{step:09d}-{time.time_ns()}"
            manifest = dict(
                schema_version=1,
                run_id=cfg["run_id"],
                publication_id=publication,
                generated_at_utc=datetime.now(UTC).isoformat(),
                status=status,
                source_sha=source,
                parent_sha256=identity["parent_sha256"],
                smoke=smoke,
                scientific_caveat="Native Gaussian working measurement model; unknown survey covariance/systematics; distance output is point only when enabled, no calibrated distance PDF; pretraining overlap not excluded. Epoch0 QA follows head-only warmup.",
            )
            tables = {
                "losses.csv": (["epoch", "step", "split", "loss", "value"], history),
                "metrics.csv": (
                    [
                        "epoch",
                        "split",
                        "target",
                        "subset",
                        "metric",
                        "value",
                        "unit",
                        "count",
                    ],
                    metrics,
                ),
                "predictions.csv": (
                    [
                        "epoch",
                        "split",
                        "target",
                        "source_id",
                        "truth",
                        "prediction",
                        "unit",
                        "lower68",
                        "upper68",
                        "lower95",
                        "upper95",
                        "survey",
                    ],
                    predictions,
                ),
                "resources.csv": (
                    [
                        "epoch",
                        "step",
                        "gpu_utilization_pct",
                        "gpu_memory_used_mb",
                        "rss_mb",
                        "examples_per_second",
                    ],
                    resources,
                ),
            }
            published = write_qa_snapshot(output / "qa", manifest, tables)
            # Separate publisher does bounded VOS copies while GPU training
            # continues. The durable outbox remains retryable after any failure.
            atomic_json(
                output / "qa-outbox.json",
                {
                    "publication_id": published["publication_id"],
                    "remote": cfg["qa_remote"],
                },
            )
            if (
                cfg["publication"]["enabled"]
                and not smoke
                and (publisher is None or publisher.poll() is not None)
            ):
                publisher_log = (output / "qa-publisher.log").open("a")
                command = [
                    *cfg["publication"]["command_prefix"],
                    "python",
                    "scripts/publish_training_qa.py",
                    "--output",
                    str(output),
                    "--remote",
                    cfg["qa_remote"],
                    "--watch",
                    "--timeout",
                    str(cfg["publication"]["timeout_seconds"]),
                ]
                publisher = subprocess.Popen(
                    command,
                    stdout=publisher_log,
                    stderr=publisher_log,
                    env={**os.environ, "PYTHONNOUSERSITE": "1"},
                )
            atomic_json(
                output / "manifest.json",
                {
                    **identity,
                    "run_id": cfg["run_id"],
                    "status": status,
                    "epoch": epoch,
                    "step": step,
                    "next_row": next_row,
                    "qa_publication": publication,
                    "config": cfg,
                    "smoke": smoke,
                },
            )

        try:
            if not resume:
                model.eval()
                head.train()
                for _ in range(cfg["head_warmup_steps"]):
                    rows = rng.choice(
                        data["fit_rows"], size=cfg["supervised_micro_batch"]
                    )
                    head_optimizer.zero_grad(set_to_none=True)
                    # Freeze encoder graph during head initialization.
                    with torch.no_grad():
                        latent = model.encoder(
                            torch.as_tensor(data["input"][rows], device=device)
                        )
                    out = head(latent)
                    loss = masked_gaussian_measurement_nll(
                        out["mean"],
                        out["scale_tril"],
                        torch.as_tensor(
                            (data["value"][rows] - data["label_center"])
                            / data["label_scale"],
                            device=device,
                            dtype=torch.float32,
                        ),
                        torch.as_tensor(
                            data["error"][rows] / data["label_scale"],
                            device=device,
                            dtype=torch.float32,
                        ),
                        eligible=torch.as_tensor(data["observed"][rows], device=device),
                    ).mean
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        head.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
                    )
                    head_optimizer.step()
                metrics += qa_tables(
                    model, head, data, device, 0, history, predictions, resources
                )
                value, spec = reconstruction_qa(
                    model, validation, features, width, device, center, scale, basis
                )
                history.extend(
                    [
                        dict(
                            epoch=0,
                            step=step,
                            split="validation",
                            loss="masked_mae",
                            value=value,
                        ),
                        dict(
                            epoch=0,
                            step=step,
                            split="validation",
                            loss="spectral_fraction",
                            value=spec,
                        ),
                    ]
                )
                save()
                export("running")
            elif (
                epoch > 0
                and (epoch % cfg["qa_every_epochs"] == 0 or epoch == cfg["epochs"])
                and not any(
                    r["epoch"] == epoch and r["split"] == "tune" for r in metrics
                )
            ):
                # The last committed step may precede its scheduled QA. Replay
                # evaluation, never a training step, after a boundary interruption.
                metrics += qa_tables(
                    model, head, data, device, epoch, history, predictions, resources
                )
                value, spec = reconstruction_qa(
                    model, validation, features, width, device, center, scale, basis
                )
                history.extend(
                    [
                        dict(
                            epoch=epoch,
                            step=step,
                            split="validation",
                            loss="masked_mae",
                            value=value,
                        ),
                        dict(
                            epoch=epoch,
                            step=step,
                            split="validation",
                            loss="spectral_fraction",
                            value=spec,
                        ),
                    ]
                )
                save()
                export("running")
            while epoch < cfg["epochs"]:
                while next_row < len(cache):
                    tick = time.monotonic()
                    raw = np.array(
                        cache[next_row : next_row + cfg["effective_batch"]], copy=True
                    )
                    target = torch.as_tensor(raw, device=device)
                    clean, visible = spectrophotometric_inputs(
                        target,
                        torch.isfinite(target),
                        features,
                        include_reddening=cfg["include_reddening"],
                    )
                    hidden, mask, finite = pt.training_mask(
                        clean, xp, 0.5, 0.2, mask_rng
                    )
                    scored = (mask & visible)[:, :width]
                    total_count = scored.sum()
                    optimizer.zero_grad(set_to_none=True)
                    head_optimizer.zero_grad(set_to_none=True)
                    model.train()
                    head.train()
                    reco = 0.0
                    core_value = 0.0
                    parallax_value = 0.0
                    for start in range(0, len(raw), cfg["micro_batch"]):
                        stop = start + cfg["micro_batch"]
                        prediction, _ = model(hidden[start:stop])
                        loss = pt.experiment_objective(
                            prediction,
                            target[start:stop, :width],
                            scored[start:stop],
                            xp,
                            {
                                "loss": "spectral_mae",
                                "spectral_weight": cfg["spectral_weight"],
                            }
                            if cfg["spectral_weight"]
                            else {"loss": "mae"},
                            total_count,
                            spectral=spectral,
                        )
                        (cfg["reconstruction_weight"] * loss).backward()
                        reco += float(loss.detach())
                    rows = rng.choice(data["fit_rows"], size=cfg["supervised_batch"])
                    core_count = int(data["observed"][rows].sum())
                    distance_count = int(data["distance_fit"][rows].sum())
                    for start in range(0, len(rows), cfg["supervised_micro_batch"]):
                        core, distance = supervised_sum(
                            model,
                            head,
                            data,
                            rows[start : start + cfg["supervised_micro_batch"]],
                            device,
                        )
                        loss = cfg["stellar_weight"] * core.numerator / max(
                            1, core_count
                        ) + cfg["parallax_weight"] * distance.numerator / max(
                            1, distance_count
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite composite objective")
                        loss.backward()
                        core_value += float(core.numerator.detach()) / max(
                            1, core_count
                        )
                        parallax_value += float(distance.numerator.detach()) / max(
                            1, distance_count
                        )
                    for module in (model, head):
                        torch.nn.utils.clip_grad_norm_(
                            module.parameters(),
                            cfg["gradient_clip"],
                            error_if_nonfinite=True,
                        )
                    optimizer.step()
                    head_optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    next_row += len(raw)
                    step += 1
                    for name, value in [
                        ("reconstruction_composite", reco),
                        ("stellar_nll_per_observed_coordinate", core_value),
                        ("parallax_nll", parallax_value),
                    ]:
                        history.append(
                            dict(
                                epoch=epoch + 1,
                                step=step,
                                split="fit",
                                loss=name,
                                value=value,
                            )
                        )
                    resources.append(
                        dict(
                            epoch=epoch + 1,
                            step=step,
                            gpu_utilization_pct="",
                            gpu_memory_used_mb=torch.cuda.max_memory_allocated()
                            / 1024**2
                            if device.type == "cuda"
                            else 0,
                            rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            / (1024**2 if sys.platform == "darwin" else 1024),
                            examples_per_second=len(raw)
                            / max(1e-9, time.monotonic() - tick),
                        )
                    )
                    if next_row >= len(cache):
                        epoch += 1
                        next_row = 0
                        boundary = True
                    else:
                        boundary = False
                    if boundary or step % cfg["checkpoint_every_steps"] == 0:
                        save()
                    if boundary and (
                        epoch % cfg["qa_every_epochs"] == 0 or epoch == cfg["epochs"]
                    ):
                        metrics += qa_tables(
                            model,
                            head,
                            data,
                            device,
                            epoch,
                            history,
                            predictions,
                            resources,
                        )
                        value, spec = reconstruction_qa(
                            model,
                            validation,
                            features,
                            width,
                            device,
                            center,
                            scale,
                            basis,
                        )
                        history.extend(
                            [
                                dict(
                                    epoch=epoch,
                                    step=step,
                                    split="validation",
                                    loss="masked_mae",
                                    value=value,
                                ),
                                dict(
                                    epoch=epoch,
                                    step=step,
                                    split="validation",
                                    loss="spectral_fraction",
                                    value=spec,
                                ),
                            ]
                        )
                        export("running")
                        save()
                    if (
                        stopped
                        or (stop_after_steps is not None and step >= stop_after_steps)
                        or time.monotonic() - started > cfg["max_wall_hours"] * 3600
                    ):
                        save()
                        export("stopped")
                        return
                    if boundary:
                        break
            if not any(r["split"] == "test" and r["epoch"] == epoch for r in metrics):
                metrics += qa_tables(
                    model,
                    head,
                    data,
                    device,
                    epoch,
                    history,
                    predictions,
                    resources,
                    final=True,
                )
            save()
            export("complete")
        except BaseException:
            # Resume only a fully committed optimizer step. Never save partial updates.
            export("failed")
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if publisher is not None:
                try:
                    publisher.wait(
                        timeout=cfg["publication"]["timeout_seconds"] * 5 + 60
                    )
                except subprocess.TimeoutExpired:
                    publisher.terminate()
                    atomic_json(
                        output / "qa-publication-status.json",
                        {
                            "status": "outbox_pending",
                            "message": "Local QA preserved; publisher exceeded bounded shutdown wait.",
                        },
                    )
            if publisher_log is not None:
                publisher_log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/joint-continuation-30e.yaml")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stop-after-steps", type=int)
    args = p.parse_args()
    run(
        yaml.safe_load(Path(args.config).read_text()),
        resume=args.resume,
        smoke=args.smoke,
        stop_after_steps=args.stop_after_steps,
    )


if __name__ == "__main__":
    main()
