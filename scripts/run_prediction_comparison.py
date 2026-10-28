"""Matched native-FeH head/loss pilots with an exact reconstruction-row budget.

This is a separate continuation; the parent/main run and benchmark are read-only.
Labels describe reported native spectroscopic FeH, not calibrated latent truth.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import run_pretrain_experiments as pretrain
import run_xp_representation_probes as probe
import torch
import yaml
from prediction_continuation import load_segment, save_epoch
from run_enriched_value import COHORT_SHA, prepare
from run_native_age import SCALER_SHA
from run_stellar_label_experiments import BENCH
from stellar_catalog_jobs import atomic_json, identity_split
from targeted_policy import dropout_offset, experiment_seed, resolve_policy

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.conditional_flow_head import (
    ConditionalFlowHead,
)
from masked_stellar_autoencoder.training.experiment_augmentation import (
    canonical_xp_columns,
    supervised_augmentation,
)
from masked_stellar_autoencoder.training.imbalance_screen import (
    imbalance_loss,
    imbalance_objective,
)
from masked_stellar_autoencoder.training.input_measurement import (
    audit_xp_error_sources,
    mean_prediction_by_lambda,
    prepare_replicas,
    rational_extrapolate,
    standardized_xp_errors,
    xp_indices,
)
from masked_stellar_autoencoder.training.large_reconstruction import (
    prepare_original_reconstruction,
)
from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_contrastive_flow,
    build_prediction_head,
    contrastive_grid,
    fit_lds_huber,
    point_prediction,
    prediction_loss,
    rank_contrast_loss,
)
from masked_stellar_autoencoder.training.prediction_mechanisms import (
    NeuralRatioEstimator,
    corrected_score_objective,
    group_dro_objective,
    ratio_grid,
)
from masked_stellar_autoencoder.training.reserved_tuning import (
    evaluate_reserved_predictions,
    load_reserved_tuning,
)
from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def replica_directory(root, replica, index, count):
    """Every SIMEX model is independently reproducible; ordinary arms stay flat."""
    if count == 1:
        return root
    path = (
        root
        / f"replica-{index:03d}-lambda-{replica['lambda_value']:g}-sim-{replica['simulation']}"
    )
    path.mkdir(exist_ok=False)
    for name in (
        "initial_reconstruction.npz",
        "dropout_policy.json",
        "optimizer_policy.json",
    ):
        source = root / name
        if source.exists():
            shutil.copy2(source, path / name)
    return path


def replay_batches(cache, budget, batch_size):
    """Sequential train-only passes, with an exact truncated final batch."""
    if min(len(cache), budget, batch_size) < 1:
        raise ValueError("Replay lengths and budget must be positive")
    seen = 0
    while seen < budget:
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache), start + budget - seen)
            if stop <= start:
                return
            yield np.arange(start, stop), np.array(cache[start:stop])
            seen += stop - start
            if seen == budget:
                return


def native_inputs(cohort, limit=None):
    data = prepare(cohort)
    fit = data["native_fit"]
    # The fit cohorts and global source-ID roles are audited again by prepare().
    test = data["tests"]["native"]
    if limit is None and (len(fit) != 19859 or len(test["used_row"]) != 100620):
        raise ValueError("Declared native cohort counts changed")
    if limit is not None:
        fit = fit[:limit]
        test = {
            "used_row": test["used_row"][:limit],
            "measurements": test["measurements"][:limit],
        }
    target = data["fit_truth"][fit]
    yc, ys = float(target.mean()), float(target.std())
    if not np.isfinite(ys) or ys <= 0:
        raise ValueError("Native fit target scale must be positive")
    values = (data["raw_values"] - data["center"]) / data["scale"]
    safe, visible = spectrophotometric_inputs(
        torch.from_numpy(values.astype("f4")),
        torch.from_numpy(np.isfinite(data["raw_values"])),
        data["feature_columns"],
    )
    x = safe.numpy()
    assert not np.isin(data["source_id"][fit], test["measurements"]["source_id"]).any()
    assert (identity_split(data["source_id"][fit]) == 0).all()
    with h5py.File(cohort) as source:
        raw_native = source["raw_layers/native/measurements"][:]
    order = np.argsort(raw_native["source_id"])
    matched = raw_native[
        order[np.searchsorted(raw_native["source_id"][order], data["source_id"][fit])]
    ]
    np.testing.assert_array_equal(matched["source_id"], data["source_id"][fit])
    np.testing.assert_array_equal(matched["value"], data["fit_truth"][fit])
    data["fit_native_measurements"] = matched
    return data, fit, test, x, visible.numpy(), yc, ys


@torch.no_grad()
def predict(model, head, values, device, batch_size=512):
    model.eval()
    head.eval()
    if isinstance(head, ConditionalFlowHead):
        batch_size = min(batch_size, 128)
    return np.concatenate(
        [
            head(model.encoder(torch.as_tensor(block, device=device))).cpu().numpy()
            for block in np.array_split(
                values, max(1, int(np.ceil(len(values) / batch_size)))
            )
        ]
    )


@torch.no_grad()
def reconstruct(model, values, mask, device, micro=512):
    model.eval()
    predictions = []
    for start in range(0, len(values), micro):
        raw = torch.as_tensor(np.array(values[start : start + micro]), device=device)
        hidden = raw.masked_fill(
            torch.as_tensor(mask[start : start + micro], device=device)
            | ~torch.isfinite(raw),
            -9999,
        )
        predictions.append(model(hidden)[0].cpu().numpy())
    return np.concatenate(predictions)


def probabilistic_metrics(outputs, truth, center, scale, label_sigma_dex=None):
    """Observed-label Gaussian diagnostics, never inferred for Huber heads."""
    out = torch.from_numpy(outputs).double()
    y = torch.from_numpy(np.asarray((truth - center) / scale, dtype="f8"))
    mean = out[:, 0, 0].numpy() * scale + center
    sigma = (out[:, 0, 1] * 0.5).exp().numpy() * scale
    metrics = dict(
        gaussian_crps_dex=float(prediction_loss(out, y, "gaussian_crps")) * scale,
        gaussian_nll_dex=float(prediction_loss(out, y, "gaussian_nll"))
        + float(np.log(scale)),
        central_68_27_coverage=float(np.mean(np.abs(truth - mean) <= sigma)),
        central_95_coverage=float(
            np.mean(np.abs(truth - mean) <= 1.959963984540054 * sigma)
        ),
        mean_sigma_dex=float(sigma.mean()),
        target="reported_native_FeH_not_error_deconvolved_truth",
        quoted_label_error_included=False,
    )
    if label_sigma_dex is None:
        return metrics
    quoted = np.asarray(label_sigma_dex, dtype="f8")
    if (
        quoted.shape != truth.shape
        or not np.isfinite(quoted).all()
        or np.any(quoted <= 0)
    ):
        raise ValueError("Test quoted FeH errors must be finite and positive")
    total = np.sqrt(sigma**2 + quoted**2)
    standardized = torch.as_tensor((quoted / scale) ** 2, dtype=torch.float64)
    metrics.update(
        quoted_label_error_included=True,
        quoted_error_is_not_total_calibrated_sigma=True,
        convolved_gaussian_nll_dex=float(
            prediction_loss(
                out, y, "gaussian_nll_label_error", label_variance=standardized
            )
        )
        + float(np.log(scale)),
        convolved_central_68_27_coverage=float(np.mean(np.abs(truth - mean) <= total)),
        convolved_central_95_coverage=float(
            np.mean(np.abs(truth - mean) <= 1.959963984540054 * total)
        ),
        mean_convolved_sigma_dex=float(total.mean()),
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/prediction-comparison-10m.yaml")
    parser.add_argument("--checkpoint", default=os.getenv("MSA_PRETRAIN_RESUME"))
    parser.add_argument("--benchmark", default=BENCH)
    parser.add_argument(
        "--cohort",
        default="/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/enriched_feature_cohort_v1/enriched-feature-cohort.h5",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Explicit local numerical integration, never a production result",
    )
    args = parser.parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR",
        "/tmp/msa-prediction-matplotlib"
        if args.smoke
        else str(args.output / ".mplcache"),
    )
    suite = yaml.safe_load(Path(args.config).read_text())
    arm = next(a for a in suite["arms"] if a["run_id"] == args.run_id)
    suite = resolve_policy(suite, arm)
    seed = experiment_seed(suite, arm)
    seed_offset = dropout_offset(int(arm.get("experiment_seed", 42)))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        smoke=args.smoke,
        policy=suite,
        arm=arm,
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        caveat="Native reported-label prediction; Gaussian scores describe observed labels, not deconvolved/calibrated stellar truth. Pretraining overlap not excluded. BatchEnsemble head is TabM-style, not official TabM.",
    )
    atomic_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        if arm.get("library_revision"):
            direct = json.loads(
                importlib.metadata.distribution("torchregress").read_text(
                    "direct_url.json"
                )
            )
            if direct.get("vcs_info", {}).get("commit_id") != arm["library_revision"]:
                raise ValueError(
                    "Installed torchregress differs from the arm's immutable library pin"
                )
            manifest["torchregress_provenance"] = direct
        checkpoint_sha = digest(args.checkpoint)
        if not args.smoke and checkpoint_sha != suite["checkpoint_sha256"]:
            raise ValueError("Pinned parent digest differs")
        if digest(args.cohort) != COHORT_SHA:
            raise ValueError("Audited cohort digest differs")
        checkpoint = torch_load_trusted(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        config["training"]["micro_batch_size"] = (
            32 if args.smoke else suite["micro_batch"]
        )
        data, fit, test, safe, visible, yc, ys = native_inputs(
            args.cohort,
            128 if args.smoke and not suite.get("continuation_checkpoint") else None,
        )
        if args.smoke and suite.get("continuation_checkpoint"):
            fit = fit[:128]
            data["fit_native_measurements"] = data["fit_native_measurements"][:128]
            test = {
                "used_row": test["used_row"][:128],
                "measurements": test["measurements"][:128],
            }
        features = data["feature_columns"]
        augmentation = arm.get("supervised_augmentation")
        if augmentation is not None:
            if set(augmentation) - {
                "noise_multiplier",
                "xp_mask_probability",
                "mask_mode",
            }:
                raise ValueError("Unknown supervised augmentation policy")
            if "input_correction" in arm or arm["loss"] == "corrected_score":
                raise ValueError(
                    "Augmentation cannot silently combine with EIV correction"
                )
        augmentation_noise = float((augmentation or {}).get("noise_multiplier", 0.0))
        augmentation_mask = float((augmentation or {}).get("xp_mask_probability", 0.0))
        if not np.isfinite(augmentation_noise) or augmentation_noise < 0:
            raise ValueError("Augmentation noise must be finite and nonnegative")
        if not np.isfinite(augmentation_mask) or not 0 <= augmentation_mask <= 1:
            raise ValueError("Augmentation mask probability must be in [0, 1]")
        if (
            "input_correction" in arm
            or arm["loss"] == "corrected_score"
            or augmentation_noise > 0
        ):
            if args.smoke:
                error_audit = dict(
                    status="smoke_only", original_error_source_verified=False
                )
            else:
                with h5py.File(args.cohort) as actual_cohort:
                    legacy_path = str(actual_cohort.attrs["legacy_input"])
                error_audit = audit_xp_error_sources(
                    args.cohort,
                    args.benchmark,
                    legacy_path,
                    data["feature_row"],
                    features,
                )
                if error_audit["error_convention"] != arm.get(
                    "input_correction", {}
                ).get("error_convention", arm.get("error_convention", "raw_xp")):
                    raise ValueError(
                        "Declared XP error convention differs from actual sources"
                    )
            atomic_json(args.output / "xp-error-source-audit.json", error_audit)
            manifest["xp_error_source_audit"] = error_audit
        with h5py.File(args.benchmark) as source:
            keys, validation_keys, center, scale = pretrain.validate_source(
                checkpoint, config, source
            )
            if not args.smoke:
                provenance = json.loads(source.attrs["provenance"])
                if (
                    provenance["source_id_sha256"]
                    != suite["benchmark_ordered_id_sha256"]
                ):
                    raise ValueError("Immutable benchmark ID provenance differs")
                if len(source["train/records"]) != 9_000_000:
                    raise ValueError("Expected original 9M train split")
            np.testing.assert_array_equal(center, data["center"])
            np.testing.assert_array_equal(scale, data["scale"])
            if (
                hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest()
                != SCALER_SHA
            ):
                raise ValueError("Parent scaling differs")
            # Local surrogate input tests explicitly do not certify original units.
            units = {"local_smoke_only": True, "fresh_original_128_check": False}
            if not args.smoke:
                with h5py.File(args.cohort) as cohort:
                    units = probe.audit_fits_xp_units(
                        args.benchmark,
                        features,
                        cohort["features/values"][:],
                        cohort["features/source_id"][:],
                        "g_normalized",
                    )
            if (
                suite.get("reconstruction_source") == "original_sample"
                and not args.smoke
            ):
                manifest["status"] = "preparing"
                atomic_json(args.output / "manifest.json", manifest)
                cache, counts, sample = prepare_original_reconstruction(
                    suite, checkpoint, features, center, scale
                )
                atomic_json(args.output / "reconstruction-sample.json", sample)
                manifest["reconstruction_sample"] = sample
                manifest["status"] = "running"
                atomic_json(args.output / "manifest.json", manifest)
            else:
                cache, _, counts = pretrain.read_prefixes(
                    source,
                    keys,
                    features,
                    None,
                    512 if args.smoke else 9_000_000,
                    center,
                    scale,
                    args.output / "replay-cache.npy",
                )
            validation, _, _ = pretrain.read_prefixes(
                source,
                validation_keys,
                features,
                None,
                128 if args.smoke else 10_000,
                center,
                scale,
                args.output / "validation-cache.npy",
            )
            validation_mask = pretrain.fixed_masks(validation, features, 42, 0.5, 0.2)[
                "common"
            ]
            wrapper, optimizer, _ = pretrain.build_wrapper(
                config,
                checkpoint,
                source,
                center,
                scale,
                {"lr_factor": 1.0},
                args.output,
            )
            model, device = wrapper.model, wrapper.device
            if suite.get("require_cuda") and not args.smoke and device.type != "cuda":
                raise RuntimeError("Large continuation requires an actual CUDA device")
            if arm.get("freeze_encoder", False):
                model.encoder.requires_grad_(False)
            continuation = None
            if suite.get("continuation_checkpoint"):
                continuation = load_segment(
                    suite["continuation_checkpoint"],
                    suite["continuation_sha256"],
                    model,
                    optimizer,
                    features=features,
                    center=yc,
                    scale=ys,
                )
                manifest["continuation_sha256"] = suite["continuation_sha256"]
                manifest["continuation_policy"] = (
                    "Model/head/both Adam states retained; no head warmup. New declared streams if predecessor RNG absent; not an interrupted exact-RNG resume."
                )
                atomic_json(args.output / "manifest.json", manifest)
            width = len(config["data"]["recon_cols"])
            if config["data"]["recon_cols"] != features[:width]:
                raise ValueError(
                    "Reconstruction order is not the declared feature prefix"
                )
            projection = pretrain.spectral_basis(features[:width])
            initial_reconstruction = reconstruct(
                model, validation, validation_mask, device
            )
            np.savez_compressed(
                args.output / "initial_reconstruction.npz",
                target=validation,
                mask=validation_mask,
                prediction=initial_reconstruction,
                scaler_center=center,
                scaler_scale=scale,
                spectral_basis=projection["basis"],
                wavelength_nm=projection["wavelength_nm"],
            )
            # Retain original Adam moments; explicitly replace scheduler with fixed LR.
            replicas = [dict(lambda_value=0.0, simulation=0, matrix=safe)]
            if "input_correction" in arm:
                with h5py.File(args.cohort) as cohort_file:
                    raw_error = cohort_file["features/reported_feature_errors"][:][
                        data["feature_row"]
                    ]
                replicas, correction_summary = prepare_replicas(
                    arm["input_correction"],
                    safe,
                    raw_error,
                    data["scale"],
                    features,
                    fit,
                    g_magnitude=data["raw_values"][:, features.index("G")],
                    error_convention=arm["input_correction"]["error_convention"],
                )
                atomic_json(args.output / "input-correction.json", correction_summary)
            dex_predictions = []
            replica_records = []
            for replica_number, replica in enumerate(replicas):
                replica_output = replica_directory(
                    args.output, replica, replica_number, len(replicas)
                )
                encoder_matrix = replica["matrix"]
                if replicas[0] is not replica:
                    wrapper, optimizer, _ = pretrain.build_wrapper(
                        config,
                        checkpoint,
                        source,
                        center,
                        scale,
                        {"lr_factor": 1.0},
                        replica_output,
                    )
                    model, device = wrapper.model, wrapper.device
                    for group in optimizer.param_groups:
                        group["lr"] = suite["encoder_lr"]
                for group in optimizer.param_groups:
                    group["lr"] = suite["encoder_lr"]
                torch.manual_seed(seed)
                contrastive = arm["loss"] in {"contrastive_flow", "neural_ratio"}
                conditional = arm["loss"] == "conditional_flow_nll"
                flow_options = None
                if conditional:
                    from torchregress.losses.nflows import recommended_tail_bound

                    # Full native fit labels only, before constructing the optimizer.
                    fit_standardized = (data["fit_truth"][fit] - yc) / ys
                    bound = recommended_tail_bound(fit_standardized)
                    flow_options = dict(arm["conditional_flow"], bound=bound)
                    atomic_json(
                        replica_output / "conditional-flow-policy.json",
                        dict(
                            **flow_options,
                            fit_rows=len(fit),
                            fit_max_abs=float(np.abs(fit_standardized).max()),
                            fit_outside_bound=int(
                                (np.abs(fit_standardized) > bound).sum()
                            ),
                            density="reported standardized native FeH conditional on encoder latent",
                            initialization="standard normal; mean and logvariance within 2e-6",
                            moments="64-panel eight-point composite Legendre; analytic identity tails",
                            refinement="128-panel mean saved for every test row; not Gaussian scores",
                        ),
                    )
                if contrastive and arm["head"] not in {
                    "contrastive_flow",
                    "neural_ratio",
                }:
                    raise ValueError(
                        "contrastive flow uses its own estimator, not a point head"
                    )
                head = (
                    ConditionalFlowHead(**flow_options).to(device)
                    if conditional
                    else NeuralRatioEstimator().to(device)
                    if arm["head"] == "neural_ratio"
                    else build_contrastive_flow(arm["contrastive_flow"]).to(device)
                    if contrastive
                    else build_prediction_head(arm["head"], input_dim=256).to(device)
                )
                head_opt = torch.optim.AdamW(
                    head.parameters(), lr=suite["head_lr"], weight_decay=1e-4
                )
                if continuation is not None:
                    head.load_state_dict(continuation["head_state_dict"], strict=True)
                    head_opt.load_state_dict(continuation["head_optimizer_state_dict"])
                labels = torch.as_tensor(
                    (data["fit_truth"][fit] - yc) / ys,
                    dtype=torch.float32,
                    device=device,
                )
                imbalance = imbalance_objective(arm["loss"], arm.get("imbalance", {}))
                if imbalance is not None:
                    imbalance = imbalance.to(device)
                    atomic_json(replica_output / "imbalance-policy.json", arm)
                lds = None
                label_variance = None
                if arm["loss"] == "lds_huber":
                    lds = fit_lds_huber(labels, arm["lds"])
                    weights = lds._weights_per_bin.detach().cpu().numpy()
                    atomic_json(
                        replica_output / "lds-weights.json",
                        dict(
                            kernel_width_bins=2,
                            n_bins=100,
                            min_weight=float(weights.min()),
                            max_weight=float(weights.max()),
                            weights=weights.tolist(),
                            note="torchregress LDS kernel width is histogram-bin sigma, not dex",
                        ),
                    )
                elif arm["loss"] == "gaussian_nll_label_error":
                    quoted = np.asarray(
                        data["fit_native_measurements"]["uncertainty"], dtype="f8"
                    )
                    if (
                        quoted.shape != (len(fit),)
                        or not np.isfinite(quoted).all()
                        or np.any(quoted <= 0)
                    ):
                        raise ValueError(
                            "Native fit FeH uncertainties must be finite and positive"
                        )
                    label_variance = torch.as_tensor(
                        (quoted / ys) ** 2, dtype=torch.float32, device=device
                    )
                    atomic_json(
                        replica_output / "label-error-summary.json",
                        dict(
                            fit_rows=int(len(quoted)),
                            min_dex=float(quoted.min()),
                            median_dex=float(np.median(quoted)),
                            max_dex=float(quoted.max()),
                            quoted_error_is_not_total_calibrated_sigma=True,
                        ),
                    )
                grid = (
                    (
                        ratio_grid(labels)
                        if arm["head"] == "neural_ratio"
                        else contrastive_grid(labels, arm["contrastive_flow"])
                    )
                    if contrastive
                    else None
                )
                if grid is not None:
                    atomic_json(
                        replica_output / "contrastive-flow-grid.json",
                        dict(
                            points=int(grid.numel()),
                            min_standardized=float(grid[0]),
                            max_standardized=float(grid[-1]),
                            estimand="grid_argmax_neural_log_likelihood_ratio"
                            if arm["head"] == "neural_ratio"
                            else "grid_mle_of_p_summary_latent_given_FeH",
                            negatives="fixed_batch_rolls",
                            not_a_training_prior_posterior=True,
                        ),
                    )
                train = torch.as_tensor(encoder_matrix[fit], device=device)
                test_x = encoder_matrix[test["used_row"]]
                augmentation_rng = torch.Generator(device=device).manual_seed(
                    seed + 1701
                )
                augmentation_digest = hashlib.sha256()
                augmentation_counts = dict(batches=0, observed_xp=0, hidden_xp=0)
                augmentation_columns = (
                    canonical_xp_columns(features) if augmentation else ()
                )
                train_visible = torch.as_tensor(visible[fit], device=device)
                augmentation_sigma = None
                if augmentation_noise > 0:
                    with h5py.File(args.cohort) as raw_cohort:
                        augmentation_errors = raw_cohort[
                            "features/reported_feature_errors"
                        ][:][data["feature_row"]]
                    augmentation_sigma = torch.as_tensor(
                        standardized_xp_errors(
                            augmentation_errors,
                            data["scale"],
                            np.asarray(augmentation_columns),
                            g_magnitude=data["raw_values"][:, features.index("G")],
                            error_convention="raw_xp",
                        )[fit],
                        dtype=train.dtype,
                        device=device,
                    )
                if (
                    continuation is not None
                    and "augmentation_rng_state" in continuation
                ):
                    augmentation_rng.set_state(
                        continuation["augmentation_rng_state"].cpu()
                    )

                def supervised_view(rows):
                    if augmentation is None:
                        return train[rows]
                    view, view_visible, audit = supervised_augmentation(
                        train[rows],
                        train_visible[rows],
                        augmentation_columns,
                        generator=augmentation_rng,
                        sigma_xp=None
                        if augmentation_sigma is None
                        else augmentation_sigma[rows],
                        noise_scale=augmentation_noise,
                        mask_probability=augmentation_mask,
                        mask_mode=augmentation.get("mask_mode", "coefficients"),
                    )
                    augmentation_digest.update(view.detach().cpu().numpy().tobytes())
                    augmentation_digest.update(view_visible.cpu().numpy().tobytes())
                    augmentation_counts["batches"] += 1
                    augmentation_counts["observed_xp"] += audit["observed_xp"]
                    augmentation_counts["hidden_xp"] += audit["hidden_xp"]
                    return view

                corrected = None
                sigma_fit = None
                dro = None
                if arm["loss"] == "corrected_score":
                    with h5py.File(args.cohort) as raw_cohort:
                        errors = raw_cohort["features/reported_feature_errors"][:][
                            data["feature_row"]
                        ]
                    columns = xp_indices(features)
                    sigma = standardized_xp_errors(
                        errors,
                        data["scale"],
                        columns,
                        g_magnitude=data["raw_values"][:, features.index("G")],
                        error_convention=arm["error_convention"],
                    )
                    sigma_fit = torch.zeros(
                        (len(fit), len(features)), device=device, dtype=torch.float32
                    )
                    sigma_fit[:, torch.as_tensor(columns, device=device)] = (
                        torch.as_tensor(sigma[fit], device=device, dtype=torch.float32)
                    )
                    corrected = corrected_score_objective(model, head)
                    atomic_json(
                        replica_output / "mechanism.json",
                        dict(
                            method="corrected_score",
                            raw_XP_error_audit=error_audit,
                            standardized_fit_XP_sigma_min=float(sigma[fit].min()),
                            standardized_fit_XP_sigma_max=float(sigma[fit].max()),
                            non_XP_sigma_assumed_zero=True,
                            shared_G_uncertainty_omitted=True,
                            off_diagonal_XP_covariance_omitted=True,
                            nonlinear_correction_is_first_order=True,
                            negative_loss_allowed=True,
                            clipping_or_physical_floor=False,
                        ),
                    )
                if arm["loss"] == "group_dro_huber":
                    dro = group_dro_objective(labels, yc, ys)
                    atomic_json(
                        replica_output / "mechanism.json",
                        dict(
                            method="group_dro_huber",
                            eta=0.01,
                            min_group_weight=0.01,
                            standardized_group_edges=dro.group_edges.detach()
                            .cpu()
                            .tolist()
                            if hasattr(dro, "group_edges")
                            else [],
                            label_bins_dex=[-2, -1, 0],
                            fit_counts=[
                                int((labels < (-2 - yc) / ys).sum()),
                                int(
                                    (
                                        (labels >= (-2 - yc) / ys)
                                        & (labels < (-1 - yc) / ys)
                                    ).sum()
                                ),
                                int(
                                    (
                                        (labels >= (-1 - yc) / ys)
                                        & (labels < (0 - yc) / ys)
                                    ).sum()
                                ),
                                int((labels >= (0 - yc) / ys).sum()),
                            ],
                        ),
                    )

                def batch_objective(latent, rows):
                    if imbalance is not None:
                        return imbalance_loss(
                            imbalance, arm["loss"], head(latent), latent, labels[rows]
                        )
                    if corrected is not None:
                        model.eval()
                        head.eval()
                        return corrected(
                            train[rows],
                            labels[rows],
                            sigma_x=torch.diag_embed(sigma_fit[rows].square()),
                        )
                    if dro is not None:
                        return dro(head(latent)[:, 0, 0], labels[rows])
                    if contrastive or conditional:
                        return head.criterion(latent, labels[rows])
                    return prediction_loss(
                        head(latent),
                        labels[rows],
                        arm["loss"],
                        lds=lds,
                        label_variance=None
                        if label_variance is None
                        else label_variance[rows],
                    )

                def current_outputs(active_model, active_head):
                    active_model.eval()
                    active_head.eval()
                    if not contrastive:
                        return predict(active_model, active_head, test_x, device)
                    chunks = []
                    with torch.no_grad():
                        for block in np.array_split(
                            test_x, max(1, int(np.ceil(len(test_x) / 512)))
                        ):
                            encoded = active_model.encoder(
                                torch.as_tensor(np.array(block), device=device)
                            )
                            chunks.append(
                                active_head.standardized_mle(encoded, grid)
                                .cpu()
                                .numpy()
                            )
                    return np.concatenate(chunks)

                def save_outputs(path, array):
                    key = "standardized_mle" if contrastive else "member_outputs"
                    np.savez_compressed(path, **{key: array})

                np.savez_compressed(
                    replica_output / "paired-cohort.npz",
                    fit_source_id=data["source_id"][fit],
                    fit_input=encoder_matrix[fit],
                    fit_target=data["fit_truth"][fit],
                    fit_measurements=data["fit_native_measurements"],
                    test_input=test_x,
                    test_measurements=test["measurements"],
                    label_center=yc,
                    label_scale=ys,
                )
                initial = current_outputs(model, head)
                if conditional:
                    np.testing.assert_allclose(initial, 0, rtol=0, atol=2e-6)
                if continuation is not None:
                    previous_predictions = np.load(
                        Path(suite["continuation_checkpoint"]).parent
                        / "predictions.npz",
                        allow_pickle=False,
                    )["member_outputs"]
                    if args.smoke:
                        previous_predictions = previous_predictions[: len(initial)]
                    np.testing.assert_allclose(
                        initial, previous_predictions, rtol=5e-5, atol=5e-5
                    )
                    manifest["continuation_initial_max_difference"] = float(
                        np.max(np.abs(initial - previous_predictions))
                    )
                    atomic_json(args.output / "manifest.json", manifest)
                save_outputs(replica_output / "initial_predictions.npz", initial)
                rng = np.random.default_rng(seed)
                if continuation is not None and "label_rng_state" in continuation:
                    rng.bit_generator.state = continuation["label_rng_state"]
                supervised_digest, row_digest, mask_digest = (
                    hashlib.sha256() for _ in range(3)
                )
                warmup = (
                    0
                    if continuation is not None
                    else 2
                    if args.smoke
                    else suite["warmup_updates"]
                )
                for _ in range(warmup):
                    rows = rng.integers(len(fit), size=suite["contrast_batch"])
                    supervised_digest.update(rows.tobytes())
                    model.eval()
                    head.train()
                    with torch.no_grad():
                        z = model.encoder(supervised_view(rows))
                    head_opt.zero_grad(set_to_none=True)
                    model.zero_grad(set_to_none=True)
                    loss = batch_objective(z, rows)
                    loss.backward()
                    head_opt.step()
                budget = 512 if args.smoke else suite["reconstruction_presentations"]
                effective = 256 if args.smoke else suite["effective_batch"]
                mask_rng = torch.Generator(device=device).manual_seed(seed)
                if continuation is not None and "mask_rng_state" in continuation:
                    mask_rng.set_state(continuation["mask_rng_state"].cpu())
                update_offset = (
                    int(
                        continuation.get(
                            "completed_updates",
                            suite.get("continuation_previous_updates", 0),
                        )
                    )
                    if continuation is not None
                    else 0
                )
                xp = [i for i, n in enumerate(features) if n.startswith(("bp_", "rp_"))]
                micro = config["training"]["micro_batch_size"]
                seen, steps = 0, 0
                interval = (
                    256
                    if args.smoke
                    else suite.get("checkpoint_interval_presentations", 0)
                )
                snapshots = (
                    list(range(interval, budget + 1, interval))
                    if interval
                    else [0, 3_000_000, 6_000_000, 9_000_000, budget]
                )
                if not args.smoke:
                    snapshots = sorted(
                        set(snapshots)
                        | set(suite.get("early_checkpoint_presentations", []))
                    )
                after_warmup = current_outputs(model, head)
                save_outputs(replica_output / "snapshot-0.npz", after_warmup)
                for indices, raw in replay_batches(cache, budget, effective):
                    row_digest.update(indices.tobytes())
                    target = torch.as_tensor(raw, device=device)
                    hidden, mask, finite = pretrain.training_mask(
                        target, xp, 0.5, 0.2, mask_rng
                    )
                    score = (mask & finite)[:, :width]
                    mask_digest.update(mask.cpu().numpy().tobytes())
                    optimizer.zero_grad(set_to_none=True)
                    head_opt.zero_grad(set_to_none=True)
                    model.train()
                    head.train()
                    torch.manual_seed(1000 + seed_offset + update_offset + steps)
                    reconstruction_total = 0.0
                    for start in range(0, len(raw), micro):
                        stop = start + micro
                        predicted, _ = model(hidden[start:stop])
                        replay = pretrain.experiment_objective(
                            predicted,
                            target[start:stop, :width],
                            score[start:stop],
                            xp,
                            {"loss": "mae"},
                            score.sum(),
                        )
                        (suite["reconstruction_weight"] * replay).backward()
                        reconstruction_total += float(replay.detach())
                    groups = 1 if args.smoke else suite["supervised_batches_per_update"]
                    supervised_total = 0.0
                    for group in range(groups):
                        rows = rng.integers(len(fit), size=suite["contrast_batch"])
                        supervised_digest.update(rows.tobytes())
                        torch.manual_seed(
                            100_000
                            + seed_offset
                            + (update_offset + steps) * groups
                            + group
                        )
                        z = model.encoder(supervised_view(rows))
                        loss = batch_objective(z, rows)
                        if arm.get("rnc_weight", 0):
                            if contrastive:
                                raise ValueError(
                                    "contrastive flow does not combine with RnC"
                                )
                            loss = loss + arm["rnc_weight"] * rank_contrast_loss(
                                z, labels[rows]
                            )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite supervised objective")
                        (loss / groups).backward()
                        supervised_total += float(loss.detach()) / groups
                    model_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    head_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        head.parameters(), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    head_opt.step()
                    seen += len(raw)
                    steps += 1
                    metric_record = dict(
                        step=steps,
                        replica_index=replica_number,
                        lambda_value=replica["lambda_value"],
                        simulation=replica["simulation"],
                        presentations=seen,
                        total_presentations=replica_number * budget + seen,
                        supervised_loss=supervised_total,
                        reconstruction_mae=reconstruction_total,
                        model_gradient_norm_before_clip=float(model_gradient_norm),
                        head_gradient_norm_before_clip=float(head_gradient_norm),
                        elapsed_s=time.monotonic() - started,
                    )
                    log_paths = [replica_output / "metrics.jsonl"]
                    if len(replicas) > 1:
                        log_paths.append(args.output / "metrics.jsonl")
                    for log_path in log_paths:
                        with log_path.open("a") as handle:
                            handle.write(json.dumps(metric_record) + "\n")
                    if any(
                        seen - len(raw) < threshold <= seen
                        for threshold in snapshots
                        if threshold
                    ):
                        outputs = current_outputs(model, head)
                        save_outputs(replica_output / f"snapshot-{seen}.npz", outputs)
                        if suite.get("checkpoint_interval_presentations"):
                            receipt = save_epoch(
                                replica_output
                                / "checkpoints"
                                / f"checkpoint-{seen:012d}.pth",
                                model,
                                head,
                                optimizer,
                                head_opt,
                                arm=arm,
                                feature_columns=features,
                                label_center=yc,
                                label_scale=ys,
                                parent_sha256=checkpoint_sha,
                                continuation_sha256=suite.get("continuation_sha256"),
                                presentations=seen,
                                completed_updates=update_offset + steps,
                                label_rng_state=rng.bit_generator.state,
                                mask_rng_state=mask_rng.get_state(),
                                augmentation_rng_state=augmentation_rng.get_state(),
                                source_sha=manifest["source_sha"],
                            )
                            atomic_json(
                                replica_output / "latest-checkpoint.json", receipt
                            )

                if arm.get("experiment_seed") is not None:
                    tuning = load_reserved_tuning(
                        args.cohort,
                        identity_split=identity_split,
                        size=64 if args.smoke else 4096,
                    )
                    if set(tuning["source_id"]) & set(data["source_id"][fit]):
                        raise ValueError("Reserved tuning overlaps fitting")
                    if set(tuning["source_id"]) & set(
                        test["measurements"]["source_id"]
                    ):
                        raise ValueError("Reserved tuning overlaps identity test")
                    tune_output = predict(
                        model,
                        head,
                        tuning["safe"],
                        device,
                        batch_size=32 if conditional else 512,
                    )
                    tune_point = (
                        point_prediction(torch.from_numpy(tune_output)).numpy() * ys
                        + yc
                    )
                    np.savez_compressed(
                        replica_output / "reserved-tuning.npz",
                        source_id=tuning["source_id"],
                        measurements=tuning["measurements"],
                        feature_row=tuning["feature_row"],
                        raw_values=tuning["raw_values"],
                        safe=tuning["safe"],
                        member_outputs=tune_output,
                        prediction=tune_point,
                        label_center=yc,
                        label_scale=ys,
                    )
                    atomic_json(
                        replica_output / "reserved-tuning.json",
                        dict(
                            **tuning["audit"],
                            metrics=evaluate_reserved_predictions(
                                tuning, tune_point, tuning["source_id"]
                            ),
                            used_for_current_training_or_stopping=False,
                        ),
                    )

                outputs = current_outputs(model, head)
                if conditional:
                    flow_diagnostics = {}
                    with torch.no_grad():
                        for block in np.array_split(
                            test_x, max(1, int(np.ceil(len(test_x) / 128)))
                        ):
                            start = sum(
                                len(item) for item in flow_diagnostics.get("cdf", [])
                            )
                            latent = model.encoder(
                                torch.as_tensor(np.array(block), device=device)
                            )
                            truth_block = torch.as_tensor(
                                (
                                    test["measurements"]["value"][
                                        start : start + len(block)
                                    ]
                                    - yc
                                )
                                / ys,
                                dtype=latent.dtype,
                                device=device,
                            )
                            for key, value in head.diagnostics(
                                latent, truth_block
                            ).items():
                                flow_diagnostics.setdefault(key, []).append(
                                    value.cpu().numpy()
                                )
                    np.savez_compressed(
                        replica_output / "conditional-flow-diagnostics.npz",
                        **{
                            key: np.concatenate(value)
                            for key, value in flow_diagnostics.items()
                        },
                    )
                state = dict(
                    model_state_dict=model.state_dict(),
                    head_state_dict=head.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    head_optimizer_state_dict=head_opt.state_dict(),
                    parent_sha256=checkpoint_sha,
                    arm=arm,
                    label_center=yc,
                    label_scale=ys,
                    presentations=seen,
                    feature_columns=features,
                    source_sha=manifest["source_sha"],
                    replica_index=replica_number,
                    lambda_value=float(replica["lambda_value"]),
                    simulation=int(replica["simulation"]),
                    completed_updates=update_offset + steps,
                    label_rng_state=rng.bit_generator.state,
                    mask_rng_state=mask_rng.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else [],
                )
                if continuation is not None:
                    state.update(
                        continuation_sha256=suite["continuation_sha256"],
                    )
                if augmentation is not None:
                    augmentation_audit = dict(
                        policy=augmentation,
                        seed=seed + 1701,
                        actual_device=str(device),
                        training_view_sha256=augmentation_digest.hexdigest(),
                        counts=augmentation_counts,
                        test_inputs_augmented=False,
                        reconstruction_inputs_augmented=False,
                        reconstruction_targets_augmented=False,
                        labels_augmented=False,
                        error_convention="raw_xp",
                        G_uncertainty_omitted=True,
                        off_diagonal_XP_covariance_omitted=True,
                        CPU_independent_random_regeneration_claimed=False,
                    )
                    atomic_json(
                        replica_output / "supervised-augmentation.json",
                        augmentation_audit,
                    )
                    manifest["supervised_augmentation"] = augmentation_audit
                    state["augmentation_rng_state"] = augmentation_rng.get_state()
                    state["supervised_augmentation"] = augmentation_audit
                if dro is not None:
                    state["objective_state_dict"] = dro.state_dict()
                if conditional:
                    state["conditional_flow_options"] = flow_options
                torch.save(state, replica_output / "checkpoint.pth")
                restored = torch_load_trusted(
                    replica_output / "checkpoint.pth",
                    map_location=device,
                    weights_only=False,
                )
                other_model, other_head = copy.deepcopy(model), copy.deepcopy(head)
                other_model.load_state_dict(restored["model_state_dict"], strict=True)
                other_head.load_state_dict(restored["head_state_dict"], strict=True)
                reloaded = current_outputs(other_model, other_head)
                np.testing.assert_array_equal(outputs, reloaded)
                final_reconstruction = reconstruct(
                    model, validation, validation_mask, device
                )
                np.testing.assert_array_equal(
                    final_reconstruction,
                    reconstruct(other_model, validation, validation_mask, device),
                )
                np.savez_compressed(
                    replica_output / "final_reconstruction.npz",
                    prediction=final_reconstruction,
                )
                reconstruction = {}
                validation_target = validation[:, :width]
                validation_score_mask = validation_mask[:, :width]
                scored = validation_score_mask & np.isfinite(validation_target)
                for name, prediction in (
                    ("initial", initial_reconstruction),
                    ("final", final_reconstruction),
                ):
                    residual = prediction - validation_target
                    reconstruction[name] = dict(
                        masked_mae=float(np.abs(residual[scored]).mean()),
                        spectral=pretrain.spectral_summary(
                            validation_target,
                            prediction,
                            validation_score_mask,
                            scale[:width],
                            center[:width],
                            projection,
                        ),
                    )
                if contrastive:
                    prediction = outputs * ys + yc
                    raw_key = "standardized_mle"
                else:
                    prediction = (
                        point_prediction(torch.from_numpy(outputs)).numpy() * ys + yc
                    )
                    raw_key = "member_outputs"
                np.savez_compressed(
                    replica_output / "predictions.npz",
                    prediction=prediction,
                    **{raw_key: outputs},
                )
                dex_predictions.append(prediction)
                reload_report = dict(
                    all_test_member_outputs_byte_exact=not contrastive,
                    all_test_standardized_mle_byte_exact=contrastive,
                    rows=len(outputs),
                    model_and_head_strict_reload=True,
                    all_validation_decoder_outputs_byte_exact=True,
                    same_device=True,
                    optimizer_state_saved_not_resumed_tested=True,
                )
                atomic_json(
                    replica_output / "checkpoint_verification.json", reload_report
                )
                record = dict(
                    index=replica_number,
                    directory=str(replica_output.relative_to(args.output)),
                    lambda_value=float(replica["lambda_value"]),
                    simulation=int(replica["simulation"]),
                    parent_sha256=checkpoint_sha,
                    source_sha=manifest["source_sha"],
                    presentations=seen,
                    optimizer_updates=steps,
                    row_sha256=row_digest.hexdigest(),
                    mask_sha256=mask_digest.hexdigest(),
                    supervised_stream_sha256=supervised_digest.hexdigest(),
                    head_parameters=sum(p.numel() for p in head.parameters()),
                    supervised_warmup_presentations=warmup * suite["contrast_batch"],
                    supervised_adaptation_presentations=steps
                    * groups
                    * suite["contrast_batch"],
                    reconstruction=reconstruction,
                    reload=reload_report,
                )
                if len(replicas) > 1:
                    record["perturbation_seed"] = int(
                        replica.get(
                            "perturbation_seed",
                            20_000
                            + correction_summary["lambdas"].index(
                                float(replica["lambda_value"])
                            )
                            * 100
                            + int(replica["simulation"]),
                        )
                    )
                    record["files"] = {
                        path.name: dict(bytes=path.stat().st_size, sha256=digest(path))
                        for path in sorted(replica_output.iterdir())
                        if path.is_file()
                    }
                    atomic_json(
                        replica_output / "replica-manifest.json",
                        {"status": "complete", **record},
                    )
                    record["manifest_sha256"] = digest(
                        replica_output / "replica-manifest.json"
                    )
                    if replica_records:
                        for key in (
                            "row_sha256",
                            "mask_sha256",
                            "supervised_stream_sha256",
                            "parent_sha256",
                            "source_sha",
                        ):
                            if record[key] != replica_records[0][key]:
                                raise ValueError(
                                    f"SIMEX replica stream/parent mismatch: {key}"
                                )
                replica_records.append(record)
                if len(replicas) > 1:
                    # Do not keep previous GPU models, Adam tensors or reload
                    # copies alive during the next replica's allocation/training.
                    del other_model, other_head, restored, state
                    del model, wrapper, optimizer, head, head_opt, train, labels
                    z = loss = predicted = replay = target = hidden = mask = finite = (
                        score
                    ) = None
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            # Replica loop ends after all actual checkpoints and outputs are stored.
            if len(replicas) > 1:
                lambdas, means = mean_prediction_by_lambda(replicas, dex_predictions)
                prediction = rational_extrapolate(lambdas, means)
                np.savez_compressed(
                    args.output / "predictions.npz",
                    prediction=prediction,
                    lambda_values=np.asarray(lambdas, dtype="f8"),
                    lambda_mean_prediction=means,
                    per_replica_prediction=np.stack(dex_predictions),
                    extrapolant="b+a/(c+lambda), c grid[1.05,8]",
                )
                np.savez_compressed(
                    args.output / "initial_predictions.npz",
                    prediction=np.full(len(prediction), yc),
                    stage="common_zero_initial_head_constant; all raw heads preserved per replica",
                )
                np.savez_compressed(
                    args.output / "paired-cohort.npz",
                    fit_source_id=data["source_id"][fit],
                    fit_input=safe[fit],
                    fit_target=data["fit_truth"][fit],
                    fit_measurements=data["fit_native_measurements"],
                    test_input=safe[test["used_row"]],
                    test_measurements=test["measurements"],
                    label_center=yc,
                    label_scale=ys,
                    input_note="unperturbed reference; actual corrected inputs are per replica",
                )
                atomic_json(
                    args.output / "replicas.json",
                    dict(
                        status="complete",
                        predictions_are_multi_model_extrapolation=True,
                        no_single_aggregate_checkpoint=True,
                        replicas=replica_records,
                    ),
                )
                reconstruction = dict(
                    available=False,
                    reason="No unique extrapolated decoder; each replica has its own reconstruction diagnostics",
                    per_replica=[
                        dict(index=r["index"], **r["reconstruction"])
                        for r in replica_records
                    ],
                )
            truth = test["measurements"]["value"]
            comparison = {
                name: probe.score(prediction[keep], truth[keep])
                for name, keep in {
                    "all": np.ones(len(truth), bool),
                    "below_minus2": truth < -2,
                    "below_minus3": truth < -3,
                }.items()
            }
            if conditional:
                density = np.concatenate(flow_diagnostics["observed_label_log_density"])
                quantiles = np.concatenate(flow_diagnostics["quantiles"]) * ys + yc
                refinement = np.concatenate(flow_diagnostics["moment_refinement_mean"])
                comparison["probabilistic"] = dict(
                    available=True,
                    family="conditional flow; moments are not Gaussian parameters",
                    observed_label_nll_dex=float(-density.mean() + np.log(ys)),
                    nominal95_coverage=float(
                        (
                            (truth >= quantiles[:, 0]) & (truth <= quantiles[:, -1])
                        ).mean()
                    ),
                    moment_refinement_max_abs_standardized=float(
                        np.max(np.abs(refinement - outputs[:, 0, 0]))
                    ),
                )
            elif arm["loss"] in {
                "gaussian_nll",
                "gaussian_crps",
                "gaussian_nll_label_error",
            }:
                quoted_test = (
                    test["measurements"]["uncertainty"]
                    if arm["loss"] == "gaussian_nll_label_error"
                    else None
                )
                comparison["probabilistic"] = probabilistic_metrics(
                    outputs, truth, yc, ys, quoted_test
                )
            else:
                comparison["probabilistic"] = {
                    "available": False,
                    "reason": (
                        "Grid MLE of p(summary(latent)|FeH); not a Gaussian on FeH"
                        if contrastive
                        else "Huber log variance is not trained"
                    ),
                }
            comparison["reconstruction"] = reconstruction
            atomic_json(args.output / "comparison.json", comparison)
            atomic_json(
                args.output / "checkpoint_verification.json",
                dict(
                    all_test_member_outputs_byte_exact=not contrastive
                    and len(replicas) == 1,
                    all_test_standardized_mle_byte_exact=contrastive,
                    rows=len(outputs),
                    model_and_head_strict_reload=True,
                    all_validation_decoder_outputs_byte_exact=True,
                    same_device=True,
                    optimizer_state_saved_not_resumed_tested=True,
                    replica_count=len(replicas),
                    all_replica_test_outputs_byte_exact=True,
                    aggregate_prediction_is_single_checkpoint=len(replicas) == 1,
                ),
            )
            manifest.update(
                status="complete",
                runtime_seconds=time.monotonic() - started,
                parent_sha256=checkpoint_sha,
                cohort_sha256=COHORT_SHA,
                unit_audit=units,
                replay_source_counts=counts,
                presentations=sum(r["presentations"] for r in replica_records),
                optimizer_updates=sum(r["optimizer_updates"] for r in replica_records),
                replica_count=len(replicas),
                presentations_per_replica=seen,
                optimizer_updates_per_replica=steps,
                row_sha256=row_digest.hexdigest(),
                mask_sha256=mask_digest.hexdigest(),
                supervised_stream_sha256=supervised_digest.hexdigest(),
                stream_hash_scope="shared per-replica stream"
                if len(replicas) > 1
                else "single model",
                head_parameters=replica_records[-1]["head_parameters"],
                supervised_warmup_presentations=sum(
                    r["supervised_warmup_presentations"] for r in replica_records
                ),
                supervised_adaptation_presentations=sum(
                    r["supervised_adaptation_presentations"] for r in replica_records
                ),
                supervised_loss=(
                    "Mean of eight 256-star InfoNCE losses per optimizer update; p(summary(latent)|FeH), fixed label rolls, declared reconstruction budget."
                    if contrastive
                    else "Mean of eight 256-star losses per optimizer update; declared reconstruction budget. RnC normalized single-view auxiliary, not full Chang recipe."
                ),
                comparison=comparison,
                test_not_used_for_selection=True,
                optimizer_policy=(
                    "Retained winner model/head and both Adam states; no warmup or scheduler steps"
                    if suite.get("continuation_checkpoint")
                    else "Retained parent Adam moments; fixed encoder/model LR and fresh warmed-up head optimizer; no scheduler steps"
                ),
                input_correction=correction_summary
                if "input_correction" in arm
                else None,
                reported_prediction=(
                    "per-star rational extrapolation of the lambda curve to -1"
                    if len(replicas) > 1
                    else "single adapted head"
                ),
                aggregate_checkpoint="replicas.json; all listed checkpoints required"
                if len(replicas) > 1
                else "checkpoint.pth",
            )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
