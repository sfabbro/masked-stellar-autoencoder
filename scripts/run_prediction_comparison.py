"""Matched native-FeH head/loss pilots with an exact reconstruction-row budget.

This is a separate continuation; the parent/main run and benchmark are read-only.
Labels describe reported native spectroscopic FeH, not calibrated latent truth.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import run_pretrain_experiments as pretrain
import run_xp_representation_probes as probe
import torch
import yaml
from run_enriched_value import COHORT_SHA, prepare
from run_native_age import SCALER_SHA
from run_stellar_label_experiments import BENCH
from stellar_catalog_jobs import atomic_json, identity_split

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_contrastive_flow,
    build_prediction_head,
    contrastive_grid,
    fit_lds_huber,
    point_prediction,
    prediction_loss,
    rank_contrast_loss,
)
from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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
        data, fit, test, safe, _, yc, ys = native_inputs(
            args.cohort, 128 if args.smoke else None
        )
        features = data["feature_columns"]
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
            for group in optimizer.param_groups:
                group["lr"] = suite["encoder_lr"]
            torch.manual_seed(42)
            contrastive = arm["loss"] == "contrastive_flow"
            if contrastive and arm["head"] != "contrastive_flow":
                raise ValueError(
                    "contrastive flow uses its own estimator, not a point head"
                )
            head = (
                build_contrastive_flow(arm["contrastive_flow"]).to(device)
                if contrastive
                else build_prediction_head(arm["head"], input_dim=256).to(device)
            )
            head_opt = torch.optim.AdamW(
                head.parameters(), lr=suite["head_lr"], weight_decay=1e-4
            )
            labels = torch.as_tensor(
                (data["fit_truth"][fit] - yc) / ys, dtype=torch.float32, device=device
            )
            lds = None
            label_variance = None
            if arm["loss"] == "lds_huber":
                lds = fit_lds_huber(labels, arm["lds"])
                weights = lds._weights_per_bin.detach().cpu().numpy()
                atomic_json(
                    args.output / "lds-weights.json",
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
                    args.output / "label-error-summary.json",
                    dict(
                        fit_rows=int(len(quoted)),
                        min_dex=float(quoted.min()),
                        median_dex=float(np.median(quoted)),
                        max_dex=float(quoted.max()),
                        quoted_error_is_not_total_calibrated_sigma=True,
                    ),
                )
            grid = (
                contrastive_grid(labels, arm["contrastive_flow"])
                if contrastive
                else None
            )
            if grid is not None:
                atomic_json(
                    args.output / "contrastive-flow-grid.json",
                    dict(
                        points=int(grid.numel()),
                        min_standardized=float(grid[0]),
                        max_standardized=float(grid[-1]),
                        estimand="grid_mle_of_p_summary_latent_given_FeH",
                        negatives="fixed_batch_rolls",
                        not_a_training_prior_posterior=True,
                    ),
                )
            train = torch.as_tensor(safe[fit], device=device)
            test_x = safe[test["used_row"]]

            def batch_objective(latent, rows):
                if contrastive:
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
                            active_head.standardized_mle(encoded, grid).cpu().numpy()
                        )
                return np.concatenate(chunks)

            def save_outputs(path, array):
                key = "standardized_mle" if contrastive else "member_outputs"
                np.savez_compressed(path, **{key: array})

            np.savez_compressed(
                args.output / "paired-cohort.npz",
                fit_source_id=data["source_id"][fit],
                fit_input=safe[fit],
                fit_target=data["fit_truth"][fit],
                fit_measurements=data["fit_native_measurements"],
                test_input=test_x,
                test_measurements=test["measurements"],
                label_center=yc,
                label_scale=ys,
            )
            initial = current_outputs(model, head)
            save_outputs(args.output / "initial_predictions.npz", initial)
            rng = np.random.default_rng(42)
            supervised_digest, row_digest, mask_digest = (
                hashlib.sha256() for _ in range(3)
            )
            warmup = 2 if args.smoke else suite["warmup_updates"]
            for _ in range(warmup):
                rows = rng.integers(len(fit), size=suite["contrast_batch"])
                supervised_digest.update(rows.tobytes())
                model.eval()
                head.train()
                with torch.no_grad():
                    z = model.encoder(train[rows])
                head_opt.zero_grad(set_to_none=True)
                loss = batch_objective(z, rows)
                loss.backward()
                head_opt.step()
            budget = 512 if args.smoke else suite["reconstruction_presentations"]
            effective = 256 if args.smoke else suite["effective_batch"]
            mask_rng = torch.Generator(device=device).manual_seed(42)
            xp = [i for i, n in enumerate(features) if n.startswith(("bp_", "rp_"))]
            micro = config["training"]["micro_batch_size"]
            seen, steps = 0, 0
            snapshots = [0, 3_000_000, 6_000_000, 9_000_000, budget]
            after_warmup = current_outputs(model, head)
            save_outputs(args.output / "snapshot-0.npz", after_warmup)
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
                torch.manual_seed(1000 + steps)
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
                    torch.manual_seed(100_000 + steps * groups + group)
                    z = model.encoder(train[rows])
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
                with (args.output / "metrics.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(
                            dict(
                                step=steps,
                                presentations=seen,
                                supervised_loss=supervised_total,
                                reconstruction_mae=reconstruction_total,
                                model_gradient_norm_before_clip=float(
                                    model_gradient_norm
                                ),
                                head_gradient_norm_before_clip=float(
                                    head_gradient_norm
                                ),
                                elapsed_s=time.monotonic() - started,
                            )
                        )
                        + "\n"
                    )
                if any(
                    seen - len(raw) < threshold <= seen
                    for threshold in snapshots
                    if threshold
                ):
                    outputs = current_outputs(model, head)
                    save_outputs(args.output / f"snapshot-{seen}.npz", outputs)
            outputs = current_outputs(model, head)
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
            )
            torch.save(state, args.output / "checkpoint.pth")
            restored = torch_load_trusted(
                args.output / "checkpoint.pth", map_location=device, weights_only=False
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
                args.output / "final_reconstruction.npz",
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
                args.output / "predictions.npz",
                prediction=prediction,
                **{raw_key: outputs},
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
            if arm["loss"] in {
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
                    all_test_member_outputs_byte_exact=not contrastive,
                    all_test_standardized_mle_byte_exact=contrastive,
                    rows=len(outputs),
                    model_and_head_strict_reload=True,
                    all_validation_decoder_outputs_byte_exact=True,
                    same_device=True,
                    optimizer_state_saved_not_resumed_tested=True,
                ),
            )
            manifest.update(
                status="complete",
                runtime_seconds=time.monotonic() - started,
                parent_sha256=checkpoint_sha,
                cohort_sha256=COHORT_SHA,
                unit_audit=units,
                replay_source_counts=counts,
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
                supervised_loss=(
                    "Mean of eight 256-star InfoNCE losses per optimizer update; p(summary(latent)|FeH), fixed label rolls, independent 10M reconstruction budget."
                    if contrastive
                    else "Mean of eight 256-star losses per optimizer update; independent 10M reconstruction budget. RnC normalized single-view auxiliary, not full Chang recipe."
                ),
                comparison=comparison,
                test_not_used_for_selection=True,
                optimizer_policy="Retained parent Adam moments; fixed encoder/model LR and fresh warmed-up head optimizer; no scheduler steps",
            )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
