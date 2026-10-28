"""CANFAR-only sparse six-target density/reconstruction recipe comparisons.

FIT-only gradients and scaling; first minimum macro TUNE observed-coordinate NLL
selects a full state. TEST/CAL are used only after selection. Source-role masks
are respected, and StarHorse posterior ages/distances are not physical truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
from pathlib import Path

import h5py
import numpy as np
import run_multilabel_screen as native
import torch
import yaml
from run_prediction_comparison import replay_batches

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.final_six_recipes import (
    NAMES,
    AdaptedTriLoss,
    SparseSixHead,
    independent_masked_views,
    normalization,
    prediction_order_loss,
    standardized_arrays,
)
from masked_stellar_autoencoder.training.input_measurement import xp_indices
from masked_stellar_autoencoder.training.xp_experiments import (
    spectral_basis,
    spectral_relative_error,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def draw_balanced_rows(rng, fit, mask, per_label=48):
    """Unique source rows, balanced observed-label availability, FIT only."""
    blocks = []
    for column in range(6):
        candidates = fit[mask[fit, column]]
        if len(candidates) < 16:
            raise ValueError("Too few usable FIT labels")
        blocks.append(
            rng.choice(candidates, min(per_label, len(candidates)), replace=False)
        )
    return np.unique(np.concatenate(blocks))


def cp_threshold(scores, alpha, *, allow_unbounded=False):
    scores = np.asarray(scores, dtype="f8")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if not 0 < alpha < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Insufficient finite calibration data for declared coverage")
    if rank > len(scores):
        if allow_unbounded:
            return float("inf")
        raise ValueError("Insufficient finite calibration data for declared coverage")
    return float(np.sort(scores)[rank - 1])


def make_auxiliary(data, fit, standardized, latent_width):
    edges = []
    for column in range(6):
        values = standardized[fit[data["quality"][fit, column]], column]
        edge = np.unique(np.quantile(values, np.linspace(0, 1, 17)))
        if len(edge) < 3:
            raise ValueError("Degenerate FIT-only auxiliary bins")
        edges.append(edge)
    return AdaptedTriLoss(latent_width, edges)


def forward_coordinates(head, z):
    return head.moments(z)[0]


def evaluate(
    model,
    head,
    data,
    rows,
    standardized,
    sigma,
    mask,
    jacobian,
    center,
    scale,
    device,
    output=None,
):
    """Isolated end-to-end FP64 prediction; native moments and marginal scores."""
    model = copy.deepcopy(model).double().eval()
    head = copy.deepcopy(head).double().eval()
    pieces = {
        key: []
        for key in (
            "mean_native",
            "variance_native",
            "intrinsic_log_prob_native",
            "observed_log_prob_native",
            "native3_intrinsic_joint_log_prob_native",
            "native3_observed_joint_log_prob_native",
        )
    }
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            rr = rows[start : start + 128]
            x = torch.tensor(data["input"][rr], device=device, dtype=torch.float64)
            z = model.encoder(x)
            mean, variance = head.native_moments(
                z,
                torch.tensor(center, device=device),
                torch.tensor(scale, device=device),
            )
            if (
                not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all()
                or (variance <= 0).any()
            ):
                raise FloatingPointError("Nonfinite native predictive moments")
            pieces["mean_native"].append(mean.cpu().numpy())
            pieces["variance_native"].append(variance.cpu().numpy())
            y = torch.tensor(standardized[rr], device=device)
            s = torch.tensor(sigma[rr], device=device)
            intrinsic = []
            observed = []
            for column in range(6):
                single = torch.zeros((len(rr), 6), device=device, dtype=torch.bool)
                single[:, column] = True
                intrinsic.append(
                    head.row_log_prob(z, y, s, single, observed=False).cpu().numpy()
                    - jacobian[rr, column]
                )
                observed.append(
                    head.row_log_prob(z, y, s, single).cpu().numpy()
                    - jacobian[rr, column]
                )
            native3 = torch.zeros((len(rr), 6), device=device, dtype=torch.bool)
            native3[:, :3] = torch.tensor(mask[rr, :3], device=device)
            complete = mask[rr, :3].all(1)
            for name, noisy in [
                ("native3_intrinsic_joint_log_prob_native", False),
                ("native3_observed_joint_log_prob_native", True),
            ]:
                scores = head.row_log_prob(
                    z, y, s, native3, observed=noisy
                ).cpu().numpy() - jacobian[rr, :3].sum(-1)
                pieces[name].append(np.where(complete, scores, np.nan))
            pieces["intrinsic_log_prob_native"].append(np.stack(intrinsic, -1))
            pieces["observed_log_prob_native"].append(np.stack(observed, -1))
    result = {key: np.concatenate(values) for key, values in pieces.items()}
    result.update(
        rows=rows,
        source_id=data["source_id"][rows],
        target_native=data["value"][rows],
        observed=mask[rows],
        label_source=data["label_source"][rows],
    )
    # Densities of unobserved values are not scores at an invented target.
    for key in ("intrinsic_log_prob_native", "observed_log_prob_native"):
        result[key] = np.where(mask[rows], result[key], np.nan)
    if output is not None:
        np.savez_compressed(output, **result)
    return result


def selection_score(result):
    scores = []
    for column in range(6):
        valid = result["observed"][:, column]
        if valid.sum() < 16:
            raise ValueError("TUNE role lacks a target")
        values = result["observed_log_prob_native"][valid, column]
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite TUNE density")
        scores.append(float(-values.mean()))
    return float(np.mean(scores)), scores


def calibrate(cal, test, output):
    """Per-label finite-sample CP and conservative simultaneous sparse coverage."""
    arrays = {}
    summary = {}
    for alpha in (0.1, 0.05):
        level = str(int(round((1 - alpha) * 100)))
        labels = []
        for column, name in enumerate(NAMES):
            valid = cal["observed"][:, column]
            scores = np.abs(
                cal["mean_native"][valid, column] - cal["target_native"][valid, column]
            )
            q = cp_threshold(scores, alpha)
            joint = cp_threshold(scores, alpha / 6, allow_unbounded=True)
            for mode, threshold in [("marginal", q), ("bonferroni", joint)]:
                arrays[f"{level}_{mode}_{name}_lower"] = (
                    test["mean_native"][:, column] - threshold
                )
                arrays[f"{level}_{mode}_{name}_upper"] = (
                    test["mean_native"][:, column] + threshold
                )
            density_q = cp_threshold(
                -cal["observed_log_prob_native"][valid, column], alpha
            )
            intrinsic_q = cp_threshold(
                -cal["intrinsic_log_prob_native"][valid, column], alpha
            )
            arrays[f"{level}_observed_density_{name}_threshold"] = np.array(density_q)
            arrays[f"{level}_intrinsic_density_{name}_threshold"] = np.array(
                intrinsic_q
            )
            test_valid = test["observed"][:, column]
            covered = (
                np.abs(
                    test["mean_native"][:, column] - test["target_native"][:, column]
                )
                <= q
            )
            labels.append(
                dict(
                    target=name,
                    calibration_count=int(valid.sum()),
                    test_count=int(test_valid.sum()),
                    threshold_native=q,
                    coverage=float(covered[test_valid].mean()),
                    width_native=2 * q,
                    observed_density_threshold_native=density_q,
                    intrinsic_density_threshold_native=intrinsic_q,
                    simultaneous_threshold_native=joint if np.isfinite(joint) else None,
                    simultaneous_interval_unbounded=not np.isfinite(joint),
                )
            )
        summary[level] = labels
    arrays["test_observed"] = test["observed"]
    arrays["calibration_observed"] = cal["observed"]
    arrays["test_source_id"] = test["source_id"]
    arrays["calibration_source_id"] = cal["source_id"]
    np.savez_compressed(output / "conformal.npz", **arrays)
    native.atomic(
        output / "conformal.json",
        dict(
            status="producer_arithmetic_only",
            levels=summary,
            physical_truth_claim=False,
        ),
    )
    return summary


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("All native model execution requires CANFAR CUDA")
    if args.output.exists():
        raise FileExistsError(
            "Refusing to reuse an output or repeat completed training"
        )
    args.output.mkdir(parents=True)
    cfg = yaml.safe_load(args.config.read_text())
    arm = next(a for a in cfg["arms"] if a["run_id"] == args.run_id)
    if arm.get("parent86_control"):
        cfg["checkpoint"] = cfg["reference_checkpoint"]
        cfg["checkpoint_sha256"] = cfg["reference_checkpoint_sha256"]
    if getattr(args, "smoke", False):
        cfg.update(
            reconstruction_presentations=512,
            effective_batch=256,
            micro_batch=128,
            warmup_updates=2,
            supervised_batches_per_update=1,
        )
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(
        status="preparing",
        run_id=args.run_id,
        source_sha=source,
        policy=cfg,
        arm=arm,
        scope="six sparse source-specific targets; label masks and source roles retained; no identified empirical six-dimensional copula",
        physical_truth_claim=False,
        smoke=getattr(args, "smoke", False),
    )
    native.atomic(args.output / "manifest.json", manifest)
    try:
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Installed library SHA changed")
        prep = json.loads(Path(cfg["preparation_manifest"]).read_text())
        if (
            prep["status"] != "sparse_six_target_source_assembly_complete"
            or prep["source_commit"] != cfg["preparation_source_sha"]
            or sha(cfg["prepared"]) != prep["input_sha256"]
            or prep["input_sha256"] != cfg["prepared_sha256"]
            or sha(cfg["preparation_manifest"]) != cfg["preparation_manifest_sha256"]
        ):
            raise ValueError("Six-target preparation not complete or binding changed")
        expected_policy = yaml.safe_load(Path(cfg["preparation_policy"]).read_text())
        if prep["source_provenance"] != expected_policy or not prep.get(
            "role_promoted_labels_excluded_from_scoring"
        ):
            raise ValueError("Preparation source/role policy differs")
        if sha(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
            raise ValueError("Parent SHA changed")
        data = dict(np.load(cfg["prepared"], allow_pickle=False))
        quality = np.asarray(data["quality"], bool) & np.asarray(data["observed"], bool)
        if not np.array_equal(
            quality, data["observed"] & data["original_role_eligible"]
        ):
            raise ValueError("Source-role eligibility was bypassed")
        ids = data["source_id"]
        roles = data["role"]
        fit = np.flatnonzero(roles == "fit")
        if len(np.unique(ids)) != len(ids) or np.asarray(
            data["target_names"]
        ).tolist() != list(NAMES):
            raise ValueError("Source IDs or target order differs")
        center, scale = normalization(data["value"], quality, roles == "fit")
        standardized, sigma, mask, jacobian = standardized_arrays(
            data["value"],
            quality,
            data["error"],
            data["error_observed"] & quality,
            center,
            scale,
        )
        if sha(cfg["reference_checkpoint"]) != cfg["reference_checkpoint_sha256"]:
            raise ValueError("Feature-reference checkpoint SHA changed")
        reference_disk = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        weights_disk = torch_load_trusted(
            cfg["checkpoint"], map_location="cpu", weights_only=False
        )
        if not arm.get("parent86_control") and (
            weights_disk.get("source_sha") != cfg["large_checkpoint_source_sha"]
            or weights_disk.get("presentations") != 4500000000
            or weights_disk.get("completed_updates") != 137340
            or weights_disk.get("parent_sha256") != cfg["reference_checkpoint_sha256"]
        ):
            raise ValueError("Final large-model checkpoint lineage or counters differ")
        if any(
            torch.is_floating_point(v) and not torch.isfinite(v).all()
            for v in weights_disk["model_state_dict"].values()
        ):
            raise ValueError("Nonfinite pretrained model weights")
        disk = dict(reference_disk, model_state_dict=weights_disk["model_state_dict"])
        signature = reference_disk["run_signature"]
        features = signature["feature_cols"]
        if list(data["feature_names"]) != features:
            raise ValueError("Prepared feature order differs from parent")
        for key in ("scaler_center", "scaler_scale"):
            np.testing.assert_array_equal(data[key], signature[key])
        device = torch.device("cuda")
        torch.set_num_threads(4)
        seed = arm["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        mask_rng = torch.Generator(device=device).manual_seed(seed)
        view_rng = torch.Generator(device=device).manual_seed(seed + 1701)
        model_cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        model = native.probe.load_model(model_cfg, disk, device)
        head = SparseSixHead(
            arm["kind"],
            latent_dim=model_cfg["model"]["layer_dims"][-1],
            noise_aware=True,
            mixed_weight=arm.get("mixed_weight", 0),
        ).to(device)
        auxiliary = (
            make_auxiliary(
                data, fit, standardized, model_cfg["model"]["layer_dims"][-1]
            ).to(device)
            if arm.get("auxiliary")
            else None
        )
        model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg["encoder_lr"])
        head_parameters = list(head.parameters()) + (
            list(auxiliary.parameters()) if auxiliary is not None else []
        )
        head_optimizer = torch.optim.Adam(head_parameters, lr=cfg["head_lr"])
        with h5py.File(cfg["benchmark"]) as file:
            keys, valid_keys, bc, bs = native.pretrain.validate_source(
                disk, model_cfg, file
            )
            provenance = json.loads(file.attrs["provenance"])
            if provenance["source_id_sha256"] != cfg["benchmark_ordered_id_sha256"]:
                raise ValueError("Immutable benchmark source-ID provenance changed")
            manifest["benchmark_source_id_sha256"] = provenance["source_id_sha256"]
            np.testing.assert_array_equal(bc, data["scaler_center"])
            np.testing.assert_array_equal(bs, data["scaler_scale"])
            if (
                len(file["train/records"]) != 9000000
                or len(file["validation/records"]) != 1000000
            ):
                raise ValueError("Fixed reconstruction sample differs")
            scratch = Path(os.environ.get("WORK", "/scratch/src")) / (
                "final-six-cache-" + args.run_id
            )
            scratch.mkdir(parents=True, exist_ok=True)
            cache, _, counts = native.pretrain.read_prefixes(
                file,
                keys,
                features,
                None,
                1024 if getattr(args, "smoke", False) else 9000000,
                bc,
                bs,
                scratch / "train.npy",
            )
        audit = native.probe.audit_fits_xp_units(
            cfg["benchmark"], features, data["raw"], ids, "g_normalized"
        )
        native.atomic(args.output / "xp-unit-audit.json", audit)
        xp = xp_indices(features).tolist()
        width = len(model_cfg["data"]["recon_cols"])
        if features[:width] != model_cfg["data"]["recon_cols"]:
            raise ValueError("Reconstruction order differs from features")
        allowed = np.array(
            [
                name in {"W1", "W2", "G", "BP", "RP"}
                or name
                in {f"{band}_{i}" for band in ("bp", "rp") for i in range(1, 56)}
                for name in features
            ]
        )
        projection = spectral_basis(features[:width])
        basis = torch.tensor(projection["basis"], device=device, dtype=torch.float32)
        wave = torch.tensor(
            projection["wavelength_nm"], device=device, dtype=torch.float32
        )
        fcenter = torch.tensor(bc[:width], device=device, dtype=torch.float32)
        fscale = torch.tensor(bs[:width], device=device, dtype=torch.float32)
        stream = hashlib.sha256()
        row_hash = hashlib.sha256()
        mask_hash = hashlib.sha256()
        metrics = []
        source_codes = np.zeros(data["label_source"].shape, dtype="i8")
        for column in range(6):
            source_codes[:, column] = np.unique(
                data["label_source"][:, column], return_inverse=True
            )[1]

        def supervised(warmup=False):
            rr = draw_balanced_rows(rng, fit, mask)
            stream.update(np.asarray(rr, dtype="<i8").tobytes())
            x = torch.tensor(data["input"][rr], device=device)
            y = torch.tensor(standardized[rr], device=device, dtype=torch.float32)
            s = torch.tensor(sigma[rr], device=device, dtype=torch.float32)
            observed = torch.tensor(mask[rr], device=device)
            identity = torch.tensor(ids[rr].astype("i8"), device=device)
            sources = torch.tensor(source_codes[rr], device=device)
            if warmup:
                with torch.no_grad():
                    z = model.encoder(x)
            else:
                z = model.encoder(x)
            loss = head.criterion(z, y, s, observed, identity)
            if not warmup and arm.get("ranking_weight", 0):
                loss = loss + arm["ranking_weight"] * prediction_order_loss(
                    forward_coordinates(head, z), y, observed, s, sources
                )
            if not warmup and auxiliary is not None:
                first, second = independent_masked_views(
                    x, torch.tensor(data["visible"][rr], device=device), xp, view_rng
                )
                loss = loss + cfg["auxiliary_weight"] * auxiliary(
                    model.encoder(first),
                    model.encoder(second),
                    y,
                    observed,
                    sources,
                    include_rnc=arm["auxiliary"] == "triloss",
                )
            return loss

        # Actual native CUDA objective/gradient allocation preflight, before SGD.
        model.eval()
        head.train()
        torch.cuda.reset_peak_memory_stats()
        probe = supervised()
        probe.backward()
        gradients = [
            p.grad
            for p in list(model.parameters()) + head_parameters
            if p.grad is not None
        ]
        if (
            not torch.isfinite(probe)
            or not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
        ):
            raise FloatingPointError("Actual CUDA gradient preflight failed")
        native.atomic(
            args.output / "cuda-objective-preflight.json",
            dict(
                before_sgd=True,
                finite_objective_and_gradients=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                library_revision=cfg["library_revision"],
                target_count=6,
                missing_labels_not_imputed=True,
            ),
        )
        model.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        # Preflight consumes no training draws or state updates.
        rng = np.random.default_rng(seed)
        stream = hashlib.sha256()
        mask_rng.manual_seed(seed)
        view_rng.manual_seed(seed + 1701)
        for warmup_step in range(cfg["warmup_updates"]):
            torch.manual_seed(100000 + seed + warmup_step)
            model.eval()
            head_optimizer.zero_grad(set_to_none=True)
            loss = supervised(True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Warmup objective nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_parameters, 1, error_if_nonfinite=True)
            head_optimizer.step()
        seen = updates = 0
        best = float("inf")
        selected = None

        def state():
            return dict(
                model_state_dict=model.state_dict(),
                head_state_dict=head.state_dict(),
                auxiliary_state_dict=auxiliary.state_dict() if auxiliary else None,
                model_optimizer_state_dict=model_optimizer.state_dict(),
                head_optimizer_state_dict=head_optimizer.state_dict(),
                rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all(),
                numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),
                supervised_rng_state=rng.bit_generator.state,
                mask_rng_state=mask_rng.get_state(),
                auxiliary_view_rng_state=view_rng.get_state(),
                label_center=center,
                label_scale=scale,
                target_names=NAMES,
                feature_columns=features,
                completed_presentations=seen,
                completed_updates=updates,
                arm=arm,
                policy=cfg,
                source_sha=source,
                parent_sha256=cfg["checkpoint_sha256"],
                input_sha256=prep["input_sha256"],
                run_signature=signature,
            )

        model.train()
        head.train()

        def role_rows(role):
            candidates = np.flatnonzero(roles == role)
            if not getattr(args, "smoke", False):
                return candidates
            return np.unique(
                np.concatenate([candidates[mask[candidates, j]][:32] for j in range(6)])
            )

        for positions, raw in replay_batches(
            cache, cfg["reconstruction_presentations"], cfg["effective_batch"]
        ):
            torch.manual_seed(1000 + seed + updates)
            row_hash.update(np.asarray(positions, dtype="<i8").tobytes())
            raw[:, ~allowed] = np.nan
            target = torch.tensor(raw, device=device)
            hidden, hidden_mask, finite = native.pretrain.training_mask(
                target, xp, 0.5, 0.2, mask_rng
            )
            mask_hash.update(hidden_mask.cpu().numpy().tobytes())
            score = (hidden_mask & finite)[:, :width]
            denominator = score.sum()
            model_optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            reco_sum = spectral_sum = supervised_sum = 0.0
            for start in range(0, len(raw), cfg["micro_batch"]):
                stop = start + cfg["micro_batch"]
                prediction, _ = model(hidden[start:stop])
                reco = native.pretrain.experiment_objective(
                    prediction,
                    target[start:stop, :width],
                    score[start:stop],
                    xp,
                    {"loss": "mae"},
                    denominator,
                )
                objective = cfg["reconstruction_weight"] * reco
                reco_sum += float(reco.detach())
                if arm.get("spectral_weight", 0):
                    fraction, valid_count = spectral_relative_error(
                        prediction,
                        target[start:stop, :width],
                        score[start:stop],
                        xp,
                        fscale,
                        fcenter,
                        basis,
                        wave,
                    )
                    spectral = fraction.sum() / len(raw)
                    objective = objective + arm["spectral_weight"] * spectral
                    spectral_sum += float(spectral.detach())
                objective.backward()
            for _ in range(cfg["supervised_batches_per_update"]):
                loss = supervised() / cfg["supervised_batches_per_update"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Sparse target objective nonfinite")
                loss.backward()
                supervised_sum += float(loss.detach())
            mn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            hn = torch.nn.utils.clip_grad_norm_(
                head_parameters, 1, error_if_nonfinite=True
            )
            model_optimizer.step()
            head_optimizer.step()
            seen += len(raw)
            updates += 1
            metrics.append(
                dict(
                    presentations=seen,
                    updates=updates,
                    reconstruction_mae=reco_sum,
                    spectral_fraction=spectral_sum,
                    supervised_objective=supervised_sum,
                    model_gradient_norm=float(mn),
                    head_gradient_norm=float(hn),
                    label_stream_sha256=stream.hexdigest(),
                    reconstruction_stream_sha256=row_hash.hexdigest(),
                    mask_stream_sha256=mask_hash.hexdigest(),
                )
            )
            if updates % 50 == 0 or seen == cfg["reconstruction_presentations"]:
                tune_rows = role_rows("tune")
                result = evaluate(
                    model,
                    head,
                    data,
                    tune_rows,
                    standardized,
                    sigma,
                    mask,
                    jacobian,
                    center,
                    scale,
                    device,
                )
                value, by_label = selection_score(result)
                receipt = native.save_periodic_checkpoint(args.output, state())
                metrics[-1].update(
                    tune_macro_native_nll=value,
                    tune_nll_by_label=by_label,
                    checkpoint=receipt,
                )
                if value < best:
                    best = value
                    selected = receipt
                    np.savez_compressed(args.output / "tune-predictions.npz", **result)
                    native.atomic(
                        args.output / "selected-checkpoint.json",
                        dict(
                            receipt=receipt,
                            selection="first minimum macro TUNE observed-coordinate native NLL",
                            tune_source_id_sha256=hashlib.sha256(
                                ids[tune_rows].tobytes()
                            ).hexdigest(),
                            score=value,
                        ),
                    )
            native.atomic(args.output / "metrics.json", metrics)
            model.train()
            head.train()
            manifest.update(
                status="running",
                completed_presentations=seen,
                completed_updates=updates,
                training_role="FIT only",
                reconstruction_actual_rows_scanned=seen,
                model_optimizer="fresh Adam",
                head_optimizer="fresh Adam",
            )
            native.atomic(args.output / "manifest.json", manifest)
        final_budget = dict(presentations=seen, updates=updates)
        if selected is None:
            raise ValueError("No finite TUNE selection")
        restored = torch_load_trusted(
            args.output / selected["file"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(restored["model_state_dict"], strict=True)
        head.load_state_dict(restored["head_state_dict"], strict=True)
        if auxiliary:
            auxiliary.load_state_dict(restored["auxiliary_state_dict"], strict=True)
        native.save_checkpoint(args.output / "checkpoint.pth", restored)
        # Full-state serialization equality is separate from numerical prediction validation.
        check = torch_load_trusted(
            args.output / "checkpoint.pth", map_location="cpu", weights_only=False
        )
        native._assert_checkpoint_equal(restored, check)
        outputs = {}
        for role in ("cal", "test"):
            rows = role_rows(role)
            outputs[role] = evaluate(
                model,
                head,
                data,
                rows,
                standardized,
                sigma,
                mask,
                jacobian,
                center,
                scale,
                device,
                args.output / (role + "-predictions.npz"),
            )
        cp = calibrate(outputs["cal"], outputs["test"], args.output)
        manifest.update(
            status="complete",
            training_budget=final_budget,
            selected_model_counters=dict(
                presentations=restored["completed_presentations"],
                updates=restored["completed_updates"],
            ),
            selected_checkpoint=selected,
            checkpoint_sha256=sha(args.output / "checkpoint.pth"),
            input_sha256=prep["input_sha256"],
            library_revision=cfg["library_revision"],
            gpu=torch.cuda.get_device_name(),
            row_stream_sha256=row_hash.hexdigest(),
            label_stream_sha256=stream.hexdigest(),
            mask_stream_sha256=mask_hash.hexdigest(),
            conformal_producer=cp,
            density_scope="Gaussian/MDN exact sparse observed marginals; scalar NSF independent product, not empirically identified six-dimensional dependence",
            coordinate_units=["K", "dex", "dex", "dex", "Gyr", "kpc"],
            age_distance_estimand="StarHorse posterior medians, not independent physical truth or raw-parallax likelihood",
            full_state_reload_equal=True,
            original_role_target_counts=prep["observed_per_role_target"],
            point_moments="Gaussian/MDN analytic native/lognormal moments; scalarNSF fixed256 antitheticQMC approximation, not exact population moments",
            target_dependence="Shared-mixture MDN normalized sparse marginals; missing joint blocks do not identify an empirical6Dcopula. ScalarNSF product explicitly assumes independence.",
            ancestral_supervision="Large parent includes native FeH supervision. New target holdout IDs may have ancestral auxiliary-label exposure; no fresh-object generalization claim without per-ID ancestry audit.",
            calibration_scope="Globally disjoint CAL preserves original per-label roles; 1024 age/distance, 490 alpha, 2046 native3. No arbitrary-tail/physical-truth guarantee; small smoke sets may produce explicit unbounded Bonferroni sets.",
            spectral_scope="Calibrated GaiaXPy BP/RP projection of hidden-coefficient decoder residuals; not a label-to-stellar-atmosphere forward likelihood.",
            contrastive_scope="Explicit per-observed-scalar InfoNCE with the same quoted-error convolution under both cyclic contexts; no unsupported automatic noise wrapper or empirically identified joint6Dcopula.",
        )
        native.atomic(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        native.atomic(args.output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="CUDA-only bounded native schema/gradient/reload gate, not a production result",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
