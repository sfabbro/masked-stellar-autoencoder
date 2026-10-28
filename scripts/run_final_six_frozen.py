"""Matched frozen-large-encoder six-target head controls on CANFAR CUDA.

No encoder/decoder gradient, reconstruction pass or large-model update occurs.
Head updates match the joint screen:250warmup +306Adam updates of8minibatches.
FIT embeddings may be cached only under exact source/input/parent SHA bindings;
TUNE/CAL/TEST scoring always re-encodes raw inputs in isolated FP64.
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

import numpy as np
import run_final_six_recipes as joint
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.final_six_recipes import (
    NAMES,
    SparseSixHead,
    normalization,
    standardized_arrays,
)


def schedule(config, *, smoke=False):
    """Explicit head-update budget; no artificial reconstruction-row budget."""
    warmup = 2 if smoke else int(config["warmup_updates"])
    updates = 2 if smoke else int(config["head_optimizer_updates"])
    groups = 1 if smoke else int(config["supervised_batches_per_update"])
    if min(warmup, updates, groups) < 1:
        raise ValueError("Positive matched head schedule required")
    return dict(
        warmup_updates=warmup,
        head_optimizer_updates=updates,
        supervised_batches_per_update=groups,
        total_head_adam_steps=warmup + updates,
        supervised_minibatch_calls=warmup + updates * groups,
        reconstruction_presentations=0,
        encoder_updates=0,
        decoder_updates=0,
    )


def cache_bindings(cfg, source, preparation_manifest, feature_names):
    return dict(
        source_sha=source,
        input_sha256=cfg["prepared_sha256"],
        parent_sha256=cfg["checkpoint_sha256"],
        reference_parent_sha256=cfg["reference_checkpoint_sha256"],
        preparation_manifest_sha256=joint.sha(preparation_manifest),
        feature_names=feature_names,
        dtype="float32",
        encoder_mode="eval_frozen_no_grad",
        encoder_architecture_config_sha256=joint.sha("configs/pretrain.canfar.yaml"),
    )


def bound_cache(path, bindings):
    """Reject changed-input cache reuse instead of silently replacing evidence."""
    binding = path.with_suffix(".bindings.json")
    if path.exists() or binding.exists():
        if not path.exists() or not binding.exists():
            raise ValueError(
                "Incomplete frozen FIT cache; preserve original diagnostic"
            )
        receipt = json.loads(binding.read_text())
        if receipt.get("bindings") != bindings or receipt.get(
            "cache_sha256"
        ) != joint.sha(path):
            raise ValueError("Changed frozen FIT cache binding or content")
        return True
    return False


def frozen_parameter_digest(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype="<i8").tobytes())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def role_rows(data, mask, role, smoke):
    rows = np.flatnonzero(data["role"] == role)
    if not smoke:
        return rows
    return np.unique(np.concatenate([rows[mask[rows, j]][:32] for j in range(6)]))


def run(args):
    if (
        not torch.cuda.is_available()
        or os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
    ):
        raise RuntimeError(
            "Frozen native-model controls execute only on CANFAR headless CUDA"
        )
    control = yaml.safe_load(args.config.read_text())
    arm = next((a for a in control["arms"] if a["run_id"] == args.run_id), None)
    if arm is None:
        raise ValueError("Unknown immutable frozen-control arm")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an attempted frozen control")
    cfg = yaml.safe_load(Path(control["source_config"]).read_text())
    if (
        control["warmup_updates"] != cfg["warmup_updates"]
        or control["supervised_batches_per_update"]
        != cfg["supervised_batches_per_update"]
        or control["head_optimizer_updates"]
        != math.ceil(cfg["reconstruction_presentations"] / cfg["effective_batch"])
    ):
        raise ValueError("Frozen head schedule differs from matched joint control")
    smoke = bool(getattr(args, "smoke", False))
    budget = schedule(control, smoke=smoke)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(
        status="preparing",
        run_id=args.run_id,
        task="frozen-large-six-target-head-control",
        source_sha=source,
        source_config_sha256=joint.sha(control["source_config"]),
        control_config_sha256=joint.sha(args.config),
        shared_evaluation_runner_sha256=joint.sha(joint.__file__),
        head_module_sha256=joint.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/final_six_recipes.py"
        ),
        frozen_runner_sha256=joint.sha(__file__),
        target_names=list(NAMES),
        target_estimands="Teff/logg/FeH native measurements; DESIalpha; StarHorseposterior age/distance, not physicaltruth",
        age_distance_coordinates="FIT-standardized ln(Gyr),ln(kpc); nativeJacobian retained",
        measurement_error_semantics="Native3+alpha quoted diagonal error only; StarHorseposterior widths not convolved",
        source_policy=cfg,
        control_policy=control,
        arm=arm,
        training_budget=budget,
        smoke=smoke,
        encoder_trainable=False,
        decoder_trainable=False,
        physical_truth_claim=False,
        independent_science_verification=False,
        scope="matched fresh head control on frozen final4.5B large encoder; sparse source-specific labels, no empirical full6Dcopula claim",
    )
    joint.native.atomic(args.output / "manifest.json", manifest)
    try:
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Frozen control installed library differs")
        prep = json.loads(Path(cfg["preparation_manifest"]).read_text())
        if (
            prep["status"] != "sparse_six_target_source_assembly_complete"
            or prep["source_commit"] != cfg["preparation_source_sha"]
            or joint.sha(cfg["prepared"]) != cfg["prepared_sha256"]
            or prep["input_sha256"] != cfg["prepared_sha256"]
            or joint.sha(cfg["preparation_manifest"])
            != cfg["preparation_manifest_sha256"]
            or prep["source_provenance"]
            != yaml.safe_load(Path(cfg["preparation_policy"]).read_text())
            or not prep.get("role_promoted_labels_excluded_from_scoring")
        ):
            raise ValueError("Frozen control source/role evidence changed")
        data = dict(np.load(cfg["prepared"], allow_pickle=False))
        quality = np.asarray(data["quality"], bool) & np.asarray(data["observed"], bool)
        if not np.array_equal(
            quality, data["observed"] & data["original_role_eligible"]
        ):
            raise ValueError("Source-specific eligible masks changed")
        ids, roles = data["source_id"], data["role"]
        if len(np.unique(ids)) != len(ids) or list(data["target_names"]) != list(NAMES):
            raise ValueError("Frozen control source identities/targets differ")
        fit = np.flatnonzero(roles == "fit")
        if set(roles.tolist()) != {"fit", "tune", "cal", "test"}:
            raise ValueError("Four disjoint source roles required")
        center, scale = normalization(data["value"], quality, roles == "fit")
        standardized, sigma, mask, jacobian = standardized_arrays(
            data["value"],
            quality,
            data["error"],
            data["error_observed"] & quality,
            center,
            scale,
        )
        if (
            joint.sha(cfg["checkpoint"]) != cfg["checkpoint_sha256"]
            or joint.sha(cfg["reference_checkpoint"])
            != cfg["reference_checkpoint_sha256"]
        ):
            raise ValueError("Frozen/reference parent SHA differs")
        reference = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        weights = torch_load_trusted(
            cfg["checkpoint"], map_location="cpu", weights_only=False
        )
        if (
            weights.get("source_sha") != cfg["large_checkpoint_source_sha"]
            or weights.get("presentations") != 4500000000
            or weights.get("completed_updates") != 137340
            or weights.get("parent_sha256") != cfg["reference_checkpoint_sha256"]
        ):
            raise ValueError("Frozen large checkpoint lineage/counters differ")
        signature = reference["run_signature"]
        features = signature["feature_cols"]
        if list(data["feature_names"]) != features:
            raise ValueError("Frozen control feature order differs")
        for key in ("scaler_center", "scaler_scale"):
            np.testing.assert_array_equal(data[key], signature[key])
        if any(
            torch.is_floating_point(v) and not torch.isfinite(v).all()
            for v in weights["model_state_dict"].values()
        ):
            raise ValueError("Nonfinite frozen large model weights")
        disk = dict(reference, model_state_dict=weights["model_state_dict"])
        model_cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        device = torch.device("cuda")
        torch.set_num_threads(4)
        seed = arm["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.default_rng(seed)
        audit = joint.native.probe.audit_fits_xp_units(
            cfg["benchmark"], features, data["raw"], ids, "g_normalized"
        )
        joint.native.atomic(args.output / "xp-unit-audit.json", audit)
        model = joint.native.probe.load_model(model_cfg, disk, device).eval()
        model.requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Frozen model has trainable parameters")
        parameter_sha = frozen_parameter_digest(model)
        head = SparseSixHead(
            arm["kind"],
            latent_dim=model_cfg["model"]["layer_dims"][-1],
            noise_aware=True,
            mixed_weight=arm.get("mixed_weight", 0),
        ).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg["head_lr"])
        manifest.update(
            library_revision=cfg["library_revision"],
            library_source=direct,
            parent_sha256=cfg["checkpoint_sha256"],
            input_sha256=prep["input_sha256"],
            feature_reference_parent_sha256=cfg["reference_checkpoint_sha256"],
            frozen_model_parameter_sha256=parameter_sha,
            gpu=torch.cuda.get_device_name(),
        )
        joint.native.atomic(args.output / "manifest.json", manifest)
        # Per-job immutable FIT cache; no calibration/test information enters training.
        cache = args.output / "fit-embeddings.npz"
        bindings = cache_bindings(cfg, source, cfg["preparation_manifest"], features)
        if bound_cache(cache, bindings):
            frozen = dict(np.load(cache, allow_pickle=False))
            np.testing.assert_array_equal(frozen["source_id"], ids[fit])
        else:
            blocks = []
            with torch.no_grad():
                for start in range(0, len(fit), 512):
                    blocks.append(
                        model.encoder(
                            torch.tensor(
                                data["input"][fit[start : start + 512]], device=device
                            )
                        )
                        .cpu()
                        .numpy()
                    )
            frozen = dict(
                source_id=ids[fit], embedding=np.concatenate(blocks), source_row=fit
            )
            np.savez_compressed(cache, **frozen)
            joint.native.atomic(
                cache.with_suffix(".bindings.json"),
                dict(bindings=bindings, cache_sha256=joint.sha(cache)),
            )
        np.testing.assert_array_equal(frozen["source_row"], fit)
        if (
            frozen["embedding"].dtype != np.float32
            or not np.isfinite(frozen["embedding"]).all()
        ):
            raise ValueError("Frozen FIT cache is not finitefloat32")
        fit_embedding = torch.tensor(frozen["embedding"], device=device)
        # Global -> FIT cache position. Only sampled FIT rows can be looked up.
        fit_position = np.full(len(ids), -1, dtype="i8")
        fit_position[fit] = np.arange(len(fit))
        stream = hashlib.sha256()
        sampled_counts = np.zeros(6, dtype="i8")

        def supervised():
            rr = joint.draw_balanced_rows(rng, fit, mask, per_label=48)
            stream.update(np.asarray(rr, dtype="<i8").tobytes())
            sampled_counts[:] += mask[rr].sum(0)
            positions = fit_position[rr]
            if (positions < 0).any():
                raise ValueError("Non-FIT source entered frozen training")
            latent = fit_embedding[torch.tensor(positions, device=device)]
            return head.criterion(
                latent,
                torch.tensor(standardized[rr], device=device, dtype=torch.float32),
                torch.tensor(sigma[rr], device=device, dtype=torch.float32),
                torch.tensor(mask[rr], device=device),
                torch.tensor(ids[rr].astype("i8"), device=device),
            )

        # Actual CUDA bounded native preflight; reset draw state before declared budget.
        head.train()
        torch.cuda.reset_peak_memory_stats()
        loss = supervised()
        loss.backward()
        if not torch.isfinite(loss) or not all(
            torch.isfinite(p.grad).all()
            for p in head.parameters()
            if p.grad is not None
        ):
            raise FloatingPointError("Frozen head CUDA gradient preflight failed")
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen model unexpectedly acquired gradients")
        with torch.no_grad():
            mean, var = head.moments(fit_embedding[:128])
            if (
                not torch.isfinite(mean).all()
                or not torch.isfinite(var).all()
                or (var <= 0).any()
            ):
                raise FloatingPointError(
                    "Frozen head CUDA moment allocation preflight failed"
                )
        joint.native.atomic(
            args.output / "cuda-objective-preflight.json",
            dict(
                before_sgd=True,
                finite_gradients_and_moments=True,
                context_rows=min(128, len(fit)),
                point_nodes=256,
                moment_node_chunk=16,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                frozen_model_gradient_absent=True,
            ),
        )
        rng = np.random.default_rng(seed)
        stream = hashlib.sha256()
        sampled_counts[:] = 0
        optimizer.zero_grad(set_to_none=True)
        for step in range(budget["warmup_updates"]):
            torch.manual_seed(100000 + seed + step)
            head.train()
            optimizer.zero_grad(set_to_none=True)
            loss = supervised()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite frozen-head warmup")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
        metrics = []
        best = float("inf")
        selected = None
        updates = 0

        def state():
            return dict(
                head_state_dict=head.state_dict(),
                head_optimizer_state_dict=optimizer.state_dict(),
                rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all(),
                numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),
                supervised_rng_state=rng.bit_generator.state,
                label_center=center,
                label_scale=scale,
                target_names=NAMES,
                feature_columns=features,
                completed_updates=updates,
                label_stream_sha256=stream.hexdigest(),
                supervised_label_presentations=sampled_counts.copy(),
                warmup_updates=budget["warmup_updates"],
                completed_head_adam_steps=budget["warmup_updates"] + updates,
                completed_presentations=0,
                encoder_updates=0,
                decoder_updates=0,
                reconstruction_presentations=0,
                arm=arm,
                policy=cfg,
                control_policy=control,
                source_sha=source,
                parent_sha256=cfg["checkpoint_sha256"],
                frozen_model_parameter_sha256=parameter_sha,
                input_sha256=prep["input_sha256"],
                run_signature=signature,
            )

        def scoring_rows(role):
            return role_rows(data, mask, role, smoke)

        for update in range(budget["head_optimizer_updates"]):
            torch.manual_seed(1000 + seed + update)
            head.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(budget["supervised_batches_per_update"]):
                loss = supervised() / budget["supervised_batches_per_update"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Frozen-head objective nonfinite")
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            updates += 1
            metric = dict(
                updates=updates,
                head_adam_steps=budget["warmup_updates"] + updates,
                head_objective=total,
                head_gradient_norm=float(norm),
                label_stream_sha256=stream.hexdigest(),
                encoder_updates=0,
                reconstruction_presentations=0,
            )
            if updates % 50 == 0 or updates == budget["head_optimizer_updates"]:
                result = joint.evaluate(
                    model,
                    head,
                    data,
                    scoring_rows("tune"),
                    standardized,
                    sigma,
                    mask,
                    jacobian,
                    center,
                    scale,
                    device,
                )
                score, by_label = joint.selection_score(result)
                checkpoint = args.output / f"checkpoint-head-update-{updates:06d}.pth"
                snapshot = copy.deepcopy(state())
                joint.native.save_checkpoint(checkpoint, snapshot)
                actual = torch_load_trusted(
                    checkpoint, map_location="cpu", weights_only=False
                )
                joint.native._assert_checkpoint_equal(snapshot, actual)
                receipt = dict(
                    file=checkpoint.name,
                    sha256=joint.sha(checkpoint),
                    bytes=checkpoint.stat().st_size,
                    head_updates=updates,
                    head_adam_steps=budget["warmup_updates"] + updates,
                    reconstruction_presentations=0,
                    encoder_updates=0,
                    all_state_values_reload_equal=True,
                )
                joint.native.atomic(checkpoint.with_suffix(".json"), receipt)
                metric.update(
                    tune_macro_native_nll=score,
                    tune_nll_by_label=by_label,
                    checkpoint=receipt,
                )
                if score < best:
                    best, selected = score, receipt
                    np.savez_compressed(args.output / "tune-predictions.npz", **result)
                    joint.native.atomic(
                        args.output / "selected-checkpoint.json",
                        dict(
                            receipt=receipt,
                            selection="first minimum macro TUNE observed-coordinate native NLL",
                            score=score,
                            tune_source_id_sha256=hashlib.sha256(
                                ids[scoring_rows("tune")].tobytes()
                            ).hexdigest(),
                        ),
                    )
            metrics.append(metric)
            joint.native.atomic(args.output / "metrics.json", metrics)
            manifest.update(
                status="running",
                completed_head_updates=updates,
                completed_head_adam_steps=budget["warmup_updates"] + updates,
                label_stream_sha256=stream.hexdigest(),
                supervised_label_presentations=sampled_counts.tolist(),
            )
            joint.native.atomic(args.output / "manifest.json", manifest)
        if selected is None:
            raise ValueError("No finite disjoint TUNE-selected head")
        if frozen_parameter_digest(model) != parameter_sha or any(
            p.grad is not None for p in model.parameters()
        ):
            raise ValueError("Frozen model changed during head control")
        selected_path = args.output / selected["file"]
        if joint.sha(selected_path) != selected["sha256"]:
            raise ValueError("Selected frozen-head artifact changed")
        restored = torch_load_trusted(
            selected_path, map_location="cpu", weights_only=False
        )
        head.load_state_dict(restored["head_state_dict"], strict=True)
        joint.native.save_checkpoint(args.output / "checkpoint.pth", restored)
        joint.native._assert_checkpoint_equal(
            restored,
            torch_load_trusted(
                args.output / "checkpoint.pth", map_location="cpu", weights_only=False
            ),
        )
        outputs = {}
        for role in ("cal", "test"):
            outputs[role] = joint.evaluate(
                model,
                head,
                data,
                scoring_rows(role),
                standardized,
                sigma,
                mask,
                jacobian,
                center,
                scale,
                device,
                args.output / (role + "-predictions.npz"),
            )
        cp = joint.calibrate(outputs["cal"], outputs["test"], args.output)
        if frozen_parameter_digest(model) != parameter_sha:
            raise ValueError("Scoring mutated frozen model")
        manifest.update(
            status="complete",
            training_budget=budget,
            selected_head_counters=dict(
                head_updates=restored["completed_updates"],
                head_adam_steps=restored["completed_head_adam_steps"],
            ),
            selected_checkpoint=selected,
            checkpoint_sha256=joint.sha(args.output / "checkpoint.pth"),
            fit_embedding_cache_sha256=joint.sha(cache),
            fit_embedding_cache_bindings=bindings,
            frozen_model_parameter_sha256_after=parameter_sha,
            frozen_model_values_unchanged=True,
            full_state_reload_equal=True,
            scoring_precision="isolated FP64 encoder+head from raw inputs",
            calibration=cp,
            supervised_label_presentations=sampled_counts.tolist(),
            training_calls=budget["supervised_minibatch_calls"],
            no_10m_training_claim=True,
        )
        joint.native.atomic(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(
            status="failed", exception_type=type(error).__name__, error=str(error)
        )
        joint.native.atomic(args.output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/final-six-frozen-20261024.yaml")
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
