"""Four matched CANFAR-only large-sample fits under the five-job total cap.

Existing immutable150M original TRAIN cache is reused read-only for three
complete passes. Internal CUDA smoke/preflight, TUNE selection and all final
prediction/diagnostic artifacts happen inside the same GPU job.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import run_final_six_recipes as shared
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.distribution_diagnostics import (
    native_coordinates,
    predictive_draws,
)
from masked_stellar_autoencoder.training.final_six_recipes import (
    NAMES,
    normalization,
    standardized_arrays,
)
from masked_stellar_autoencoder.training.five_wave_heads import (
    QUANTILES,
    build_head,
    masked_native_median_score,
    modality_dropout,
    native_fit_point_scale,
    native_quantiles,
)
from masked_stellar_autoencoder.training.input_measurement import xp_indices
from masked_stellar_autoencoder.training.reconstruction_sample import (
    open_reconstruction_sample,
)
from masked_stellar_autoencoder.training.xp_experiments import (
    spectral_basis,
    spectral_relative_error,
)


def budget_contract(cfg):
    rows, epochs, presentations = (
        cfg["reconstruction_training_rows"],
        cfg["reconstruction_epochs"],
        cfg["reconstruction_presentations"],
    )
    if rows != 150000000 or epochs != 3 or presentations != rows * epochs:
        raise ValueError(
            "Exactly three complete passes of the existing150M sample required"
        )
    if (
        cfg["supervised_whole_xp_dropout"] != 0.25
        or cfg["reconstruction_weight"] != 0.03
    ):
        raise ValueError("Matched modality/reconstruction policy changed")
    return dict(
        distinct_original_train_rows=rows,
        epochs=epochs,
        presentations=presentations,
        expected_encoder_updates=epochs * math.ceil(rows / cfg["effective_batch"]),
    )


def availability(data, xp):
    """Model input availability is distinct from official Gaia availability."""
    present = np.asarray(data["visible"], bool)[:, xp].any(-1)
    real = np.asarray(data["real_xp_present"], bool)
    absent = np.asarray(data["real_no_xp"], bool)
    if (
        real.shape != present.shape
        or absent.shape != present.shape
        or (real & absent).any()
    ):
        raise ValueError("Conflicting official/natural XP evidence")
    if (absent & present).any():
        raise ValueError("Officially absent XP unexpectedly entered model inputs")
    if "input_has_xp" in data and not np.array_equal(present, data["input_has_xp"]):
        raise ValueError("Prepared model XP flag differs from actual input mask")
    return dict(input_has_xp=present, real_xp_present=real, real_no_xp=absent)


def load_inputs(cfg, source):
    direct = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
        raise ValueError("Installed immutable library pin differs")
    prep = json.loads(Path(cfg["preparation_manifest"]).read_text())
    expected_source = (
        source
        if cfg["preparation_source_sha"] == "same_as_runner_source"
        else cfg["preparation_source_sha"]
    )
    expected_status = cfg.get(
        "preparation_status", "five_job_source_preparation_complete"
    )
    if (
        prep["status"] != expected_status
        or prep["source_commit"] != expected_source
        or shared.sha(cfg["prepared"]) != prep["input_sha256"]
    ):
        raise ValueError("Broad source preparation identity or terminal status differs")
    policy = yaml.safe_load(Path(cfg["preparation_policy"]).read_text())
    if prep.get("source_provenance") != policy and prep.get(
        "policy_sha256"
    ) != shared.sha(cfg["preparation_policy"]):
        raise ValueError("Broad source preparation policy differs")
    data = dict(np.load(cfg["prepared"], allow_pickle=False))
    ids, roles = data["source_id"], data["role"]
    quality = np.asarray(data["quality"], bool) & np.asarray(data["observed"], bool)
    if not np.array_equal(quality, data["observed"] & data["original_role_eligible"]):
        raise ValueError("Source-role eligible masks changed")
    if (
        len(np.unique(ids)) != len(ids)
        or list(data["target_names"]) != list(NAMES)
        or set(roles.tolist()) != {"fit", "tune", "cal", "test"}
    ):
        raise ValueError(
            "Broad source identities, target order or role contract differs"
        )
    fit = np.flatnonzero(roles == "fit")
    center, scale = normalization(data["value"], quality, roles == "fit")
    y, sigma, mask, jacobian = standardized_arrays(
        data["value"],
        quality,
        data["error"],
        data["error_observed"] & quality,
        center,
        scale,
    )
    point_scale = native_fit_point_scale(data["value"], mask, fit)
    if (
        shared.sha(cfg["reference_checkpoint"]) != cfg["reference_checkpoint_sha256"]
        or shared.sha(cfg["checkpoint"]) != cfg["checkpoint_sha256"]
    ):
        raise ValueError("Reference or final large parent SHA changed")
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
        raise ValueError("Final4.5B large-model lineage/counters differ")
    if any(
        torch.is_floating_point(v) and not torch.isfinite(v).all()
        for v in weights["model_state_dict"].values()
    ):
        raise ValueError("Nonfinite parent model weights")
    signature = reference["run_signature"]
    if list(data["feature_names"]) != signature["feature_cols"]:
        raise ValueError("Broad source feature order differs from reference scaler")
    for key in ("scaler_center", "scaler_scale"):
        np.testing.assert_array_equal(data[key], signature[key])
    return (
        data,
        dict(reference, model_state_dict=weights["model_state_dict"]),
        prep,
        direct,
        center,
        scale,
        y,
        sigma,
        mask,
        jacobian,
        point_scale,
    )


def reuse_existing_cache(cfg, signature):
    if not Path(cfg["reconstruction_cache"]).exists():
        raise FileNotFoundError(
            "Existing150M cache missing; no restaging is authorized by this runner"
        )
    expected = dict(
        rows=cfg["reconstruction_training_rows"],
        features=signature["feature_cols"],
        center=np.asarray(signature["scaler_center"]).tolist(),
        scale=np.asarray(signature["scaler_scale"]).tolist(),
        train_keys=signature["train_keys"],
        validation_keys=cfg["reconstruction_validation_keys"],
        parent_sha256=cfg["reference_checkpoint_sha256"],
        seed=42,
    )
    return open_reconstruction_sample(
        cfg["reconstruction_cache"],
        expected_request_sha256=cfg["reconstruction_request_sha256"],
        expected_identity=expected,
    )


def paired_inputs(data, rows, xp, masked):
    values = np.asarray(data["input"][rows], dtype="f4").copy()
    if masked:
        values[:, xp] = -9999
    return values


def tuning(
    model, head, data, y, sigma, mask, jacobian, center, scale, point_scale, xp, device
):
    """One comparable point criterion for quantile and distribution heads."""
    model = copy.deepcopy(model).double().eval()
    head = copy.deepcopy(head).double().eval()
    candidates = np.flatnonzero(data["role"] == "tune")
    available = availability(data, xp)
    details = []
    with torch.no_grad():
        for mode, rows in [
            ("natural", candidates),
            (
                "xp_masked_ablation",
                candidates[
                    (available["input_has_xp"] & available["real_xp_present"])[
                        candidates
                    ]
                ],
            ),
        ]:
            pieces = []
            for start in range(0, len(rows), 128):
                rr = rows[start : start + 128]
                x = torch.tensor(
                    paired_inputs(data, rr, xp, mode != "natural"),
                    device=device,
                    dtype=torch.float64,
                )
                z = model.encoder(x)
                median = native_quantiles(
                    head,
                    z,
                    torch.tensor(center, device=device),
                    torch.tensor(scale, device=device),
                )[:, 1]
                pieces.append(median.cpu().numpy())
            prediction = np.concatenate(pieces)
            score, by_label = masked_native_median_score(
                prediction, data["value"][rows], mask[rows], point_scale
            )
            details.append(
                dict(mode=mode, score=score, by_label=by_label, count=len(rows))
            )
    return float(np.mean([v["score"] for v in details])), details


def internal_cuda_preflight(
    model, head, data, fit, y, sigma, mask, cache, cfg, xp, device, output
):
    """Two copy-only SGD updates in the SAME job; production states untouched."""
    model = copy.deepcopy(model).train()
    head = copy.deepcopy(head).train()
    mo = torch.optim.Adam(model.parameters(), lr=cfg["encoder_lr"])
    ho = torch.optim.Adam(head.parameters(), lr=cfg["head_lr"])
    rng = np.random.default_rng(918273)
    generator = torch.Generator(device=device).manual_seed(918273)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(cfg["internal_preflight_updates"]):
        raw = np.array(cache[step * 128 : (step + 1) * 128], copy=True)
        allowed = np.array(
            [
                n in {"W1", "W2", "G", "BP", "RP"}
                or n in {f"{b}_{j}" for b in ("bp", "rp") for j in range(1, 56)}
                for n in data["feature_names"]
            ]
        )
        raw[:, ~allowed] = np.nan
        target = torch.tensor(raw, device=device)
        hidden, hiddenmask, finite = shared.native.pretrain.training_mask(
            target, xp, 0.5, 0.2, generator
        )
        mo.zero_grad(set_to_none=True)
        ho.zero_grad(set_to_none=True)
        prediction, _ = model(hidden)
        width = prediction.shape[-1]
        score = (hiddenmask & finite)[:, :width]
        reco = shared.native.pretrain.experiment_objective(
            prediction, target[:, :width], score, xp, {"loss": "mae"}, score.sum()
        )
        rr = shared.draw_balanced_rows(rng, fit, mask)
        x = torch.tensor(data["input"][rr], device=device)
        x, _, _ = modality_dropout(
            x, torch.tensor(data["visible"][rr], device=device), xp, 0.25, generator
        )
        supervised = head.criterion(
            model.encoder(x),
            torch.tensor(y[rr], device=device, dtype=torch.float32),
            torch.tensor(sigma[rr], device=device, dtype=torch.float32),
            torch.tensor(mask[rr], device=device),
            torch.tensor(data["source_id"][rr].astype("i8"), device=device),
        )
        loss = cfg["reconstruction_weight"] * reco + supervised
        if not torch.isfinite(loss):
            raise FloatingPointError("Actual CUDA preflight objective nonfinite")
        loss.backward()
        mn = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1, error_if_nonfinite=True
        )
        hn = torch.nn.utils.clip_grad_norm_(
            head.parameters(), 1, error_if_nonfinite=True
        )
        mo.step()
        ho.step()
        records.append(
            dict(
                update=step + 1,
                presentations=len(raw),
                loss=float(loss),
                model_gradient_norm=float(mn),
                head_gradient_norm=float(hn),
            )
        )
    with torch.no_grad():
        z = model.encoder(torch.tensor(data["input"][fit[:128]], device=device))
        quantiles = native_quantiles(
            head, z, torch.zeros(6, device=device), torch.ones(6, device=device)
        )
        if (
            not torch.isfinite(quantiles).all()
            or not (quantiles[:, 1:] >= quantiles[:, :-1]).all()
        ):
            raise FloatingPointError("Native CUDA quantile/inverse preflight failed")
        if head.kind != "quantile3":
            mean, variance = head.native_moments(
                z, torch.zeros(6, device=device), torch.ones(6, device=device)
            )
            if (
                not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all()
                or (variance <= 0).any()
            ):
                raise FloatingPointError(
                    "Native CUDA fixed256point moment allocation preflight failed"
                )
    state = dict(
        model_state_dict=model.state_dict(),
        head_state_dict=head.state_dict(),
        model_optimizer_state_dict=mo.state_dict(),
        head_optimizer_state_dict=ho.state_dict(),
        rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all(),
    )
    path = output / "internal-preflight-copy-state.pth"
    shared.native.save_checkpoint(path, state)
    shared.native._assert_checkpoint_equal(
        state, torch_load_trusted(path, map_location="cpu", weights_only=False)
    )
    shared.native.atomic(
        output / "internal-preflight.json",
        dict(
            status="passed",
            copy_only_updates=cfg["internal_preflight_updates"],
            production_state_updates=0,
            actual_cuda=True,
            actual_native_labels=True,
            records=records,
            strict_state_reload_equal=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    del model, head, mo, ho
    gc.collect()
    torch.cuda.empty_cache()


def mode_metadata(data, rows, xp, mode):
    metadata = {k: np.asarray(v)[rows] for k, v in availability(data, xp).items()}
    metadata["evaluation_mode"] = np.full(len(rows), mode)
    if mode != "natural":
        metadata["input_has_xp"][:] = False
    for name in (
        "xp_status",
        "official_xp_unknown",
        "label_reference_kind",
        "andrae_prediction_native",
        "andrae_matched",
        "andrae_membership_known",
        "andrae_training_membership_known",
        "ancestor_feh_supervised_fit_exposed",
        "ancestor_150m_reconstruction_exposed",
        "andrae_in_training",
    ):
        if name in data:
            metadata[name] = data[name][rows]
    if "andrae_training_membership_known" in metadata:
        metadata["andrae_membership_known"] = metadata[
            "andrae_training_membership_known"
        ]
    return metadata


def load_evaluation_batch(path, source_id):
    receipt = path.with_suffix(".sha256.json")
    if path.exists() != receipt.exists():
        raise ValueError("Incomplete evaluation batch cache; retain diagnostic")
    if not path.exists():
        return None
    if json.loads(receipt.read_text())["sha256"] != shared.sha(path):
        raise ValueError("Changed evaluation batch bytes")
    part = dict(np.load(path, allow_pickle=False))
    np.testing.assert_array_equal(part.pop("_source_id"), source_id)
    return part


def evaluation(
    model,
    head,
    data,
    rows,
    y,
    sigma,
    mask,
    jacobian,
    center,
    scale,
    xp,
    device,
    mode,
    output,
    seed,
):
    """Final FP64 raw-input scoring; immutable cached arrays before assertions."""
    model = copy.deepcopy(model).double().eval()
    head = copy.deepcopy(head).double().eval()
    center_t, scale_t = (
        torch.tensor(center, device=device),
        torch.tensor(scale, device=device),
    )
    folder = output.parent / "evaluation-batches" / output.stem
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(
        selected_checkpoint_sha256=shared.sha(output.parent / "checkpoint.pth"),
        runner_sha256=shared.sha(__file__),
        head_module_sha256=shared.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/five_wave_heads.py"
        ),
        distribution_helper_sha256=shared.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/distribution_diagnostics.py"
        ),
        input_role_sha256=hashlib.sha256(
            np.ascontiguousarray(data["input"][rows]).tobytes()
        ).hexdigest(),
        target_sha256=hashlib.sha256(
            np.ascontiguousarray(data["value"][rows]).tobytes()
        ).hexdigest(),
        observed_sha256=hashlib.sha256(
            np.ascontiguousarray(mask[rows]).tobytes()
        ).hexdigest(),
        ordered_source_id_sha256=hashlib.sha256(
            data["source_id"][rows].tobytes()
        ).hexdigest(),
        mode=mode,
        seed=seed,
        precision="float64_encoder_and_head",
    )
    binding_path = folder / "bindings.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError(
            "Changed scoring/source binding; never reuse cached predictions"
        )
    shared.native.atomic(binding_path, binding)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(rows), 128):
            rr = rows[start : start + 128]
            cache = folder / f"batch-{start:07d}.npz"
            part = load_evaluation_batch(cache, data["source_id"][rr])
            if part is None:
                z = model.encoder(
                    torch.tensor(
                        paired_inputs(data, rr, xp, mode != "natural"),
                        device=device,
                        dtype=torch.float64,
                    )
                )
                q = native_quantiles(head, z, center_t, scale_t)
                part = dict(
                    point_native=q[:, 1].cpu().numpy(), quantiles_native=q.cpu().numpy()
                )
                if head.kind != "quantile3":
                    from run_final_six_distribution_metrics import calculate_batch

                    mean, var = head.native_moments(z, center_t, scale_t)
                    part.update(
                        mean_native=mean.cpu().numpy(),
                        variance_native=var.cpu().numpy(),
                    )
                    targets = torch.tensor(y[rr], device=device)
                    errors = torch.tensor(sigma[rr], device=device)
                    observed = torch.tensor(mask[rr], device=device)
                    intrinsic = []
                    observed_density = []
                    for column in range(6):
                        single = torch.zeros_like(observed)
                        single[:, column] = True
                        intrinsic.append(
                            head.row_log_prob(
                                z, targets, errors, single, observed=False
                            )
                            .cpu()
                            .numpy()
                            - jacobian[rr, column]
                        )
                        observed_density.append(
                            head.row_log_prob(z, targets, errors, single).cpu().numpy()
                            - jacobian[rr, column]
                        )
                    part["intrinsic_log_prob_native"] = np.where(
                        mask[rr], np.stack(intrinsic, -1), np.nan
                    )
                    part["observed_log_prob_native"] = np.where(
                        mask[rr], np.stack(observed_density, -1), np.nan
                    )
                    diagnostics, draws = calculate_batch(
                        head,
                        z,
                        targets,
                        errors,
                        torch.tensor(data["value"][rr], device=device),
                        observed,
                        center_t,
                        scale_t,
                        seed + int(start) * 31 + (1000001 if mode != "natural" else 0),
                    )
                    part.update(
                        {key: value.cpu().numpy() for key, value in diagnostics.items()}
                    )
                    part["observed_quantiles_native"] = np.stack(
                        [
                            part[f"observed_quantile_{level:g}_native"]
                            for level in QUANTILES
                        ],
                        1,
                    )
                    part["pit_observed_valid"] = mask[rr] & data["error_observed"][rr]
                    if start == 0:
                        pilot = folder / "bounded-metric-pilot.npz"
                        count = min(32, len(rr))
                        arrays = dict(
                            source_id=data["source_id"][rr[:count]],
                            target_native=data["value"][rr[:count]],
                            mask=mask[rr[:count]],
                            error_known=data["error_observed"][rr[:count]],
                            center=center,
                            scale=scale,
                        )
                        arrays.update(
                            {
                                f"{law}_replica{rep}": values[:, :count].cpu().numpy()
                                for law, members in draws.items()
                                for rep, values in enumerate(members)
                            }
                        )
                        arrays.update(
                            {
                                "score_" + key: value[:count]
                                for key, value in part.items()
                            }
                        )
                        np.savez_compressed(pilot, **arrays)
                        shared.native.atomic(
                            pilot.with_suffix(".sha256.json"),
                            dict(
                                sha256=shared.sha(pilot),
                                replica_draw_count=512,
                                pilot_count=count,
                                model_unchanged=True,
                            ),
                        )
                np.savez_compressed(cache, _source_id=data["source_id"][rr], **part)
                shared.native.atomic(
                    cache.with_suffix(".sha256.json"),
                    dict(sha256=shared.sha(cache), count=len(rr)),
                )
            # Every computed batch is preserved BEFORE validation, including failures.
            if (
                not np.isfinite(part["point_native"]).all()
                or not np.isfinite(part["quantiles_native"]).all()
                or (
                    part["quantiles_native"][:, 1:] < part["quantiles_native"][:, :-1]
                ).any()
            ):
                raise FloatingPointError(
                    "Invalid actual predictive quantiles/points; batch retained"
                )
            if (part["quantiles_native"][..., 4:] <= 0).any():
                raise FloatingPointError(
                    "Nonpositive actual age/distance quantiles; batch retained"
                )
            pieces.append(part)
            shared.native.atomic(
                output.with_suffix(".progress.json"),
                dict(
                    status="running",
                    mode=mode,
                    processed=start + len(rr),
                    total=len(rows),
                    density_available=head.kind != "quantile3",
                ),
            )
    result = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
    result.update(
        source_id=data["source_id"][rows],
        target_native=data["value"][rows],
        observed=mask[rows],
        label_source=data["label_source"][rows],
        error=data["error"][rows],
        error_observed=data["error_observed"][rows] & mask[rows],
        quantile_levels=np.array(QUANTILES),
        **mode_metadata(data, rows, xp, mode),
    )
    np.savez_compressed(output, **result)
    shared.native.atomic(
        output.with_suffix(".sha256.json"),
        dict(
            sha256=shared.sha(output),
            count=len(rows),
            mode=mode,
            source_id_sha256=hashlib.sha256(
                data["source_id"][rows].tobytes()
            ).hexdigest(),
        ),
    )
    return result


def sample_corners(
    model, head, data, rows, mask, center, scale, xp, device, seed, output
):
    if head.kind == "quantile3":
        return None
    model = copy.deepcopy(model).double().eval()
    head = copy.deepcopy(head).double().eval()
    center_t, scale_t = (
        torch.tensor(center, device=device),
        torch.tensor(scale, device=device),
    )
    rr = np.asarray(data["corner_rows"], dtype="i8")
    if (
        rr.ndim != 1
        or len(rr) > 8
        or len(np.unique(rr)) != len(rr)
        or np.any(rr < 0)
        or np.any(rr >= len(data["source_id"]))
        or not (data["role"][rr] == "test").all()
    ):
        raise ValueError("Predeclared corner rows must be unique eligible TEST rows")
    np.testing.assert_array_equal(data["corner_source_id"], data["source_id"][rr])
    limits = np.asarray(data["corner_plot_limits"], dtype="f8")
    if (
        limits.shape != (len(rr), 6, 2)
        or not np.isfinite(limits).all()
        or not (limits[..., 1] > limits[..., 0]).all()
    ):
        raise ValueError("Predeclared common corner axes differ")
    if not len(rr):
        return None
    with torch.no_grad():
        z = model.encoder(
            torch.tensor(data["input"][rr], device=device, dtype=torch.float64)
        )
        generator = torch.Generator(device=device).manual_seed(seed + 170271)
        standardized = predictive_draws(head, z, 8192, generator)
        native = (
            native_coordinates(standardized, center_t, scale_t)
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
        )
    plot = native.copy()
    plot[:, :, 4:] = np.log(plot[:, :, 4:])
    corners = dict(
        source_id=data["source_id"][rr],
        samples_native=native,
        samples_plot_coordinates=plot,
        target_native=data["value"][rr],
        observed=mask[rr],
        error=data["error"][rr],
        error_observed=data["error_observed"][rr] & mask[rr],
        label_source=data["label_source"][rr],
        plot_limits=limits.copy(),
    )
    np.savez_compressed(output, **corners)
    return corners


def run(args):
    if (
        not torch.cuda.is_available()
        or os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
    ):
        raise RuntimeError("All native model execution requires CANFAR headless CUDA")
    cfg = yaml.safe_load(args.config.read_text())
    budget = budget_contract(cfg)
    arm = next(a for a in cfg["arms"] if a["run_id"] == args.run_id)
    if args.output.exists():
        raise FileExistsError("Never reuse an attempted training output")
    args.output.mkdir(parents=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(
        status="preparing",
        run_id=args.run_id,
        source_sha=source,
        policy=cfg,
        arm=arm,
        training_budget=budget,
        hard_total_wave_job_cap=5,
        density_available=arm["kind"] != "quantile3",
        point_functional="intrinsic conditional median",
        physical_truth_claim=False,
        independent_verification=False,
    )
    shared.native.atomic(args.output / "manifest.json", manifest)
    try:
        (
            data,
            disk,
            prep,
            direct,
            center,
            scale,
            y,
            sigma,
            mask,
            jacobian,
            point_scale,
        ) = load_inputs(cfg, source)
        ids, roles = data["source_id"], data["role"]
        fit = np.flatnonzero(roles == "fit")
        features = disk["run_signature"]["feature_cols"]
        xp = xp_indices(features).tolist()
        available = availability(data, xp)
        model_cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        width = len(model_cfg["data"]["recon_cols"])
        if features[:width] != model_cfg["data"]["recon_cols"]:
            raise ValueError("Reconstruction feature order changed")
        cache, cache_ids, provenance, cachemeta = reuse_existing_cache(
            cfg, disk["run_signature"]
        )
        if len(cache) != 150000000:
            raise ValueError("Reconstruction cache is not150M distinct objects")
        shared.native.atomic(
            args.output / "reconstruction-source-binding.json",
            dict(
                existing_readonly_cache=True,
                request_sha256=cachemeta["request_sha256"],
                manifest_sha256=shared.sha(
                    Path(cfg["reconstruction_cache"]) / "manifest.json"
                ),
                files=cachemeta["files"],
                sample_source_ids_unique=cachemeta["sample_source_ids_unique"],
                original_validation_ids_excluded=cachemeta[
                    "original_validation_ids_excluded"
                ],
                model_parent_sha256=cfg["checkpoint_sha256"],
                cache_reference_parent_sha256=cfg["reference_checkpoint_sha256"],
            ),
        )
        device = torch.device("cuda")
        torch.set_num_threads(4)
        seed = arm["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = shared.native.probe.load_model(model_cfg, disk, device)
        head = build_head(arm, model_cfg["model"]["layer_dims"][-1]).to(device)
        rng = np.random.default_rng(seed)
        mask_rng = torch.Generator(device=device).manual_seed(seed)
        dropout_rng = torch.Generator(device=device).manual_seed(seed + 1701)
        cuda_state = torch.cuda.get_rng_state_all()
        torch_state = torch.get_rng_state()
        internal_cuda_preflight(
            model, head, data, fit, y, sigma, mask, cache, cfg, xp, device, args.output
        )
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["encoder_lr"])
        head_optimizer = torch.optim.Adam(head.parameters(), lr=cfg["head_lr"])
        row_hash, mask_hash, label_hash, dropout_hash = [
            hashlib.sha256() for _ in range(4)
        ]
        label_counts = np.zeros(6, dtype="i8")
        dropout_counts = np.zeros(3, dtype="i8")
        allowed = np.array(
            [
                name in {"W1", "W2", "G", "BP", "RP"}
                or name
                in {f"{band}_{j}" for band in ("bp", "rp") for j in range(1, 56)}
                for name in features
            ]
        )
        spectral = None
        if cfg.get("spectral_weight", 0):
            projection = spectral_basis(features[:width])
            spectral = dict(
                scale=torch.tensor(
                    data["scaler_scale"][:width], device=device, dtype=torch.float32
                ),
                center=torch.tensor(
                    data["scaler_center"][:width], device=device, dtype=torch.float32
                ),
                basis=torch.tensor(
                    projection["basis"], device=device, dtype=torch.float32
                ),
                wavelength=torch.tensor(
                    projection["wavelength_nm"], device=device, dtype=torch.float32
                ),
            )
        manifest.update(
            status="running",
            input_sha256=prep["input_sha256"],
            preparation_manifest_sha256=shared.sha(cfg["preparation_manifest"]),
            library_revision=cfg["library_revision"],
            library_source=direct,
            parent_sha256=cfg["checkpoint_sha256"],
            gpu=torch.cuda.get_device_name(),
            point_scale_fit_native=point_scale.tolist(),
            role_counts={
                r: int((roles == r).sum()) for r in ("fit", "tune", "cal", "test")
            },
            modality_counts={k: int(v.sum()) for k, v in available.items()},
            internal_preflight_copy_only=True,
        )
        shared.native.atomic(args.output / "manifest.json", manifest)
        np.savez_compressed(
            args.output / "normalization.npz",
            center=center,
            scale=scale,
            point_scale_fit_native=point_scale,
            target_names=np.array(NAMES),
        )

        def supervised(warmup=False):
            rr = shared.draw_balanced_rows(rng, fit, mask)
            label_hash.update(np.asarray(rr, dtype="<i8").tobytes())
            label_counts[:] += mask[rr].sum(0)
            x = torch.tensor(data["input"][rr], device=device)
            x, _, flags = modality_dropout(
                x,
                torch.tensor(data["visible"][rr], device=device),
                xp,
                cfg["supervised_whole_xp_dropout"],
                dropout_rng,
            )
            dropout_hash.update(flags["xp_dropped"].cpu().numpy().tobytes())
            dropout_counts[:] += np.array(
                [
                    int(flags[k].sum())
                    for k in ("input_xp_before", "input_xp_after", "xp_dropped")
                ]
            )
            if warmup:
                with torch.no_grad():
                    z = model.encoder(x)
            else:
                z = model.encoder(x)
            return head.criterion(
                z,
                torch.tensor(y[rr], device=device, dtype=torch.float32),
                torch.tensor(sigma[rr], device=device, dtype=torch.float32),
                torch.tensor(mask[rr], device=device),
                torch.tensor(ids[rr].astype("i8"), device=device),
            )

        for step in range(cfg["warmup_updates"]):
            torch.manual_seed(100000 + seed + step)
            model.eval()
            head.train()
            head_optimizer.zero_grad(set_to_none=True)
            loss = supervised(True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Headwarmup objective nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1, error_if_nonfinite=True
            )
            head_optimizer.step()
        seen = updates = 0
        selected = None
        best = float("inf")
        metrics = []

        def state():
            return dict(
                model_state_dict=model.state_dict(),
                head_state_dict=head.state_dict(),
                model_optimizer_state_dict=optimizer.state_dict(),
                head_optimizer_state_dict=head_optimizer.state_dict(),
                rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state_all(),
                numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),
                supervised_rng_state=rng.bit_generator.state,
                mask_rng_state=mask_rng.get_state(),
                modality_dropout_rng_state=dropout_rng.get_state(),
                label_center=center,
                label_scale=scale,
                point_scale_fit_native=point_scale,
                target_names=NAMES,
                feature_columns=features,
                completed_presentations=seen,
                completed_updates=updates,
                warmup_updates=cfg["warmup_updates"],
                arm=arm,
                policy=cfg,
                source_sha=source,
                parent_sha256=cfg["checkpoint_sha256"],
                input_sha256=prep["input_sha256"],
                run_signature=disk["run_signature"],
                label_stream_sha256=label_hash.hexdigest(),
                reconstruction_stream_sha256=row_hash.hexdigest(),
                mask_stream_sha256=mask_hash.hexdigest(),
                modality_dropout_stream_sha256=dropout_hash.hexdigest(),
            )

        for positions, raw in shared.replay_batches(
            cache, cfg["reconstruction_presentations"], cfg["effective_batch"]
        ):
            torch.manual_seed(1000 + seed + updates)
            row_hash.update(np.asarray(positions, dtype="<i8").tobytes())
            raw[:, ~allowed] = np.nan
            target = torch.tensor(raw, device=device)
            hidden, hiddenmask, finite = shared.native.pretrain.training_mask(
                target, xp, 0.5, 0.2, mask_rng
            )
            mask_hash.update(hiddenmask.cpu().numpy().tobytes())
            score = (hiddenmask & finite)[:, :width]
            optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            model.train()
            head.train()
            reco_total = spectral_total = supervised_total = 0.0
            for start in range(0, len(raw), cfg["micro_batch"]):
                stop = start + cfg["micro_batch"]
                prediction, _ = model(hidden[start:stop])
                reco = shared.native.pretrain.experiment_objective(
                    prediction,
                    target[start:stop, :width],
                    score[start:stop],
                    xp,
                    {"loss": "mae"},
                    score.sum(),
                )
                objective = cfg["reconstruction_weight"] * reco
                reco_total += float(reco.detach())
                if spectral is not None:
                    fraction, _ = spectral_relative_error(
                        prediction,
                        target[start:stop, :width],
                        score[start:stop],
                        xp,
                        **spectral,
                    )
                    extra = fraction.sum() / len(raw)
                    objective = objective + cfg["spectral_weight"] * extra
                    spectral_total += float(extra.detach())
                objective.backward()
            for _ in range(cfg["supervised_batches_per_update"]):
                loss = supervised() / cfg["supervised_batches_per_update"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Supervised density/pinball objective nonfinite"
                    )
                loss.backward()
                supervised_total += float(loss.detach())
            mn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            hn = torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            head_optimizer.step()
            before = seen
            seen += len(raw)
            updates += 1
            record = dict(
                presentations=seen,
                updates=updates,
                original_sample_pass=math.ceil(seen / len(cache)),
                reconstruction_mae=reco_total,
                spectral_fraction=spectral_total,
                supervised_loss=supervised_total,
                model_gradient_norm=float(mn),
                head_gradient_norm=float(hn),
                label_stream_sha256=label_hash.hexdigest(),
                modality_dropout_stream_sha256=dropout_hash.hexdigest(),
            )
            if any(before < point <= seen for point in cfg["tuning_presentations"]):
                score_tune, details = tuning(
                    model,
                    head,
                    data,
                    y,
                    sigma,
                    mask,
                    jacobian,
                    center,
                    scale,
                    point_scale,
                    xp,
                    device,
                )
                receipt = shared.native.save_periodic_checkpoint(args.output, state())
                record.update(
                    tune_macro_native_scaled_median_mae=score_tune,
                    tune_details=details,
                    checkpoint=receipt,
                )
                if score_tune < best:
                    best, selected = score_tune, receipt
                    shared.native.atomic(
                        args.output / "selected-checkpoint.json",
                        dict(
                            receipt=receipt,
                            selection=cfg["selection_policy"],
                            score=best,
                            tune_details=details,
                        ),
                    )
            metrics.append(record)
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            manifest.update(
                status="running",
                completed_presentations=seen,
                completed_updates=updates,
                supervised_label_presentations=label_counts.tolist(),
                modality_dropout_counts=dict(
                    zip(
                        ("xp_before", "xp_after", "xp_dropped"),
                        dropout_counts.tolist(),
                        strict=True,
                    )
                ),
            )
            if updates % 25 == 0 or seen == budget["presentations"]:
                shared.native.atomic(args.output / "manifest.json", manifest)
        if (
            seen != budget["presentations"]
            or updates != budget["expected_encoder_updates"]
            or selected is None
        ):
            raise ValueError("Final exact three-pass budget differs")
        restored = torch_load_trusted(
            args.output / selected["file"], map_location="cpu", weights_only=False
        )
        model.load_state_dict(restored["model_state_dict"], strict=True)
        head.load_state_dict(restored["head_state_dict"], strict=True)
        shared.native.save_checkpoint(args.output / "checkpoint.pth", restored)
        shared.native._assert_checkpoint_equal(
            restored,
            torch_load_trusted(
                args.output / "checkpoint.pth", map_location="cpu", weights_only=False
            ),
        )
        evals = {}
        for role in ("cal", "test"):
            candidates = np.flatnonzero(roles == role)
            for mode, rr in [
                ("natural", candidates),
                (
                    "xp_masked_ablation",
                    candidates[
                        (available["input_has_xp"] & available["real_xp_present"])[
                            candidates
                        ]
                    ],
                ),
            ]:
                evals[role + ":" + mode] = evaluation(
                    model,
                    head,
                    data,
                    rr,
                    y,
                    sigma,
                    mask,
                    jacobian,
                    center,
                    scale,
                    xp,
                    device,
                    mode,
                    args.output / f"{role}-{mode}-predictions.npz",
                    seed,
                )
        tests = [evals["test:" + mode] for mode in ("natural", "xp_masked_ablation")]
        predictions = {
            key: (
                tests[0][key]
                if key == "quantile_levels"
                else np.concatenate([part[key] for part in tests])
            )
            for key in tests[0]
        }
        np.savez_compressed(args.output / "test-all-modalities.npz", **predictions)
        corners = sample_corners(
            model,
            head,
            data,
            np.flatnonzero(roles == "test"),
            mask,
            center,
            scale,
            xp,
            device,
            seed,
            args.output / "corner-samples.npz",
        )
        from five_wave_report import make_report

        make_report(
            predictions,
            args.output / "report",
            run_id=args.run_id,
            head_kind=head.kind,
            corners=corners,
            metadata=dict(
                point_functional="intrinsic conditional median",
                quantile_baseline_pdf_supported=False,
                pit_available=head.kind != "quantile3",
                corner_density_available=head.kind != "quantile3",
                model_input_dropout_training=0.25,
                source_manifest_sha256=shared.sha(cfg["preparation_manifest"]),
                input_sha256=prep["input_sha256"],
                state_checkpoint_sha256=shared.sha(args.output / "checkpoint.pth"),
                physical_truth_claim=False,
            ),
        )
        shared.native.atomic(args.output / "metrics.json", metrics)
        manifest.update(
            status="complete",
            training_budget=budget,
            completed_presentations=seen,
            completed_updates=updates,
            selected_checkpoint=selected,
            selected_model_counters=dict(
                presentations=restored["completed_presentations"],
                updates=restored["completed_updates"],
            ),
            checkpoint_sha256=shared.sha(args.output / "checkpoint.pth"),
            label_stream_sha256=label_hash.hexdigest(),
            reconstruction_stream_sha256=row_hash.hexdigest(),
            mask_stream_sha256=mask_hash.hexdigest(),
            modality_dropout_stream_sha256=dropout_hash.hexdigest(),
            full_state_reload_equal=True,
            scoring_precision="isolated FP64 rawencoder+head",
            report_status="producer plots require visual independent review",
            no_local_model_execution=True,
        )
        shared.native.atomic(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        shared.native.atomic(args.output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/five-wave-training-20261024.yaml")
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Internal copy-only preflight is always executed inside the same GPU job; does not create another job",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
