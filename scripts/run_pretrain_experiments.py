"""Bounded, paired continuation experiments from a trusted full MSA checkpoint.

The sample is a sequential prefix from every configured shard, not a random
sample of the catalogue. Run with ``pixi run python scripts/run_pretrain_experiments.py``.
"""

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import shutil
import subprocess
import time
from pathlib import Path
from types import MethodType

import h5py
import numpy as np
import torch
import yaml
from rtdl_num_embeddings import PeriodicEmbeddings
from sklearn.preprocessing import RobustScaler

from masked_stellar_autoencoder.models.blocks import MaskedPeriodicEmbeddings
from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.models.model import (
    TabResnetWrapper,
    _atomic_torch_save,
    _capture_rng_state,
    _restore_rng_state,
    make_model,
)
from masked_stellar_autoencoder.training.config_paths import expand_config_paths
from masked_stellar_autoencoder.training.hdf5_io import _clean_column
from masked_stellar_autoencoder.training.pretrain_msa import _validate_error_columns


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def feature_groups(names):
    xp = [i for i, name in enumerate(names) if name.startswith(("bp_", "rp_"))]
    astro = [
        i
        for i, name in enumerate(names)
        if name in {"PARALLAX", "EBV", "pmra", "pmdec", "RA", "DEC"}
    ]
    photo = [i for i in range(len(names)) if i not in xp and i not in astro]
    return {
        name: indices
        for name, indices in {"xp": xp, "photo": photo, "astro": astro}.items()
        if indices
    }


def availability(cache, names):
    xp = [i for i, name in enumerate(names) if name.startswith(("bp_", "rp_"))]
    counts = np.zeros(len(names), dtype=np.int64)
    xp_all = xp_none = xp_partial = 0
    for start in range(0, len(cache), 65536):
        finite = np.isfinite(cache[start : start + 65536])
        counts += finite.sum(axis=0)
        if xp:
            present = finite[:, xp].sum(axis=1)
            xp_all += int((present == len(xp)).sum())
            xp_none += int((present == 0).sum())
            xp_partial += int(((present > 0) & (present < len(xp))).sum())
    return {
        "feature_names": names,
        "finite_feature_counts": counts.tolist(),
        "xp_feature_count": len(xp),
        "xp_all_finite_rows": xp_all,
        "xp_none_finite_rows": xp_none,
        "xp_partial_finite_rows": xp_partial,
    }


def latent_summary(latent):
    centered = latent.astype(np.float64) - latent.mean(axis=0)
    std = centered.std(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / max(len(latent), 1)).clip(
        min=0
    )
    total = eigenvalues.sum()
    probability = eigenvalues[eigenvalues > 0] / total if total > 0 else np.zeros(0)
    return {
        "dimension_std": std.tolist(),
        "std_min": float(std.min()),
        "std_median": float(np.median(std)),
        "std_max": float(std.max()),
        "near_constant_dimensions": int((std < 1e-6).sum()),
        "effective_rank": float(np.exp(-np.sum(probability * np.log(probability))))
        if total > 0
        else 0.0,
        "effective_rank_definition": "exp(entropy of centered latent covariance eigenvalue fractions)",
    }


def balanced_mae(prediction, target, mask, groups, counts):
    """Equal group weights, with denominators from the full optimizer batch."""
    errors = (prediction - target).masked_fill(~mask, 0).abs()
    active = torch.stack([count > 0 for count in counts]).sum().clamp_min(1)
    return (
        sum(
            errors[:, indices].sum() / count.clamp_min(1)
            for indices, count in zip(groups.values(), counts, strict=True)
        )
        / active
    )


def validate_source(checkpoint, config, source):
    signature = checkpoint.get("run_signature")
    if not signature:
        raise ValueError("Experiments require a full checkpoint with a run_signature")
    for name in ("feature_cols", "recon_cols"):
        if signature[name] != config["data"][name]:
            raise ValueError(f"Checkpoint {name} differs from experiment config")
    for name in ("optimizer_state_dict", "scheduler_state_dict", "rng_state"):
        if name not in checkpoint:
            raise ValueError(f"Checkpoint lacks reloadable full training state: {name}")
    train_keys = list(signature["train_keys"])
    validation_keys = list(config["data"]["valid_keys"])
    if set(train_keys) != set(source) - set(validation_keys):
        raise ValueError("Checkpoint training shards differ from source/config holdout")
    for key in [*train_keys, *validation_keys]:
        if key not in source:
            raise ValueError(f"Source shard absent: {key}")
        saved_count = (signature.get("train_rows_by_key") or {}).get(key, 0)
        if len(source[key]) < saved_count:
            raise ValueError(f"Source shard shorter than checkpoint coverage: {key}")
    features = config["data"]["feature_cols"]
    errors = config["data"]["error_cols"]
    _validate_error_columns(features, errors)
    for feature, error in zip(features, errors, strict=True):
        if feature in {"G", "BP", "RP"} and error not in {None, f"e_{feature}"}:
            raise ValueError(
                f"{feature} requires a magnitude sigma e_{feature}; raw flux sigma has incompatible units"
            )
    center = np.asarray(signature["scaler_center"], dtype=np.float64)
    scale = np.asarray(signature["scaler_scale"], dtype=np.float64)
    if (
        center.shape != (len(features),)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Checkpoint scaler is invalid")
    return train_keys, validation_keys, center, scale


def read_prefixes(source, keys, features, errors, max_rows, center, scale, destination):
    """Stage only bounded sequential prefixes, never scan the full catalogue."""
    per_key = math.ceil(max_rows / len(keys))
    counts = {}
    remaining = max_rows
    for key in keys:
        counts[key] = min(len(source[key]), per_key, remaining)
        remaining -= counts[key]
    count = sum(counts.values())
    x = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(count, len(features))
    )
    sigma = np.full(x.shape, np.nan, dtype=np.float32) if errors is not None else None
    fields = list(dict.fromkeys([*features, *(name for name in errors or [] if name)]))
    offset = 0
    for key in keys:
        missing = set(fields) - set(source[key].dtype.names or ())
        if missing:
            raise ValueError(f"Missing source fields in {key}: {sorted(missing)}")
        for start in range(0, counts[key], 65536):
            stop = min(start + 65536, counts[key])
            records = source[key].fields(fields)[start:stop]
            values = np.column_stack(
                [_clean_column(name, records[name]) for name in features]
            )
            values[~np.isfinite(values)] = np.nan
            x[offset : offset + len(values)] = (values - center) / scale
            if sigma is not None:
                for column, error in enumerate(errors):
                    if error is not None:
                        raw = _clean_column(error, records[error])
                        sigma[offset : offset + len(values), column] = np.where(
                            np.isfinite(raw) & (raw > 0), raw / scale[column], np.nan
                        )
            offset += len(values)
        print(f"Cached prefix {key}: {counts[key]} rows", flush=True)
    x.flush()
    return x, sigma, counts


def summarize(residual, target, median, sigma=None):
    valid = np.isfinite(residual) & np.isfinite(target)
    values = residual[valid].astype(np.float64)
    observed = target[valid].astype(np.float64)
    baseline = np.broadcast_to(median, target.shape)[valid]
    result = {"count": int(values.size), "finite": bool(np.isfinite(values).all())}
    if not len(values):
        return {
            **result,
            **dict.fromkeys(
                (
                    "mae",
                    "bias",
                    "rmse",
                    "p50",
                    "p84",
                    "p95",
                    "zero_mae",
                    "median_mae",
                    "skill_zero",
                    "skill_median",
                )
            ),
        }
    absolute = abs(values)
    mae, zero, med = (
        float(absolute.mean()),
        float(abs(observed).mean()),
        float(abs(observed - baseline).mean()),
    )
    result.update(
        mae=mae,
        bias=float(values.mean()),
        rmse=float(np.sqrt(np.mean(values * values))),
        p50=float(np.quantile(absolute, 0.5)),
        p84=float(np.quantile(absolute, 0.84)),
        p95=float(np.quantile(absolute, 0.95)),
        zero_mae=zero,
        median_mae=med,
        skill_zero=1 - mae / zero if zero > 0 else None,
        skill_median=1 - mae / med if med > 0 else None,
    )
    if sigma is not None:
        sigma_valid = valid & np.isfinite(sigma) & (sigma > 0)
        normalized = residual[sigma_valid].astype(np.float64) / sigma[sigma_valid]
        result["uncertainty_normalized"] = {
            "count": int(len(normalized)),
            "bias": float(normalized.mean()) if len(normalized) else None,
            "rmse": float(np.sqrt(np.mean(normalized * normalized)))
            if len(normalized)
            else None,
            "p50": float(np.quantile(abs(normalized), 0.5))
            if len(normalized)
            else None,
            "p95": float(np.quantile(abs(normalized), 0.95))
            if len(normalized)
            else None,
        }
    return result


def fixed_masks(x, features, seed, xp_ratio, photo_ratio):
    rng = np.random.default_rng(seed)
    xp = [i for i, name in enumerate(features) if name.startswith(("bp_", "rp_"))]
    common = rng.random(x.shape) < photo_ratio
    common[:, xp] = False
    rows = rng.permutation(len(x))[: int(len(x) * xp_ratio)]
    common[np.ix_(rows, xp)] = True
    xp_on = common.copy()
    xp_on[:, xp] = False
    xp_off = common.copy()
    xp_off[:, xp] = True
    return {"common": common, "xp_on": xp_on, "xp_off": xp_off}


def training_mask(target, xp_indices, xp_ratio, photo_ratio, generator):
    """Mask RNG is independent of dropout so all paired arms share mask draws."""
    mask = (
        torch.rand(target.shape, device=target.device, generator=generator)
        < photo_ratio
    )
    mask[:, xp_indices] = False
    rows = torch.randperm(len(target), device=target.device, generator=generator)[
        : int(len(target) * xp_ratio)
    ]
    mask[rows[:, None], xp_indices] = True
    finite = torch.isfinite(target)
    return target.masked_fill(mask | ~finite, -9999), mask, finite


def evaluate(
    wrapper,
    validation,
    sigma,
    masks,
    median,
    center,
    scale,
    output,
    validation_id,
    rows_seen,
    verified_sigma=(),
):
    model, device = wrapper.model, wrapper.device
    names = wrapper.recon_cols
    width = len(names)
    target = validation[:, :width]
    sigmas = sigma[:, :width].copy()
    sigmas[:, [i for i, name in enumerate(names) if name not in verified_sigma]] = (
        np.nan
    )
    groups = feature_groups(names)
    for lower, upper in ((1, 10), (11, 30), (31, 55)):
        groups[f"xp_{lower}_{upper}"] = [
            i
            for i, name in enumerate(names)
            if name.startswith(("bp_", "rp_"))
            and lower <= int(name.split("_")[1]) <= upper
        ]
    qa = {
        "validation_id": validation_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_seen_total": rows_seen,
        "units": "saved RobustScaler units",
        "uncertainty_units": "verified only for named features"
        if verified_sigma
        else "unverified",
        "verified_sigma_features": list(verified_sigma),
        "score_policy": "Only artificially held-out finite targets; XP targets scored only when hidden",
        "snr_units": "verified"
        if "PARALLAX" in verified_sigma
        else "unverified; source e_parallax ratio is provisional",
        "regimes": {},
    }
    snapshots = {
        "targets": target,
        "sigma": sigmas,
        "source_sigma": sigma[:, :width],
        "feature_names": np.asarray(names),
    }
    prior = _capture_rng_state()
    model.eval()
    with torch.no_grad():
        for regime, mask in masks.items():
            predictions = []
            latents = []
            for start in range(0, len(validation), wrapper._pretrain_micro_batch_size):
                batch = torch.as_tensor(
                    np.array(
                        validation[start : start + wrapper._pretrain_micro_batch_size]
                    ),
                    device=device,
                )
                hidden = torch.as_tensor(
                    mask[start : start + len(batch)], device=device
                ) | ~torch.isfinite(batch)
                prediction, latent = model(batch.masked_fill(hidden, -9999))
                predictions.append(prediction.cpu().numpy())
                if regime == "common":
                    latents.append(latent.cpu().numpy())
            prediction = np.concatenate(predictions)
            scored = mask[:, :width] & np.isfinite(target)
            residual = np.where(scored, prediction - target, np.nan)
            finite_prediction = bool(np.isfinite(prediction).all())
            if not finite_prediction:
                raise FloatingPointError(
                    f"Nonfinite validation predictions in {regime}"
                )
            if latents:
                latent = np.concatenate(latents)
                if not np.isfinite(latent).all():
                    raise FloatingPointError("Nonfinite validation latent values")
                qa["latent"] = latent_summary(latent)
            overall = summarize(residual, target, median[:width], sigmas)
            overall["masked_mae"] = (
                float(abs(residual[scored]).mean()) if scored.any() else None
            )
            overall["prediction_finite"] = finite_prediction
            features = []
            for i, name in enumerate(names):
                summary = summarize(
                    residual[:, i], target[:, i], median[i], sigmas[:, i]
                )
                summary["name"] = name
                summary["uncertainty_units"] = (
                    "verified" if name in verified_sigma else "unverified"
                )
                features.append(summary)
            blocks = {
                name: summarize(
                    residual[:, indices],
                    target[:, indices],
                    median[indices],
                    sigmas[:, indices],
                )
                for name, indices in groups.items()
                if indices
            }
            primary_groups = [
                blocks[name]
                for name in ("xp", "photo", "astro")
                if name in blocks and blocks[name]["count"]
            ]
            overall["group_mae"] = (
                float(np.mean([group["mae"] for group in primary_groups]))
                if primary_groups
                else None
            )
            overall["group_skill_median"] = (
                float(
                    np.mean(
                        [
                            group["skill_median"]
                            for group in primary_groups
                            if group["skill_median"] is not None
                        ]
                    )
                )
                if any(group["skill_median"] is not None for group in primary_groups)
                else None
            )
            overall["group_skill_zero"] = (
                float(
                    np.mean(
                        [
                            group["skill_zero"]
                            for group in primary_groups
                            if group["skill_zero"] is not None
                        ]
                    )
                )
                if any(group["skill_zero"] is not None for group in primary_groups)
                else None
            )
            histograms = {}
            for name, indices in groups.items():
                values = residual[:, indices]
                counts, edges = np.histogram(
                    values[np.isfinite(values)], bins=np.linspace(-5, 5, 81)
                )
                histograms[name] = {
                    "edges": edges.tolist(),
                    "counts": counts.tolist(),
                    "outside_range_count": int(
                        np.sum(np.isfinite(values) & (abs(values) > 5))
                    ),
                }
            bins = {}
            if "G" in wrapper.feature_cols:
                i = wrapper.feature_cols.index("G")
                magnitude = validation[:, i] * scale[i] + center[i]
                bins["magnitude"] = []
                for lower, upper in (
                    (-100, 12),
                    (12, 15),
                    (15, 18),
                    (18, 21),
                    (21, 100),
                ):
                    selected = (
                        np.isfinite(magnitude)
                        & (magnitude >= lower)
                        & (magnitude < upper)
                    )
                    bins["magnitude"].append(
                        {
                            "lower": lower,
                            "upper": upper,
                            "row_count": int(selected.sum()),
                            **summarize(
                                residual[selected], target[selected], median[:width]
                            ),
                        }
                    )
            if "PARALLAX" in wrapper.feature_cols:
                i = wrapper.feature_cols.index("PARALLAX")
                raw = validation[:, i] * scale[i] + center[i]
                raw_sigma = sigma[:, i] * scale[i]
                snr = np.divide(
                    raw,
                    raw_sigma,
                    out=np.full(len(raw), np.nan),
                    where=np.isfinite(raw_sigma) & (raw_sigma > 0),
                )
                bins["snr"] = []
                for lower, upper in ((-1e6, 0), (0, 2), (2, 5), (5, 10), (10, 1e6)):
                    selected = np.isfinite(snr) & (snr >= lower) & (snr < upper)
                    bins["snr"].append(
                        {
                            "lower": lower,
                            "upper": upper,
                            "row_count": int(selected.sum()),
                            **summarize(
                                residual[selected], target[selected], median[:width]
                            ),
                        }
                    )
            qa["regimes"][regime] = {
                "overall": overall,
                "features": features,
                "blocks": blocks,
                "histograms": histograms,
                "bins": bins,
            }
            snapshots[f"{regime}_prediction"] = prediction
            snapshots[f"{regime}_mask"] = mask[:, :width]
    _restore_rng_state(prior)
    write_json(output / "residual_latest.json", qa)
    temporary = output / "residual_latest.tmp.npz"
    np.savez_compressed(temporary, **snapshots)
    temporary.replace(output / "residual_latest.npz")
    model.train()
    return qa["regimes"]["common"]["overall"]["masked_mae"], qa


def build_wrapper(config, checkpoint, source, center, scale, arm, output):
    model_config = config["model"]
    data = config["data"]
    training = config["training"]
    model = make_model(
        len(data["feature_cols"]),
        model_config["layer_dims"],
        len(data["recon_cols"]),
        model_config["pt_activ_func"],
        model_config["rtdl_embed"],
        model_config["norm"],
        decoder_dims=model_config.get("decoder_dims"),
        encoder_type=model_config.get("encoder_type", "resnet"),
        cosine_latent=model_config.get("cosine_latent", False),
        heteroscedastic=training.get("heteroscedastic", False),
    )
    # Derive legacy sentinel buffers on the same device that computes the control.
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]), strict=True)
    if arm.get("embedding", "stable") == "legacy":
        for module in model.modules():
            if isinstance(module, MaskedPeriodicEmbeddings):
                module.forward = MethodType(PeriodicEmbeddings.forward, module)
    if "dropout" in arm:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = float(arm["dropout"])
    scaler = RobustScaler()
    scaler.center_, scaler.scale_, scaler.n_features_in_ = (
        center.copy(),
        scale.copy(),
        len(center),
    )
    wrapper = TabResnetWrapper(
        model=model,
        datafile=source,
        scaler=scaler,
        feature_cols=data["feature_cols"],
        error_cols=data["error_cols"],
        recon_cols=data["recon_cols"],
        xp_masking_ratio=float(
            arm.get("xp_masking_ratio", training["xp_masking_ratio"])
        ),
        m_masking_ratio=float(arm.get("m_masking_ratio", training["m_masking_ratio"])),
        lr=training["lr"],
        wd=training["weight_decay"],
        optimizer=training["optimizer"],
        lf="mae",
        lasso=0,
        micro_batch_size=training["micro_batch_size"],
        pt_save_str=str(output / "checkpoint.pth"),
    )
    optimizer, scheduler = wrapper._setup_pretrain_optimizer()
    # Adam state tensors must be copied: load_state_dict can alias the source.
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
    scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler_state_dict"]))
    _restore_rng_state(checkpoint["rng_state"])
    return wrapper, optimizer, scheduler


def iter_batches(cache, batch_size, rng):
    # ponytail: shuffle 65K-row blocks and rows within each; use a reservoir for wider mixing.
    blocks = list(range(0, len(cache), 65536))
    rng.shuffle(blocks)
    for start in blocks:
        block = np.array(cache[start : start + 65536])
        rng.shuffle(block)
        for offset in range(0, len(block), batch_size):
            yield block[offset : offset + batch_size]


def run_arm(
    config,
    checkpoint,
    source,
    cache,
    validation,
    sigma,
    center,
    scale,
    median,
    masks,
    arm,
    args,
    manifest,
):
    arm_id = arm["arm_id"]
    output = args.output / arm_id
    output.mkdir()
    wrapper, optimizer, scheduler = build_wrapper(
        config, checkpoint, source, center, scale, arm, output
    )
    device, model = wrapper.device, wrapper.model
    frequency = [
        module.periodic.weight
        for module in model.modules()
        if isinstance(module, MaskedPeriodicEmbeddings)
    ]
    groups = feature_groups(wrapper.recon_cols)
    width = len(wrapper.recon_cols)
    micro = wrapper._pretrain_micro_batch_size
    rows, steps, interval_steps = 0, 0, 0
    sums = torch.zeros(4, device=device)
    interval_rows = 0
    train_seconds = 0.0
    interval_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    mask_generator = torch.Generator(device=device).manual_seed(args.seed)
    xp_indices = [
        i
        for i, name in enumerate(wrapper.feature_cols)
        if name.startswith(("bp_", "rp_"))
    ]
    parent_rows = int(checkpoint.get("rows_seen_total", 0))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def emit(phase):
        nonlocal sums, interval_steps, interval_rows, interval_start, train_seconds
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - interval_start
        train_seconds += elapsed
        statistics = sums.cpu().tolist()
        if not all(math.isfinite(value) for value in statistics):
            raise FloatingPointError("Nonfinite training statistics")
        mae, qa = evaluate(
            wrapper,
            validation,
            sigma,
            masks,
            median,
            center,
            scale,
            output,
            manifest["validation_id"],
            rows,
            config["data"].get("uncertainty_units_verified", []),
        )
        common = qa["regimes"]["common"]["overall"]
        entry = {
            "suite_id": manifest["suite_id"],
            "run_id": arm_id,
            "arm_id": arm_id,
            "arm": arm_id,
            "validation_id": manifest["validation_id"],
            "phase": phase,
            "event": phase,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows_seen_total": rows,
            "parent_rows_seen_total": parent_rows,
            "optimizer_steps": steps,
            "train_loss": statistics[0] / interval_rows if interval_rows else None,
            "sampled_validation_mae": mae,
            "common_scaled_mae": mae,
            "median_mae": common["median_mae"],
            "zero_mae": common["zero_mae"],
            "group_mae": common["group_mae"],
            "group_skill_median": common["group_skill_median"],
            "finite": common["prediction_finite"],
            "gradient_norm_mean": statistics[1] / interval_steps
            if interval_steps
            else None,
            "frequency_gradient_fraction": statistics[2] / interval_steps
            if interval_steps
            else None,
            "clipping_rate": statistics[3] / interval_steps if interval_steps else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "rows_per_second": interval_rows / max(elapsed, 1e-9),
            "train_seconds": train_seconds,
            "validation_seconds": time.perf_counter() - interval_start - elapsed,
            "regimes": {
                name: value["overall"] for name, value in qa["regimes"].items()
            },
            **wrapper._resource_metrics(),
        }
        for path in (output / "metrics.jsonl", args.output / "metrics.jsonl"):
            with path.open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(json.dumps(entry, allow_nan=False), flush=True)
        sums.zero_()
        interval_rows = interval_steps = 0
        interval_start = time.perf_counter()

    emit("init")
    for suffix in ("json", "npz"):
        shutil.copyfile(
            output / f"residual_latest.{suffix}", output / f"residual_init.{suffix}"
        )
    model.train()
    balanced = arm.get("loss", "mae") == "group_mae"
    while rows < args.presentations:
        for raw in iter_batches(cache, config["training"]["mini_batch_size"], rng):
            raw = raw[: args.presentations - rows]
            if not len(raw):
                break
            target = torch.as_tensor(raw, device=device)
            hidden, mask, finite = training_mask(
                target,
                xp_indices,
                wrapper.xp_masking_ratio,
                wrapper.m_masking_ratio,
                mask_generator,
            )
            score = mask[:, :width] & finite[:, :width]
            counts = [score[:, indices].sum() for indices in groups.values()]
            global_count = score.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for start in range(0, len(raw), micro):
                stop = min(start + micro, len(raw))
                prediction, _ = model(hidden[start:stop])
                if balanced:
                    loss = balanced_mae(
                        prediction,
                        target[start:stop, :width],
                        score[start:stop],
                        groups,
                        counts,
                    )
                else:
                    loss = (prediction - target[start:stop, :width]).masked_fill(
                        ~score[start:stop], 0
                    ).abs().sum() / global_count
                loss.backward()
                batch_loss.add_(loss.detach())
            frequency_squared = sum(
                parameter.grad.detach().square().sum()
                for parameter in frequency
                if parameter.grad is not None
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            rows += len(raw)
            steps += 1
            # ponytail: hold the resumed LR for this bounded pilot; resume the source schedule for full training.
            sums[0].add_(batch_loss * len(raw))
            sums[1].add_(norm.detach())
            sums[2].add_(frequency_squared / norm.detach().square().clamp_min(1e-30))
            sums[3].add_((norm.detach() > 1).float())
            interval_rows += len(raw)
            interval_steps += 1
            if interval_rows >= args.log_rows or rows == args.presentations:
                emit("complete" if rows == args.presentations else "interval")
            if rows == args.presentations:
                break
    signature = {
        **checkpoint["run_signature"],
        "loader_mode": "experiment-prefix",
        "train_rows_per_epoch": len(cache),
        "train_rows_by_key": manifest["sample"]["train_rows_by_key"],
    }
    _atomic_torch_save(
        {
            "epoch": checkpoint["epoch"],
            "model_state_dict": model.state_dict(),
            "experiment_only": True,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": _capture_rng_state(),
            "mask_rng_state": mask_generator.get_state(),
            "epoch_loss": 0.0,
            "loss_div": 0.0,
            "rows_seen_total": parent_rows + rows,
            "run_signature": signature,
            "experiment": {
                "suite_id": manifest["suite_id"],
                "arm": arm,
                "pilot_presentations": rows,
                "parent_checkpoint_sha256": manifest["checkpoint_sha256"],
                "source_run_signature": checkpoint["run_signature"],
            },
        },
        output / "checkpoint.pth",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain.canfar.yaml")
    parser.add_argument("--experiments", default="configs/pretrain.experiments.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--train-rows", type=int, default=3_000_000)
    parser.add_argument("--validation-rows", type=int, default=10_000)
    parser.add_argument("--presentations", type=int, default=5_000_000)
    parser.add_argument("--log-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args()
    if (
        min(args.train_rows, args.validation_rows, args.presentations, args.log_rows)
        < 1
    ):
        parser.error("All row budgets must be positive")
    if args.output.exists():
        parser.error("Use a new output directory for every experiment suite")
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    expand_config_paths(config)
    with open(args.experiments) as stream:
        arms = yaml.safe_load(stream)["arms"]
    if args.arms:
        arms = [arm for arm in arms if arm["arm_id"] in args.arms]
        if {arm["arm_id"] for arm in arms} != set(args.arms):
            parser.error("Unknown arm requested")
    if not arms or len({arm["arm_id"] for arm in arms}) != len(arms):
        parser.error("Arm IDs must be unique and nonempty")
    for arm in arms:
        if Path(arm["arm_id"]).name != arm["arm_id"] or arm["arm_id"] in {".", ".."}:
            parser.error("Arm IDs must be single safe path components")
        if arm.get("embedding", "stable") not in {"legacy", "stable"} or arm.get(
            "loss", "mae"
        ) not in {"mae", "group_mae"}:
            parser.error("Unknown embedding/loss arm")
        for field in ("dropout", "xp_masking_ratio", "m_masking_ratio"):
            if field in arm and not 0 <= arm[field] <= 1:
                parser.error(f"{field} must be between zero and one")
    if (
        config["model"]["norm"] != "layer"
        or config["training"].get("pert_features")
        or config["training"].get("heteroscedastic")
        or config["training"].get("mask_mixture_xp_full_frac", 0)
    ):
        parser.error(
            "These bounded experiments require LayerNorm, plain MAE and no perturbation"
        )
    if (
        config["data"]["recon_cols"]
        != config["data"]["feature_cols"][: len(config["data"]["recon_cols"])]
    ):
        parser.error("Reconstruction features must be the input feature prefix")
    if any(
        name not in config["data"]["feature_cols"]
        for name in config["data"].get("uncertainty_units_verified", [])
    ):
        parser.error("Verified sigma features must be configured inputs")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch_load_trusted(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if any(arm.get("embedding") == "legacy" for arm in arms) and any(
        "missing_periodic_encoding" in name for name in checkpoint["model_state_dict"]
    ):
        parser.error(
            "The paired legacy control needs a checkpoint from before stable sentinel encoding"
        )
    manifest = {
        "suite_id": args.output.name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "preparing",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "optimizer_policy": "full inherited AdamW moments retained per arm; frequency moments may remain dominated by prior sentinel gradients, so short-pilot adaptation is limited",
        "source_code_sha256": {
            name: sha256(Path(name))
            for name in (
                "src/masked_stellar_autoencoder/models/model.py",
                "src/masked_stellar_autoencoder/models/blocks.py",
            )
        },
        "runner_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(Path(args.config)),
        "experiments_sha256": sha256(Path(args.experiments)),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "config": config,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "seed": args.seed,
        "mask_policy": "separate device generator seeded once per arm; independent of dropout RNG",
        "scheduler_policy": "checkpoint LR held fixed for paired bounded presentations; saved scheduler state unchanged",
        "selection_metric": "group_skill_median",
        "selection_policy": "higher mean of XP/photo/astro skill against training medians, completed arms only, identical presentations and optimizer steps",
        "runtime": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        },
        "presentations_per_arm": args.presentations,
        "arms": [
            {**arm, "status": "pending", "output_root": arm["arm_id"]} for arm in arms
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    try:
        with h5py.File(config["data"]["datafile"], "r") as source:
            keys, holdout, center, scale = validate_source(checkpoint, config, source)
            expected_cache_bytes = (
                (args.train_rows + 2 * args.validation_rows) * len(center) * 4
            )
            cache_limit = min(
                150_000_000_000, int(shutil.disk_usage(args.cache_dir).free * 0.8)
            )
            manifest["cache_limit_bytes"] = cache_limit
            manifest["projected_cache_bytes"] = expected_cache_bytes
            write_json(args.output / "manifest.json", manifest)
            if expected_cache_bytes > cache_limit:
                raise ValueError(
                    f"Projected pilot cache {expected_cache_bytes} exceeds scratch budget {cache_limit}"
                )
            cache, _, train_counts = read_prefixes(
                source,
                keys,
                config["data"]["feature_cols"],
                None,
                args.train_rows,
                center,
                scale,
                args.cache_dir / f"{args.output.name}-train.npy",
            )
            validation, sigma, validation_counts = read_prefixes(
                source,
                holdout,
                config["data"]["feature_cols"],
                config["data"]["error_cols"],
                args.validation_rows,
                center,
                scale,
                args.cache_dir / f"{args.output.name}-validation.npy",
            )
            if not len(cache) or not len(validation):
                raise ValueError("Training and holdout samples must be nonempty")
            manifest["sample"] = {
                "policy": "sequential_prefix_per_shard_v1; deliberately bounded, not population random sampling",
                "train_rows": len(cache),
                "validation_rows": len(validation),
                "train_rows_by_key": train_counts,
                "validation_rows_by_key": validation_counts,
                "source_rows_by_key": {key: len(source[key]) for key in source},
                "source_path": str(config["data"]["datafile"]),
                "source_size_bytes": Path(source.filename).stat().st_size,
                "cache_bytes": cache.nbytes + validation.nbytes + sigma.nbytes,
                "median_baseline_rows": min(100_000, len(cache)),
                "median_baseline_policy": "evenly spaced deterministic rows across the bounded pilot cache",
                "sigma_policy": "only named finite positive source errors; missing/invalid is NaN",
                "train_availability": availability(
                    cache, config["data"]["feature_cols"]
                ),
                "validation_availability": availability(
                    validation, config["data"]["feature_cols"]
                ),
            }
            validation_hash = hashlib.sha256(np.asarray(validation).tobytes())
            validation_hash.update(np.asarray(sigma).tobytes())
            validation_hash.update(str(args.seed).encode())
            masks = fixed_masks(
                validation,
                config["data"]["feature_cols"],
                args.seed,
                config["training"]["xp_masking_ratio"],
                config["training"]["m_masking_ratio"],
            )
            for name, mask in masks.items():
                validation_hash.update(name.encode())
                validation_hash.update(mask.tobytes())
            validation_hash.update(json.dumps(config["data"]["feature_cols"]).encode())
            validation_hash.update(center.tobytes())
            validation_hash.update(scale.tobytes())
            manifest["validation_id"] = validation_hash.hexdigest()
            # ponytail: a bounded 100K training-row baseline; use stratified medians for population calibration.
            baseline_sample = np.asarray(
                cache[
                    np.linspace(
                        0, len(cache) - 1, min(100_000, len(cache)), dtype=np.int64
                    )
                ]
            )
            median = np.array(
                [
                    np.median(values[np.isfinite(values)])
                    if np.isfinite(values).any()
                    else 0
                    for values in baseline_sample.T
                ],
                dtype=np.float32,
            )
            manifest["status"] = "running"
            write_json(args.output / "manifest.json", manifest)
            for arm in manifest["arms"]:
                arm["status"] = "running"
                write_json(args.output / "manifest.json", manifest)
                run_arm(
                    config,
                    checkpoint,
                    source,
                    cache,
                    validation,
                    sigma,
                    center,
                    scale,
                    median,
                    masks,
                    arm,
                    args,
                    manifest,
                )
                arm["status"] = "complete"
                write_json(args.output / "manifest.json", manifest)
            manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        for arm in manifest["arms"]:
            if arm["status"] == "running":
                arm["status"] = "failed"
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
