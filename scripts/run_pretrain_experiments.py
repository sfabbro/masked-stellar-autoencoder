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
from masked_stellar_autoencoder.training.xp_experiments import (
    apply_optimizer_warmup,
    configure_optimizer,
    experiment_objective,
    noise_reliability,
    spectral_basis,
    spectral_summary,
)
from masked_stellar_autoencoder.training.xp_experiments import (
    project_spectral_residual as project_spectral_residual,
)
from masked_stellar_autoencoder.training.xp_photometry import (
    PhotometryProjection,
    fit_calibration,
    normalized_observed_flux,
    photometry_consistency_sum,
)
from masked_stellar_autoencoder.training.xp_representation import (
    continuum_operator,
    paired_latent_regularizer,
)


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


def cgroup_memory_metrics():
    result = {}
    for source, name in (
        ("memory.current", "cgroup_memory_current_bytes"),
        ("memory.peak", "cgroup_memory_peak_bytes"),
        ("memory.max", "cgroup_memory_limit_bytes"),
    ):
        try:
            value = Path("/sys/fs/cgroup", source).read_text().strip()
        except OSError:
            value = None
        if value is not None and value.isdecimal():
            value = int(value)
        result[name] = value
    return result


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
    grouped = "train/records" in source and "validation/records" in source
    if grouped:
        scaler = source["reference_scaler"]
        for field, saved in (("center", "scaler_center"), ("scale", "scaler_scale")):
            np.testing.assert_array_equal(scaler[field][:], signature[saved])
        if json.loads(scaler.attrs["feature_cols"]) != signature["feature_cols"]:
            raise ValueError("Benchmark feature order differs from checkpoint")
        saved_validation = json.loads(source["validation"].attrs["source_shard_names"])
        if set(saved_validation) != set(validation_keys):
            raise ValueError("Benchmark validation membership differs")
        train_keys, validation_keys = ["train/records"], ["validation/records"]
    elif set(train_keys) != set(source) - set(validation_keys):
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


def read_prefixes(
    source,
    keys,
    features,
    errors,
    max_rows,
    center,
    scale,
    destination,
    *,
    correct_xp_errors=False,
):
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
                        if correct_xp_errors and features[column].startswith(
                            ("bp_", "rp_")
                        ):
                            flux = 10 ** (
                                (8.5 - _clean_column("G", records["G"])) / 2.5
                            )
                            raw = raw / flux
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


def fixed_masks(x, features, seed, xp_ratio, photo_ratio, extended=False):
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
    masks = {"common": common, "xp_on": xp_on, "xp_off": xp_off}
    if extended:
        for mode in ("coefficients", "block", "channel"):
            generator = torch.Generator().manual_seed(seed)
            masks[f"partial_{mode}"] = training_mask(
                torch.from_numpy(np.asarray(x)),
                xp,
                1.0,
                photo_ratio,
                generator,
                xp_mode=mode,
                features=features,
                whole_fraction=0,
            )[1].numpy()
    return masks


def training_mask(
    target,
    xp_indices,
    xp_ratio,
    photo_ratio,
    generator,
    *,
    xp_mode="whole",
    features=None,
    whole_fraction=0.5,
):
    """Mask RNG is independent of dropout so all paired experiments share mask draws."""
    mask = (
        torch.rand(target.shape, device=target.device, generator=generator)
        < photo_ratio
    )
    mask[:, xp_indices] = False
    permutation = torch.randperm(len(target), device=target.device, generator=generator)
    rows = permutation[: int(len(target) * xp_ratio)]
    if xp_mode == "whole":
        mask[rows[:, None], xp_indices] = True
    else:
        n_whole = int(len(rows) * whole_fraction)
        mask[rows[:n_whole, None], xp_indices] = True
        count = max(1, len(xp_indices) // 2)
        # Preserve the baseline's hidden-coefficient budget where enough rows exist.
        n_partial = min(
            len(target) - n_whole,
            round((len(rows) - n_whole) * len(xp_indices) / count),
        )
        partial = permutation[n_whole : n_whole + n_partial]
        if xp_mode == "coefficients":
            scores = torch.rand(
                (len(partial), len(xp_indices)),
                device=target.device,
                generator=generator,
            )
            selected = torch.topk(scores, count, dim=1).indices
            columns = torch.as_tensor(xp_indices, device=target.device)[selected]
            mask[partial[:, None], columns] = True
        elif xp_mode == "block":
            starts = torch.randint(
                len(xp_indices) - count + 1,
                (len(partial),),
                device=target.device,
                generator=generator,
            )
            columns = torch.as_tensor(xp_indices, device=target.device)[
                starts[:, None] + torch.arange(count, device=target.device)
            ]
            mask[partial[:, None], columns] = True
        elif xp_mode == "channel":
            band = torch.randint(
                2, (len(partial),), device=target.device, generator=generator
            )
            for value, prefix in enumerate(("bp_", "rp_")):
                columns = [i for i in xp_indices if features[i].startswith(prefix)]
                mask[partial[band == value, None], columns] = True
        else:
            raise ValueError(f"Unknown XP mask mode: {xp_mode}")
    finite = torch.isfinite(target)
    return target.masked_fill(mask | ~finite, -9999), mask, finite


def paired_training_view(target, primary_hidden, features, photo_ratio, generator):
    """Change photometry masks only; preserve primary G, XP and astrometry."""
    if len(features) != target.shape[1] or target.shape != primary_hidden.shape:
        raise ValueError("Paired-view features must match target and primary shapes")
    if not math.isfinite(photo_ratio) or not 0 <= photo_ratio <= 1:
        raise ValueError("Photometry mask ratio must be finite and in [0, 1]")
    columns = [
        i
        for i, name in enumerate(features)
        if name in {"W1", "W2", "BP", "RP", "J", "H", "KS"}
        or name.endswith(("_SMSS", "_SDSS", "_PS1"))
    ]
    secondary = primary_hidden.clone()
    hidden = (
        torch.rand(
            (len(target), len(columns)), device=target.device, generator=generator
        )
        < photo_ratio
    )
    values = target[:, columns]
    secondary[:, columns] = values.masked_fill(hidden | ~torch.isfinite(values), -9999)
    return secondary


def prepare_photometry(
    cache, train_sigma, center, scale, names, projection_path, output
):
    """Fixed calibration from 100K training-only rows; no learned targets."""
    projection = PhotometryProjection.load(projection_path)
    xp = [names.index(f"{band}_{i}") for band in ("bp", "rp") for i in range(1, 56)]
    photo = [names.index(name) for name in projection.columns]
    g = names.index("G")
    indices = np.linspace(0, len(cache) - 1, min(100_000, len(cache)), dtype=np.int64)
    values = np.asarray(cache[indices], dtype=np.float64) * scale + center
    calibration = fit_calibration(
        values[:, xp], values[:, photo], values[:, g], projection, min_count=1000
    )
    metadata = {
        "columns": projection.columns,
        "bands": projection.bands,
        "calibration": calibration.as_dict(),
        "rows": len(indices),
        "selection": "training-only evenly spaced deterministic benchmark rows",
        "projection_sha256": sha256(Path(projection_path)),
        "scaler_center": center[xp].tolist(),
        "scaler_scale": scale[xp].tolist(),
        "normalization": "raw linear XP projection and observed flux divided by 10**((8.5-G)/2.5); true G used only in loss targets, no model-input substitution",
        "correction": "training-only per-band additive median residual in reference-flux units; no standardised GaiaXPy offsets",
        "floor": "max(1.4826*MAD training residual, 0.02 median band flux); observed magnitude errors added in quadrature",
        "decoder": "predicted photometry targets detached; ordinary observed MAE remains anchor",
    }
    # Baseline measures the observational mismatch before learning any correction.
    observed = normalized_observed_flux(
        values[:, photo], values[:, g], projection.zero_points
    )
    synthetic = values[:, xp] @ projection.matrix
    baseline = []
    for i, column in enumerate(projection.columns):
        valid = (
            np.isfinite(observed[:, i])
            & (observed[:, i] > 0)
            & np.isfinite(synthetic[:, i])
            & (synthetic[:, i] > 0)
        )
        dm = -2.5 * np.log10(synthetic[valid, i] / observed[valid, i])
        baseline.append(
            {
                "column": column,
                "count": int(valid.sum()),
                "median_synthetic_minus_observed_mag": float(np.median(dm)),
                "median_absolute_mag": float(np.median(abs(dm))),
            }
        )
    metadata["observed_xp_baseline"] = baseline
    write_json(output / "photometry_calibration.json", metadata)
    return {
        "projection": projection,
        "calibration": calibration,
        "xp": xp,
        "photo": photo,
        "g": g,
        "center": center,
        "scale": scale,
    }


def photometry_terms(prediction, target, score, uncertainty, context, *, decoder=False):
    xp, photo, g = context["xp"], context["photo"], context["g"]
    projection = context["projection"]
    kw = {"device": prediction.device, "dtype": prediction.dtype}
    scale = torch.as_tensor(context["scale"], **kw)
    center = torch.as_tensor(context["center"], **kw)
    # Dimensionless band/G flux target, evaluated with one exponent to avoid tiny divisors.
    magnitudes = target[:, photo] * scale[photo] + center[photo]
    g_mag = target[:, g] * scale[g] + center[g]
    zp = torch.as_tensor(projection.zero_points, **kw)
    observed = 10.0 ** (-0.4 * (magnitudes - zp + 8.5 - g_mag[:, None]))
    errors = (
        uncertainty[:, photo] * scale[photo]
        if uncertainty is not None
        else torch.full_like(observed, torch.nan)
    )
    decoder_flux = None
    band_mask = None
    if decoder:
        mag_prediction = prediction[:, photo] * scale[photo] + center[photo]
        decoder_flux = 10.0 ** (-0.4 * (mag_prediction - zp + 8.5 - g_mag[:, None]))
        band_mask = score[:, photo]
    return photometry_consistency_sum(
        prediction[:, xp],
        target[:, xp],
        score[:, xp],
        center[xp],
        scale[xp],
        projection.matrix,
        observed,
        errors,
        context["calibration"],
        band_mask=band_mask,
        decoder_flux=decoder_flux,
    )[0]


def photometry_summary(target, prediction, mask, context):
    projection, xp = context["projection"], context["xp"]
    photo, g = context["photo"], context["g"]
    scale, center = context["scale"], context["center"]
    raw = target * scale[: target.shape[1]] + center[: target.shape[1]]
    combined = np.where(mask[:, xp], prediction[:, xp], target[:, xp])
    synthetic = (combined * scale[xp] + center[xp]) @ projection.matrix
    observed = normalized_observed_flux(
        raw[:, photo], raw[:, g], projection.zero_points
    )
    active = mask[:, xp].any(1) & np.isfinite(target[:, xp]).all(1)
    calibration = context["calibration"]
    result = {
        "policy": "hybrid visible observed plus hidden predicted XP; actual survey photometry target; raw magnitude residual and bias-corrected flux residual",
        "bands": [],
    }
    for i, column in enumerate(projection.columns):
        valid = (
            active
            & np.isfinite(observed[:, i])
            & (observed[:, i] > 0)
            & np.isfinite(synthetic[:, i])
        )
        residual = (synthetic[:, i] - observed[:, i]) / calibration.reference_flux[
            i
        ] - calibration.bias[i]
        positive = valid & (synthetic[:, i] > 0)
        dm = -2.5 * np.log10(synthetic[positive, i] / observed[positive, i])
        item = {
            "column": column,
            "count": int(valid.sum()),
            "nonpositive_synthetic_count": int((valid & (synthetic[:, i] <= 0)).sum()),
            "reference_flux_absolute_error_mean": float(np.mean(abs(residual[valid])))
            if valid.any()
            else None,
            "magnitude_mae": float(np.mean(abs(dm))) if len(dm) else None,
            "magnitude_median_absolute_error": float(np.median(abs(dm)))
            if len(dm)
            else None,
        }
        for state, selected in (
            ("hidden", mask[:, photo[i]]),
            ("visible", ~mask[:, photo[i]]),
        ):
            rows = valid & selected
            item[state] = {
                "count": int(rows.sum()),
                "reference_flux_absolute_error_mean": float(
                    np.mean(abs(residual[rows]))
                )
                if rows.any()
                else None,
            }
        result["bands"].append(item)
    return result


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
    diagnostic_rows=100_000,
    spectral_projection=None,
    photometry=None,
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
        "validation_rows_total": len(validation),
        "diagnostic_snapshot": {
            "rows": min(diagnostic_rows, len(validation)),
            "validation_rows": len(validation),
            "selection": "evenly spaced deterministic validation rows; diagnostics only",
        },
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
    snapshot_indices = np.linspace(
        0, len(validation) - 1, min(diagnostic_rows, len(validation)), dtype=np.int64
    )
    snapshots = {
        "targets": target[snapshot_indices],
        "sigma": sigmas[snapshot_indices],
        "source_sigma": sigma[snapshot_indices, :width],
        "feature_names": np.asarray(names),
        "validation_row_index": snapshot_indices,
    }
    prior = _capture_rng_state()
    model.eval()
    with torch.no_grad():
        for regime, mask in masks.items():
            predictions = []
            latents = []
            latent_cursor = 0
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
                    next_cursor = np.searchsorted(
                        snapshot_indices, start + len(batch), side="left"
                    )
                    if next_cursor > latent_cursor:
                        rows = snapshot_indices[latent_cursor:next_cursor] - start
                        latents.append(latent[rows].cpu().numpy())
                        latent_cursor = next_cursor
            prediction = np.concatenate(predictions)
            del predictions
            scored = mask[:, :width] & np.isfinite(target)
            residual = np.where(scored, prediction - target, np.nan)
            finite_prediction = bool(np.isfinite(prediction).all())
            if not finite_prediction:
                raise FloatingPointError(
                    f"Nonfinite validation predictions in {regime}"
                )
            if latents:
                latent = np.concatenate(latents)
                del latents
                if not np.isfinite(latent).all():
                    raise FloatingPointError("Nonfinite validation latent values")
                qa["latent"] = latent_summary(latent)
                qa["latent"]["sample_rows"] = len(latent)
                qa["latent"]["sample_policy"] = qa["diagnostic_snapshot"]["selection"]
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
                            "xp_only": summarize(
                                residual[selected][:, groups["xp"]],
                                target[selected][:, groups["xp"]],
                                median[groups["xp"]],
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
            snapshots[f"{regime}_prediction"] = prediction[snapshot_indices]
            snapshots[f"{regime}_mask"] = mask[snapshot_indices, :width]
            if spectral_projection is not None:
                qa["regimes"][regime]["spectral"] = spectral_summary(
                    target[snapshot_indices],
                    prediction[snapshot_indices],
                    mask[snapshot_indices, :width],
                    scale[:width],
                    center[:width],
                    spectral_projection,
                )
            if photometry is not None:
                qa["regimes"][regime]["photometry_consistency"] = photometry_summary(
                    target[snapshot_indices],
                    prediction[snapshot_indices],
                    mask[snapshot_indices, :width],
                    photometry,
                )
            xp_columns = groups.get("xp", [])
            xp_target = target[:, xp_columns] * scale[xp_columns] + center[xp_columns]
            xp_sigma = sigma[:, xp_columns] * scale[xp_columns]
            snr = np.divide(
                abs(xp_target),
                xp_sigma,
                out=np.full_like(xp_target, np.nan),
                where=np.isfinite(xp_sigma) & (xp_sigma > 0),
            )
            qa["regimes"][regime]["xp_coefficient_snr"] = [
                {
                    "lower": lower,
                    "upper": upper,
                    **summarize(
                        np.where(
                            (snr >= lower) & (snr < upper),
                            residual[:, xp_columns],
                            np.nan,
                        ),
                        target[:, xp_columns],
                        median[xp_columns],
                    ),
                }
                for lower, upper in ((0, 1), (1, 3), (3, 10), (10, 1e30))
            ]
    _restore_rng_state(prior)
    write_json(output / "residual_latest.json", qa)
    temporary = output / "residual_latest.tmp.npz"
    np.savez_compressed(temporary, **snapshots)
    temporary.replace(output / "residual_latest.npz")
    model.train()
    return qa["regimes"]["common"]["overall"]["masked_mae"], qa


def configure_experiment_dropout(model, experiment):
    """Override experiment dropout only; encoder/decoder keys take precedence."""
    for key in ("dropout", "encoder_dropout", "decoder_dropout"):
        if key in experiment:
            value = experiment[key]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and between zero and one")
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Dropout):
            continue
        key = (
            "encoder_dropout"
            if name.startswith("encoder.")
            else "decoder_dropout"
            if name.startswith("decoder.")
            else "dropout"
        )
        if key in experiment or "dropout" in experiment:
            module.p = float(experiment.get(key, experiment.get("dropout")))
    return {
        name: module.p
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout)
    }


def experiment_seed(experiment, default):
    seed = experiment.get("seed", default)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Experiment seed must be an integer in [0, 2**32)")
    return seed


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
    dropout_policy = configure_experiment_dropout(model, arm)
    write_json(output / "dropout_policy.json", dropout_policy)
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
    write_json(output / "optimizer_policy.json", configure_optimizer(optimizer, arm))
    _restore_rng_state(checkpoint["rng_state"])
    return wrapper, optimizer, scheduler


def iter_batches(cache, batch_size, rng, sigma=None, correlations=None):
    # ponytail: shuffle 65K-row blocks and rows within each; use a reservoir for wider mixing.
    blocks = list(range(0, len(cache), 65536))
    rng.shuffle(blocks)
    for start in blocks:
        indices = np.arange(start, min(start + 65536, len(cache)))
        rng.shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            if correlations is not None:
                yield (
                    np.array(cache[selected]),
                    np.array(sigma[selected]),
                    np.array(correlations[selected]),
                )
            elif sigma is None:
                yield np.array(cache[selected])
            else:
                yield np.array(cache[selected]), np.array(sigma[selected])


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
    train_sigma=None,
    spectral_projection=None,
    train_correlations=None,
    photometry=None,
):
    arm_id = arm["run_id"]
    output = args.output / arm_id
    output.mkdir()
    wrapper, optimizer, scheduler = build_wrapper(
        config, checkpoint, source, center, scale, arm, output
    )
    seed = experiment_seed(arm, args.seed)
    if args.independent_rng or "seed" in arm:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    optimizer_policy = json.loads((output / "optimizer_policy.json").read_text())
    optimizer_policy["seed"] = seed
    optimizer_policy["independent_rng"] = args.independent_rng or "seed" in arm
    write_json(output / "optimizer_policy.json", optimizer_policy)
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
    rng = np.random.default_rng(seed)
    mask_generator = torch.Generator(device=device).manual_seed(seed)
    xp_indices = [
        i
        for i, name in enumerate(wrapper.feature_cols)
        if name.startswith(("bp_", "rp_"))
    ]
    reliability = None
    spectral_loss = None
    if arm.get("loss") in {
        "spectral_mae",
        "photometry_spectral_mae",
        "multiscale_spectral_mae",
        "full_decoder_spectral_mae",
    }:
        if spectral_projection is None:
            raise ValueError("Spectral loss requires --xp-diagnostics")
        spectral_loss = {
            "scale": torch.as_tensor(scale[:width], dtype=torch.float32, device=device),
            "center": torch.as_tensor(
                center[:width], dtype=torch.float32, device=device
            ),
            "basis": torch.as_tensor(
                spectral_projection["basis"], dtype=torch.float32, device=device
            ),
            "wavelength": torch.as_tensor(
                spectral_projection["wavelength_nm"], dtype=torch.float32, device=device
            ),
        }
        if arm.get("loss") == "multiscale_spectral_mae":
            spectral_loss["smoothing"] = continuum_operator(
                spectral_loss["wavelength"], arm.get("spectral_smoothing_nm", 40.0)
            )
    latent_scale = None
    second_mask_generator = None
    if arm.get("latent_weight", 0) > 0:
        indices = np.linspace(
            0, len(cache) - 1, min(10_000, len(cache)), dtype=np.int64
        )
        calibration_generator = torch.Generator(device=device).manual_seed(42)
        calibration = torch.as_tensor(np.array(cache[indices]), device=device)
        calibration_hidden, _, _ = training_mask(
            calibration,
            xp_indices,
            wrapper.xp_masking_ratio,
            wrapper.m_masking_ratio,
            calibration_generator,
            features=wrapper.feature_cols,
        )
        model.eval()
        with torch.no_grad():
            calibration_latent = torch.cat(
                [
                    model(calibration_hidden[start : start + micro])[1]
                    for start in range(0, len(indices), micro)
                ]
            )
        latent_scale = calibration_latent.std(0, unbiased=True)
        if not torch.isfinite(latent_scale).all() or not (latent_scale > 0).all():
            raise ValueError(
                "Parent training latent scales must be finite and positive"
            )
        write_json(
            output / "latent_calibration.json",
            {
                "rows": len(indices),
                "training_only": True,
                "mask_seed": 42,
                "parent_sha256": manifest["checkpoint_sha256"],
                "reference_scale": latent_scale.cpu().tolist(),
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "statistics_microbatch": micro,
                "pair_policy": "same G/XP/astrometry visibility; independent photometry masks",
            },
        )
        second_mask_generator = torch.Generator(device=device).manual_seed(
            seed + 1_000_000
        )
    if arm.get("loss") == "signal_weighted_mae":
        indices = np.linspace(
            0, len(cache) - 1, min(100_000, len(cache)), dtype=np.int64
        )
        reliability = torch.as_tensor(
            noise_reliability(
                np.asarray(cache[indices]), np.asarray(train_sigma[indices])
            ),
            device=device,
        )
        write_json(
            output / "training_reliability.json",
            {
                "feature_names": wrapper.feature_cols,
                "weights": reliability.cpu().tolist(),
                "rows": len(indices),
                "selection": "training-only evenly spaced prefix rows",
            },
        )
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
            args.diagnostic_rows,
            spectral_projection,
            photometry,
        )
        common = qa["regimes"]["common"]["overall"]
        entry = {
            "suite_id": manifest["suite_id"],
            "run_id": arm_id,
            "validation_id": manifest["validation_id"],
            "phase": phase,
            "event": phase,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows_seen_total": rows,
            "parent_rows_seen_total": parent_rows,
            "optimizer_steps": steps,
            "seed": seed,
            "train_loss": statistics[0] / interval_rows if interval_rows else None,
            "sampled_validation_mae": mae,
            "common_scaled_mae": mae,
            "all_columns_mae": mae,
            "xp_only_mae": qa["regimes"]["common"]["blocks"]["xp"]["mae"],
            "per_column_mae": {
                value["name"]: value["mae"]
                for value in qa["regimes"]["common"]["features"]
            },
            "per_column_count": {
                value["name"]: value["count"]
                for value in qa["regimes"]["common"]["features"]
            },
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
                name: {**value["overall"], "xp_only_mae": value["blocks"]["xp"]["mae"]}
                for name, value in qa["regimes"].items()
            },
            **wrapper._resource_metrics(),
            "loader_mode": "experiment-prefix-cache",
            "optimizer_batch_size": config["training"]["mini_batch_size"],
            "cache_bytes": manifest["sample"]["cache_bytes"],
            "projected_cache_bytes": manifest["projected_cache_bytes"],
            "scratch_free_bytes": shutil.disk_usage(args.cache_dir).free,
            **cgroup_memory_metrics(),
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
    if getattr(args, "evaluation_only", False):
        checkpoint_hash_after = sha256(args.checkpoint)
        if checkpoint_hash_after != manifest["checkpoint_sha256"]:
            raise ValueError(
                "Read-only evaluation checkpoint changed during evaluation"
            )
        reloaded = copy.deepcopy(model)
        reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        reloaded.eval()
        with torch.no_grad():
            probe = torch.as_tensor(np.array(validation[:16]), device=device)
            probe = probe.masked_fill(~torch.isfinite(probe), -9999)
            torch.testing.assert_close(
                model(probe)[0], reloaded(probe)[0], rtol=0, atol=0
            )
        write_json(
            output / "evaluation_verification.json",
            {
                "evaluation_only": True,
                "optimizer_steps": 0,
                "training_presentations": 0,
                "checkpoint_sha256": checkpoint_hash_after,
                "reload_predictions_equal": True,
                "source_unchanged": True,
            },
        )
        return
    model.train()
    balanced = arm.get("loss", "mae") == "group_mae"
    while rows < args.presentations:
        for values in iter_batches(
            cache,
            config["training"]["mini_batch_size"],
            rng,
            train_sigma,
            train_correlations,
        ):
            packed = None
            if train_correlations is not None:
                raw, uncertainty, packed = values
            else:
                raw, uncertainty = values if train_sigma is not None else (values, None)
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
                xp_mode=arm.get("xp_mask_mode", "whole"),
                features=wrapper.feature_cols,
            )
            second_hidden = None
            if latent_scale is not None:
                second_hidden = paired_training_view(
                    target,
                    hidden,
                    wrapper.feature_cols,
                    wrapper.m_masking_ratio,
                    second_mask_generator,
                )
            uncertainty = (
                torch.as_tensor(uncertainty[: len(raw)], device=device)
                if uncertainty is not None
                else None
            )
            score = mask[:, :width] & finite[:, :width]
            counts = [score[:, indices].sum() for indices in groups.values()]
            global_count = score.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.zeros((), device=device)
            for start in range(0, len(raw), micro):
                stop = min(start + micro, len(raw))
                prediction, latent = model(hidden[start:stop])
                if balanced:
                    loss = balanced_mae(
                        prediction,
                        target[start:stop, :width],
                        score[start:stop],
                        groups,
                        counts,
                    )
                else:
                    loss = experiment_objective(
                        prediction,
                        target[start:stop, :width],
                        score[start:stop],
                        [i for i in xp_indices if i < width],
                        {
                            **arm,
                            "loss": "spectral_mae"
                            if arm.get("loss") == "photometry_spectral_mae"
                            else "mae",
                        }
                        if arm.get("loss")
                        in {
                            "photometry_mae",
                            "photometry_spectral_mae",
                            "photometry_decoder_mae",
                        }
                        else arm,
                        global_count,
                        sigma=uncertainty[start:stop, :width]
                        if uncertainty is not None
                        else None,
                        spectral=spectral_loss,
                        reliability=reliability,
                        correlations=torch.as_tensor(packed[start:stop], device=device)
                        if packed is not None
                        else None,
                    )
                if arm.get("loss") in {
                    "photometry_mae",
                    "photometry_spectral_mae",
                    "photometry_decoder_mae",
                }:
                    if photometry is None:
                        raise ValueError(
                            "Photometry objective requires calibrated projection"
                        )
                    observed_sum = photometry_terms(
                        prediction,
                        target[start:stop, :width],
                        score[start:stop],
                        uncertainty[start:stop, :width]
                        if uncertainty is not None
                        else None,
                        photometry,
                    )
                    loss = (
                        loss
                        + arm.get("photometry_weight", 0.01)
                        * observed_sum
                        / global_count
                    )
                    if arm.get("loss") == "photometry_decoder_mae":
                        decoder_sum = photometry_terms(
                            prediction,
                            target[start:stop, :width],
                            score[start:stop],
                            uncertainty[start:stop, :width]
                            if uncertainty is not None
                            else None,
                            photometry,
                            decoder=True,
                        )
                        loss = (
                            loss
                            + arm.get("decoder_weight", 0.002)
                            * decoder_sum
                            / global_count
                        )
                if latent_scale is not None:
                    if stop - start < 2:
                        raise ValueError(
                            "Paired latent regularization requires >=2 rows"
                        )
                    # The auxiliary forward must not advance primary dropout RNG.
                    devices = (
                        [
                            device.index
                            if device.index is not None
                            else torch.cuda.current_device()
                        ]
                        if device.type == "cuda"
                        else []
                    )
                    with torch.random.fork_rng(devices=devices):
                        _, second_latent = model(second_hidden[start:stop])
                    loss = loss + arm["latent_weight"] * paired_latent_regularizer(
                        latent,
                        second_latent,
                        latent_scale,
                    ) * ((stop - start) / len(raw))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite experiment objective")
                loss.backward()
                batch_loss.add_(loss.detach())
            frequency_squared = sum(
                parameter.grad.detach().square().sum()
                for parameter in frequency
                if parameter.grad is not None
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            apply_optimizer_warmup(optimizer, optimizer_policy, steps)
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
                "settings": arm,
                "pilot_presentations": rows,
                "parent_checkpoint_sha256": manifest["checkpoint_sha256"],
                "source_run_signature": checkpoint["run_signature"],
            },
        },
        output / "checkpoint.pth",
    )
    saved = torch_load_trusted(
        output / "checkpoint.pth", map_location=device, weights_only=False
    )
    reloaded = copy.deepcopy(model)
    reloaded.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        probe = torch.as_tensor(np.array(validation[:16]), device=device)
        probe = probe.masked_fill(~torch.isfinite(probe), -9999)
        torch.testing.assert_close(model(probe)[0], reloaded(probe)[0], rtol=0, atol=0)
    write_json(
        output / "checkpoint_verification.json",
        {
            "reload_predictions_equal": True,
            "sha256": sha256(output / "checkpoint.pth"),
            "parent_sha256": manifest["checkpoint_sha256"],
            "probe_rows": len(probe),
        },
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
    parser.add_argument("--diagnostic-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--independent-rng", action="store_true")
    parser.add_argument("--runs", "--arms", dest="runs", nargs="+")
    parser.add_argument("--xp-diagnostics", action="store_true")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--photometry-preflight", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.train_rows,
            args.validation_rows,
            args.presentations,
            args.log_rows,
            args.diagnostic_rows,
        )
        < 1
    ):
        parser.error("All row budgets must be positive")
    if args.output.exists():
        parser.error("Use a new output directory for every experiment suite")
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    expand_config_paths(config)
    if args.benchmark:
        config["data"]["datafile"] = str(args.benchmark)
    if args.micro_batch_size:
        config["training"]["micro_batch_size"] = args.micro_batch_size
    if args.xp_diagnostics:
        config["data"]["uncertainty_units_verified"] = [
            name
            for name in config["data"]["feature_cols"]
            if name.startswith(("bp_", "rp_"))
        ]
        config["data"]["xp_uncertainty_policy"] = (
            "benchmark raw errors divided by G flux factor then scaler scale; BP/RP packed correlations available; G uncertainty and BP/RP cross covariance excluded"
            if args.benchmark
            else "legacy error normalization unverified"
        )
    with open(args.experiments) as stream:
        specification = yaml.safe_load(stream)
        arms = specification.get("experiments", specification.get("arms"))
        arms = [
            {
                **{key: value for key, value in arm.items() if key != "arm_id"},
                "run_id": arm.get("run_id", arm.get("arm_id")),
            }
            for arm in arms
        ]
    if args.runs:
        arms = [arm for arm in arms if arm["run_id"] in args.runs]
        if {arm["run_id"] for arm in arms} != set(args.runs):
            parser.error("Unknown experiment requested")
    if not arms or len({arm["run_id"] for arm in arms}) != len(arms):
        parser.error("Experiment IDs must be unique and nonempty")
    for arm in arms:
        if Path(arm["run_id"]).name != arm["run_id"] or arm["run_id"] in {".", ".."}:
            parser.error("Experiment IDs must be single safe path components")
        if arm.get("embedding", "stable") not in {"legacy", "stable"} or arm.get(
            "loss", "mae"
        ) not in {
            "mae",
            "group_mae",
            "xp_mse",
            "torchregress_noise",
            "signal_weighted_mae",
            "torchregress_covariance",
            "spectral_mae",
            "photometry_mae",
            "photometry_spectral_mae",
            "photometry_decoder_mae",
            "multiscale_spectral_mae",
            "full_decoder_spectral_mae",
        }:
            parser.error("Unknown embedding/loss experiment")
        if arm.get("xp_mask_mode", "whole") not in {
            "whole",
            "coefficients",
            "block",
            "channel",
        }:
            parser.error("Unknown XP mask mode")
        if (
            not math.isfinite(arm.get("variance_floor", 0.1))
            or arm.get("variance_floor", 0.1) <= 0
        ):
            parser.error("Variance floor must be finite and positive")
        for field in (
            "dropout",
            "encoder_dropout",
            "decoder_dropout",
            "xp_masking_ratio",
            "m_masking_ratio",
        ):
            if field in arm and (
                not math.isfinite(arm[field]) or not 0 <= arm[field] <= 1
            ):
                parser.error(f"{field} must be between zero and one")
        try:
            experiment_seed(arm, args.seed)
        except ValueError as error:
            parser.error(str(error))
        warmup = arm.get("warmup_steps", 0)
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            parser.error("Warmup steps must be a nonnegative integer")
        if not math.isfinite(arm.get("lr_factor", 1)) or arm.get("lr_factor", 1) <= 0:
            parser.error("Learning-rate factor must be finite and positive")
        if (
            not math.isfinite(arm.get("spectral_weight", 0.1))
            or arm.get("spectral_weight", 0.1) < 0
        ):
            parser.error("Spectral weight must be finite and nonnegative")
        if (
            not math.isfinite(arm.get("latent_weight", 0))
            or arm.get("latent_weight", 0) < 0
        ):
            parser.error("Latent weight must be finite and nonnegative")
        if (
            not math.isfinite(arm.get("spectral_smoothing_nm", 40))
            or arm.get("spectral_smoothing_nm", 40) <= 0
        ):
            parser.error("Spectral smoothing width must be finite and positive")
    photometry_path = specification.get("photometry_projection")
    if args.photometry_preflight and not photometry_path:
        parser.error("Photometry preflight requires a projection asset")
    for arm in arms:
        for field in ("photometry_weight", "decoder_weight"):
            if not math.isfinite(arm.get(field, 0.01)) or arm.get(field, 0.01) < 0:
                parser.error(f"{field} must be finite and nonnegative")
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
        "optimizer_policy": "inherit checkpoint AdamW groups; optional per-experiment moment reset and LR factor recorded in optimizer_policy.json",
        "source_code_sha256": {
            name: sha256(Path(name))
            for name in (
                "src/masked_stellar_autoencoder/models/model.py",
                "src/masked_stellar_autoencoder/models/blocks.py",
            )
        },
        "runner_sha256": sha256(Path(__file__)),
        "xp_helpers_sha256": sha256(
            Path("src/masked_stellar_autoencoder/training/xp_experiments.py")
        ),
        "representation_helpers_sha256": sha256(
            Path("src/masked_stellar_autoencoder/training/xp_representation.py")
        ),
        "experiment_environment_lock_sha256": sha256(Path("experiments/xp/pixi.lock"))
        if args.xp_diagnostics
        else None,
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
        "validation_seed": args.validation_seed,
        "independent_rng": args.independent_rng,
        "mask_policy": "separate device generator seeded once per experiment; independent of dropout RNG",
        "scheduler_policy": "inherited LR times per-experiment lr_factor; optional linear warmup_steps then fixed; saved parent scheduler state unchanged",
        "selection_metric": "xp_only_mae",
        "selection_policy": "compare ordinary all-column and XP-only held-out MAE; completed experiments only, identical checkpoint, masks, presentations and optimizer steps",
        "runtime": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu",
        },
        "evaluation_only": args.evaluation_only,
        "presentations_per_experiment": 0
        if args.evaluation_only
        else args.presentations,
        "diagnostic_validation_rows": min(args.validation_rows, args.diagnostic_rows),
        "diagnostic_snapshot_policy": "evenly spaced deterministic validation rows; diagnostics only; JSON evaluation uses every validation row",
        "experiments": [
            {**arm, "status": "pending", "output_root": arm["run_id"]} for arm in arms
        ],
    }
    write_json(args.output / "manifest.json", manifest)
    try:
        with h5py.File(config["data"]["datafile"], "r") as source:
            keys, holdout, center, scale = validate_source(checkpoint, config, source)
            needs_sigma = bool(photometry_path) or any(
                arm.get("loss")
                in {
                    "torchregress_noise",
                    "signal_weighted_mae",
                    "torchregress_covariance",
                }
                for arm in arms
            )
            expected_cache_bytes = (
                ((2 if needs_sigma else 1) * args.train_rows + 2 * args.validation_rows)
                * len(center)
                * 4
            )
            needs_covariance = any(
                arm.get("loss") == "torchregress_covariance" for arm in arms
            )
            if needs_covariance:
                if not args.benchmark:
                    raise ValueError(
                        "Covariance experiments require the grouped benchmark"
                    )
                expected_cache_bytes += args.train_rows * 2 * 1485 * 4
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
            cache, train_sigma, train_counts = read_prefixes(
                source,
                keys,
                config["data"]["feature_cols"],
                config["data"]["error_cols"] if needs_sigma else None,
                args.train_rows,
                center,
                scale,
                args.cache_dir / f"{args.output.name}-train.npy",
                correct_xp_errors=bool(args.benchmark),
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
                correct_xp_errors=bool(args.benchmark),
            )
            train_correlations = None
            if args.benchmark:
                audit = Path(str(args.benchmark) + ".manifest.json")
                benchmark_manifest = json.loads(audit.read_text())
                if benchmark_manifest["status"] != "complete":
                    raise ValueError("Benchmark manifest is not complete")
                manifest["benchmark_source_id_sha256"] = benchmark_manifest[
                    "source_id_sha256"
                ]
                manifest["benchmark_manifest_sha256"] = sha256(audit)
                if (
                    len(cache) != args.train_rows
                    or len(validation) != args.validation_rows
                ):
                    raise ValueError(
                        "Benchmark row budgets exceed grouped split counts"
                    )
            if needs_covariance:
                xp_names = [
                    n
                    for n in config["data"]["recon_cols"]
                    if n.startswith(("bp_", "rp_"))
                ]
                if xp_names != [
                    f"{band}_{i}" for band in ("bp", "rp") for i in range(1, 56)
                ]:
                    raise ValueError(
                        "Covariance requires canonical BP/RP reconstruction order"
                    )
                train_correlations = np.lib.format.open_memmap(
                    args.cache_dir / "train-correlations.npy",
                    mode="w+",
                    dtype="f4",
                    shape=(len(cache), 2, 1485),
                )
                for start in range(0, len(cache), 8192):
                    stop = min(start + 8192, len(cache))
                    if not source["train/xp_covariance_available"][start:stop].all():
                        raise ValueError("Benchmark covariance is unavailable")
                    train_correlations[start:stop] = source["train/xp_correlations"][
                        start:stop
                    ]
                train_correlations.flush()
            if not len(cache) or not len(validation):
                raise ValueError("Training and holdout samples must be nonempty")
            manifest["sample"] = {
                "policy": "sequential_prefix_per_shard_v1; deliberately bounded, not population random sampling",
                "train_rows": len(cache),
                "validation_rows": len(validation),
                "train_rows_by_key": train_counts,
                "validation_rows_by_key": validation_counts,
                "source_rows_by_key": {
                    key: len(source[key]) for key in [*keys, *holdout]
                },
                "source_path": str(config["data"]["datafile"]),
                "source_size_bytes": Path(source.filename).stat().st_size,
                "cache_bytes": cache.nbytes
                + validation.nbytes
                + sigma.nbytes
                + (train_sigma.nbytes if train_sigma is not None else 0),
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
            validation_hash.update(str(args.validation_seed).encode())
            masks = fixed_masks(
                validation,
                config["data"]["feature_cols"],
                args.validation_seed,
                config["training"]["xp_masking_ratio"],
                config["training"]["m_masking_ratio"],
                extended=args.xp_diagnostics,
            )
            spectral_projection = (
                spectral_basis(config["data"]["recon_cols"])
                if args.xp_diagnostics
                else None
            )
            if spectral_projection is not None:
                np.savez_compressed(
                    args.output / "spectral_projection.npz",
                    wavelength_nm=spectral_projection["wavelength_nm"],
                    basis=spectral_projection["basis"],
                )
                manifest["spectral_projection"] = {
                    key: value
                    for key, value in spectral_projection.items()
                    if key not in {"basis", "wavelength_nm"}
                }
                manifest["spectral_projection"]["sha256"] = sha256(
                    args.output / "spectral_projection.npz"
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
            photometry = None
            if photometry_path:
                photometry = prepare_photometry(
                    cache,
                    train_sigma,
                    center,
                    scale,
                    config["data"]["feature_cols"],
                    photometry_path,
                    args.output,
                )
                manifest["photometry_calibration_sha256"] = sha256(
                    args.output / "photometry_calibration.json"
                )
                manifest["photometry_projection_sha256"] = sha256(Path(photometry_path))
                manifest["photometry_helpers_sha256"] = sha256(
                    Path("src/masked_stellar_autoencoder/training/xp_photometry.py")
                )
            if args.photometry_preflight:
                manifest["status"] = "complete"
                manifest["preflight_only"] = True
                return
            manifest["status"] = "running"
            write_json(args.output / "manifest.json", manifest)
            for arm in manifest["experiments"]:
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
                    train_sigma,
                    spectral_projection,
                    train_correlations,
                    photometry,
                )
                arm["status"] = "complete"
                write_json(args.output / "manifest.json", manifest)
            manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        for arm in manifest["experiments"]:
            if arm["status"] == "running":
                arm["status"] = "failed"
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
