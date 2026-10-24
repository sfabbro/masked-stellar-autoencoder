"""CANFAR CPU independent sparse-six functional head/source/CP verification.

No production distribution, moment, marginal or loss operators. Reuses only
an encoder architecture loader and fixed pure functional spline/head kernels.
Original source-role proofs are transferred by exact policy/input hashes;
CUDA training masks and exact optimizer trajectories remain separate gates.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("scripts").resolve()))
import independent_lowdim_nsf_kernel as kernel
import run_xp_representation_probes as encoder_architecture
import verify_validation_points as functional

from masked_stellar_autoencoder.models import model as model_architecture
from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compare(actual, expected):
    """Missing-score NaNs must match exactly; all scored values stay finite."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("Prediction shape mismatch")
    missing = np.isnan(expected)
    valid = ~missing
    bad = (np.isnan(actual) != missing) | (
        valid & (~np.isfinite(actual) | ~np.isfinite(expected))
    )
    bad |= valid & ~np.isclose(actual, expected, atol=5e-5, rtol=5e-5, equal_nan=True)
    differences = np.abs(actual[valid] - expected[valid])
    return dict(
        passed=not bool(bad.any()),
        violations=int(bad.sum()),
        maximum_absolute_difference=float(differences.max())
        if differences.size and np.isfinite(differences).all()
        else None,
    )


def independent_coordinates(value, quality, errors, error_known, fit):
    value = np.asarray(value, dtype="f8")
    if np.any(np.asarray(quality, bool) & ~np.isfinite(value)):
        raise ValueError("Observed source evidence contains a nonfinite label")
    mask = np.asarray(quality, bool) & np.isfinite(value)
    if value.shape != mask.shape or value.shape[1] != 6:
        raise ValueError("Six aligned native target columns required")
    coordinate = np.where(mask, value, np.nan)
    for column in (4, 5):
        if np.any(mask[:, column] & (value[:, column] <= 0)):
            raise ValueError("Observed ages/distances must be positive")
        coordinate[:, column] = np.where(
            mask[:, column],
            np.log(np.where(mask[:, column], value[:, column], 1)),
            np.nan,
        )
    center, scale = [], []
    for column in range(6):
        selected = coordinate[np.asarray(fit, bool) & mask[:, column], column]
        if len(selected) < 16 or not np.isfinite(selected).all() or selected.std() <= 0:
            raise ValueError("Insufficient finite FIT target evidence")
        center.append(selected.mean())
        scale.append(selected.std())
    center, scale = np.asarray(center), np.asarray(scale)
    target = np.where(mask, (coordinate - center) / scale, 0)
    known = np.asarray(error_known, bool) & mask
    errors = np.asarray(errors, dtype="f8")
    if known[:, 4:].any():
        raise ValueError("Posterior ages/distances cannot be measurement-error sigmas")
    if np.any(known & (~np.isfinite(errors) | (errors <= 0))):
        raise ValueError("Invalid known native measurement errors")
    sigma = np.where(known, np.where(np.isfinite(errors), errors, 0) / scale, 0)
    jacobian = np.broadcast_to(np.log(scale), value.shape).copy()
    for column in (4, 5):
        jacobian[:, column] += np.log(np.where(mask[:, column], value[:, column], 1))
    return target, sigma, mask, jacobian, center, scale


def substate(state, prefix):
    result = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not result:
        raise ValueError("Required frozen head state missing")
    return result


def normal_log_prob(target, mean, variance):
    return -0.5 * (
        (target - mean).square() / variance + variance.log() + math.log(2 * math.pi)
    )


def lognormal_moments(mean, variance):
    return (mean + variance / 2).exp(), variance.expm1() * (2 * mean + variance).exp()


def independent_head(latent, target, sigma, mask, jacobian, center, scale, state, kind):
    """Native moments, scalar densities and genuine observed native3 joints."""
    if kind in {"gaussian", "mdn"}:
        raw = functional.raw_head(latent, substate(state, "head."))
        if kind == "gaussian":
            mean, lv = raw.chunk(2, -1)
            variance = lv.exp()
            intrinsic = normal_log_prob(target, mean, variance)
            observed = normal_log_prob(target, mean, variance + sigma.square())
            native_mean = center + scale * mean
            native_variance = scale.square() * variance
            age_mean, age_var = lognormal_moments(
                native_mean[:, 4:], native_variance[:, 4:]
            )
            native_mean = torch.cat((native_mean[:, :4], age_mean), -1)
            native_variance = torch.cat((native_variance[:, :4], age_var), -1)
            joint_intrinsic, joint_observed = (
                intrinsic[:, :3].sum(-1),
                observed[:, :3].sum(-1),
            )
        else:
            components = 4
            weight = raw[:, :components].log_softmax(-1)
            means = raw[:, components : components + components * 6].reshape(
                -1, components, 6
            )
            variances = (
                2 * raw[:, components + components * 6 :].reshape(-1, components, 6)
            ).exp()
            component_intrinsic = normal_log_prob(target[:, None], means, variances)
            component_observed = normal_log_prob(
                target[:, None], means, variances + sigma[:, None].square()
            )
            intrinsic = torch.logsumexp(weight[:, :, None] + component_intrinsic, 1)
            observed = torch.logsumexp(weight[:, :, None] + component_observed, 1)
            first = center + scale * means
            component_variance = scale.square() * variances
            age_mean, age_var = lognormal_moments(
                first[..., 4:], component_variance[..., 4:]
            )
            first = torch.cat((first[..., :4], age_mean), -1)
            component_variance = torch.cat((component_variance[..., :4], age_var), -1)
            native_mean = (weight.exp()[:, :, None] * first).sum(1)
            native_variance = (
                weight.exp()[:, :, None]
                * (component_variance + (first - native_mean[:, None]).square())
            ).sum(1)
            joint_intrinsic = torch.logsumexp(
                weight + component_intrinsic[..., :3].sum(-1), 1
            )
            joint_observed = torch.logsumexp(
                weight + component_observed[..., :3].sum(-1), 1
            )
    elif kind == "factorized_nsf":
        native_means, native_variances, intrinsic_columns, observed_columns = (
            [],
            [],
            [],
            [],
        )
        for column in range(6):
            scalar = substate(state, f"heads.{column}.")
            context = functional.raw_head(latent, scalar)
            y = target[:, column : column + 1]
            intrinsic_columns.append(kernel.density(y, context, scalar))
            shifted = (
                y[None]
                + sigma[None, :, column : column + 1]
                * scalar["measurement_nodes"][:, None]
            )
            observed_columns.append(
                torch.logsumexp(kernel.density(shifted, context, scalar), 0)
                - math.log(len(scalar["measurement_nodes"]))
            )
            draws = torch.cat(
                [
                    kernel.inverse(
                        nodes[:, None].expand(-1, len(latent), -1), context, scalar
                    )
                    for nodes in scalar["point_nodes"].split(16)
                ],
                0,
            )
            draws = center[column] + scale[column] * draws
            if column in (4, 5):
                draws = draws.exp()
            mean = draws.mean(0)
            native_means.append(mean)
            native_variances.append((draws - mean).square().mean(0))
        native_mean, native_variance = (
            torch.cat(native_means, -1),
            torch.cat(native_variances, -1),
        )
        intrinsic, observed = (
            torch.stack(intrinsic_columns, -1),
            torch.stack(observed_columns, -1),
        )
        joint_intrinsic, joint_observed = (
            intrinsic[:, :3].sum(-1),
            observed[:, :3].sum(-1),
        )
    else:
        raise ValueError("Unsupported independently verified six-target family")
    complete = mask[:, :3].all(1)
    result = dict(
        mean_native=native_mean,
        variance_native=native_variance,
        intrinsic_log_prob_native=(intrinsic - jacobian).masked_fill(~mask, torch.nan),
        observed_log_prob_native=(observed - jacobian).masked_fill(~mask, torch.nan),
        native3_intrinsic_joint_log_prob_native=(
            joint_intrinsic - jacobian[:, :3].sum(-1)
        ).masked_fill(~complete, torch.nan),
        native3_observed_joint_log_prob_native=(
            joint_observed - jacobian[:, :3].sum(-1)
        ).masked_fill(~complete, torch.nan),
    )
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def independent_row_log_prob(latent, target, sigma, mask, state, kind):
    """Differentiable exact sparse observed likelihood, independent of heads."""
    if kind == "gaussian":
        mean, lv = functional.raw_head(latent, substate(state, "head.")).chunk(2, -1)
        return (
            normal_log_prob(target, mean, lv.exp() + sigma.square())
            .masked_fill(~mask, 0)
            .sum(-1)
        )
    if kind == "mdn":
        raw = functional.raw_head(latent, substate(state, "head."))
        weights = raw[:, :4].log_softmax(-1)
        means = raw[:, 4:28].reshape(-1, 4, 6)
        variance = (2 * raw[:, 28:].reshape(-1, 4, 6)).exp() + sigma[:, None].square()
        component = (
            normal_log_prob(target[:, None], means, variance)
            .masked_fill(~mask[:, None], 0)
            .sum(-1)
        )
        return torch.logsumexp(weights + component, -1)
    total = latent.new_zeros(len(latent))
    for column in range(6):
        keep = mask[:, column]
        if keep.any():
            scalar = substate(state, f"heads.{column}.")
            context = functional.raw_head(latent[keep], scalar)
            shifted = (
                target[None, keep, column : column + 1]
                + sigma[None, keep, column : column + 1]
                * scalar["measurement_nodes"][:, None]
            )
            lp = torch.logsumexp(
                kernel.density(shifted, context, scalar), 0
            ) - math.log(len(scalar["measurement_nodes"]))
            total = total.index_add(0, keep.nonzero().squeeze(1), lp)
    return total


def independent_criterion(
    latent, target, sigma, mask, source_ids, state, kind, mixed_weight=0
):
    total = -independent_row_log_prob(latent, target, sigma, mask, state, kind).sum()
    if mixed_weight:
        if kind != "factorized_nsf":
            raise ValueError("Mixed contrastive loss requires factorized NSF")
        for column in range(6):
            index = mask[:, column].nonzero().squeeze(1)
            if len(index) < 2:
                continue
            if len(torch.unique(source_ids[index])) != len(index):
                raise ValueError("Repeated source context")
            scalar = substate(state, f"heads.{column}.")
            y, error = (
                target[index, column : column + 1],
                sigma[index, column : column + 1],
            )
            shifted = y[None] + error[None] * scalar["measurement_nodes"][:, None]
            contexts = [latent[index], latent[index].roll(1, 0)]
            scores = [
                torch.logsumexp(
                    kernel.density(
                        shifted, functional.raw_head(context, scalar), scalar
                    ),
                    0,
                )
                - math.log(len(scalar["measurement_nodes"]))
                for context in contexts
            ]
            total = (
                total
                + mixed_weight
                * (torch.logaddexp(scores[0], scores[1]) - scores[0]).sum()
            )
    return total / mask.sum()


def validate_qmc_nodes(state, kind):
    if kind != "factorized_nsf":
        return
    for column in range(6):
        for name, count in [("point_nodes", 256), ("measurement_nodes", 32)]:
            uniform = (
                torch.quasirandom.SobolEngine(1, scramble=True, seed=2026)
                .draw(count // 2)
                .double()
            )
            normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
            expected = torch.cat([normal, -normal]).float().double()
            if not torch.equal(state[f"heads.{column}.{name}"], expected):
                raise ValueError("Saved fixed QMC nodes differ from regenerated nodes")


def source_layout_checks(data, prep):
    observed = np.asarray(data["observed"], bool)
    expected = observed & (data["label_source_role"] == data["role"][:, None])
    if not np.array_equal(data["quality"], expected) or not np.array_equal(
        data["original_role_eligible"], expected
    ):
        raise ValueError("Original per-label roles do not imply the saved eligibility")
    if np.any(data["benchmark_partition"][np.isin(data["role"], ["fit", "tune"])] == 1):
        raise ValueError("Benchmark validation entered global FIT/TUNE")
    for role, counts in prep["observed_per_role_target"].items():
        if expected[data["role"] == role].sum(0).tolist() != counts:
            raise ValueError("Source-preparation per-target role counts differ")
    features = list(data["feature_names"])
    allowed = np.array(
        [
            name in {"W1", "W2", "G", "BP", "RP"}
            or name in {f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)}
            for name in features
        ]
    )
    raw = data["raw"]
    visible = allowed[None] & np.isfinite(raw) & (raw != -9999)
    scaled = (raw - data["scaler_center"]) / data["scaler_scale"]
    visible &= np.isfinite(scaled)
    np.testing.assert_array_equal(data["visible"], visible)
    np.testing.assert_array_equal(
        data["input"], np.where(visible, scaled, -9999).astype("f4")
    )
    return dict(
        original_role_eligibility_pass=True,
        original_target_counts_pass=True,
        photometry_xp_only_input_transform_pass=True,
        original_raw_source_receipt_transfer=True,
    )


def finite_quantile(scores, alpha, *, unbounded=False):
    scores = np.asarray(scores, dtype="f8")
    if not len(scores) or not np.isfinite(scores).all() or not 0 < alpha < 1:
        raise ValueError("Finite nonempty calibration evidence required")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        if unbounded:
            return float("inf")
        raise ValueError("Finite threshold unavailable")
    return float(np.partition(scores, rank - 1)[rank - 1])


def check_calibration(cal, test, saved):
    checks = {}
    for alpha in (0.1, 0.05):
        level = str(int(round((1 - alpha) * 100)))
        for column, name in enumerate(NAMES):
            valid = cal["observed"][:, column]
            scores = np.abs(
                cal["mean_native"][valid, column] - cal["target_native"][valid, column]
            )
            q = finite_quantile(scores, alpha)
            simultaneous = finite_quantile(scores, alpha / 6, unbounded=True)
            for mode, threshold in [("marginal", q), ("bonferroni", simultaneous)]:
                for bound, sign in [("lower", -1), ("upper", 1)]:
                    expected = test["mean_native"][:, column] + sign * threshold
                    actual = saved[f"{level}_{mode}_{name}_{bound}"]
                    if np.isinf(expected).any():
                        passed = bool(np.array_equal(actual, expected))
                        checks[f"{level}:{mode}:{name}:{bound}"] = dict(
                            passed=passed, violations=int((actual != expected).sum())
                        )
                    else:
                        checks[f"{level}:{mode}:{name}:{bound}"] = compare(
                            actual, expected
                        )
            for estimand in ["intrinsic", "observed"]:
                density = finite_quantile(
                    -cal[estimand + "_log_prob_native"][valid, column], alpha
                )
                checks[f"{level}:{estimand}_density:{name}"] = compare(
                    saved[f"{level}_{estimand}_density_{name}_threshold"],
                    np.asarray(density),
                )
    np.testing.assert_array_equal(saved["test_observed"], test["observed"])
    np.testing.assert_array_equal(saved["calibration_observed"], cal["observed"])
    np.testing.assert_array_equal(saved["test_source_id"], test["source_id"])
    np.testing.assert_array_equal(saved["calibration_source_id"], cal["source_id"])
    return checks


def check_state_and_streams(checkpoint, manifest, data, mask, metrics, policy):
    selected = int(checkpoint["completed_updates"])
    budget = math.ceil(
        policy["reconstruction_presentations"] / policy["effective_batch"]
    )
    if manifest["training_budget"]["updates"] != budget or len(metrics) != budget:
        raise ValueError("Actual update budget differs")
    if (
        checkpoint["source_sha"] != manifest["source_sha"]
        or checkpoint["arm"] != manifest["arm"]
    ):
        raise ValueError("Checkpoint producer identity differs")
    if selected != manifest["selected_model_counters"]["updates"]:
        raise ValueError("Selected checkpoint counter differs")
    scores = [row for row in metrics if "tune_macro_native_nll" in row]
    if min(scores, key=lambda row: row["tune_macro_native_nll"])["updates"] != selected:
        raise ValueError("Selection was not first TUNE minimum")
    row_stream = hashlib.sha256()
    expected_rows = []
    seen = 0
    while seen < policy["reconstruction_presentations"]:
        for start in range(0, 9000000, policy["effective_batch"]):
            stop = min(
                start + policy["effective_batch"],
                9000000,
                start + policy["reconstruction_presentations"] - seen,
            )
            if stop <= start:
                break
            row_stream.update(np.arange(start, stop, dtype="<i8").tobytes())
            seen += stop - start
            expected_rows.append((seen, row_stream.hexdigest()))
            if seen == policy["reconstruction_presentations"]:
                break
    if (
        len(expected_rows) != len(metrics)
        or row_stream.hexdigest() != manifest["row_stream_sha256"]
    ):
        raise ValueError("Independent reconstruction position stream differs")
    for update, (presentations, stream_sha) in enumerate(expected_rows, 1):
        row = metrics[update - 1]
        if (
            row["updates"] != update
            or row["presentations"] != presentations
            or row["reconstruction_stream_sha256"] != stream_sha
        ):
            raise ValueError("Actual row counters or cumulative position hashes differ")
    if checkpoint["completed_presentations"] != expected_rows[selected - 1][0]:
        raise ValueError("Selected presentation counter differs from actual stream")
    if manifest["mask_stream_sha256"] != metrics[-1]["mask_stream_sha256"]:
        raise ValueError("Mask receipt disagrees with final metrics")
    rng = np.random.default_rng(manifest["arm"]["seed"])
    fit = np.flatnonzero(data["role"] == "fit")
    stream = hashlib.sha256()
    selected_rng = None
    for draw in range(
        policy["warmup_updates"] + budget * policy["supervised_batches_per_update"]
    ):
        blocks = []
        for column in range(6):
            candidates = fit[mask[fit, column]]
            blocks.append(
                rng.choice(candidates, min(48, len(candidates)), replace=False)
            )
        rows = np.unique(np.concatenate(blocks))
        stream.update(np.asarray(rows, dtype="<i8").tobytes())
        after_warmup = draw + 1 - policy["warmup_updates"]
        if (
            after_warmup > 0
            and after_warmup % policy["supervised_batches_per_update"] == 0
        ):
            update_index = after_warmup // policy["supervised_batches_per_update"] - 1
            if stream.hexdigest() != metrics[update_index]["label_stream_sha256"]:
                raise ValueError("Intermediate FIT label stream hash differs")
        if (
            draw + 1
            == policy["warmup_updates"]
            + selected * policy["supervised_batches_per_update"]
        ):
            selected_rng = json.loads(json.dumps(rng.bit_generator.state))
    if stream.hexdigest() != manifest["label_stream_sha256"]:
        raise ValueError("FIT-only balanced label draw stream differs")
    if checkpoint["supervised_rng_state"] != selected_rng:
        raise ValueError("Selected supervised RNG state differs")
    model_steps = [
        float(item["step"])
        for item in checkpoint["model_optimizer_state_dict"]["state"].values()
    ]
    head_steps = [
        float(item["step"])
        for item in checkpoint["head_optimizer_state_dict"]["state"].values()
    ]
    if not model_steps or set(model_steps) != {float(selected)}:
        raise ValueError("Model Adam steps differ from selected updates")
    expected_head = policy["warmup_updates"] + selected
    allowed = (
        {float(expected_head), float(selected)}
        if manifest["arm"].get("auxiliary")
        else {float(expected_head)}
    )
    if (
        not head_steps
        or float(expected_head) not in head_steps
        or not set(head_steps) <= allowed
    ):
        raise ValueError("Head/auxiliary Adam steps differ")
    for key in [
        "rng_state",
        "cuda_rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "mask_rng_state",
        "auxiliary_view_rng_state",
    ]:
        if key not in checkpoint:
            raise ValueError("Missing full-state RNG receipt")
    for optimizer_name in ["model_optimizer_state_dict", "head_optimizer_state_dict"]:
        for item in checkpoint[optimizer_name]["state"].values():
            for name, value in item.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError("Nonfinite selected Adam state")
                if name == "exp_avg_sq" and (value < 0).any():
                    raise ValueError("Negative selected Adam second moment")
    return dict(
        label_draw_stream_pass=True,
        selected_rng_pass=True,
        first_tune_minimum_pass=True,
        model_adam_steps=selected,
        head_adam_steps=expected_head,
        reconstruction_position_stream_pass=True,
        reconstruction_raw_source_gate="source SHA transferred; actual benchmark target/mask tensors not replayed",
        cuda_training_mask_gate="not regenerated on CPU; saved SHA retained",
        exact_trajectory_gate="not replayed",
    )


def check_frozen_state_and_streams(checkpoint, manifest, data, mask, metrics, control):
    selected = int(checkpoint["completed_updates"])
    warmup = int(control["warmup_updates"])
    updates = int(control["head_optimizer_updates"])
    groups = int(control["supervised_batches_per_update"])
    budget = manifest["training_budget"]
    if (
        len(metrics) != updates
        or budget["head_optimizer_updates"] != updates
        or budget["warmup_updates"] != warmup
        or budget["supervised_batches_per_update"] != groups
    ):
        raise ValueError("Frozen head schedule differs from declared budget")
    for key in ["encoder_updates", "decoder_updates", "reconstruction_presentations"]:
        if checkpoint.get(key) != 0 or budget.get(key) != 0:
            raise ValueError("Frozen control claimed model or reconstruction updates")
    if checkpoint["completed_presentations"] != 0 or not manifest.get(
        "no_10m_training_claim"
    ):
        raise ValueError("Frozen head control cannot claim10M reconstruction")
    if (
        checkpoint["source_sha"] != manifest["source_sha"]
        or checkpoint["arm"] != manifest["arm"]
    ):
        raise ValueError("Frozen selected state producer identity differs")
    if (
        selected != manifest["selected_head_counters"]["head_updates"]
        or checkpoint["completed_head_adam_steps"] != warmup + selected
        or manifest["selected_head_counters"]["head_adam_steps"] != warmup + selected
    ):
        raise ValueError("Frozen selected head counters differ")
    scores = [row for row in metrics if "tune_macro_native_nll" in row]
    if min(scores, key=lambda row: row["tune_macro_native_nll"])["updates"] != selected:
        raise ValueError("Frozen selection is not first TUNE minimum")
    fit = np.flatnonzero(data["role"] == "fit")
    rng = np.random.default_rng(manifest["arm"]["seed"])
    stream = hashlib.sha256()
    selected_rng = selected_hash = selected_counts = None
    counts = np.zeros(6, dtype="i8")
    for draw in range(warmup + updates * groups):
        blocks = []
        for column in range(6):
            candidates = fit[mask[fit, column]]
            blocks.append(
                rng.choice(candidates, min(48, len(candidates)), replace=False)
            )
        rows = np.unique(np.concatenate(blocks))
        counts += mask[rows].sum(0)
        stream.update(np.asarray(rows, dtype="<i8").tobytes())
        after_warmup = draw + 1 - warmup
        if after_warmup > 0 and after_warmup % groups == 0:
            update = after_warmup // groups
            row = metrics[update - 1]
            if (
                row["updates"] != update
                or row["head_adam_steps"] != warmup + update
                or row["label_stream_sha256"] != stream.hexdigest()
                or row["encoder_updates"] != 0
                or row["reconstruction_presentations"] != 0
            ):
                raise ValueError("Frozen head intermediate stream/counters differ")
            if update == selected:
                selected_rng = json.loads(json.dumps(rng.bit_generator.state))
                selected_hash = stream.hexdigest()
                selected_counts = counts.copy()
    if (
        stream.hexdigest() != manifest["label_stream_sha256"]
        or counts.tolist() != manifest["supervised_label_presentations"]
    ):
        raise ValueError("Frozen final FIT draw stream/counts differ")
    if (
        checkpoint["supervised_rng_state"] != selected_rng
        or checkpoint["label_stream_sha256"] != selected_hash
    ):
        raise ValueError("Frozen selected source RNG/hash differs")
    np.testing.assert_array_equal(
        checkpoint["supervised_label_presentations"], selected_counts
    )
    states = checkpoint["head_optimizer_state_dict"]["state"]
    if not states or {float(item["step"]) for item in states.values()} != {
        float(warmup + selected)
    }:
        raise ValueError("Frozen selected head Adam steps differ")
    for item in states.values():
        for name, value in item.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite frozen head Adam state")
            if name == "exp_avg_sq" and (value < 0).any():
                raise ValueError("Negative frozen head Adam second moment")
    for key in ["rng_state", "cuda_rng_state", "numpy_rng_state", "python_rng_state"]:
        if key not in checkpoint:
            raise ValueError("Frozen selected state missing RNG receipt")
    if (
        not manifest.get("frozen_model_values_unchanged")
        or manifest["frozen_model_parameter_sha256"]
        != manifest["frozen_model_parameter_sha256_after"]
        or checkpoint["frozen_model_parameter_sha256"]
        != manifest["frozen_model_parameter_sha256"]
    ):
        raise ValueError("Frozen encoder producer state digest differs")
    return dict(
        frozen_control=True,
        encoder_updates=0,
        decoder_updates=0,
        reconstruction_presentations=0,
        head_adam_steps=warmup + selected,
        full_declared_head_adam_budget=warmup + updates,
        selected_rng_pass=True,
        label_draw_stream_pass=True,
        first_tune_minimum_pass=True,
        cuda_training_mask_gate="not applicable to frozen control",
        exact_trajectory_gate="not replayed",
    )


def encoder_source_digest():
    directory = Path(model_architecture.__file__).parent
    result = hashlib.sha256()
    for source in sorted(directory.rglob("*.py")):
        result.update(str(source.relative_to(directory)).encode())
        result.update(source.read_bytes())
    return result.hexdigest()


def model_parameter_digest(model):
    result = hashlib.sha256()
    for key, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        result.update(key.encode())
        result.update(str(value.dtype).encode())
        result.update(np.asarray(value.shape, dtype="<i8").tobytes())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def encode_inputs(model, data, rows, latent_cache=None):
    """Shared immutable frozen-encoder proof cache, not a trainable-head cache."""
    if latent_cache is None:
        return model.encoder(torch.tensor(data["input"][rows], dtype=torch.float64))
    ids = np.asarray(data["source_id"][rows], dtype="<i8")
    key = hashlib.sha256(ids.tobytes()).hexdigest()
    path = latent_cache / (key + ".npz")
    lock = path.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if path.exists():
            saved = dict(np.load(path, allow_pickle=False))
            np.testing.assert_array_equal(saved["source_id"], ids)
            if (
                saved["latent"].dtype != np.float64
                or not np.isfinite(saved["latent"]).all()
            ):
                raise ValueError("Corrupt shared frozen encoder cache")
            return torch.tensor(saved["latent"], dtype=torch.float64)
        latent = model.encoder(torch.tensor(data["input"][rows], dtype=torch.float64))
        values = latent.detach().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite frozen encoder output")
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, source_id=ids, latent=values)
        temporary.replace(path)
        return latent
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)


def bounded_objective_contract(
    model, state, data, y, sigma, mask, arm, output, latent_cache=None
):
    candidates = np.flatnonzero(data["role"] == "fit")
    rows = np.unique(
        np.concatenate(
            [candidates[mask[candidates, column]][:16] for column in range(6)]
        )
    )
    with torch.no_grad():
        values = encode_inputs(model, data, rows, latent_cache)
    latent = values.detach().requires_grad_(True)
    args = (
        torch.tensor(y[rows]),
        torch.tensor(sigma[rows]),
        torch.tensor(mask[rows]),
        torch.tensor(data["source_id"][rows].astype("i8")),
        state,
        arm["kind"],
        arm.get("mixed_weight", 0),
    )
    loss = independent_criterion(latent, *args)
    gradient = torch.autograd.grad(loss, latent)[0]
    array = dict(
        rows=rows,
        source_id=data["source_id"][rows],
        latent=values.numpy(),
        objective=np.asarray(float(loss)),
        latent_gradient=gradient.detach().numpy(),
    )
    np.savez_compressed(output / "manual-objective-contract.npz", **array)
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite independent actual FIT objective/gradient contract")
    position = int(gradient.abs().argmax())
    coordinate = np.unravel_index(position, gradient.shape)
    lower, upper = values.clone(), values.clone()
    lower[coordinate] -= 1e-6
    upper[coordinate] += 1e-6
    with torch.no_grad():
        finite_difference = float(
            (independent_criterion(upper, *args) - independent_criterion(lower, *args))
            / 2e-6
        )
    derivative = float(gradient[coordinate])
    derivative_pass = math.isclose(
        finite_difference, derivative, abs_tol=5e-5, rel_tol=5e-5
    )
    report = dict(
        status="manual_contract_pass" if derivative_pass else "manual_contract_failed",
        actual_fit_rows=len(rows),
        objective=float(loss),
        derivative=derivative,
        independent_central_difference=finite_difference,
        derivative_tolerance=dict(atol=5e-5, rtol=5e-5),
        sparse_likelihood_denominator="total observed-coordinate count",
        both_positive_negative_contexts_convolved=bool(arm.get("mixed_weight", 0)),
        actual_producer_gradient_replay_or_optimizer_trajectory=False,
    )
    atomic(output / "manual-objective-contract.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final-six-independent-20261024.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError(
            "Native independent model verification must run on CANFAR CPU headless"
        )
    cfg = yaml.safe_load(args.config.read_text())
    suite = yaml.safe_load(Path(cfg["training_policy"]).read_text())
    library_source = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if library_source.get("vcs_info", {}).get("commit_id") != suite["library_revision"]:
        raise ValueError("Verifier environment library SHA differs")
    directory = Path(cfg["base"]) / args.run_id
    args.output.mkdir(parents=True, exist_ok=True)
    singleton = (args.output / "singleton.lock").open("w")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["source_sha"] != args.expected_source_sha
        or manifest.get("smoke")
    ):
        raise ValueError("Actual complete production manifest/source required")
    frozen_policy = yaml.safe_load(Path(cfg["frozen_policy"]).read_text())
    joint_arms = {value["run_id"]: value for value in suite["arms"]}
    frozen_arms = {value["run_id"]: value for value in frozen_policy["arms"]}
    if args.run_id not in joint_arms and args.run_id not in frozen_arms:
        raise ValueError("Unknown immutable six-target audit arm")
    frozen_control = args.run_id in frozen_arms
    arm = (frozen_arms if frozen_control else joint_arms)[args.run_id]
    if (
        manifest["arm"] != arm
        or manifest["library_revision"] != suite["library_revision"]
    ):
        raise ValueError("Immutable family/library policy differs")
    if sha(directory / "checkpoint.pth") != manifest["checkpoint_sha256"]:
        raise ValueError("Saved checkpoint SHA differs")
    prep = json.loads(Path(suite["preparation_manifest"]).read_text())
    if (
        sha(suite["prepared"]) != suite["prepared_sha256"]
        or prep["input_sha256"] != suite["prepared_sha256"]
    ):
        raise ValueError("Source-prepared SHA differs")
    if (
        sha(suite["preparation_manifest"]) != suite["preparation_manifest_sha256"]
        or prep["source_commit"] != suite["preparation_source_sha"]
    ):
        raise ValueError("Source-preparation provenance differs")
    expected_preparation = yaml.safe_load(Path(suite["preparation_policy"]).read_text())
    if prep["source_provenance"] != expected_preparation or not prep.get(
        "role_promoted_labels_excluded_from_scoring"
    ):
        raise ValueError("Original source policy or heldout-label exclusions differ")
    bindings = dict(
        source=args.expected_source_sha,
        checkpoint=manifest["checkpoint_sha256"],
        input=suite["prepared_sha256"],
        policy=sha(cfg["training_policy"]),
        frozen_policy=sha(cfg["frozen_policy"]) if frozen_control else None,
        frozen_parent=suite["checkpoint_sha256"] if frozen_control else None,
        frozen_reference=suite["reference_checkpoint_sha256"]
        if frozen_control
        else None,
        producer_manifest=sha(directory / "manifest.json"),
        test=sha(directory / "test-predictions.npz"),
        cal=sha(directory / "cal-predictions.npz"),
        metrics=sha(directory / "metrics.json"),
        conformal=sha(directory / "conformal.npz"),
        verifier=sha(__file__),
        kernel=sha(kernel.__file__),
        functional_helper=sha(functional.__file__),
        encoder_architecture=sha(encoder_architecture.__file__),
        encoder_implementation=encoder_source_digest(),
        torch_version=torch.__version__,
    )
    bindings_path = args.output / "bindings.json"
    if bindings_path.exists() and json.loads(bindings_path.read_text()) != bindings:
        raise ValueError("Changed input/code bindings; never reuse cached passes")
    atomic(bindings_path, bindings)
    report_path = args.output / ("pilot-report.json" if args.pilot else "report.json")
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("status") in {
            "pilot_strict_pass",
            "complete_strict_pass",
            "complete_strict_failed",
            "pilot_strict_failed",
        }:
            return
    archive = args.output / "independent-code"
    archive.mkdir(exist_ok=True)
    for source in [Path(__file__), Path(kernel.__file__), Path(functional.__file__)]:
        shutil.copyfile(source, archive / source.name)
    data = dict(np.load(suite["prepared"], allow_pickle=False))
    ids, roles = data["source_id"], data["role"]
    if (
        len(np.unique(ids)) != len(ids)
        or tuple(data["target_names"]) != NAMES
        or set(roles) != {"fit", "tune", "cal", "test"}
    ):
        raise ValueError("Source IDs, roles or target order differ")
    source_checks = source_layout_checks(data, prep)
    quality = np.asarray(data["quality"], bool)
    if not np.array_equal(quality, data["observed"] & data["original_role_eligible"]):
        raise ValueError("Original per-label source-role eligibility differs")
    y, sigma, mask, jacobian, center, scale = independent_coordinates(
        data["value"],
        quality,
        data["error"],
        data["error_observed"] & quality,
        roles == "fit",
    )
    checkpoint = torch_load_trusted(
        directory / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    np.testing.assert_array_equal(checkpoint["label_center"], center)
    np.testing.assert_array_equal(checkpoint["label_scale"], scale)
    if (
        checkpoint["input_sha256"] != suite["prepared_sha256"]
        or tuple(checkpoint["target_names"]) != NAMES
    ):
        raise ValueError("Checkpoint normalization/input identity differs")
    state = {
        key: value.double() if value.is_floating_point() else value
        for key, value in checkpoint["head_state_dict"].items()
    }
    if any(
        value.is_floating_point() and not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Nonfinite trained head state")
    validate_qmc_nodes(state, arm["kind"])
    metrics = json.loads((directory / "metrics.json").read_text())
    if frozen_control:
        if (
            manifest["control_policy"] != frozen_policy
            or manifest["source_policy"] != suite
        ):
            raise ValueError("Frozen source/control policy differs")
        training_checks = check_frozen_state_and_streams(
            checkpoint, manifest, data, mask, metrics, frozen_policy
        )
        if (
            sha(suite["checkpoint"]) != suite["checkpoint_sha256"]
            or sha(suite["reference_checkpoint"])
            != suite["reference_checkpoint_sha256"]
        ):
            raise ValueError(
                "Frozen encoder or feature-reference checkpoint SHA differs"
            )
        weights = torch_load_trusted(
            suite["checkpoint"], map_location="cpu", weights_only=False
        )
        reference = torch_load_trusted(
            suite["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        if (
            weights.get("source_sha") != suite["large_checkpoint_source_sha"]
            or weights.get("presentations") != 4500000000
            or weights.get("completed_updates") != 137340
            or weights.get("parent_sha256") != suite["reference_checkpoint_sha256"]
        ):
            raise ValueError("Frozen large-model lineage differs")
        if (
            checkpoint["parent_sha256"] != suite["checkpoint_sha256"]
            or manifest["parent_sha256"] != suite["checkpoint_sha256"]
        ):
            raise ValueError("Frozen head parent binding differs")
        model_checkpoint = dict(reference, model_state_dict=weights["model_state_dict"])
    else:
        training_checks = check_state_and_streams(
            checkpoint, manifest, data, mask, metrics, suite
        )
        model_checkpoint = checkpoint
    for name in ["scaler_center", "scaler_scale"]:
        np.testing.assert_array_equal(
            data[name], model_checkpoint["run_signature"][name]
        )
    if list(data["feature_names"]) != model_checkpoint["run_signature"]["feature_cols"]:
        raise ValueError("Frozen encoder feature signature differs")
    torch.set_num_threads(cfg.get("threads", 4))
    model = encoder_architecture.load_model(
        yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
        model_checkpoint,
        "cpu",
    )
    if any(
        value.is_floating_point() and not torch.isfinite(value).all()
        for value in model.state_dict().values()
    ):
        raise ValueError("Nonfinite selected encoder/model state")
    if (
        frozen_control
        and model_parameter_digest(model) != manifest["frozen_model_parameter_sha256"]
    ):
        raise ValueError("Independently loaded frozen model value digest differs")
    model = model.double().eval()
    latent_cache = None
    if frozen_control:
        latent_bindings = dict(
            parent=suite["checkpoint_sha256"],
            reference=suite["reference_checkpoint_sha256"],
            input=suite["prepared_sha256"],
            encoder_loader=sha(encoder_architecture.__file__),
            encoder_implementation=encoder_source_digest(),
            architecture_config=sha("configs/pretrain.canfar.yaml"),
            torch_version=torch.__version__,
            verifier_encoder_cache_version="independent-frozen-encoder-fp64-v1",
        )
        key = hashlib.sha256(
            json.dumps(latent_bindings, sort_keys=True).encode()
        ).hexdigest()
        latent_cache = Path(cfg["frozen_encoder_cache"]) / key
        latent_cache.mkdir(parents=True, exist_ok=True)
        cache_bindings_path = latent_cache / "bindings.json"
        binding_lock = cache_bindings_path.with_suffix(".lock").open("w")
        fcntl.flock(binding_lock, fcntl.LOCK_EX)
        try:
            if cache_bindings_path.exists():
                if json.loads(cache_bindings_path.read_text()) != latent_bindings:
                    raise ValueError("Shared frozen encoder cache binding differs")
            else:
                atomic(cache_bindings_path, latent_bindings)
        finally:
            fcntl.flock(binding_lock, fcntl.LOCK_UN)
    objective_path = args.output / "manual-objective-contract.json"
    objective_report = (
        json.loads(objective_path.read_text())
        if objective_path.exists()
        else bounded_objective_contract(
            model, state, data, y, sigma, mask, arm, args.output, latent_cache
        )
    )
    report = dict(
        status="running",
        bindings=bindings,
        checks={},
        training_checks=training_checks,
        source_checks=source_checks,
        independent_manual_objective_contract=objective_report,
        producer_density_scope=manifest.get("density_scope", manifest["scope"]),
        pilot=args.pilot,
        frozen_control=frozen_control,
        shared_frozen_encoder_cache=str(latent_cache)
        if latent_cache is not None
        else None,
        unresolved=[
            "CUDA training masks not independently regenerated"
            if not frozen_control
            else "No CUDA training-mask/reconstruction gate is applicable to frozen controls",
            "raw benchmark reconstruction targets/trajectory not replayed"
            if not frozen_control
            else "Frozen FIT FP32 embeddings not independently replayed; final raw-input FP64 predictions independently checked",
            "ancestral per-ID auxiliary supervision not independently excluded",
        ],
        physical_truth_claim=False,
        empirical_six_dimensional_copula_identified=False,
    )
    role_results = {}
    role_specs = {}
    tasks = []
    failed = objective_report["status"] != "manual_contract_pass"
    if failed:
        report["status"] = "pilot_strict_failed"
        atomic(report_path, report)
        return
    for role in ["cal", "test"]:
        all_rows = np.flatnonzero(roles == role)
        rows = all_rows[:128] if args.pilot else all_rows
        saved = dict(np.load(directory / f"{role}-predictions.npz", allow_pickle=False))
        np.testing.assert_array_equal(saved["source_id"], ids[all_rows])
        np.testing.assert_array_equal(saved["observed"], mask[all_rows])
        np.testing.assert_array_equal(saved["target_native"], data["value"][all_rows])
        np.testing.assert_array_equal(
            saved["label_source"], data["label_source"][all_rows]
        )
        role_specs[role] = dict(rows=rows, saved=saved, parts=[])
        for start in range(0, len(rows), 128):
            tasks.append((start != 0, role, start, rows[start : start + 128]))
    # Both first-role pilots must pass before any remaining cal/test model batch.
    tasks.sort(key=lambda item: (item[0], item[1], item[2]))
    pilot_roles_passed = set()
    for _, role, start, batch in tasks:
        saved = role_specs[role]["saved"]
        cache = args.output / f"{role}-{start:06d}.npz"
        if cache.exists():
            result = dict(np.load(cache, allow_pickle=False))
            np.testing.assert_array_equal(result.pop("source_id"), ids[batch])
        else:
            with torch.no_grad():
                latent = encode_inputs(model, data, batch, latent_cache)
                result = independent_head(
                    latent,
                    torch.tensor(y[batch]),
                    torch.tensor(sigma[batch]),
                    torch.tensor(mask[batch]),
                    torch.tensor(jacobian[batch]),
                    torch.tensor(center),
                    torch.tensor(scale),
                    state,
                    arm["kind"],
                )
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, source_id=ids[batch], **result)
            temporary.replace(cache)
        for key, value in result.items():
            verdict = compare(value, saved[key][start : start + len(batch)])
            report["checks"][f"{role}:{start}:{key}"] = verdict
            failed |= not verdict["passed"]
        role_specs[role]["parts"].append((start, result))
        atomic(
            args.output / "progress.json",
            dict(
                role=role,
                processed=start + len(batch),
                total=len(role_specs[role]["rows"]),
                any_strict_failure=failed,
                bindings=bindings,
            ),
        )
        if start == 0:
            if failed:
                report["status"] = "pilot_strict_failed"
                atomic(report_path, report)
                atomic(args.output / "pilot-report.json", report)
                return
            pilot_roles_passed.add(role)
            if pilot_roles_passed == {"cal", "test"}:
                pilot_report = dict(report, status="pilot_strict_pass", pilot=True)
                atomic(args.output / "pilot-report.json", pilot_report)
    for role, spec in role_specs.items():
        ordered = [
            piece for _, piece in sorted(spec["parts"], key=lambda item: item[0])
        ]
        role_results[role] = {
            key: np.concatenate([part[key] for part in ordered]) for key in ordered[0]
        }
        rows = spec["rows"]
        role_results[role].update(
            source_id=ids[rows], target_native=data["value"][rows], observed=mask[rows]
        )
    if not args.pilot:
        calibration = check_calibration(
            role_results["cal"],
            role_results["test"],
            dict(np.load(directory / "conformal.npz", allow_pickle=False)),
        )
        report["calibration_checks"] = calibration
        failed |= any(not value["passed"] for value in calibration.values())
    report["status"] = (
        ("pilot_strict_failed" if failed else "pilot_strict_pass")
        if args.pilot
        else ("complete_strict_failed" if failed else "complete_strict_pass")
    )
    report["library_version"] = importlib.metadata.version("torchregress")
    report["library_source"] = library_source
    report["full_pipeline_or_physical_truth_pass"] = False
    atomic(report_path, report)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            failure_directory = Path(sys.argv[sys.argv.index("--output") + 1])
            failure_directory.mkdir(parents=True, exist_ok=True)
            atomic(
                failure_directory / "failure.json",
                dict(
                    status="failed", error_type=type(error).__name__, error=str(error)
                ),
            )
        raise
