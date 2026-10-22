"""Bounded audited native three-label MSA experiments, one immutable arm per job.

Joint densities use complete Teff/logg/FeH rows. Forward inversion is a separate
XP-only experiment. CP and label-free TTT are inference-only postprocessing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import run_pretrain_experiments as pretrain
import run_xp_representation_probes as probe
import torch
import yaml
from run_prediction_comparison import reconstruct, replay_batches

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.input_measurement import (
    audit_xp_error_sources,
    xp_indices,
)
from masked_stellar_autoencoder.training.multilabel_experiments import (
    NON_DENSITY_KINDS,
    MultiLabelHead,
    complete_roles,
    conformal_intervals,
    hash_selected,
    label_scaling,
    native_xp_error_subset,
    qmc_normal_nodes,
    refine_one,
    supervised_fit_rows,
)
from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)
from masked_stellar_autoencoder.training.targets import (
    REGISTRY,
    density_offsets,
    log_abs_det_jacobian,
    model_coordinates,
    native_from_standardized,
    validate_targets,
)
from masked_stellar_autoencoder.training.targets import (
    TARGET_NAMES as REGISTRY_TARGET_NAMES,
)

COHORT_SHA = "321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120"


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def interval_crossed(previous, current, interval):
    """A short boundary batch still commits exactly one crossed checkpoint."""
    if not interval:
        return False
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("Checkpoint/snapshot intervals must be positive integers")
    return previous // interval < current // interval


def _assert_checkpoint_equal(expected, actual):
    if torch.is_tensor(expected):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert torch.equal(expected.detach().cpu(), actual.cpu())
    elif isinstance(expected, np.ndarray):
        assert expected.dtype == actual.dtype
        np.testing.assert_array_equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_checkpoint_equal(expected[key], actual[key])
    elif isinstance(expected, list | tuple):
        assert type(expected) is type(actual) and len(expected) == len(actual)
        for left, right in zip(expected, actual, strict=True):
            _assert_checkpoint_equal(left, right)
    else:
        assert expected == actual


def save_periodic_checkpoint(output, state):
    """Keep immutable full states, reload every tensor, publish receipt last."""
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"checkpoint-{state['completed_presentations']:012d}.pth"
    if path.exists():
        raise FileExistsError(f"Refusing to replace periodic checkpoint: {path}")
    save_checkpoint(path, state)
    restored = torch_load_trusted(path, map_location="cpu", weights_only=False)
    _assert_checkpoint_equal(state, restored)
    receipt = {
        "file": str(path.relative_to(output)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "presentations": state["completed_presentations"],
        "updates": state["completed_updates"],
        "all_saved_state_values_reload_equal": True,
    }
    atomic(path.with_suffix(".json"), receipt)
    atomic(output / "latest-checkpoint.json", receipt)
    return receipt


def select_by_reserved_tuning(output, receipt, data, rows, result):
    """Select only by the declared disjoint tuning likelihood.

    The density estimand is explicit: a noise-aware arm is selected on the
    observed-label density, a plain density arm on the intrinsic one. The
    observed-label array is NaN (never silently aliased) when no measurement
    model was declared.
    """
    dim = len(np.asarray(data["label_center"]))
    observed = np.asarray(
        result.get("observed_log_prob_native", np.full(len(rows), np.nan)), dtype="f8"
    )
    if "observed_log_prob_available" in result:
        available = bool(np.asarray(result["observed_log_prob_available"]).item())
    else:
        # Legacy in-memory mapping: the observed array was the selection key.
        available = True
    if available and np.isfinite(observed).all():
        values, density_kind = observed, "observed_label"
    else:
        values = np.asarray(
            result.get("intrinsic_log_prob_native", np.full(len(rows), np.nan)),
            dtype="f8",
        )
        density_kind = "intrinsic"
    if values.shape != (len(rows),) or not len(rows) or not np.isfinite(values).all():
        raise ValueError("Finite joint tuning densities required for selection")
    score = float(-values.mean() / dim)
    prediction = native_from_standardized(
        target_names(data), result["mean"], data["label_center"], data["label_scale"]
    )
    path = output / f"tuning-{receipt['presentations']:012d}.npz"
    np.savez_compressed(
        path,
        rows=rows,
        source_id=data["source_id"][rows],
        value=data["value"][rows],
        error=data["error"][rows],
        **result,
    )
    summary = {
        **receipt,
        "selection_policy": "reserved_tune_mean_observed_nll",
        "tuning_mean_observed_nll_per_label_native": score,
        "tuning_density_kind": density_kind,
        "tuning_loss_denominator": str(
            np.asarray(result.get("loss_denominator", "per_coordinate")).item()
        ),
        "tuning_mae_native": np.abs(prediction - data["value"][rows]).mean(0).tolist(),
        "tuning_rows": len(rows),
        "tuning_ordered_source_id_sha256": hashlib.sha256(
            data["source_id"][rows].astype("<i8").tobytes()
        ).hexdigest(),
        "tuning_predictions_file": path.name,
        "tuning_predictions_sha256": sha(path),
        "calibration_or_test_used_for_selection": False,
    }
    atomic(path.with_suffix(".json"), summary)
    selected_path = output / "selected-checkpoint.json"
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else None
    if (
        selected is None
        or score < selected["tuning_mean_observed_nll_per_label_native"]
    ):
        atomic(selected_path, summary)
    return summary


def load_native(cfg, cohort, sidecar, qa, features, center, scale, *, synthetic=False):
    with h5py.File(cohort) as f:
        ids = f["features/source_id"][:]
        raw = f["features/values"][:]
        feature_names = json.loads(f.attrs["feature_columns_json"])
        assert feature_names == features
        x, visible = spectrophotometric_inputs(
            torch.tensor(((raw - center) / scale).astype("f4")),
            torch.tensor(np.isfinite(raw)),
            features,
            include_reddening=False,
        )
        errors = f["features/reported_feature_errors"][:]
    if synthetic:
        # Only a plumbing fixture: no native-label/source/GPU certification.
        keep = np.flatnonzero(visible.numpy()[:, xp_indices(features)].sum(1) >= 2)[
            :256
        ]
        ids = ids[keep]
        raw = raw[keep]
        errors = errors[keep]
        x = x[keep]
        visible = visible[keep]
        values = np.column_stack(
            [
                4500 + 100 * np.arange(len(ids)) / len(ids),
                2 + np.sin(np.arange(len(ids))),
                -0.5 + np.cos(np.arange(len(ids))),
            ]
        )
        uncertainty = np.full_like(values, 0.05)
        observed = np.ones_like(values, bool)
        quality = observed.copy()
        roles = {
            key: np.zeros(len(ids), bool)
            for key in ["fit", "tune", "calibration", "test"]
        }
        for key, bounds in zip(
            roles, [(0, 128), (128, 160), (160, 208), (208, len(ids))], strict=True
        ):
            roles[key][slice(*bounds)] = True
        survey = np.full((len(ids), 3), b"synthetic")
        provenance = {"synthetic_only": True}
    else:
        report = json.loads(Path(qa).read_text())
        required = [
            "native_labels_verified",
            "all_original_source_target_values_errors_flags_rows_verified",
            "all_source_id_roles_and_reference_bans_verified",
            "all_target_quality_masks_verified",
        ]
        if report.get("status") != "passed" or not all(
            report.get(k) is True for k in required
        ):
            raise ValueError(
                "Independent original-source native quality audit incomplete"
            )
        if report.get("sidecar_sha256") != sha(sidecar) or report.get(
            "cohort_sha256"
        ) != sha(cohort):
            raise ValueError("Native independent QA hashes differ")
        with h5py.File(sidecar) as f:
            if (
                not f.attrs.get("complete", False)
                or f.attrs.get("schema") != "joint-native-labels-v1"
            ):
                raise ValueError("Joint sidecar incomplete/schema differs")
            np.testing.assert_array_equal(ids, f["source_id"][:])
            values = f["targets/value"][:]
            uncertainty = f["targets/uncertainty"][:]
            observed = f["targets/observed"][:]
            quality = f["targets/quality_pass"][:]
            survey = f["targets/survey"][:]
            roles = {
                role: f["roles/" + role][:]
                for role in ["fit", "tune", "calibration", "test"]
            }
            provenance = json.loads(f.attrs["provenance_json"])
            for role in ["fit", "tune", "calibration"]:
                if np.any(
                    roles[role]
                    & (
                        (f["roles/benchmark_partition"][:] != 0)
                        | f["roles/reference_holdout"][:]
                    )
                ):
                    raise ValueError(
                        "Supervised role leaks benchmark validation/reference objects"
                    )
    selected = complete_roles(observed, quality, roles)
    for role, rows in selected.items():
        if len(rows) < (4 if synthetic else cfg["minimum_role_rows"]):
            raise ValueError("Insufficient complete native rows: " + role)
    yc, ys = label_scaling(values, selected["fit"])
    if (
        not np.isfinite(uncertainty[np.r_[tuple(selected.values())]]).all()
        or not (uncertainty[np.r_[tuple(selected.values())]] > 0).all()
    ):
        raise ValueError("Complete targets need positive actual formal errors")
    return dict(
        source_id=ids,
        input=x.numpy(),
        visible=visible.numpy(),
        raw=raw,
        feature_errors=errors,
        value=values,
        error=uncertainty,
        observed=observed,
        quality=quality,
        survey=survey,
        roles=selected,
        label_center=yc,
        label_scale=ys,
        provenance=provenance,
    )


def predictions(model, head, data, rows, device, batch=128):
    """Score fixed weights in float64 without changing training modules.

    Float32 encoder/context roundoff can be amplified by a trained spline's
    log-Jacobian: even a different CPU batch size can change a density beyond
    the declared replay tolerance. Promoting only the final log density does
    not recover the lost context precision. Keep the entire evaluation path,
    including label standardization and native-unit conversion, in float64.
    Saved quadrature nodes retain their values; no estimator is resampled.

    The native-unit conversion is now **per row and per observed coordinate**.
    For a declared transform ``T`` the exact relation is::

        log p_Y(y|x) = log p_U(u|x) - sum_obs log w - sum_obs log|T'(y)|

    A single scalar ``sum(log(scale))`` is only correct for identity
    transforms; with ``log_kpc``/``log_gyr`` it is wrong by ``+log d`` /
    ``+log age`` on every row. The emitting keys ``density_kind``,
    ``loss_denominator`` and ``jacobian*`` make the two estimands
    (intrinsic versus observed-label) explicit instead of aliased.
    """
    encoder = copy.deepcopy(model.encoder).to(device=device, dtype=torch.float64)
    evaluation_head = copy.deepcopy(head).to(device=device, dtype=torch.float64)
    encoder.eval()
    evaluation_head.eval()
    names = target_names(data)
    dim = len(names)
    label_center = np.asarray(data["label_center"], dtype="f8")
    label_scale = np.asarray(data["label_scale"], dtype="f8")
    if label_center.shape != (dim,) or label_scale.shape != (dim,):
        raise ValueError("label center/scale must match the declared target names")
    observed_source = data.get("observed")
    quality_source = data.get("quality")
    has_density = bool(getattr(head, "has_density", head.kind not in NON_DENSITY_KINDS))
    density_kind = getattr(head, "density_kind", None)
    loss_denominator = getattr(head, "loss_denominator", "per_coordinate")
    means = []
    variances = []
    observed_variances = []
    intrinsic = []
    intrinsic_standardized = []
    observed = []
    observed_standardized = []
    jacobian = []
    transform_terms = []
    observed_rows_all = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            selected = rows[start : start + batch]
            latent = encoder(
                torch.as_tensor(
                    data["input"][selected], device=device, dtype=torch.float64
                )
            )
            mean, variance = evaluation_head.moments(latent)
            if not torch.isfinite(mean).all():
                raise FloatingPointError("Nonfinite multilabel point predictions")
            if has_density and (
                not torch.isfinite(variance).all() or not (variance > 0).all()
            ):
                raise FloatingPointError("Invalid predictive variance")
            native = np.asarray(data["value"][selected], dtype="f8")
            if native.shape[1] != dim:
                raise ValueError("native target width differs from the registry names")
            observed_rows = np.isfinite(native)
            if observed_source is not None:
                observed_rows &= np.asarray(observed_source)[selected].astype(bool)
            if quality_source is not None:
                observed_rows &= np.asarray(quality_source)[selected].astype(bool)
            y = torch.tensor(
                (model_coordinates(names, native) - label_center) / label_scale,
                device=device,
                dtype=torch.float64,
            )
            sigma = torch.tensor(
                np.asarray(data["error"][selected], dtype="f8") / label_scale,
                device=device,
                dtype=torch.float64,
            )
            if has_density:
                lp = evaluation_head.log_prob(latent, y)
                if density_kind == "observed_label":
                    observed_lp = evaluation_head.log_prob(latent, y, sigma)
                else:
                    # No measurement model was declared: the observed-label
                    # density is NOT identified. Store NaN, never the intrinsic
                    # value under the observed name.
                    observed_lp = torch.full_like(lp, float("nan"))
                if not torch.isfinite(lp).all():
                    raise FloatingPointError("Nonfinite intrinsic density")
                if (
                    density_kind == "observed_label"
                    and not torch.isfinite(observed_lp).all()
                ):
                    raise FloatingPointError("Nonfinite observed-label density")
            else:
                lp = torch.full((len(selected),), float("nan"), device=device)
                observed_lp = lp.clone()
            offsets = density_offsets(
                names, native, label_scale, observed=observed_rows
            )
            transform = log_abs_det_jacobian(names, native, observed=observed_rows)
            jacobian.append(offsets)
            transform_terms.append(transform)
            observed_rows_all.append(observed_rows)
            means.append(mean.cpu().numpy())
            variances.append(variance.cpu().numpy())
            if density_kind == "observed_label":
                observed_variances.append((variance + sigma.square()).cpu().numpy())
            else:
                observed_variances.append(np.full_like(variance.cpu().numpy(), np.nan))
            reduction = offsets.sum(1)
            intrinsic_standardized.append(lp.cpu().numpy())
            observed_standardized.append(observed_lp.cpu().numpy())
            intrinsic.append(lp.cpu().numpy() - reduction)
            observed.append(observed_lp.cpu().numpy() - reduction)
    intrinsic, observed = np.concatenate(intrinsic), np.concatenate(observed)
    return dict(
        mean=np.concatenate(means),
        variance=np.concatenate(variances),
        observed_variance=np.concatenate(observed_variances),
        intrinsic_log_prob_standardized=np.concatenate(intrinsic_standardized),
        observed_log_prob_standardized=np.concatenate(observed_standardized),
        intrinsic_log_prob_native=intrinsic,
        observed_log_prob_native=observed,
        jacobian=np.concatenate(jacobian),
        jacobian_transform=np.concatenate(transform_terms),
        target_observed=np.concatenate(observed_rows_all),
        density_kind=np.array(density_kind if density_kind else "none"),
        loss_denominator=np.array(loss_denominator),
        observed_log_prob_available=np.array(density_kind == "observed_label"),
        target_names=np.array(names),
        label_scale=np.asarray(label_scale, dtype="f8"),
    )


def target_names(data):
    """Declared target names for this arm; registry-validated, never positional."""
    return validate_targets(tuple(data.get("target_names", REGISTRY_TARGET_NAMES[:3])))


def _point_estimator(head):
    """Declare how the reported point estimate was formed, per head family."""
    if head.kind == "flow":
        return (
            "Fixed antithetic Sobol inverse-base sample mean, not exact analytic "
            "marginal mean"
        )
    if head.kind == "mdn":
        return "exact analytic mixture mean and mixture variance"
    if head.kind in {"mvgaussian", "lowrank"}:
        return "exact analytic joint mean and marginal variances"
    if head.kind == "quantile":
        return (
            "0.5-quantile output; variance from a Gaussian location-scale fit to "
            "the predicted levels (diagnostic, not a density)"
        )
    if head.kind == "expectile":
        return (
            "0.5-expectile output; variance from a Gaussian location-scale fit to "
            "the predicted expectiles (diagnostic, not a density)"
        )
    if head.kind == "huber":
        return "Huber point output; no predictive variance"
    return f"{int(getattr(head, 'target_dim', 3))} conditional point means"


def calibrate_and_save(output, data, test, cal, means_test, means_cal, prefix=""):
    center = np.asarray(data["label_center"], dtype="f8")
    scale = np.asarray(data["label_scale"], dtype="f8")
    names = target_names(data)
    calibration_target = (
        model_coordinates(names, np.asarray(data["value"][cal], dtype="f8")) - center
    ) / scale
    intervals = conformal_intervals(
        torch.tensor(means_cal),
        torch.tensor(calibration_target),
        torch.tensor(means_test),
    )
    np.savez_compressed(
        output / (prefix + "conformal.npz"),
        calibration_source_id=data["source_id"][cal],
        test_source_id=data["source_id"][test],
        **{key: value.cpu().numpy() for key, value in intervals.items()},
    )
    y = (
        model_coordinates(names, np.asarray(data["value"][test], dtype="f8")) - center
    ) / scale
    lower, upper = [
        intervals[key].numpy() for key in ("marginal_lower", "marginal_upper")
    ]
    jl, ju = [
        intervals[key].numpy() for key in ("simultaneous_lower", "simultaneous_upper")
    ]
    return {
        "marginal_coverage": ((y >= lower) & (y <= upper)).mean(0).tolist(),
        "simultaneous_box_coverage": float(((y >= jl) & (y <= ju)).all(1).mean()),
        "alpha": 0.1,
        "simultaneous_method": "Bonferroni alpha/3 marginal torchregress MultiTargetConformal",
        "interpretation": "Observed native-label coverage under survey dependence; no physical-truth certificate",
    }


def postprocess(output, model, head, data, test, cal, device, cfg, smoke):
    xp = xp_indices(cfg["feature_columns"]).tolist()
    # Eligibility is input-only; target values never select adaptation objects.
    test_eligible = test[data["visible"][test][:, xp].sum(1) >= 2]
    cal_eligible = cal[data["visible"][cal][:, xp].sum(1) >= 2]
    if min(len(test_eligible), len(cal_eligible)) < 2:
        raise ValueError("Insufficient observed XP for label-free refinement")
    test_rows = test_eligible[
        hash_selected(
            data["source_id"][test_eligible], 4 if smoke else cfg["ttt_objects"]
        )
    ]
    cal_rows = cal_eligible[
        hash_selected(
            data["source_id"][cal_eligible],
            4 if smoke else cfg["ttt_calibration_objects"],
            seed=1043,
        )
    ]
    actual = {}
    diagnostics = {}
    before = {key: v.detach().clone() for key, v in model.state_dict().items()}
    for role, rows in [("test", test_rows), ("calibration", cal_rows)]:
        values = []
        ds = []
        for row in rows:
            p, d = refine_one(
                model,
                head,
                torch.tensor(data["input"][row : row + 1], device=device),
                torch.tensor(data["visible"][row : row + 1], device=device),
                xp,
                int(data["source_id"][row]),
                steps=cfg["ttt_steps"],
            )
            values.append(p.cpu().numpy()[0])
            ds.append(d)
        actual[role] = np.stack(values)
        diagnostics[role] = ds
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    np.savez_compressed(
        output / "ttt-predictions.npz",
        test_rows=test_rows,
        calibration_rows=cal_rows,
        test_source_id=data["source_id"][test_rows],
        calibration_source_id=data["source_id"][cal_rows],
        test_standardized_prediction=actual["test"],
        calibration_standardized_prediction=actual["calibration"],
    )
    atomic(
        output / "ttt-diagnostics.json",
        {
            "steps": cfg["ttt_steps"],
            "per_object_reset_verified": True,
            "test_below_minus2_count": int((data["value"][test_rows, 2] < -2).sum()),
            "diagnostics": diagnostics,
            "calibration_after_refinement": True,
            "labels_enter_refinement_objective": False,
        },
    )
    result = calibrate_and_save(
        output,
        data,
        test_rows,
        cal_rows,
        actual["test"],
        actual["calibration"],
        prefix="ttt-",
    )
    baseline_test = predictions(model, head, data, test_rows, device)["mean"]
    baseline_cal = predictions(model, head, data, cal_rows, device)["mean"]
    # Preserve the exact separately batched means used in these CP intervals.
    # Earlier full-test batches can differ by float32 rounding.
    np.savez_compressed(
        output / "ttt-matched-clean-predictions.npz",
        test_rows=test_rows,
        calibration_rows=cal_rows,
        test_source_id=data["source_id"][test_rows],
        calibration_source_id=data["source_id"][cal_rows],
        test_standardized_prediction=baseline_test,
        calibration_standardized_prediction=baseline_cal,
    )
    result["matched_clean_cp"] = calibrate_and_save(
        output,
        data,
        test_rows,
        cal_rows,
        baseline_test,
        baseline_cal,
        prefix="ttt-matched-clean-",
    )
    result["test_rows"] = len(test_rows)
    result["test_XP_eligible_count"] = len(test_eligible)
    result["calibration_XP_eligible_count"] = len(cal_eligible)
    result["test_below_minus2_count"] = int((data["value"][test_rows, 2] < -2).sum())
    return result


def library_joint_tta(output, model, head, data, test, cal, device, smoke):
    """Actual library weighted joint CP control, zero pseudo-label gradient rounds."""
    from torchregress.test_time import JointDistributionalTTA

    class Predictor(torch.nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.feature_extractor = encoder
            self.head = head

        def forward(self, x):
            mean, var = self.head.moments(self.feature_extractor(x))
            return mean, var

    test = test[hash_selected(data["source_id"][test], 4 if smoke else 512)]
    cal = cal[hash_selected(data["source_id"][cal], 8 if smoke else 1024, 1043)]
    adapted = (
        Predictor(copy.deepcopy(model.encoder), copy.deepcopy(head)).to(device).eval()
    )
    adapted.feature_extractor.requires_grad_(False)
    wrapper = JointDistributionalTTA(
        alpha=0.1, align_features=False, pseudo_label_rounds=0
    )
    result = wrapper.adapt_and_calibrate(
        adapted,
        torch.tensor(data["input"][cal], device=device),
        torch.tensor(
            ((data["value"][cal] - data["label_center"]) / data["label_scale"]).astype(
                "f4"
            ),
            device=device,
        ),
        torch.tensor(data["input"][test], device=device),
    )
    result.diagnostics["protocol_scope"] = (
        "Weighted joint CP only: pseudo_label_rounds0, align_featuresFalse; no gradient adaptation claimed"
    )
    outputs = wrapper.predict_intervals(
        result, torch.tensor(data["input"][test], device=device)
    )
    np.savez_compressed(
        output / "library-joint-tta.npz",
        calibration_source_id=data["source_id"][cal],
        test_source_id=data["source_id"][test],
        **{key: value.detach().cpu().numpy() for key, value in outputs.items()},
    )
    save_checkpoint(
        output / "library-joint-tta-checkpoint.pth",
        {
            "model_state_dict": adapted.state_dict(),
            "diagnostics": result.diagnostics,
            "conformal_threshold": result.conformal.threshold_,
        },
    )
    atomic(output / "library-joint-tta-diagnostics.json", result.diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/multilabel-screen-20261020.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--cohort")
    parser.add_argument("--sidecar")
    parser.add_argument("--native-qa")
    parser.add_argument("--benchmark")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    arm = next(a for a in cfg["arms"] if a["run_id"] == args.run_id)
    cfg.update(
        {
            k: getattr(args, k) or cfg[k]
            for k in ["checkpoint", "cohort", "sidecar", "native_qa", "benchmark"]
        }
    )
    if args.synthetic_smoke and not args.smoke:
        raise ValueError("Synthetic labels are allowed only in explicitly marked smoke")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "status": "preparing",
        "run_id": args.run_id,
        "policy": cfg,
        "arm": arm,
        "smoke": args.smoke,
        "synthetic_labels": args.synthetic_smoke,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "scope": "Three native high-resolution reported labels, complete-case joint density; no distance/age/alpha/astrometry",
        "prediction_evaluation": {
            "dtype": "float64",
            "modules": "isolated encoder and head copies; training state unchanged",
            "targets": "float64 standardization and per-row, per-observed-coordinate native log-Jacobian",
            "quadrature": "original saved nodes promoted exactly, not resampled",
        },
        "target_registry": {
            spec.name: {
                "unit": spec.unit,
                "transform": spec.transform,
                "positive": spec.positive,
                "error_kind": spec.error_kind,
                "source_tag": spec.source_tag,
                "observable_kind": spec.observable_kind,
                "status": spec.status,
            }
            for spec in REGISTRY.values()
        },
    }
    atomic(args.output / "manifest.json", manifest)
    try:
        if not args.smoke and not torch.cuda.is_available():
            raise ValueError("Production requires actual CUDA")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
        if (
            sha(cfg["checkpoint"]) != cfg["checkpoint_sha256"]
            or sha(cfg["cohort"]) != COHORT_SHA
        ):
            raise ValueError("Pinned actual parent/cohort differs")
        metadata = importlib.metadata.distribution("torchregress")
        direct = json.loads(metadata.read_text("direct_url.json") or "{}")
        if (
            not args.smoke
            and direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]
        ):
            raise ValueError(
                "Installed torchregress differs from immutable library pin"
            )
        disk = torch_load_trusted(
            cfg["checkpoint"], map_location="cpu", weights_only=False
        )
        model_cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        features = disk["run_signature"]["feature_cols"]
        center = np.asarray(disk["run_signature"]["scaler_center"])
        scale = np.asarray(disk["run_signature"]["scaler_scale"])
        cfg["feature_columns"] = features
        data = load_native(
            cfg,
            cfg["cohort"],
            cfg["sidecar"],
            cfg["native_qa"],
            features,
            center,
            scale,
            synthetic=args.synthetic_smoke,
        )
        roles = data["roles"]
        fit, test, cal = roles["fit"], roles["test"], roles["calibration"]
        if args.smoke:
            fit = fit[:64]
            test = test[:16]
            cal = cal[:16]
        np.savez_compressed(
            args.output / "paired-cohort.npz",
            **{
                key: data[key]
                for key in [
                    "source_id",
                    "input",
                    "visible",
                    "value",
                    "error",
                    "observed",
                    "quality",
                    "survey",
                    "label_center",
                    "label_scale",
                ]
            },
            **{key + "_rows": value for key, value in roles.items()},
        )
        atomic(args.output / "native-provenance.json", data["provenance"])
        manifest.update(
            native_counts={key: len(value) for key, value in roles.items()},
            device=str(device),
            gpu_name=torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            parent_sha256=sha(cfg["checkpoint"]),
            cohort_sha256=COHORT_SHA,
            torchregress_provenance=direct,
            complete_case_policy=True,
            include_reddening=False,
        )
        atomic(args.output / "manifest.json", manifest)
        if not args.smoke:
            with h5py.File(cfg["benchmark"]) as f:
                keys, validation_keys, bc, bs = pretrain.validate_source(
                    disk, model_cfg, f
                )
                np.testing.assert_array_equal(center, bc)
                np.testing.assert_array_equal(scale, bs)
                if (
                    len(f["train/records"]) != 9000000
                    or len(f["validation/records"]) != 1000000
                ):
                    raise ValueError("Original fixed9M/1M sample required")
                with h5py.File(cfg["cohort"]) as cohort:
                    unit_audit = probe.audit_fits_xp_units(
                        cfg["benchmark"],
                        features,
                        cohort["features/values"][:],
                        cohort["features/source_id"][:],
                        "g_normalized",
                    )
                atomic(args.output / "xp-unit-audit.json", unit_audit)
                if arm["kind"] != "forward":
                    cache, _, counts = pretrain.read_prefixes(
                        f,
                        keys,
                        features,
                        None,
                        9000000,
                        center,
                        scale,
                        args.output / "replay-cache.npy",
                    )
                    validation, _, _ = pretrain.read_prefixes(
                        f,
                        validation_keys,
                        features,
                        None,
                        10000,
                        center,
                        scale,
                        args.output / "validation-cache.npy",
                    )
        elif arm["kind"] != "forward":
            cache = data["input"].copy()
            cache[cache == -9999] = np.nan
            validation = cache[:32].copy()
            counts = {"synthetic_replay": len(cache)}
        if arm["kind"] == "forward":
            from torchregress.algorithms import ForwardInversionRegressor

            xp = xp_indices(features)
            used_xp_rows, sigma = native_xp_error_subset(
                data["feature_errors"],
                scale,
                xp,
                data["raw"][:, features.index("G")],
                np.r_[fit, test, cal],
            )
            np.savez_compressed(
                args.output / "forward-xp-inputs.npz",
                feature_rows=used_xp_rows,
                source_id=data["source_id"][used_xp_rows],
                standardized_input=data["input"][used_xp_rows][:, xp],
                standardized_sigma=sigma,
            )

            if not args.smoke:
                with h5py.File(cfg["cohort"]) as f:
                    legacy = str(f.attrs["legacy_input"])
                audit = audit_xp_error_sources(
                    cfg["cohort"],
                    cfg["benchmark"],
                    legacy,
                    np.unique(np.r_[fit, test, cal]),
                    features,
                )
                atomic(args.output / "xp-error-source-audit.json", audit)

            def tensors(rows):
                positions = np.searchsorted(used_xp_rows, rows)
                np.testing.assert_array_equal(used_xp_rows[positions], rows)
                return (
                    torch.tensor(data["input"][rows][:, xp], device=device),
                    torch.tensor(
                        (
                            (data["value"][rows] - data["label_center"])
                            / data["label_scale"]
                        ).astype("f4"),
                        device=device,
                    ),
                    torch.tensor(sigma[positions], device=device),
                )

            X, y, s = tensors(fit)
            reg = ForwardInversionRegressor(
                n_labels=3,
                n_observables=110,
                epochs=2 if args.smoke else cfg["forward_epochs"],
                batch_size=256,
                hidden_dims=(128, 128),
                device=device,
                seed=arm["seed"],
                validation_fraction=0.0,
                learn_noise=True,
                likelihood="gaussian",
            )
            reg.fit(X, y, sigma_x=s)
            outputs = {}
            inverse_info = {}
            for role, rows in [("test", test), ("calibration", cal)]:
                xt, yt, st = tensors(rows)
                prediction, info = reg.predict(
                    xt,
                    sigma_x=st,
                    max_iter=5 if args.smoke else cfg["forward_iterations"],
                    chunk_size=128,
                    return_info=True,
                )
                outputs[role] = prediction.detach().cpu().numpy()
                inverse_info[role] = {
                    key: (
                        value.detach().cpu().numpy()
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in info.items()
                }
            np.savez_compressed(
                args.output / "forward-inversion-diagnostics.npz",
                **{
                    role + "_" + key: value
                    for role, info in inverse_info.items()
                    for key, value in info.items()
                },
            )
            inverse_summary = {}
            for role, info in inverse_info.items():
                inverse_summary[role] = {}
                for key, value in info.items():
                    arr = np.asarray(value)
                    finite = arr[np.isfinite(arr)]
                    inverse_summary[role][key] = {
                        "count": int(arr.size),
                        "nonfinite_count": int(arr.size - finite.size),
                        "maximum_finite": float(finite.max()) if finite.size else None,
                        "median_finite": float(np.median(finite))
                        if finite.size
                        else None,
                    }
                inverse_summary[role]["relative_stationarity_fraction_1e-3"] = float(
                    np.mean(info["relative_gradient_norm"] <= 1e-3)
                )
            save_checkpoint(
                args.output / "checkpoint.pth",
                {
                    "forward_regressor": reg,
                    "label_center": data["label_center"],
                    "label_scale": data["label_scale"],
                    "policy": cfg,
                    "arm": arm,
                    "parent_sha256": manifest["parent_sha256"],
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None,
                },
            )
            reloaded = torch_load_trusted(
                args.output / "checkpoint.pth", map_location=device, weights_only=False
            )["forward_regressor"]
            xt, _, st = tensors(test)
            again = reloaded.predict(
                xt,
                sigma_x=st,
                max_iter=5 if args.smoke else cfg["forward_iterations"],
                chunk_size=128,
            )
            np.testing.assert_array_equal(outputs["test"], again.detach().cpu().numpy())
            atomic(
                args.output / "forward-diagnostics.json",
                {
                    "scope": "Actual library3-label→110XP likelihood/inversion, zero MSA encoder/reconstruction SGD; noisy-label integration is NOT implemented here",
                    "fit_epochs": 2 if args.smoke else cfg["forward_epochs"],
                    "fit_labels": len(fit),
                    "no_10M_reconstruction_claim": True,
                    "library_inversion_info": inverse_summary,
                    "stationarity_is_not_global_MLE_certificate": True,
                },
            )
            cp = calibrate_and_save(
                args.output, data, test, cal, outputs["test"], outputs["calibration"]
            )
            np.savez_compressed(
                args.output / "predictions.npz",
                test_rows=test,
                calibration_rows=cal,
                test_standardized_prediction=outputs["test"],
                calibration_standardized_prediction=outputs["calibration"],
            )
            manifest.update(
                status="complete",
                presentations=0,
                optimizer_updates=None,
                forward_epochs=cfg["forward_epochs"],
                conformal=cp,
                checkpoint_sha256=sha(args.output / "checkpoint.pth"),
                all_test_reload_byte_equal=True,
                runtime_seconds=time.monotonic() - started,
            )
            atomic(args.output / "manifest.json", manifest)
            return
        # Projection uses the same named allowlist for supervised and reconstruction paths.
        allowed, _ = spectrophotometric_inputs(
            torch.zeros((1, len(features))),
            torch.ones((1, len(features)), dtype=torch.bool),
            features,
            include_reddening=False,
        )
        allowed_columns = allowed.numpy()[0] != -9999
        validation[:, ~allowed_columns] = np.nan
        model = probe.load_model(model_cfg, disk, device).to(device)
        torch.manual_seed(arm["seed"])
        head = MultiLabelHead(
            arm["kind"],
            target_dim=len(data["label_center"]),
            flow_type=arm.get("flow_type", "nsf"),
            noise_aware=arm.get("noise_aware", False),
            mc_samples=cfg["noise_mc_samples"],
            flow_objective=arm.get("flow_objective", "nll"),
            contrastive_weight=arm.get("contrastive_weight", 0.1),
            quantiles=tuple(arm.get("quantiles", (0.16, 0.5, 0.84))),
            expectiles=tuple(arm.get("expectiles", (0.1, 0.5, 0.9))),
            quantile_crossover=arm.get("quantile_crossover", True),
            quantile_crossover_penalty=arm.get("quantile_crossover_penalty", 10.0),
            mdn_components=arm.get("mdn_components", 4),
            lowrank_rank=arm.get("lowrank_rank", 2),
        ).to(device)
        if cfg.get("density_conformal", False) or arm.get("density_conformal", False):
            # Fail before SGD if the published environment lacks post-fit APIs.
            from torchregress.losses.conformal import (
                ConformalizedDensity,
                ContrastiveFlowConformal,
            )

            assert ConformalizedDensity and ContrastiveFlowConformal
            if len(np.unique(data["source_id"][fit])) != len(fit):
                raise ValueError("FIT source IDs must be unique for negative contexts")
            manifest["density_api_preflight"] = {
                "passed": True,
                "before_sgd": True,
                "group_feature": "actual apparent G; boundaries FIT-only",
                "contrastive_negatives": cfg["contrastive_negative_policy"],
            }
        head.point_nodes = qmc_normal_nodes(
            8 if args.smoke else cfg["point_samples"], dimensions=head.target_dim
        ).to(device)
        if arm.get("recovery_of") or cfg.get("flow_memory_preflight", False):
            # Exercise the original failing evaluation shape before new SGD.
            model.eval()
            head.eval()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                preflight_latent = model.encoder(
                    torch.tensor(data["input"][fit[:128]], device=device)
                )
                preflight_mean, preflight_variance = head.moments(preflight_latent)
                if not (
                    torch.isfinite(preflight_mean).all()
                    and torch.isfinite(preflight_variance).all()
                    and (preflight_variance > 0).all()
                ):
                    raise FloatingPointError("Recovery moment memory preflight failed")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            atomic(
                args.output / "flow-moment-memory-preflight.json",
                {
                    "recovery_of": arm.get("recovery_of"),
                    "context_rows": len(preflight_latent),
                    "fixed_point_nodes": len(head.point_nodes),
                    "inverse_node_chunk": 16,
                    "device": str(device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                    "finite_positive_moments": True,
                    "before_supervised_sgd": True,
                },
            )
            del preflight_latent, preflight_mean, preflight_variance
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["encoder_lr"])
        head_optimizer = torch.optim.Adam(head.parameters(), lr=cfg["head_lr"])
        rng = np.random.default_rng(arm["seed"])
        batch = 256
        groups = cfg["supervised_batches_per_update"]
        mask_rng = torch.Generator(device=device).manual_seed(arm["seed"])
        xp = xp_indices(features).tolist()
        width = len(model_cfg["data"]["recon_cols"])
        validation_mask = pretrain.fixed_masks(validation, features, 42, 0.5, 0.2)[
            "common"
        ]
        initial = reconstruct(model, validation, validation_mask, device)
        np.savez_compressed(
            args.output / "initial_reconstruction.npz",
            target=validation,
            prediction=initial,
            mask=validation_mask,
        )

        supervised_hash = hashlib.sha256()

        def supervised(warmup=False):
            rows = supervised_fit_rows(rng, fit, batch)
            supervised_hash.update(rows.astype("i8").tobytes())
            x = torch.tensor(data["input"][rows], device=device)
            y = torch.tensor(
                (
                    (
                        model_coordinates(target_names(data), data["value"][rows])
                        - data["label_center"]
                    )
                    / data["label_scale"]
                ).astype("f4"),
                device=device,
            )
            sigma = torch.tensor(
                (data["error"][rows] / data["label_scale"]).astype("f4"), device=device
            )
            if warmup:
                with torch.no_grad():
                    latent = model.encoder(x)
            else:
                latent = model.encoder(x)
            return head.criterion(
                latent,
                y,
                sigma,
                source_ids=torch.tensor(
                    data["source_id"][rows].astype("i8"), device=device
                ),
            )

        model.eval()
        head.train()
        for step in range(2 if args.smoke else cfg["head_warmup_updates"]):
            torch.manual_seed(100000 + arm["seed"] + step)
            head_optimizer.zero_grad(set_to_none=True)
            loss = supervised(True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite head warmup")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            head_optimizer.step()
        row_hash = hashlib.sha256()
        mask_hash = hashlib.sha256()
        seen = 0
        steps = 0
        budget = 512 if args.smoke else cfg["reconstruction_presentations"]
        effective = 256 if args.smoke else cfg["effective_batch"]
        micro = 32 if args.smoke else cfg["micro_batch"]
        weight = float(arm["reconstruction_weight"])
        manifest.update(
            status="running",
            input_policy="Named photometry+110XP only for prediction and reconstruction; EBV/parallax/positions/PM withheld, raw fields retained; no native target aliases in benchmark",
            noise_likelihood_evaluation=(
                "Intrinsic and observed-label log densities are saved under explicit "
                "names and stamped with density_kind: a noise-aware head convolves the "
                "quoted diagonal errors exactly (Gaussian analytic / fixed-QMC flows), "
                "a plain density head declares intrinsic only and its observed-label "
                "array is NaN rather than an alias. Native density subtracts the "
                "per-row, per-observed-coordinate Jacobian (log w + log|T'|)"
            ),
            optimizer_policy="Fresh matched Adam states for model/head at same parent86; no inherited optimizer moments",
            reconstruction_weight=weight,
            reconstruction_presentations=budget,
            replay_rows_scanned=budget,
            reconstruction_training_presentations=budget if weight else 0,
        )
        atomic(args.output / "manifest.json", manifest)
        checkpoint_interval = cfg.get("checkpoint_interval_presentations", 0)
        snapshot_interval = cfg.get("snapshot_interval_presentations", 0)
        selection_policy = cfg.get("selection_policy")
        if selection_policy not in {None, "reserved_tune_mean_observed_nll"}:
            raise ValueError("Unknown checkpoint selection policy")
        if selection_policy and (not checkpoint_interval or not head.has_density):
            raise ValueError(
                "Tuning density selection requires a density head and checkpoints"
            )
        tuning_rows = roles["tune"][:16] if args.smoke else roles["tune"]
        for other in (fit, test, cal):
            if selection_policy and np.intersect1d(tuning_rows, other).size:
                raise ValueError("Tuning selection overlaps another source role")
        selection_provenance = None
        if args.smoke:
            checkpoint_interval = 256 if checkpoint_interval else 0
            snapshot_interval = 256 if snapshot_interval else 0

        def training_state():
            return {
                "model_state_dict": model.state_dict(),
                "head_state_dict": head.state_dict(),
                "model_optimizer_state_dict": optimizer.state_dict(),
                "head_optimizer_state_dict": head_optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None,
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "supervised_rng_state": rng.bit_generator.state,
                "mask_rng_state": mask_rng.get_state(),
                "label_center": data["label_center"],
                "label_scale": data["label_scale"],
                "feature_columns": features,
                "policy": cfg,
                "arm": arm,
                "completed_presentations": seen,
                "completed_updates": steps,
                "replay_rows": len(cache),
                "next_replay_row": seen % len(cache),
                "parent_sha256": manifest["parent_sha256"],
                "source_sha": manifest["source_sha"],
                "selection_provenance": selection_provenance,
            }

        for indices, raw in replay_batches(cache, budget, effective):
            row_hash.update(indices.tobytes())
            raw = np.array(raw, copy=True)
            raw[:, ~allowed_columns] = np.nan
            target = torch.tensor(raw, device=device)
            hidden, mask, finite = pretrain.training_mask(
                target, xp, 0.5, 0.2, mask_rng
            )
            mask_hash.update(mask.cpu().numpy().tobytes())
            score = (mask & finite)[:, :width]
            optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            model.train()
            head.train()
            torch.manual_seed(1000 + arm["seed"] + steps)
            reco_total = 0.0
            if weight:
                for start in range(0, len(raw), micro):
                    prediction, _ = model(hidden[start : start + micro])
                    reco = pretrain.experiment_objective(
                        prediction,
                        target[start : start + micro, :width],
                        score[start : start + micro],
                        xp,
                        {"loss": "mae"},
                        score.sum(),
                    )
                    (weight * reco).backward()
                    reco_total += float(reco.detach())
            supervised_total = 0.0
            for _ in range(1 if args.smoke else groups):
                loss = supervised()
                denominator = 1 if args.smoke else groups
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite joint density objective")
                (loss / denominator).backward()
                supervised_total += float(loss.detach()) / denominator
            model_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            head_norm = torch.nn.utils.clip_grad_norm_(
                head.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            head_optimizer.step()
            seen += len(raw)
            steps += 1
            record = {
                "presentations": seen,
                "updates": steps,
                "reconstruction_mae": reco_total,
                "supervised_loss": supervised_total,
                "model_gradient_norm": float(model_norm),
                "head_gradient_norm": float(head_norm),
                "gpu_allocated": torch.cuda.memory_allocated(device)
                if device.type == "cuda"
                else 0,
            }
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            if checkpoint_interval and (
                seen >= budget
                or interval_crossed(seen - len(raw), seen, checkpoint_interval)
            ):
                receipt = save_periodic_checkpoint(args.output, training_state())
                if selection_policy:
                    tune = predictions(model, head, data, tuning_rows, device)
                    select_by_reserved_tuning(
                        args.output, receipt, data, tuning_rows, tune
                    )
            scheduled_snapshot = (
                interval_crossed(seen - len(raw), seen, snapshot_interval)
                if snapshot_interval
                else steps % 92 == 0
            )
            if steps == 1 or seen >= budget or scheduled_snapshot:
                snapshot = predictions(model, head, data, test, device)
                np.savez_compressed(
                    args.output / f"snapshot-{seen}.npz", test_rows=test, **snapshot
                )
        training_seen, training_steps = seen, steps
        if selection_policy:
            selection_provenance = json.loads(
                (args.output / "selected-checkpoint.json").read_text()
            )
            selected_path = args.output / selection_provenance["file"]
            if sha(selected_path) != selection_provenance["sha256"]:
                raise ValueError("Selected checkpoint checksum differs")
            selected_state = torch_load_trusted(
                selected_path, map_location=device, weights_only=False
            )
            if selected_state["source_sha"] != manifest["source_sha"]:
                raise ValueError("Selected checkpoint source differs")
            model.load_state_dict(selected_state["model_state_dict"], strict=True)
            head.load_state_dict(selected_state["head_state_dict"], strict=True)
            optimizer.load_state_dict(selected_state["model_optimizer_state_dict"])
            head_optimizer.load_state_dict(selected_state["head_optimizer_state_dict"])
            torch.set_rng_state(selected_state["rng_state"].cpu())
            if selected_state["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(
                    [item.cpu() for item in selected_state["cuda_rng_state"]]
                )
            np.random.set_state(selected_state["numpy_rng_state"])
            random.setstate(selected_state["python_rng_state"])
            rng.bit_generator.state = selected_state["supervised_rng_state"]
            mask_rng.set_state(selected_state["mask_rng_state"].cpu())
            seen = selected_state["completed_presentations"]
            steps = selected_state["completed_updates"]
        output_test = predictions(model, head, data, test, device)
        output_cal = predictions(model, head, data, cal, device)
        model.eval()
        head.eval()
        final_reconstruction = reconstruct(model, validation, validation_mask, device)
        np.savez_compressed(
            args.output / "final_reconstruction.npz", prediction=final_reconstruction
        )
        save_checkpoint(
            args.output / "checkpoint.pth",
            training_state(),
        )
        restored = torch_load_trusted(
            args.output / "checkpoint.pth", map_location=device, weights_only=False
        )
        model.load_state_dict(restored["model_state_dict"], strict=True)
        head.load_state_dict(restored["head_state_dict"], strict=True)
        after = predictions(model, head, data, test, device)
        for key in output_test:
            np.testing.assert_array_equal(output_test[key], after[key])
        np.testing.assert_array_equal(
            final_reconstruction,
            reconstruct(model, validation, validation_mask, device),
        )
        cp = calibrate_and_save(
            args.output, data, test, cal, output_test["mean"], output_cal["mean"]
        )
        np.savez_compressed(
            args.output / "predictions.npz",
            test_rows=test,
            calibration_rows=cal,
            test_source_id=data["source_id"][test],
            calibration_source_id=data["source_id"][cal],
            **{"test_" + k: v for k, v in output_test.items()},
            **{"calibration_" + k: v for k, v in output_cal.items()},
        )
        if cfg.get("density_conformal", False) or arm.get("density_conformal", False):
            from masked_stellar_autoencoder.training.density_calibration import (
                calibrate_density_scores,
                save_density_calibration,
            )

            apparent_g = data["raw"][:, features.index("G")]
            density_summary, density_arrays = calibrate_density_scores(
                calibration_scores=-output_cal["intrinsic_log_prob_standardized"],
                test_scores=-output_test["intrinsic_log_prob_standardized"],
                fit_g=apparent_g[fit],
                calibration_g=apparent_g[cal],
                test_g=apparent_g[test],
                fit_ids=data["source_id"][fit],
                calibration_ids=data["source_id"][cal],
                test_ids=data["source_id"][test],
                provenance={
                    "checkpoint_sha256": sha(args.output / "checkpoint.pth"),
                    "scaler_sha256": hashlib.sha256(
                        np.r_[data["label_center"], data["label_scale"]].tobytes()
                    ).hexdigest(),
                    "source_sha256": sha(cfg["cohort"]),
                },
                api="density" if head.flow_objective == "nll" else "contrastive",
            )
            save_density_calibration(args.output, density_summary, density_arrays)
            manifest["joint_density_conformal"] = density_summary
        ttt = (
            postprocess(
                args.output, model, head, data, test, cal, device, cfg, args.smoke
            )
            if arm.get("ttt", True)
            else None
        )
        if arm["kind"] == "gaussian" and arm.get("library_tta", False):
            library_joint_tta(
                args.output, model, head, data, test, cal, device, args.smoke
            )
        manifest.update(
            status="complete",
            presentations=training_seen,
            optimizer_updates=training_steps,
            reconstruction_optimizer_updates=training_steps if weight else 0,
            selected_model_presentations=seen,
            selected_model_updates=steps,
            checkpoint_selection=selection_provenance,
            row_sha256=row_hash.hexdigest(),
            mask_sha256=mask_hash.hexdigest(),
            supervised_stream_sha256=supervised_hash.hexdigest(),
            row_source_counts=counts,
            conformal=cp,
            ttt=ttt,
            checkpoint_sha256=sha(args.output / "checkpoint.pth"),
            all_test_reload_byte_equal=True,
            all_validation_decoder_reload_byte_equal=True,
            runtime_seconds=time.monotonic() - started,
            point_estimator=_point_estimator(head),
            density_kind=output_test["density_kind"].item(),
            loss_denominator=output_test["loss_denominator"].item(),
            jacobian={
                "form": "per_row_per_observed_coordinate",
                "applied": True,
                "target_names": output_test["target_names"].tolist(),
                "target_transforms": [
                    REGISTRY[name].transform
                    for name in output_test["target_names"].tolist()
                ],
                "terms": "log w + log|T'(y)|, zero for unobserved coordinates",
                "scale_log": np.log(np.asarray(output_test["label_scale"])).tolist(),
                "arrays": "predictions.npz:test_jacobian,test_jacobian_transform",
            },
            observed_log_prob_available=bool(
                output_test["observed_log_prob_available"].item()
            ),
            density_contract={
                "intrinsic": "intrinsic_log_prob_{standardized,native}",
                "observed_label": "observed_log_prob_{standardized,native}"
                if bool(output_test["observed_log_prob_available"].item())
                else "NaN (no measurement model declared)",
                "never_alias_intrinsic_as_observed": True,
                "loss_denominator": "per_coordinate",
                "unit_convention": "native target units "
                + str(
                    [
                        REGISTRY[name].unit
                        for name in output_test["target_names"].tolist()
                    ]
                ),
            },
            noise_aware_scope="Diagonal native target measurement errors only; no input EIV claim"
            if head.noise_aware
            else "reported-label predictive loss",
        )
        atomic(args.output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            runtime_seconds=time.monotonic() - started,
        )
        atomic(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
