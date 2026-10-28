"""Independent frozen-alpha head replay on CANFAR, never local native models.

Pure functional head/NSF computations, exact source-bound embeddings, native
Jacobians, row streams, selected Adam counters and independent split conformal.
Does NOT repeat raw-source audits or certify the frozen encoder derivation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import independent_lowdim_nsf_kernel as kernel
import verify_validation_points as functions

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def order_threshold(scores, alpha):
    scores = np.asarray(scores, dtype="f8")
    if not len(scores) or not np.isfinite(scores).all() or not 0 < alpha < 1:
        raise ValueError("Finite nonempty scores and valid alpha required")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError(
            "Unbounded finite-sample threshold; do not invent a finite set"
        )
    return float(np.sort(scores)[rank - 1])


def head_arrays(x, y, state, kind):
    raw = functions.raw_head(x, state)
    if kind != "nsf":
        mean, variance, lp, _ = functions.analytic(raw, y, torch.zeros_like(y), kind)
        return mean, variance, lp
    lp = kernel.density(y, raw, state)
    samples = [
        kernel.inverse(nodes[:, None, :].expand(-1, len(x), -1), raw, state)
        for nodes in state["point_nodes"].split(16)
    ]
    sample = torch.cat(samples, 0)
    mean = sample.mean(0)
    return mean, (sample - mean).square().mean(0), lp


def comparison(actual, saved):
    if actual.shape != saved.shape:
        raise ValueError("Saved/computed shape mismatch")
    bad = (
        ~np.isfinite(actual)
        | ~np.isfinite(saved)
        | ~np.isclose(actual, saved, atol=5e-5, rtol=5e-5)
    )
    return dict(
        passed=not bool(bad.any()),
        violations=int(bad.sum()),
        maximum_absolute_difference=float(np.max(np.abs(actual - saved)))
        if np.isfinite(actual).all() and np.isfinite(saved).all()
        else None,
    )


def audit_model(base, output, policy, model, arrays, normalizer):
    run = model["run_id"]
    directory = base / run
    target = arrays["target"]
    ids = arrays["source_id"]
    roles = arrays["role"]
    center, scale = normalizer
    destination = output / run
    destination.mkdir(exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["source_commit"] != model["source_commit"]
        or manifest["library_revision"] != policy["library_revision"]
        or manifest["input_sha256"] != policy["input_sha256"]
        or manifest["kind"] != model["kind"]
        or manifest["seed"] != model["seed"]
    ):
        raise ValueError("Producer/source/library/input evidence changed")
    if (
        digest(directory / "checkpoint.pth") != model["checkpoint_sha256"]
        or manifest["checkpoint_sha256"] != model["checkpoint_sha256"]
    ):
        raise ValueError("Frozen checkpoint SHA changed")
    bindings = dict(
        checkpoint=model["checkpoint_sha256"],
        manifest=digest(directory / "manifest.json"),
        metrics=digest(directory / "metrics.json"),
        policy=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
        predictions=digest(directory / "predictions.npz"),
        input=policy["input_sha256"],
        verifier=digest(Path(__file__)),
        kernel=digest(Path(kernel.__file__)),
        functional_helper=digest(Path(functions.__file__)),
    )
    binding_path = destination / "bindings.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != bindings:
        raise ValueError("Changed bindings; never reuse completed verification")
    atomic(binding_path, bindings)
    report_path = destination / "report.json"
    if report_path.exists() and json.loads(report_path.read_text()).get("status") in {
        "head_numerics_cp_pass",
        "strict_failed",
    }:
        return json.loads(report_path.read_text())
    checkpoint = torch_load_trusted(
        directory / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    np.testing.assert_array_equal(checkpoint["center"], center)
    np.testing.assert_array_equal(checkpoint["scale"], scale)
    assert checkpoint["source_input_sha256"] == policy["input_sha256"]
    assert checkpoint["kind"] == model["kind"]
    assert checkpoint["seed"] == model["seed"]
    assert checkpoint["training_steps"] == manifest["training_steps"] == 1000
    state = {
        k: v.double() if v.is_floating_point() else v
        for k, v in checkpoint["head_state_dict"].items()
    }
    if any(
        v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()
    ):
        raise ValueError("Nonfinite head state")
    fit = np.flatnonzero(roles == "fit")
    rng = np.random.default_rng(model["seed"])
    stream = hashlib.sha256()
    for _ in range(checkpoint["training_steps"]):
        stream.update(
            np.asarray(
                rng.choice(fit, size=min(256, len(fit)), replace=False), dtype="<i8"
            ).tobytes()
        )
    assert stream.hexdigest() == manifest["row_stream_sha256"]
    history = json.loads((directory / "metrics.json").read_text())
    winner = min(history, key=lambda row: row["tune_nll_per_target"])
    assert winner["step"] == checkpoint["selected_step"] == manifest["selected_step"]
    steps = [
        float(v["step"])
        for v in checkpoint["selected_optimizer_state_dict"]["state"].values()
    ]
    assert steps and all(step == checkpoint["selected_step"] for step in steps)
    saved = dict(np.load(directory / "predictions.npz", allow_pickle=False))
    report = dict(
        run_id=run,
        status="running",
        bindings=bindings,
        scope="source-bound frozen head numerics/streams/selected Adam/mean-residual and density-set CP; encoder derivation and optimization trajectory not replayed",
        checks={},
    )
    role_arrays = {}
    tasks = []
    for role in ["test", "cal"]:
        index = np.flatnonzero(roles == role)
        np.testing.assert_array_equal(saved[role + "_source_id"], ids[index])
        np.testing.assert_array_equal(saved[role + "_target_native"], target[index])
        for start in range(0, len(index), 128):
            tasks.append((start != 0, role, start, index[start : start + 128]))
    # Test/calibration pilots FIRST; full remaining roles reuse their exact caches.
    tasks.sort(key=lambda task: (task[0], task[1], task[2]))
    results = {"test": [], "cal": []}
    for _, role, start, index in tasks:
        cache = destination / f"{role}-{start:06d}.npz"
        if cache.exists():
            cached = dict(np.load(cache, allow_pickle=False))
            np.testing.assert_array_equal(cached["source_id"], ids[index])
        else:
            x = torch.tensor(arrays["embeddings"][index], dtype=torch.float64)
            y = torch.tensor((target[index] - center) / scale, dtype=torch.float64)
            with torch.no_grad():
                mean, variance, lp = head_arrays(x, y, state, model["kind"])
            cached = dict(
                source_id=ids[index],
                mean_native=center + scale * mean.numpy(),
                variance_native=scale**2 * variance.numpy(),
                lp_standardized=lp.numpy(),
                lp_native=lp.numpy() - np.log(scale).sum(),
            )
            np.savez_compressed(
                cache, **cached
            )  # Preserve computed arrays BEFORE comparisons.
        checks = {
            "mean_native": saved[role + "_mean_native"][start : start + len(index)],
            "variance_native": saved[role + "_variance_native"][
                start : start + len(index)
            ],
            "lp_standardized": saved[role + "_joint_log_prob_standardized"][
                start : start + len(index)
            ],
            "lp_native": saved[role + "_joint_log_prob_native"][
                start : start + len(index)
            ],
        }
        for key, expected in checks.items():
            verdict = comparison(cached[key], expected)
            report["checks"][f"{role}:{start}:{key}"] = verdict
            if not verdict["passed"]:
                report["status"] = "strict_failed"
                atomic(report_path, report)
                return report
        results[role].append((start, cached))
        atomic(
            output / "progress.json",
            dict(run_id=run, role=role, processed=start + len(index)),
        )
    for role in results:
        ordered = [values for _, values in sorted(results[role])]
        role_arrays[role] = {
            key: np.concatenate([v[key] for v in ordered]) for key in ordered[0]
        }
    index_cal = np.flatnonzero(roles == "cal")
    index_test = np.flatnonzero(roles == "test")
    cp = {}
    for alpha in [0.1, 0.05]:
        residual = np.abs(
            target[index_cal, 0] - role_arrays["cal"]["mean_native"][:, 0]
        )
        q = order_threshold(residual, alpha)
        density_q = order_threshold(-role_arrays["cal"]["lp_native"], alpha)
        lower = role_arrays["test"]["mean_native"][:, 0] - q
        upper = role_arrays["test"]["mean_native"][:, 0] + q
        np.savez_compressed(
            destination / f"conformal-{int(round((1 - alpha) * 100))}.npz",
            calibration_source_id=ids[index_cal],
            test_source_id=ids[index_test],
            residual_scores=residual,
            residual_threshold=np.array(q),
            lower_native=lower,
            upper_native=upper,
            density_threshold=np.array(density_q),
            density_membership=-role_arrays["test"]["lp_native"] <= density_q,
        )
        cp[str(alpha)] = dict(
            calibration_count=len(index_cal),
            test_count=len(index_test),
            residual_threshold_native=q,
            width_native=2 * q,
            residual_coverage=float(
                (
                    (target[index_test, 0] >= lower) & (target[index_test, 0] <= upper)
                ).mean()
            ),
            density_level_set_coverage=float(
                (-role_arrays["test"]["lp_native"] <= density_q).mean()
            ),
        )
    report.update(
        status="head_numerics_cp_pass",
        conformal=cp,
        test_count=len(index_test),
        calibration_count=len(index_cal),
        stream_and_selected_adam_pass=True,
        physical_truth_claim=False,
        raw_encoder_derivation_independent=False,
    )
    atomic(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError(
            "Native-model verification must execute on CANFAR CPU headless"
        )
    verifier_git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "singleton.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = yaml.safe_load(args.config.read_text())
    actual_library = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if actual_library.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
        raise ValueError("Independent verifier library pin differs from policy")
    base = Path(cfg["base"])
    prepared = Path(cfg["prepared"])
    if digest(prepared) != cfg["input_sha256"]:
        raise ValueError("Prepared input SHA changed")
    data = dict(np.load(prepared, allow_pickle=False))
    ids = data["source_id"]
    roles = data["role"]
    target = data["target"]
    if (
        target.shape != (len(ids), 1)
        or len(np.unique(ids)) != len(ids)
        or set(roles.tolist()) != {"fit", "tune", "cal", "test"}
    ):
        raise ValueError("Invalid independent source/roles/dimensions")
    if (data["benchmark_partition"][np.isin(roles, ["fit", "tune"])] == 1).any():
        raise ValueError("Benchmark validation entered FIT/TUNE")
    if (data["raw_feh"][np.isin(roles, ["fit", "tune"])] <= -2).any():
        raise ValueError("Unrecommended alpha labels entered FIT/TUNE")
    if not np.isfinite(target).all() or not np.isfinite(data["embeddings"]).all():
        raise ValueError("Nonfinite source inputs")
    fit = target[roles == "fit"]
    normalizer = (fit.mean(0), fit.std(0))
    reports = []
    for model in cfg["models"]:
        reports.append(audit_model(base, args.output, cfg, model, data, normalizer))
        atomic(
            args.output / "report.json", dict(status="running", model_reports=reports)
        )
    atomic(
        args.output / "report.json",
        dict(
            status="scoped_head_audit_pass"
            if all(r["status"] == "head_numerics_cp_pass" for r in reports)
            else "scoped_head_audit_failed",
            model_reports=reports,
            completed_at=dt.datetime.now(dt.UTC).isoformat(),
            library_version=importlib.metadata.version("torchregress"),
            library_source=actual_library,
            verifier_source_sha=verifier_git_sha,
            full_pipeline_or_physical_truth_pass=False,
        ),
    )


if __name__ == "__main__":
    main()
