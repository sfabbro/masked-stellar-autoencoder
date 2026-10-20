"""Independent raw-source, point-head, Gaussian density and split-CP replay.

No fitting or production head/CP operators. This gate deliberately leaves
training-stream, reconstruction-source, TTT, flow and ensemble audits pending.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import run_xp_representation_probes as architecture

BASE = ROOT / "experiments/xp/results-stellar-improvement"
RTOL = ATOL = 5e-5


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def reference_head(state, latent, kind):
    hidden = F.gelu(F.linear(latent, state["input.weight"], state["input.bias"]))
    residual = F.layer_norm(
        hidden,
        (hidden.shape[-1],),
        state["block.0.weight"],
        state["block.0.bias"],
        eps=1e-5,
    )
    residual = F.gelu(
        F.linear(residual, state["block.1.weight"], state["block.1.bias"])
    )
    residual = F.linear(residual, state["block.3.weight"], state["block.3.bias"])
    output = F.linear(hidden + residual, state["output.weight"], state["output.bias"])
    if kind == "huber":
        return output, torch.full_like(output, float("nan"))
    mean, logvar = output.chunk(2, -1)
    return mean, logvar.exp()


def order_threshold(prediction, labels, alpha):
    scores = np.abs(prediction - labels)
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    return (
        np.full(scores.shape[1], np.inf)
        if k > len(scores)
        else np.sort(scores, axis=0)[k - 1]
    )


def verify(run):
    directory = BASE / run
    manifest = json.loads((directory / "manifest.json").read_text())
    assert (
        manifest["status"] == "complete"
        and not manifest["smoke"]
        and not manifest["synthetic_labels"]
    )
    arm = manifest["arm"]
    assert arm["kind"] in {"huber", "gaussian"}, (
        "Point-head gate cannot certify flows or inversion"
    )
    checkpoint = directory / "checkpoint.pth"
    assert digest(checkpoint) == manifest["checkpoint_sha256"]
    parent = BASE / "main-checkpoints/checkpoint-86.pth"
    cohort = BASE / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    sidecar = BASE / "joint-native-preparation-v2/joint-native-labels.h5"
    proof = json.loads(
        (
            BASE
            / "joint-native-preparation-v2-quality-v1/native-label-quality-independent-verification.json"
        ).read_text()
    )
    assert (
        digest(parent)
        == manifest["parent_sha256"]
        == manifest["policy"]["checkpoint_sha256"]
    )
    assert digest(cohort) == manifest["cohort_sha256"] == proof["cohort_sha256"]
    assert digest(sidecar) == proof["sidecar_sha256"]
    assert all(
        proof[k] is True
        for k in [
            "native_labels_verified",
            "all_target_quality_masks_verified",
            "all_original_source_target_values_errors_flags_rows_verified",
            "all_source_id_roles_and_reference_bans_verified",
        ]
    )
    bindings = {
        name: digest(directory / name)
        for name in [
            "manifest.json",
            "checkpoint.pth",
            "paired-cohort.npz",
            "predictions.npz",
            "conformal.npz",
        ]
    }
    bindings["verifier"] = digest(__file__)
    report_path = directory / "independent-point-source-cp.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        assert (
            old["bindings"] == bindings
            and old["status"] == "point_source_cp_passed_remaining_gates_pending"
        )
        return old
    progress = directory / "independent-point-progress.json"
    atomic(progress, {"phase": "raw_source", "bindings": bindings})
    disk = torch.load(parent, map_location="cpu", weights_only=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    features = disk["run_signature"]["feature_cols"]
    center = np.asarray(disk["run_signature"]["scaler_center"])
    scale = np.asarray(disk["run_signature"]["scaler_scale"])
    paired = np.load(directory / "paired-cohort.npz", allow_pickle=False)
    final = np.load(directory / "predictions.npz", allow_pickle=False)
    with h5py.File(cohort) as f:
        ids = f["features/source_id"][:]
        raw = f["features/values"][:]
        assert json.loads(f.attrs["feature_columns_json"]) == features
    # Independently enumerate the declared allowlist, with no production projector.
    names = {"w1", "w2", "g", "bp", "rp", "j", "h", "ks"}
    names |= {b + "_smss" for b in ["u", "v", "g", "r", "i", "z"]}
    names |= {b + "_sdss" for b in ["u", "g", "r", "i", "z"]}
    names |= {b + "_ps1" for b in ["g", "r", "i", "z", "y"]}
    names |= {f"{a}_{i}" for a in ["bp", "rp"] for i in range(1, 56)}
    allowed = np.array([str(n).casefold() in names for n in features])
    standardized = ((raw - center) / scale).astype("f4")
    visible = (
        np.isfinite(raw)
        & np.isfinite(standardized)
        & (standardized != -9999)
        & allowed[None, :]
    )
    inputs = np.where(visible, standardized, np.float32(-9999))
    np.testing.assert_array_equal(paired["source_id"], ids)
    np.testing.assert_array_equal(paired["visible"], visible)
    np.testing.assert_array_equal(paired["input"], inputs)
    roles = {}
    with h5py.File(sidecar) as f:
        np.testing.assert_array_equal(f["source_id"][:], ids)
        for name, key in [
            ("value", "value"),
            ("uncertainty", "error"),
            ("observed", "observed"),
            ("quality_pass", "quality"),
            ("survey", "survey"),
        ]:
            np.testing.assert_array_equal(f["targets/" + name][:], paired[key])
        complete = (f["targets/observed"][:] & f["targets/quality_pass"][:]).all(1)
        forbidden = (f["roles/benchmark_partition"][:] != 0) | f[
            "roles/reference_holdout"
        ][:]
        for role in ["fit", "tune", "calibration", "test"]:
            rows = np.flatnonzero(f["roles/" + role][:] & complete)
            np.testing.assert_array_equal(rows, paired[role + "_rows"])
            if role != "test":
                assert not forbidden[rows].any()
            roles[role] = rows
    for i, left in enumerate(roles):
        for right in list(roles)[i + 1 :]:
            assert not np.intersect1d(roles[left], roles[right]).size
    values = paired["value"]
    errors = paired["error"]
    label_center = values[roles["fit"]].mean(0)
    label_scale = values[roles["fit"]].std(0)
    np.testing.assert_array_equal(label_center, paired["label_center"])
    np.testing.assert_array_equal(label_scale, paired["label_scale"])
    np.testing.assert_array_equal(label_center, saved["label_center"])
    np.testing.assert_array_equal(label_scale, saved["label_scale"])
    del raw, standardized, visible
    config = yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text())
    model = architecture.load_model(config, disk, torch.device("cpu"))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    outputs = {}
    maxima = {}
    with torch.no_grad():
        for role in ["test", "calibration"]:
            rows = roles[role]
            parts = []
            variances = []
            np.testing.assert_array_equal(rows, final[role + "_rows"])
            np.testing.assert_array_equal(ids[rows], final[role + "_source_id"])
            for start in range(0, len(rows), 256):
                latent = model.encoder(
                    torch.from_numpy(inputs[rows[start : start + 256]])
                )
                mean, var = reference_head(
                    saved["head_state_dict"], latent, arm["kind"]
                )
                parts.append(mean.numpy())
                variances.append(var.numpy())
                if start % 10240 == 0:
                    atomic(
                        progress,
                        {
                            "phase": "encoder_head",
                            "role": role,
                            "rows_done": min(start + 256, len(rows)),
                            "rows_total": len(rows),
                            "bindings": bindings,
                        },
                    )
            mean = np.concatenate(parts)
            var = np.concatenate(variances)
            outputs[role] = mean
            for key, actual in [("mean", mean), ("variance", var)]:
                expected = final[role + "_" + key]
                np.testing.assert_allclose(
                    actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True
                )
                maxima[role + "_" + key] = (
                    None
                    if arm["kind"] == "huber" and key == "variance"
                    else float(np.max(np.abs(actual - expected)))
                )
            if arm["kind"] == "gaussian":
                target = ((values[rows] - label_center) / label_scale).astype("f4")
                sigma = (errors[rows] / label_scale).astype("f4")
                for name, variance in [
                    ("intrinsic", var),
                    ("observed", var + sigma**2 if arm.get("noise_aware") else var),
                ]:
                    lp = (
                        -0.5
                        * (
                            (target - mean) ** 2 / variance
                            + np.log(variance)
                            + np.float32(math.log(2 * math.pi))
                        )
                    ).sum(-1)
                    for unit, actual in [
                        ("standardized", lp),
                        ("native", lp - float(np.log(label_scale).sum())),
                    ]:
                        expected = final[f"{role}_{name}_log_prob_{unit}"]
                        np.testing.assert_allclose(
                            actual, expected, rtol=RTOL, atol=ATOL
                        )
                        maxima[f"{role}_{name}_{unit}"] = float(
                            np.max(np.abs(actual - expected))
                        )
    cp = np.load(directory / "conformal.npz")
    cal = roles["calibration"]
    test = roles["test"]
    target = ((values[cal] - label_center) / label_scale).astype("f4")
    np.testing.assert_array_equal(cp["calibration_source_id"], ids[cal])
    np.testing.assert_array_equal(cp["test_source_id"], ids[test])
    for prefix, alpha in [("marginal", 0.1), ("simultaneous", 0.1 / 3)]:
        # Exact equation gate uses saved GPU means to isolate CP from encoder rounding.
        q = order_threshold(final["calibration_mean"], target, alpha)
        np.testing.assert_array_equal(q, cp[prefix + "_threshold"])
        for suffix, sign in [("lower", -1), ("upper", 1)]:
            np.testing.assert_array_equal(
                final["test_mean"] + sign * q, cp[prefix + "_" + suffix]
            )
        cpu_q = order_threshold(outputs["calibration"], target, alpha)
        np.testing.assert_allclose(cpu_q, q, rtol=RTOL, atol=ATOL)
    np.savez_compressed(
        directory / "independent-point-predictions.npz",
        test_rows=test,
        calibration_rows=cal,
        test_mean=outputs["test"],
        calibration_mean=outputs["calibration"],
    )
    report = {
        "status": "point_source_cp_passed_remaining_gates_pending",
        "bindings": bindings,
        "source_rows": len(ids),
        "test_rows": len(test),
        "calibration_rows": len(cal),
        "max_errors": maxima,
        "rtol": RTOL,
        "atol": ATOL,
        "pending": [
            "training_stream_optimizer",
            "raw_reconstruction_source_and_decoder",
            "ttt_gradients_reset_and_calibration",
            "library_weighted_cp",
            "ensemble_composition",
            "paired_reports_visual_qa",
        ],
        "physical_truth_calibration_claim": False,
    }
    atomic(report_path, report)
    atomic(progress, {"phase": "point_source_cp_complete", "report": str(report_path)})
    return report


def self_test():
    p = np.zeros((9, 3), dtype="f4")
    y = np.arange(27, dtype="f4").reshape(9, 3)
    np.testing.assert_array_equal(
        order_threshold(p, y, 0.2), np.sort(np.abs(y), axis=0)[7]
    )
    assert np.isinf(order_threshold(p, y, 0.01)).all()
    h = 4
    s = {
        "input.weight": torch.eye(h),
        "input.bias": torch.zeros(h),
        "block.0.weight": torch.ones(h),
        "block.0.bias": torch.zeros(h),
        "block.1.weight": torch.zeros(1, h),
        "block.1.bias": torch.zeros(1),
        "block.3.weight": torch.zeros(h, 1),
        "block.3.bias": torch.zeros(h),
        "output.weight": torch.zeros(6, h),
        "output.bias": torch.tensor([1.0, 2.0, 3.0, 0.0, math.log(2), math.log(3)]),
    }
    mean, var = reference_head(s, torch.ones(2, h), "gaussian")
    torch.testing.assert_close(mean, torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1))
    torch.testing.assert_close(var, torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.self_test:
        self_test()
        print("reference head/finite-sample order tests PASS")
    else:
        if not args.run:
            parser.error("--run required")
        try:
            print(json.dumps(verify(args.run), indent=2), flush=True)
        except Exception as exc:
            path = BASE / args.run / "independent-point-failure.json"
            atomic(path, {"error_type": type(exc).__name__, "error": str(exc)})
            raise
