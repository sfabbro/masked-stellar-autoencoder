"""Independent functional NSF: masked MLPs, rational-quadratic splines, sequential inverse.

No torchregress/Zuko/MSA-head forward, distribution, density or moment helpers.
Only the established encoder architecture constructor is reused for state replay.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path("experiments/xp/monitor").resolve()))
import verify_multilabel_points as point

B = point.BASE
torch.set_num_threads(4)


def cli():
    global _args, P, _report_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--reuse-components-from-report",
        type=Path,
        help="Explicit source-bound component transfer; never bypass comparison",
    )
    _args = parser.parse_args()
    P = _args.directory
    _report_path = P / (
        "independent-wave-noisy-nsf-pilot-report.json"
        if _args.pilot
        else "independent-wave-noisy-nsf-report.json"
    )


def strict_compare(value, expected):
    """Unchanged relative+absolute 5e-5 gate; nonfinite is always failure."""
    if value.shape != expected.shape:
        raise ValueError("Independent and saved output shapes differ")
    bad = (
        ~np.isclose(value, expected, rtol=5e-5, atol=5e-5)
        | ~np.isfinite(value)
        | ~np.isfinite(expected)
    )
    return {
        "passed": not bool(bad.any()),
        "violating_elements": int(bad.sum()),
        "max_absolute_difference": float(np.max(np.abs(value - expected)))
        if np.isfinite(value).all() and np.isfinite(expected).all()
        else None,
    }


def measurement_nodes(count=32, dimensions=3, seed=2026):
    """Regenerate specified original float32 antithetic nodes, then promote."""
    if count != 32 or dimensions != 3:
        raise ValueError("Only archived3D32-node contract supported")
    uniform = (
        torch.quasirandom.SobolEngine(dimensions, scramble=True, seed=seed)
        .draw(count // 2)
        .double()
    )
    if not ((uniform > 0) & (uniform < 1)).all():
        raise ValueError("Nonfinite inverse-normal endpoint")
    z = math.sqrt(2.0) * torch.erfinv(2 * uniform - 1)
    return torch.cat([z, -z]).float()


def manual_label_convolution(target, sigma, nodes, intrinsic_log_density):
    """Finite-QMC diagonal Gaussian measurement convolution; not exact integral."""
    if target.shape != sigma.shape or target.shape[-1] != 3 or nodes.shape != (32, 3):
        raise ValueError("Expected matched3D target/sigma and32nodes")
    if not torch.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("Finite nonnegative quoted label errors required")
    perturbed = target[None] + sigma[None] * nodes[:, None]
    return torch.logsumexp(intrinsic_log_density(perturbed), 0) - math.log(len(nodes))


def context(latent, s):
    h = F.gelu(F.linear(latent, s["input.weight"], s["input.bias"]))
    b = F.layer_norm(
        h, (h.shape[-1],), s["block.0.weight"], s["block.0.bias"], eps=1e-5
    )
    b = F.gelu(F.linear(b, s["block.1.weight"], s["block.1.bias"]))
    b = F.linear(b, s["block.3.weight"], s["block.3.bias"])
    return F.linear(h + b, s["output.weight"], s["output.bias"])


import importlib.util

_kernel_path = Path(__file__).with_name("independent_nsf_kernel.py")
_spec = importlib.util.spec_from_file_location("independent_nsf_kernel", _kernel_path)
_kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kernel)
transform, inverse, density = _kernel.transform, _kernel.inverse, _kernel.density


def selftest(s):
    sd = {k: v.double() if v.is_floating_point() else v for k, v in s.items()}
    torch.manual_seed(2026)
    x = torch.randn(4, 3, dtype=torch.float64)
    c = torch.randn(4, 16, dtype=torch.float64) * 0.2
    z, ld = transform(x, c, sd)
    torch.testing.assert_close(inverse(z, c, sd), x, rtol=1e-9, atol=1e-10)
    for i in range(4):
        J = torch.autograd.functional.jacobian(
            lambda q: transform(q[None], c[i : i + 1], sd)[0][0], x[i]
        )
        sign, det = torch.linalg.slogdet(J)
        assert sign == 1
        torch.testing.assert_close(det, ld[i], rtol=1e-9, atol=1e-10)
    print(
        "Independent NSF roundtrip and autograd change-of-variables tests PASS",
        flush=True,
    )


def density_calibration_checks(P, B, a, f, actual_roles, m):
    report = {}
    dc = dict(np.load(P / "density-calibration-arrays.npz"))
    ds = json.loads((P / "density-calibration-summary.json").read_text())
    import h5py

    cohort = B / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    assert point.digest(cohort) == m["cohort_sha256"]
    with h5py.File(cohort) as original:
        features = json.loads(original.attrs["feature_columns_json"])
        g = original["features/values"][:, features.index("G")]
    boundaries = np.quantile(g[a["fit_rows"]], [1 / 3, 2 / 3], method="linear")
    np.testing.assert_array_equal(boundaries, dc["g_boundaries"])
    for role in ["fit", "calibration", "test"]:
        rr = a[role + "_rows"]
        np.testing.assert_array_equal(a["source_id"][rr].astype(str), dc[role + "_ids"])
        np.testing.assert_array_equal(g[rr], dc[role + "_g"])
        np.testing.assert_array_equal(
            np.searchsorted(boundaries, g[rr], side="right"), dc[role + "_groups"]
        )
    for role in ["calibration", "test"]:
        np.testing.assert_array_equal(
            dc[role + "_scores"], -f[role + "_intrinsic_log_prob_standardized"]
        )
        np.testing.assert_array_equal(
            dc[role + "_weights"], np.ones(len(dc[role + "_scores"]))
        )
    report["joint_density_cp"] = {}
    for entry in ds["calibrations"]:
        key = entry["key"]
        grouped = entry["grouped"]
        thresholds = []
        cpu_thresholds = []
        for group in range(3 if grouped else 1):
            selected = (
                (dc["calibration_groups"] == group)
                if grouped
                else np.ones(len(dc["calibration_scores"]), bool)
            )
            n = int(selected.sum())
            rank = math.ceil((n + 1) * (1 - entry["alpha"]))
            thresholds.append(
                np.sort(dc["calibration_scores"][selected])[rank - 1]
                if rank <= n
                else np.inf
            )
            cpu_scores = -actual_roles["calibration"][
                "intrinsic_log_prob_standardized"
            ][selected]
            cpu_thresholds.append(
                np.sort(cpu_scores)[rank - 1] if rank <= n else np.inf
            )
        thresholds = np.asarray(thresholds)
        np.testing.assert_array_equal(thresholds, dc[key + "_thresholds"])
        per = (
            thresholds[dc["test_groups"]]
            if grouped
            else np.repeat(thresholds[0], len(dc["test_scores"]))
        )
        np.testing.assert_array_equal(per, dc[key + "_per_test_threshold"])
        np.testing.assert_array_equal(dc["test_scores"] <= per, dc[key + "_membership"])
        bad = ~np.isclose(cpu_thresholds, thresholds, rtol=5e-5, atol=5e-5)
        cpu_thresholds = np.asarray(cpu_thresholds)
        finite_pair = np.isfinite(cpu_thresholds) & np.isfinite(thresholds)
        max_difference = float(
            np.max(
                np.abs(cpu_thresholds[finite_pair] - thresholds[finite_pair]),
                initial=0.0,
            )
        )
        if np.any(np.isfinite(cpu_thresholds) != np.isfinite(thresholds)):
            max_difference = None
        report["joint_density_cp"][key] = {
            "saved_scores_groups_quantiles_membership_exact": True,
            "cpu_quantile_passed": not bad.any(),
            "cpu_thresholds": [
                float(v) if np.isfinite(v) else None for v in cpu_thresholds
            ],
            "cpu_unbounded": np.isinf(cpu_thresholds).tolist(),
            "max_threshold_difference": max_difference,
        }
    return report["joint_density_cp"]


def main():
    m = json.loads((P / "manifest.json").read_text())
    assert not m["smoke"] and not m["synthetic_labels"]
    assert m["policy"]["target_names"] == ["teff", "logg", "fe_h"]
    assert m["prediction_evaluation"]["dtype"] == "float64"
    assert m["source_sha"] == "a45ae15759fa158026ff6473b0e02816bb828802"
    assert m["policy"]["library_revision"] == "1d735c6a94456e5edbb4fd53f404121598c85830"
    assert (
        m["status"] == "complete"
        and m["arm"]["kind"] == "flow"
        and m["arm"]["flow_type"] == "nsf"
        and m["arm"].get("noise_aware", False)
    )
    bindings = {
        n: point.digest(P / n)
        for n in [
            "manifest.json",
            "checkpoint.pth",
            "paired-cohort.npz",
            "predictions.npz",
            "conformal.npz",
        ]
    }
    joint_cp = bool(
        m["policy"].get("density_conformal", False)
        or m["arm"].get("density_conformal", False)
    )
    if joint_cp:
        for name in (
            "density-calibration-summary.json",
            "density-calibration-arrays.npz",
        ):
            bindings[name] = point.digest(P / name)
    bindings["verifier"] = point.digest(__file__)
    bindings["kernel"] = point.digest(_kernel_path)
    bindings["point_helper"] = point.digest(point.__file__)
    bindings["encoder_constructor"] = point.digest(point.architecture.__file__)
    bindings["model_config"] = point.digest(Path("configs/pretrain.canfar.yaml"))
    assert bindings["checkpoint.pth"] == m["checkpoint_sha256"]
    oldpath = _report_path
    if oldpath.exists():
        old = json.loads(oldpath.read_text())
        if old["bindings"] != bindings:
            archive = oldpath.with_name(
                oldpath.stem + "-" + old["bindings"]["verifier"][:12] + ".json"
            )
            if not archive.exists():
                archive.write_bytes(oldpath.read_bytes())
            if not _args.reuse_components_from_report:
                raise ValueError(
                    "Changed verifier needs explicit archived source-bound component transfer"
                )
            old = {}
        if old.get("status") in [
            "complete_strict_pass",
            "complete_strict_failed",
            "pilot_strict_pass",
            "pilot_strict_failed",
        ]:
            print(json.dumps(old, indent=2))
            return
    (P / "independent-wave-noisy-nsf-verifier.py").write_bytes(
        Path(__file__).read_bytes()
    )
    (P / "independent_nsf_kernel.py").write_bytes(_kernel_path.read_bytes())
    disk = torch.load(
        B / "main-checkpoints/checkpoint-86.pth", map_location="cpu", weights_only=False
    )
    s = torch.load(P / "checkpoint.pth", map_location="cpu", weights_only=False)
    assert point.digest(B / "main-checkpoints/checkpoint-86.pth") == m["parent_sha256"]
    head = {
        k.replace("loss_fn.base_loss.", "loss_fn."): v
        for k, v in s["head_state_dict"].items()
    }
    assert len(head["measurement_nodes"]) == 32
    measure = measurement_nodes()
    torch.testing.assert_close(measure, head["measurement_nodes"], rtol=0, atol=0)
    selftest(head)
    # Sobol+normal mapping is specified independently; all 256 stored nodes retained.
    u = torch.quasirandom.SobolEngine(3, scramble=True, seed=2026).draw(128).double()
    node = math.sqrt(2) * torch.erfinv(2 * u - 1)
    nodes = torch.cat([node, -node]).float()
    torch.testing.assert_close(nodes, head["point_nodes"], rtol=0, atol=0)
    a = dict(np.load(P / "paired-cohort.npz"))
    f = dict(np.load(P / "predictions.npz"))
    refpath = B / "multi_gaussian_recon_s42_v1"
    proof = json.loads((refpath / "independent-point-source-cp.json").read_text())
    assert (
        point.digest(refpath / "paired-cohort.npz")
        == proof["bindings"]["paired-cohort.npz"]
    )
    ref = np.load(refpath / "paired-cohort.npz")
    for key in ref.files:
        np.testing.assert_array_equal(a[key], ref[key])
    assert a["value"].shape[1] == 3
    for key in ["label_center", "label_scale"]:
        np.testing.assert_array_equal(a[key], s[key])
    token = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    cache = P / "independent-wave-noisy-nsf-cache" / token
    cache.mkdir(parents=True, exist_ok=True)
    transfer_cache = None
    if _args.reuse_components_from_report:
        prior = json.loads(_args.reuse_components_from_report.read_text())
        before = dict(prior["bindings"])
        after = dict(bindings)
        before.pop("verifier")
        after.pop("verifier")
        assert before == after, (
            "Component transfer requires exact unchanged source/input/kernel bindings"
        )
        prior_token = hashlib.sha256(
            json.dumps(prior["bindings"], sort_keys=True).encode()
        ).hexdigest()
        transfer_cache = P / "independent-wave-noisy-nsf-cache" / prior_token
        assert transfer_cache.exists()
        point.atomic(
            cache / "component-transfer.json",
            {
                "prior_report_sha256": point.digest(_args.reuse_components_from_report),
                "prior_bindings": prior["bindings"],
                "new_bindings": bindings,
                "scope": "Only output-schema handling changed; same functional double components; all gates rerun",
            },
        )
    model = point.architecture.load_model(
        yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
        disk,
        torch.device("cpu"),
    )
    model.load_state_dict(s["model_state_dict"], strict=True)
    model.double().eval()
    head = {k: v.double() if v.is_floating_point() else v for k, v in head.items()}
    nodes = nodes.double()
    measure = measure.double()
    report = {
        "status": "running",
        "pilot_only": _args.pilot,
        "processed_role_counts": {},
        "bindings": bindings,
        "source_identity_inputs_scalers_control_proof_sha256": point.digest(
            refpath / "independent-point-source-cp.json"
        ),
        "all256_nodes_independently_checked": True,
        "roles": {},
        "limits": [
            "Same fixed 256-node moment quadrature, not exact population moments",
            "Corrected end-to-end float64 evaluation of unchanged trained weights",
            "Observed density uses fixed32 antithetic measurement nodes and diagonal quoted label errors; not input EIV, exact integral or physical-truth calibration",
            "No production spline/NSF/head helper used; encoder architecture constructor reused",
            "Core model/source/density/CP only; decoder, training objective and stream audit remain separate gates",
            "Native noisy labels, no physical-truth calibration or conditional-coverage guarantee",
        ],
    }
    point.atomic(_report_path, report)
    outputs = {}
    actual_roles = {}
    for role in ["test", "calibration"]:
        all_rows = a[role + "_rows"]
        rows = all_rows[:128] if _args.pilot else all_rows
        np.testing.assert_array_equal(all_rows, f[role + "_rows"])
        np.testing.assert_array_equal(a["source_id"][all_rows], f[role + "_source_id"])
        assert bool(f[role + "_observed_log_prob_available"])
        parts = {
            k: []
            for k in [
                "mean",
                "variance",
                "observed_variance",
                "intrinsic_log_prob_standardized",
                "observed_log_prob_standardized",
            ]
        }
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                dest = cache / f"{role}-{start:06d}.npz"
                if dest.exists():
                    chunk = dict(np.load(dest))
                elif (
                    transfer_cache is not None and (transfer_cache / dest.name).exists()
                ):
                    chunk = dict(np.load(transfer_cache / dest.name))
                    np.savez_compressed(dest, **chunk)
                else:
                    rr = rows[start : start + 128]
                    c = context(
                        model.encoder(torch.from_numpy(a["input"][rr]).double()), head
                    )
                    samples = []
                    for point_batch in nodes.split(16):
                        samples.append(
                            inverse(
                                point_batch[:, None, :].expand(-1, len(rr), -1), c, head
                            )
                        )
                    sample = torch.cat(samples, 0)
                    mean = sample.mean(0)
                    variance = (sample - mean).square().mean(0)
                    y = torch.from_numpy(
                        (
                            (
                                a["value"][rr].astype("f8")
                                - a["label_center"].astype("f8")
                            )
                            / a["label_scale"].astype("f8")
                        ).astype("f8")
                    )
                    lp = density(y, c, head)
                    sigma = torch.from_numpy(
                        a["error"][rr].astype("f8") / a["label_scale"].astype("f8")
                    )
                    observed = manual_label_convolution(
                        y, sigma, measure, lambda q: density(q, c, head)
                    )
                    chunk = {
                        "mean": mean.numpy(),
                        "variance": variance.numpy(),
                        "observed_variance": (variance + sigma.square()).numpy(),
                        "intrinsic_log_prob_standardized": lp.numpy(),
                        "observed_log_prob_standardized": observed.numpy(),
                    }
                    np.savez_compressed(dest, **chunk)
                for key in parts:
                    parts[key].append(chunk[key])
                if start % 2048 == 0:
                    point.atomic(
                        P / "independent-wave-noisy-nsf-progress.json",
                        {
                            "phase": "functional_encoder_nsf_density_moments",
                            "role": role,
                            "rows_done": min(start + 128, len(rows)),
                            "rows_total": len(rows),
                            "bindings": bindings,
                        },
                    )
        actual = {key: np.concatenate(value) for key, value in parts.items()}
        np.savez_compressed(
            P
            / f"independent-wave-noisy-nsf-{role}{'-pilot' if _args.pilot else ''}.npz",
            **actual,
        )
        outputs[role] = actual["mean"]
        actual_roles[role] = actual
        checks = {}
        for key, value in actual.items():
            expected = f[role + "_" + key][: len(value)]
            checks[key] = strict_compare(value, expected)
        jac = float(np.log(a["label_scale"].astype("f8")).sum())
        for field in [
            "intrinsic_log_prob_native",
            "observed_log_prob_native",
            "observed_log_prob_standardized",
        ]:
            standardized = (
                actual["observed_log_prob_standardized"]
                if field.startswith("observed_")
                else actual["intrinsic_log_prob_standardized"]
            )
            value = (
                standardized if field.endswith("standardized") else standardized - jac
            )
            expected = f[role + "_" + field][: len(value)]
            if field.startswith("observed_") and not bool(
                f[role + "_observed_log_prob_available"]
            ):
                checks[field] = {
                    "passed": bool(np.isnan(expected).all()),
                    "status": "unavailable_no_measurement_model_nan_required",
                    "not_a_density_certificate": True,
                }
            else:
                checks[field] = strict_compare(value, expected)
        report["roles"][role] = checks
        report["processed_role_counts"][role] = len(rows)
        point.atomic(_report_path, report)
    if _args.pilot:
        ok = all(
            v["passed"] for role in report["roles"].values() for v in role.values()
        )
        report["status"] = "pilot_strict_pass" if ok else "pilot_strict_failed"
        report["limits"].append(
            "Only128 test and128 calibration rows; not a full model or CP certificate"
        )
        point.atomic(_report_path, report)
        point.atomic(
            P / "independent-wave-noisy-nsf-progress.json",
            {"phase": "pilot_complete", "status": report["status"]},
        )
        print(json.dumps(report, indent=2), flush=True)
        return
    cp = np.load(P / "conformal.npz")
    target = (
        (
            a["value"][a["calibration_rows"]].astype("f8")
            - a["label_center"].astype("f8")
        )
        / a["label_scale"].astype("f8")
    ).astype("f8")
    report["cp"] = {}
    for name, alpha in [("marginal", 0.1), ("simultaneous", 0.1 / 3)]:
        q = point.order_threshold(f["calibration_mean"], target, alpha)
        np.testing.assert_array_equal(q, cp[name + "_threshold"])
        for suffix, sign in [("lower", -1), ("upper", 1)]:
            np.testing.assert_array_equal(
                f["test_mean"] + sign * q, cp[name + "_" + suffix]
            )
        cpuq = point.order_threshold(outputs["calibration"], target, alpha)
        bad = ~np.isclose(cpuq, q, rtol=5e-5, atol=5e-5)
        report["cp"][name] = {
            "saved_arithmetic_exact": True,
            "cpu_quantile_passed": not bad.any(),
            "max_threshold_difference": float(np.abs(cpuq - q).max()),
        }
    report["joint_density_cp"] = (
        density_calibration_checks(P, B, a, f, actual_roles, m) if joint_cp else {}
    )
    report["status"] = (
        "complete_strict_pass"
        if all(v["passed"] for role in report["roles"].values() for v in role.values())
        and all(v["cpu_quantile_passed"] for v in report["cp"].values())
        and all(v["cpu_quantile_passed"] for v in report["joint_density_cp"].values())
        else "complete_strict_failed"
    )
    point.atomic(_report_path, report)
    point.atomic(
        P / "independent-wave-noisy-nsf-progress.json",
        {"phase": "complete", "status": report["status"]},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    cli()
    try:
        main()
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "error": str(exc),
            "verifier_sha256": point.digest(__file__),
        }
        point.atomic(
            P
            / (
                "independent-wave-noisy-nsf-failure-"
                + failure["verifier_sha256"][:12]
                + ".json"
            ),
            failure,
        )
        report_path = _report_path
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if (
                previous.get("bindings", {}).get("verifier")
                == failure["verifier_sha256"]
            ):
                previous.update(status="strict_failed_exception", failure=failure)
                point.atomic(report_path, previous)
        point.atomic(
            P / "independent-wave-noisy-nsf-progress.json",
            {"phase": "failed", **failure},
        )
        raise
