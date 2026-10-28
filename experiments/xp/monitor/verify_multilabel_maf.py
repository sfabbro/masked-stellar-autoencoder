"""Independent functional MAF: masked MLPs, bounded affine maps, sequential inverse.

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
P = B / "multi_maf_recon_s42_v1"
torch.set_num_threads(4)


def context(latent, s):
    h = F.gelu(F.linear(latent, s["input.weight"], s["input.bias"]))
    b = F.layer_norm(
        h, (h.shape[-1],), s["block.0.weight"], s["block.0.bias"], eps=1e-5
    )
    b = F.gelu(F.linear(b, s["block.1.weight"], s["block.1.bias"]))
    b = F.linear(b, s["block.3.weight"], s["block.3.bias"])
    return F.linear(h + b, s["output.weight"], s["output.bias"])


def affine(x, c, s, k):
    prefix = f"loss_fn.flow.transform.transforms.{k}.hyper."
    c = c.expand(*x.shape[:-1], c.shape[-1])
    h = torch.cat([x, c], -1)
    for i in [0, 2, 4]:
        h = F.linear(
            h,
            s[prefix + f"{i}.weight"] * s[prefix + f"{i}.mask"],
            s[prefix + f"{i}.bias"],
        )
        if i < 4:
            h = F.relu(h)
    h = h.reshape(*h.shape[:-1], 3, 2)
    shift = h[..., 0]
    raw = h[..., 1]
    log_scale = raw / (1 + torch.abs(raw / math.log(1e-3)))
    return shift, log_scale


def transform(x, c, s):
    logdet = torch.zeros(x.shape[:-1], dtype=x.dtype)
    for k in [0, 1, 2]:
        shift, ls = affine(x, c, s, k)
        x = x * ls.exp() + shift
        logdet += ls.sum(-1)
    return x, logdet


def inverse(z, c, s):
    for k in [2, 1, 0]:
        x = torch.zeros_like(z)
        order = s[f"loss_fn.flow.transform.transforms.{k}.order"]
        indices = torch.argsort(order).tolist()
        for j in indices:
            shift, ls = affine(x, c, s, k)
            v = (z[..., j] - shift[..., j]) * torch.exp(-ls[..., j])
            x = x.clone()
            x[..., j] = v
        z = x
    return z


def density(x, c, s):
    z, ld = transform(x, c, s)
    loc = s["loss_fn.flow.base.loc"]
    scale = s["loss_fn.flow.base.scale"]
    return (
        -0.5 * (((z - loc) / scale) ** 2 + math.log(2 * math.pi)) - torch.log(scale)
    ).sum(-1) + ld


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
        "Independent MAF roundtrip and autograd change-of-variables tests PASS",
        flush=True,
    )


def main():
    m = json.loads((P / "manifest.json").read_text())
    assert (
        m["status"] == "complete"
        and m["arm"]["kind"] == "flow"
        and m["arm"]["flow_type"] == "maf"
        and not m["arm"].get("noise_aware", False)
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
    bindings["verifier"] = point.digest(__file__)
    assert bindings["checkpoint.pth"] == m["checkpoint_sha256"]
    report_path = P / "independent-maf-report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        original = previous.get("bindings", {})
        for key, value in bindings.items():
            if key != "verifier" and original.get(key) != value:
                raise ValueError(
                    "Existing audit input binding differs; refusing to repeat or overwrite"
                )
        archive = P / "independent-maf-verifier.py"
        if original.get("verifier") != bindings["verifier"]:
            if not archive.exists() or point.digest(archive) != original.get(
                "verifier"
            ):
                raise ValueError("Existing audit verifier archive does not match")
            if previous.get("status") not in [
                "complete_strict_pass",
                "complete_strict_failed",
            ]:
                raise ValueError(
                    "Resume the exact archived verifier for an incomplete audit"
                )
        if previous.get("status") in ["complete_strict_pass", "complete_strict_failed"]:
            print(json.dumps(previous, indent=2), flush=True)
            return previous
    disk = torch.load(
        B / "main-checkpoints/checkpoint-86.pth", map_location="cpu", weights_only=False
    )
    s = torch.load(P / "checkpoint.pth", map_location="cpu", weights_only=False)
    assert point.digest(B / "main-checkpoints/checkpoint-86.pth") == m["parent_sha256"]
    head = s["head_state_dict"]
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
    for key in ["label_center", "label_scale"]:
        np.testing.assert_array_equal(a[key], s[key])
    token = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    cache = P / "independent-maf-cache" / token
    cache.mkdir(parents=True, exist_ok=True)
    model = point.architecture.load_model(
        yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
        disk,
        torch.device("cpu"),
    )
    model.load_state_dict(s["model_state_dict"], strict=True)
    model.eval()
    report = {
        "status": "running",
        "bindings": bindings,
        "source_identity_inputs_scalers_control_proof_sha256": point.digest(
            refpath / "independent-point-source-cp.json"
        ),
        "all256_nodes_independently_checked": True,
        "roles": {},
        "limits": [
            "Same fixed 256-node moment quadrature, not exact population moments",
            "No underlying production affine/MAF helper used; encoder architecture constructor reused",
            "Native noisy labels, no physical-truth calibration or conditional-coverage guarantee",
        ],
    }
    point.atomic(P / "independent-maf-report.json", report)
    outputs = {}
    for role in ["test", "calibration"]:
        rows = a[role + "_rows"]
        np.testing.assert_array_equal(rows, f[role + "_rows"])
        np.testing.assert_array_equal(a["source_id"][rows], f[role + "_source_id"])
        parts = {k: [] for k in ["mean", "variance", "intrinsic_log_prob_standardized"]}
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                dest = cache / f"{role}-{start:06d}.npz"
                if dest.exists():
                    chunk = dict(np.load(dest))
                else:
                    rr = rows[start : start + 128]
                    c = context(model.encoder(torch.from_numpy(a["input"][rr])), head)
                    samples = []
                    for point_batch in nodes.split(16):
                        samples.append(
                            inverse(
                                point_batch[:, None, :].expand(-1, len(rr), -1), c, head
                            )
                        )
                    sample = torch.cat(samples, 0)
                    mean = sample.double().mean(0).float()
                    variance = (sample - mean).square().double().mean(0).float()
                    y = torch.from_numpy(
                        (
                            (a["value"][rr] - a["label_center"]) / a["label_scale"]
                        ).astype("f4")
                    )
                    lp = density(y, c, head)
                    chunk = {
                        "mean": mean.numpy(),
                        "variance": variance.numpy(),
                        "intrinsic_log_prob_standardized": lp.numpy(),
                    }
                    np.savez_compressed(dest, **chunk)
                for key in parts:
                    parts[key].append(chunk[key])
                if start % 2048 == 0:
                    point.atomic(
                        P / "independent-maf-progress.json",
                        {
                            "phase": "functional_encoder_maf_density_moments",
                            "role": role,
                            "rows_done": min(start + 128, len(rows)),
                            "rows_total": len(rows),
                            "bindings": bindings,
                        },
                    )
        actual = {key: np.concatenate(value) for key, value in parts.items()}
        np.savez_compressed(P / f"independent-maf-{role}.npz", **actual)
        outputs[role] = actual["mean"]
        checks = {}
        for key, value in actual.items():
            expected = f[role + "_" + key]
            bad = ~np.isclose(value, expected, rtol=5e-5, atol=5e-5)
            checks[key] = {
                "passed": not bad.any(),
                "violating_elements": int(bad.sum()),
                "max_absolute_difference": float(np.abs(value - expected).max()),
            }
        report["roles"][role] = checks
        point.atomic(P / "independent-maf-report.json", report)
    cp = np.load(P / "conformal.npz")
    target = (
        (a["value"][a["calibration_rows"]] - a["label_center"]) / a["label_scale"]
    ).astype("f4")
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
    report["status"] = (
        "complete_strict_pass"
        if all(v["passed"] for role in report["roles"].values() for v in role.values())
        and all(v["cpu_quantile_passed"] for v in report["cp"].values())
        else "complete_strict_failed"
    )
    point.atomic(P / "independent-maf-report.json", report)
    point.atomic(
        P / "independent-maf-progress.json",
        {"phase": "complete", "status": report["status"]},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="multi_maf_recon_s42_v1")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        parser.error("--run must be a single run-directory name")
    P = B / args.run
    try:
        main()
    except Exception as exc:
        point.atomic(
            P / "independent-maf-failure.json",
            {"type": type(exc).__name__, "error": str(exc)},
        )
        raise
