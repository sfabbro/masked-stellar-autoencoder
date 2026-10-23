"""Independent double Gaussian/MDN replay; source transfer and per-batch caches.

No production head, distribution, moment or calibration operators. Encoder
architecture constructor only. Old verified source arrays are transferred by
exact equality; label normalization is independently recomputed in float64.
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

sys.path.insert(0, str(Path("scripts").resolve()))
import run_xp_representation_probes as architecture

BASE = Path("experiments/xp/results-stellar-improvement")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def raw_head(x, s):
    h = F.gelu(F.linear(x, s["input.weight"], s["input.bias"]))
    r = F.layer_norm(
        h, (h.shape[-1],), s["block.0.weight"], s["block.0.bias"], eps=1e-5
    )
    r = F.gelu(F.linear(r, s["block.1.weight"], s["block.1.bias"]))
    r = F.linear(r, s["block.3.weight"], s["block.3.bias"])
    return F.linear(h + r, s["output.weight"], s["output.bias"])


def analytic(raw, y, sigma, kind, components=4):
    if kind == "gaussian":
        mean, lv = raw.chunk(2, -1)
        v = lv.exp()

        def lp(var):
            return (
                -0.5 * ((y - mean).square() / var + var.log() + math.log(2 * math.pi))
            ).sum(-1)

        return mean, v, lp(v), lp(v + sigma.square())
    if kind != "mdn":
        raise ValueError("Gaussian/MDN only")
    k = components
    dim = y.shape[-1]
    w = raw[:, :k].log_softmax(-1)
    means = raw[:, k : k + k * dim].reshape(-1, k, dim)
    v = (2 * raw[:, k + k * dim :].reshape(-1, k, dim)).exp()
    mean = (w.exp()[..., None] * means).sum(1)
    variance = (w.exp()[..., None] * (v + (means - mean[:, None]).square())).sum(1)

    def lp(var):
        component = (
            -0.5
            * ((y[:, None] - means).square() / var + var.log() + math.log(2 * math.pi))
        ).sum(-1)
        return torch.logsumexp(w + component, -1)

    return mean, variance, lp(v), lp(v + sigma[:, None].square())


def source_proof(paired, directory):
    reference = BASE / "multi_gaussian_recon_s42_v1"
    proof = json.loads((reference / "independent-point-source-cp.json").read_text())
    assert proof["status"] == "point_source_cp_passed_remaining_gates_pending"
    assert (
        digest(reference / "paired-cohort.npz")
        == proof["bindings"]["paired-cohort.npz"]
    )
    with np.load(reference / "paired-cohort.npz") as old:
        for key in old.files:
            if key not in ("label_center", "label_scale"):
                np.testing.assert_array_equal(
                    old[key], paired[key], err_msg="source transfer " + key
                )
    values = paired["value"].astype("f8")
    fit = paired["fit_rows"]
    np.testing.assert_allclose(
        paired["label_center"], values[fit].mean(0), rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        paired["label_scale"], values[fit].std(0), rtol=0, atol=1e-10
    )
    return {
        "original_proof_sha256": digest(reference / "independent-point-source-cp.json"),
        "transfer": "all original raw source/quality/role/input keys exactly equal; f64 labelstats recomputed",
    }


def verify(run, pilot=False):
    torch.set_num_threads(4)
    d = BASE / run
    m = json.loads((d / "manifest.json").read_text())
    arm = m["arm"]
    assert (
        m["status"] == "complete"
        and m["source_sha"] == "a45ae15759fa158026ff6473b0e02816bb828802"
        and not m["smoke"]
    )
    assert arm["kind"] in ("gaussian", "mdn")
    assert digest(d / "checkpoint.pth") == m["checkpoint_sha256"]
    names = ["manifest.json", "checkpoint.pth", "paired-cohort.npz", "predictions.npz"]
    bindings = {n: digest(d / n) for n in names}
    bindings["code"] = digest(__file__)
    snapshot = d / ("independent-double-point-code-" + bindings["code"] + ".py")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    mode = "pilot" if pilot else "full"
    out = d / ("independent-double-point-" + mode + ".json")
    if out.exists():
        old = json.loads(out.read_text())
        assert old["bindings"] == bindings
        return old
    cache = d / ("independent-double-point-" + mode + "-cache")
    cache.mkdir(exist_ok=True)
    binding_path = cache / "bindings.json"
    if binding_path.exists():
        assert json.loads(binding_path.read_text()) == bindings, (
            "Cached code/input bindings changed; preserve cache"
        )
    binding_path.write_text(json.dumps(bindings, indent=2) + "\n")
    with np.load(d / "paired-cohort.npz") as a:
        paired = {k: a[k] for k in a.files}
    with np.load(d / "predictions.npz") as a:
        final = {k: a[k] for k in a.files}
    proof = source_proof(paired, d)
    parent = BASE / "main-checkpoints/checkpoint-86.pth"
    assert digest(parent) == m["parent_sha256"]
    saved = torch.load(d / "checkpoint.pth", map_location="cpu", weights_only=False)
    model = architecture.load_model(
        yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
        torch.load(parent, map_location="cpu", weights_only=False),
        torch.device("cpu"),
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    encoder = model.encoder.double().eval()
    s = {
        k: v.double() if v.is_floating_point() else v
        for k, v in saved["head_state_dict"].items()
    }
    errors = []
    maxima = {}
    for role in ("test", "calibration"):
        rows = paired[role + "_rows"]
        np.testing.assert_array_equal(
            final[role + "_source_id"], paired["source_id"][rows]
        )
        if pilot:
            rows = rows[:128]
        for start in range(0, len(rows), 128):
            selected = rows[start : start + 128]
            path = cache / f"{role}-{start:06d}.npz"
            if path.exists():
                with np.load(path) as a:
                    computed = {k: a[k] for k in a.files}
            else:
                with torch.no_grad():
                    latent = encoder(
                        torch.tensor(paired["input"][selected], dtype=torch.float64)
                    )
                    y = torch.tensor(
                        (
                            paired["value"][selected].astype("f8")
                            - paired["label_center"].astype("f8")
                        )
                        / paired["label_scale"].astype("f8")
                    )
                    sigma = torch.tensor(
                        paired["error"][selected].astype("f8")
                        / paired["label_scale"].astype("f8")
                    )
                    mean, v, lp, obs = analytic(
                        raw_head(latent, s),
                        y,
                        sigma,
                        arm["kind"],
                        arm.get("mdn_components", 4),
                    )
                computed = {
                    "mean": mean.numpy(),
                    "variance": v.numpy(),
                    "intrinsic_log_prob_standardized": lp.numpy(),
                }
                if arm.get("noise_aware"):
                    computed["observed_log_prob_standardized"] = obs.numpy()
                jac = float(np.log(paired["label_scale"].astype("f8")).sum())
                for kind in ("intrinsic", "observed"):
                    key = kind + "_log_prob_standardized"
                    if key in computed:
                        computed[kind + "_log_prob_native"] = computed[key] - jac
                np.savez_compressed(path, **computed)
            for key, actual in computed.items():
                expected = final[role + "_" + key][start : start + len(selected)]
                delta = np.abs(actual - expected)
                bad = (
                    ~np.isfinite(actual)
                    | ~np.isfinite(expected)
                    | (delta > 5e-5 + 5e-5 * np.abs(expected))
                )
                label = role + "_" + key
                maxima[label] = max(maxima.get(label, 0.0), float(delta.max()))
                if bad.any():
                    errors.append(
                        {
                            "role": role,
                            "start": start,
                            "field": key,
                            "violations": int(bad.sum()),
                            "max_delta": float(delta.max()),
                        }
                    )
            (d / "independent-double-point-progress.json").write_text(
                json.dumps(
                    {
                        "role": role,
                        "rows_done": start + len(selected),
                        "total": len(rows),
                        "mode": mode,
                        "bindings": bindings,
                    }
                )
                + "\n"
            )
    report = {
        "status": mode + "_strict_failed"
        if errors
        else mode + "_point_density_passed_source_transferred_remaining_gates_pending",
        "bindings": bindings,
        "source_proof": proof,
        "maxima": maxima,
        "failures": errors,
        "scope": "No decoder/stream/objective/CP certificate; clean observed-label likelihood unavailable, never aliased",
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    r = verify(a.run, a.pilot)
    print(json.dumps(r))
    sys.exit(bool(r["failures"]))
