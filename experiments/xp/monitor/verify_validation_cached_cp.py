"""Independent residual split-CP audit using completed Gaussian/MDN CPU caches.

No model construction/forward, fitting, or production calibration helper.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASE = Path("experiments/xp/results-stellar-improvement")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def threshold(mean, target, alpha):
    if mean.shape != target.shape or mean.ndim != 2:
        raise ValueError("matching two-dimensional mean and target required")
    if not np.isfinite(mean).all() or not np.isfinite(target).all():
        raise ValueError("finite calibration arrays required")
    rank = math.ceil((len(mean) + 1) * (1 - alpha))
    if rank > len(mean) or rank < 1:
        raise ValueError("insufficient calibration rows")
    return np.sort(np.abs(target - mean), axis=0)[rank - 1]


def verify(directory):
    directory = Path(directory)
    proof_path = directory / "independent-double-point-full.json"
    proof = json.loads(proof_path.read_text())
    assert (
        proof["status"]
        == "full_point_density_passed_source_transferred_remaining_gates_pending"
    )
    assert not proof["failures"]
    names = ["manifest.json", "checkpoint.pth", "paired-cohort.npz", "predictions.npz"]
    for name in names:
        assert digest(directory / name) == proof["bindings"][name]
    assert (
        digest(Path(__file__).with_name("verify_validation_points.py"))
        == proof["bindings"]["code"]
    )
    bindings = {
        name: digest(directory / name)
        for name in names + ["conformal.npz", proof_path.name]
    }
    bindings["verifier"] = digest(__file__)
    cache = directory / "independent-double-point-full-cache"
    assert json.loads((cache / "bindings.json").read_text()) == proof["bindings"]
    paths = sorted(cache.glob("calibration-*.npz"))
    bindings["calibration_caches"] = {p.name: digest(p) for p in paths}
    output = directory / "independent-cached-residual-cp.json"
    if output.exists():
        old = json.loads(output.read_text())
        assert old["bindings"] == bindings
        return old
    try:
        with np.load(directory / "paired-cohort.npz") as p:
            rows = {r: p[r + "_rows"] for r in ["test", "calibration"]}
            ids = p["source_id"]
            center, scale = (
                p["label_center"].astype("f8"),
                p["label_scale"].astype("f8"),
            )
            y = {r: p["value"][rows[r]].astype("f8") for r in rows}
        assert not np.intersect1d(ids[rows["test"]], ids[rows["calibration"]]).size
        with np.load(directory / "predictions.npz") as p:
            means = {r: p[r + "_mean"] for r in rows}
            for role in rows:
                np.testing.assert_array_equal(p[role + "_source_id"], ids[rows[role]])
                np.testing.assert_array_equal(p[role + "_rows"], rows[role])
        assert len(rows["test"]) == 97695 and len(rows["calibration"]) == 2046
        cpu = []
        for start in range(0, len(rows["calibration"]), 128):
            path = cache / f"calibration-{start:06d}.npz"
            with np.load(path) as p:
                block = p["mean"]
            assert block.shape == (min(128, len(rows["calibration"]) - start), 3)
            cpu.append(block)
        cpu = np.concatenate(cpu)
        np.testing.assert_allclose(cpu, means["calibration"], rtol=5e-5, atol=5e-5)
        target = (y["calibration"] - center) / scale
        report = {
            "status": "cached_mean_residual_cp_passed",
            "bindings": bindings,
            "protocols": {},
            "scope": "Residual CP only; not density level sets or physical-truth coverage. No model forward.",
            "coordinate": "saved standardized intervals; native widths multiply label_scale",
        }
        with np.load(directory / "conformal.npz") as cp:
            for name, alpha in [("marginal", 0.1), ("simultaneous", 0.1 / 3)]:
                q = threshold(means["calibration"], target, alpha)
                np.testing.assert_array_equal(q, cp[name + "_threshold"])
                cpu_q = threshold(cpu, target, alpha)
                np.testing.assert_allclose(cpu_q, q, rtol=5e-5, atol=5e-5)
                lo, hi = means["test"] - q, means["test"] + q
                np.testing.assert_array_equal(lo, cp[name + "_lower"])
                np.testing.assert_array_equal(hi, cp[name + "_upper"])
                native_lo, native_hi = center + scale * lo, center + scale * hi
                covered = (y["test"] >= native_lo) & (y["test"] <= native_hi)
                populations = {}
                for group, selected in [
                    ("all", np.ones(len(covered), bool)),
                    ("FeH_below_minus2", y["test"][:, 2] < -2),
                ]:
                    populations[group] = {
                        "n": int(selected.sum()),
                        "marginal_covered": covered[selected].sum(0).tolist(),
                        "joint_covered": int(covered[selected].all(1).sum()),
                        "mean_width_native_K_dex_dex": (
                            native_hi[selected] - native_lo[selected]
                        )
                        .mean(0)
                        .tolist(),
                    }
                report["protocols"][name] = {
                    "alpha_per_target": alpha,
                    "threshold_standardized": q.tolist(),
                    "cpu_threshold_max_difference": float(np.max(np.abs(cpu_q - q))),
                    "populations": populations,
                }
    except Exception as error:
        report = {"status": "strict_failed", "bindings": bindings, "error": repr(error)}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    result = verify(BASE / args.run)
    print(json.dumps(result))
    raise SystemExit(result["status"] != "cached_mean_residual_cp_passed")
