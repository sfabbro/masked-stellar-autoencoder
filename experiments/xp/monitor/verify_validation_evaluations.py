"""Independent NumPy verification of saved CPU validation evaluations; no model forwards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

BASE = Path("experiments/xp/results-stellar-improvement")
META = BASE / "validation-runtime-20261023T0405"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def rank_quantile(scores, alpha):
    scores = np.asarray(scores)
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if not 1 <= k <= len(scores):
        raise ValueError("insufficient calibration")
    return np.sort(scores)[k - 1]


def gamma_selection(residual, sigma, family, alpha):
    n = len(residual)
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(0).permutation(n)] = np.arange(n) % 5
    costs = []
    for gamma in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        den = np.maximum(sigma + gamma * family, 1e-12)
        total = 0.0
        for k in range(5):
            mask = fold == k
            q = rank_quantile(residual[~mask] / den[~mask], alpha)
            half = q * den[mask]
            total += np.sum(2 * half + 2 / alpha * np.maximum(residual[mask] - half, 0))
        costs.append((total / n, gamma))
    return min(costs)[1]


def stat(e):
    return np.array([np.abs(e).mean(), e.mean(), np.median(np.abs(e - np.median(e)))])


def bootstrap(a, b, repetitions):
    rng = np.random.default_rng(20261022)
    out = []
    for _ in range(repetitions):
        idx = rng.integers(0, len(a), len(a))
        out.append(stat(a[idx]) - stat(b[idx]))
    return np.quantile(out, [0.025, 0.975], axis=0)


def catalogue(case):
    path = BASE / "catalogue-comparison-input-v1" / Path(case["input"]).name
    assert sha(path) == case["input_npz_sha256"]
    d = load(path)
    assert d["csv_sha256"].item() == case["source_csv_sha256"]
    ids = d["source_id"]
    assert ids.dtype.kind in "iu" and (ids > 0).all() and len(set(ids)) == len(ids)
    report = json.loads((META / (case["run_id"] + ".report.json")).read_text())
    assert report["input_npz_sha256"] == sha(path)
    masks = {"all": np.ones(len(ids), bool), "FeH_below_minus2": d["feh_true"] < -2}
    if d["heldout_available"].item():
        masks["catalogue_training_heldout"] = d["heldout"]
        masks["catalogue_training_heldout_FeH_below_minus2"] = d["heldout"] & (
            d["feh_true"] < -2
        )
    rows = []
    for name, mask in masks.items():
        for j, target in enumerate(case["targets"]):
            keep = (
                mask
                & np.isfinite(d["y"][:, j])
                & np.isfinite(d["ours"][:, j])
                & np.isfinite(d["other"][:, j])
            )
            a = d["ours"][keep, j] - d["y"][keep, j]
            b = d["other"][keep, j] - d["y"][keep, j]
            if len(a):
                av, bv = stat(a), stat(b)
                ci = bootstrap(a, b, report["bootstrap"]["repetitions"])
            for k, metric in enumerate(("MAE", "bias", "MedAD")):
                saved = next(
                    r
                    for r in report["metrics"]
                    if r["subset"] == name
                    and r["target"] == target
                    and r["metric"] == metric
                )
                assert saved["n"] == len(a)
                row = {
                    "run_id": case["run_id"],
                    "subset": name,
                    "target": target,
                    "metric": metric,
                    "n": len(a),
                }
                if len(a):
                    vals = [av[k], bv[k], av[k] - bv[k], ci[0, k], ci[1, k]]
                    keys = [
                        "ours",
                        "catalogue",
                        "difference_ours_minus_catalogue",
                        "paired_bootstrap_95_low",
                        "paired_bootstrap_95_high",
                    ]
                    np.testing.assert_allclose(
                        vals, [saved[key] for key in keys], atol=1e-10, rtol=1e-12
                    )
                    row.update(zip(keys, map(float, vals)))
                rows.append(row)
    return rows, {
        "status": "independent_paired_metrics_and_bootstrap_passed",
        "input_sha256": sha(path),
        "source_csv_sha256": case["source_csv_sha256"],
        "report_sha256": sha(META / (case["run_id"] + ".report.json")),
    }


def conformal(case, report_directory=None):
    src = BASE / case["source_run"]
    report_path = (
        (META / (case["run_id"] + ".report.json"))
        if report_directory is None
        else Path(report_directory) / case["run_id"] / "report.json"
    )
    report = json.loads(report_path.read_text())
    native_coordinate = (
        report.get("calibration_coordinate") == "native_Teff_K_logg_dex_FeH_dex"
    )
    for name, digest in case["sha256"].items():
        assert sha(src / name) == digest
    manifest = json.loads((src / "manifest.json").read_text())
    assert (
        manifest["source_sha"] == case["source_sha"]
        and manifest["status"] == "complete"
    )
    d, p = load(src / "paired-cohort.npz"), load(src / "predictions.npz")
    seen = set()
    for role in ("fit", "tune", "calibration", "test"):
        ids = d["source_id"][d[role + "_rows"]]
        assert len(set(ids)) == len(ids) and not seen.intersection(ids)
        seen.update(ids)
        if role in ("calibration", "test"):
            np.testing.assert_array_equal(ids, p[role + "_source_id"])
            np.testing.assert_array_equal(d[role + "_rows"], p[role + "_rows"])
    if native_coordinate:
        for role in ("test", "calibration"):
            p[role + "_mean"] = (
                p[role + "_mean"].astype(np.float64) * d["label_scale"]
                + d["label_center"]
            )
            p[role + "_variance"] = (
                p[role + "_variance"].astype(np.float64) * d["label_scale"] ** 2
            )
    family = {r: np.zeros_like(p[r + "_mean"]) for r in ("test", "calibration")}
    if case.get("family_members"):
        members = {r: [] for r in family}
        for member in case["family_members"]:
            path = BASE / member["run_id"] / "predictions.npz"
            assert sha(path) == member["sha256"]
            assert (
                sha(BASE / member["run_id"] / "paired-cohort.npz")
                == case["sha256"]["paired-cohort.npz"]
            )
            f = load(path)
            for r in family:
                np.testing.assert_array_equal(f[r + "_source_id"], p[r + "_source_id"])
                members[r].append(
                    f[r + "_mean"].astype(np.float64) * d["label_scale"]
                    + d["label_center"]
                    if native_coordinate
                    else f[r + "_mean"]
                )
        family = {r: np.std(members[r], axis=0, ddof=1) for r in family}
    ids = p["calibration_source_id"]
    order = np.argsort(ids, kind="stable")
    t, c = order[::2], order[1::2]
    assert not set(ids[t]).intersection(ids[c])
    ycal, yt = d["value"][d["calibration_rows"]], d["value"][d["test_rows"]]
    cs, ts = d["survey"][d["calibration_rows"]], d["survey"][d["test_rows"]]
    sig = {r: np.maximum(np.sqrt(p[r + "_variance"]), 1e-6) for r in family}
    sig = {r: v.astype(np.float64) for r, v in sig.items()}
    family = {r: v.astype(np.float64) for r, v in family.items()}
    intervals = {}
    for record in report["protocols"]:
        g, label, alpha = record["group"], record["label"], record["alpha"]
        ci = c if g is None else c[np.array([str(x) for x in cs[c]]) == g]
        ti = t if g is None else t[np.array([str(x) for x in cs[t]]) == g]
        pick = ti if record["mode"] == "independent_tuning" else ci
        residual = abs(
            ycal[:, label].astype(np.float64)
            - p["calibration_mean"][:, label].astype(np.float64)
        )
        gamma = (
            0.0
            if record["mode"] == "fixed_gamma0"
            else gamma_selection(
                residual[pick],
                sig["calibration"][pick, label],
                family["calibration"][pick, label],
                alpha,
            )
        )
        assert gamma == record["gamma"]
        den = np.maximum(
            sig["calibration"][ci, label] + gamma * family["calibration"][ci, label],
            1e-12,
        )
        q = rank_quantile(residual[ci] / den, alpha)
        np.testing.assert_allclose(q, record["threshold"], atol=1e-10, rtol=1e-12)
        assert record["finite_sample_exact"] == (
            record["mode"] != "calibration_selected_empirical"
        )
        key = (record["grouping"] + "/" + record["mode"], alpha)
        lo, hi = intervals.setdefault(key, (np.empty_like(yt), np.empty_like(yt)))
        mask = (
            np.ones(len(yt), bool) if g is None else np.array([str(x) for x in ts]) == g
        )
        half = q * np.maximum(
            sig["test"][mask, label] + gamma * family["test"][mask, label], 1e-12
        )
        lo[mask, label], hi[mask, label] = (
            p["test_mean"][mask, label] - half,
            p["test_mean"][mask, label] + half,
        )
    rows = []
    for saved in report["metrics"]:
        joint = saved["target"] == "joint_bonferroni"
        lo, hi = intervals[(saved["method"], 0.1 / 3 if joint else 0.1)]
        mask = np.ones(len(yt), bool) if saved["population"] == "all" else yt[:, 2] < -2
        j = 0 if joint else ["Teff_K", "logg_dex", "FeH_dex"].index(saved["target"])
        covered = (
            ((yt >= lo) & (yt <= hi)).all(1)
            if joint
            else (yt[:, j] >= lo[:, j]) & (yt[:, j] <= hi[:, j])
        )
        assert saved["n"] == int(mask.sum()) and saved["covered"] == int(
            covered[mask].sum()
        )
        np.testing.assert_allclose(saved["coverage"], covered[mask].mean(), atol=1e-14)
        if not joint:
            np.testing.assert_allclose(
                saved["mean_width"], (hi[:, j] - lo[:, j])[mask].mean(), atol=1e-10
            )
        rows.append(dict(run_id=case["run_id"], **saved))
    return rows, {
        "status": "independent_roles_gamma_quantile_coverage_passed",
        "source_artifact_sha256": case["sha256"],
        "report_sha256": sha(report_path),
        "scientific_estimand_status": "native_coordinate_checked"
        if native_coordinate
        else "INVALID_mixed_standardized_moments_native_labels_arithmetic_only",
        "density_verification": False,
        "physical_truth_coverage": False,
    }


def age_pilots(output):
    """State/identity/selection only; never certifies age likelihood or accuracy."""
    import torch

    config = yaml.safe_load(Path("configs/target-pilot-wave-20261022.yaml").read_text())
    src = BASE / "joint-age-native-pilot-input-v3"
    assert sha(src / "pilot.npz") == config["input_sha256"]
    assert sha(src / "manifest.json") == config["manifest_sha256"]
    data = load(src / "pilot.npz")
    ids, roles = data["source_id"], data["role"]
    assert ids.dtype.kind in "iu" and len(set(ids)) == len(ids)
    assert set(roles) == {"fit", "tune", "test"}
    overlap = np.isfinite(data["teacher_tau"]).sum(axis=0) > 1
    assert not overlap.any()
    reports = {}
    for arm in config["arms"]:
        run = arm["run_id"]
        manifest_path = BASE / run / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        files = [BASE / run / name for name in ("checkpoint.pth", "predictions.npz")]
        if not all(f.exists() for f in files):
            reports[run] = {
                "status": "pending_missing_selected_checkpoint_or_predictions"
            }
            continue
        assert sha(files[0]) == manifest["checkpoint_sha256"]
        assert sha(files[1]) == manifest["predictions_sha256"]
        assert manifest["input_sha256"] == config["input_sha256"]
        assert manifest["preparation_manifest_sha256"] == config["manifest_sha256"]
        history = manifest["history"]
        assert len(history) == manifest["completed_steps"] == 100
        assert [x["step"] for x in history] == list(range(1, 101))
        assert np.isfinite([[x["fit_nll"], x["tune_nll"]] for x in history]).all()
        assert (
            history[int(np.argmin([x["tune_nll"] for x in history]))]["step"]
            == manifest["selected_step"]
        )
        state = torch.load(files[0], map_location="cpu", weights_only=True)
        assert state["step"] == manifest["selected_step"]

        def finite(value):
            if isinstance(value, torch.Tensor):
                assert bool(torch.isfinite(value).all())
            elif isinstance(value, dict):
                for v in value.values():
                    finite(v)
            elif isinstance(value, tuple | list):
                for v in value:
                    finite(v)

        finite(state)
        saved = load(files[1])
        for role in ("fit", "tune", "test"):
            np.testing.assert_array_equal(
                saved[role + "_source_id"], ids[roles == role]
            )
            assert (
                np.isfinite(saved[role + "_mean"]).all()
                and np.isfinite(saved[role + "_log_scale"]).all()
            )
        reports[run] = {
            "status": "independent_selected_state_roles_finiteness_passed_likelihood_not_audited",
            "checkpoint_sha256": sha(files[0]),
            "predictions_sha256": sha(files[1]),
            "selected_step": state["step"],
            "teacher_overlap_rows": 0,
            "test_label_isolation": "producer_source_fit_and_tune_only_control_flow; no counterfactual rerun",
            "scope": "native384 frozen embedding code-path pilot only; no joint-vs-composite identifiability claim",
        }
    (output / "age-state-report.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"verifier_sha256": sha(Path(__file__)), "runs": {}}
    for kind in ("catalogue", "conformal"):
        config = Path("configs") / (
            "catalogue-comparison-wave-20261022.yaml"
            if kind == "catalogue"
            else "conformal-validation-wave-20261022.yaml"
        )
        all_rows = []
        for case in yaml.safe_load(config.read_text())["cases"]:
            try:
                rows, verdict = (catalogue if kind == "catalogue" else conformal)(case)
                all_rows.extend(rows)
            except Exception as error:
                verdict = {"status": "failed", "error": repr(error)}
            summary["runs"][case["run_id"]] = verdict
            (args.output / "progress.json").write_text(json.dumps(summary, indent=2))
        if all_rows:
            with (args.output / (kind + "-metrics.csv")).open("w") as f:
                w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
                w.writeheader()
                w.writerows(all_rows)
    (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
