"""Unattended saved-array QA/report after the decision experiments finish."""

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import kstest

TARGETS = ["teff", "logg", "feh", "alpha", "age", "distance"]
ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def calibration(pit):
    x = np.asarray(pit)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    return dict(
        n=len(x),
        pit_mean=float(x.mean()),
        coverage50=float(np.mean((x >= 0.25) & (x <= 0.75))),
        coverage68=float(np.mean((x >= 0.16) & (x <= 0.84))),
        coverage90=float(np.mean((x >= 0.05) & (x <= 0.95))),
        coverage95=float(np.mean((x >= 0.025) & (x <= 0.975))),
        uniform_ks_statistic=float(kstest(x, "uniform").statistic),
        scope="Empirical sample PIT against retained reference values; known-noise PIT separate; not independent physical truth or certified joint six-D coverage",
    )


def score_run(run_id):
    out = ROOT / run_id
    manifest = json.loads((out / "manifest.json").read_text())
    result = dict(
        run_id=run_id,
        source_manifest_status=manifest.get("status"),
        selected_update=manifest.get("selected_update"),
        completed_updates=manifest.get("completed_updates"),
        best_tune_score=manifest.get("best_tune_score"),
        target_results={},
        artifacts={},
    )
    for name in [
        "development-predictions.npz",
        "tune-predictions.npz",
        "whole-vector-corners.npz",
        "normalization.npz",
    ]:
        p = out / name
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:
                raise ValueError("NPZ CRC failed")
        result["artifacts"][name] = dict(bytes=p.stat().st_size, sha256=sha(p))
    with np.load(out / "development-predictions.npz", allow_pickle=False) as f:
        ids = f["source_id"]
        value = f["value"]
        pred = f["native_median"]
        observed = f["observed"].astype(bool)
        if (
            len(np.unique(ids)) != len(ids)
            or pred.shape != value.shape
            or value.shape != observed.shape
        ):
            raise ValueError("Prediction identity/layout differs")
        for j, target in enumerate(TARGETS):
            mask = observed[:, j] & np.isfinite(value[:, j]) & np.isfinite(pred[:, j])
            d = pred[mask, j] - value[mask, j]
            med = float(np.median(d)) if len(d) else None
            result["target_results"][target] = dict(
                n=int(mask.sum()),
                mae=float(abs(d).mean()) if len(d) else None,
                median_bias=med,
                nmad=float(1.4826 * np.median(abs(d - med))) if len(d) else None,
                intrinsic_reference_compatibility=calibration(
                    f["empirical_intrinsic_pit"][mask, j]
                ),
                quoted_noise_reference_compatibility=calibration(
                    f["empirical_observed_pit"][mask, j]
                ),
            )
        result["input_slices"] = {
            name: int(f[name].astype(bool).sum())
            for name in ["real_xp_present", "real_no_xp", "measured_parallax_observed"]
        }
        result["complete_six_reference_rows"] = int(observed.all(1).sum())
    with np.load(out / "whole-vector-corners.npz", allow_pickle=False) as f:
        draws = f["native_draws"]
        if draws.ndim != 3 or draws.shape[-1] != 6 or not np.isfinite(draws).all():
            raise ValueError("Invalid joint six-vector draws")
        recomputed = np.array(
            [np.cov(draws[:, i, :], rowvar=False) for i in range(draws.shape[1])]
        )
        np.testing.assert_allclose(
            recomputed, f["native_covariance"], rtol=1e-6, atol=1e-8
        )
        result["whole_vector_check"] = dict(
            draws=draws.shape[0],
            stars=draws.shape[1],
            targets=6,
            covariance_recomputed=True,
            finite=True,
        )
    result["selected_checkpoint_sha256"] = sha(out / "selected-checkpoint.pth")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-hours", type=float, default=24)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.runs.read_text())
    pending = set(config["run_ids"])
    results = {}
    deadline = time.monotonic() + a.wait_hours * 3600
    while pending and time.monotonic() < deadline:
        for run_id in sorted(pending.copy()):
            path = ROOT / run_id / "manifest.json"
            if not path.exists():
                continue
            try:
                m = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if m.get("status") == "remediation_pilot_complete":
                try:
                    results[run_id] = dict(status="qa_complete", **score_run(run_id))
                except Exception as e:
                    results[run_id] = dict(status="qa_failed", error=repr(e))
                pending.remove(run_id)
            elif m.get("status") == "failed":
                results[run_id] = dict(
                    status="experiment_failed",
                    completed_updates=m.get("completed_updates"),
                    error=m.get("error"),
                    error_type=m.get("error_type"),
                )
                pending.remove(run_id)
        atomic(
            a.output / "manifest.json",
            dict(
                status="waiting" if pending else "decision_array_qa_complete",
                pending=sorted(pending),
                results=results,
                model_loaded=False,
                training_performed=False,
            ),
        )
        if pending:
            time.sleep(45)
    if pending:
        for run_id in pending:
            results[run_id] = dict(status="not_complete_by_24h_report_deadline")
    ranked = sorted(
        [r for r in results.values() if r["status"] == "qa_complete"],
        key=lambda r: r["best_tune_score"],
    )
    report = [
        "# Six-target decision wave — 6 November 2026",
        "",
        "Decision target: 7 November 2026 at approximately 19:30 Australia/Perth.",
        "",
        "Selection ordering below uses TUNE macro normalized median MAE only. Inspect reference-compatibility coverage, subgroup errors, source-map results, input coverage and failures before selecting final training. Age/distance are posterior-reference targets; chemical conventions and sparse joint support remain explicit. The controlled pilots use historical FIT pools, including known historical benchmark/teacher exposure; they do not define the protected final FIT bundle.",
        "",
        "| Run | Updates | Selected | TUNE score |",
        "|---|---:|---:|---:|",
    ]
    for r in ranked:
        report.append(
            "| "
            + r["run_id"]
            + " | "
            + str(r["completed_updates"])
            + " | "
            + str(r["selected_update"])
            + " | "
            + str(r["best_tune_score"])
            + " |"
        )
    report += [
        "",
        "## Final-training gates",
        "",
        "- Protected star/cluster/dependency roles and benchmark exclusions; no inference from an inspected devtest as fresh evidence.",
        "- Decide whether measured parallax is enabled; admit it for expanded IDs before scaling.",
        "- Evaluate new-source mappings and common-reference stability; source-only ingestion is not training.",
        "- Require finite full-vector draws, acceptable marginal calibration and domain/teacher strata; sparse six-way overlap does not certify all correlations.",
        "- Complete native label units/quantity/error contracts; age/distance posterior teachers are distinct from physical anchors.",
        "- Define all-Gaia input coverage, missing-input/OOD flags, joint PDF representation, shard failures and throughput/storage budget.",
        "",
        "Full per-target results and SHA/CRC receipts are in manifest.json. No final large training job was launched automatically.",
    ]
    (a.output / "decision-report.md").write_text("\n".join(report) + "\n")
    atomic(
        a.output / "manifest.json",
        dict(
            status="decision_array_qa_complete"
            if not pending
            else "decision_report_incomplete_experiments",
            results=results,
            pending=sorted(pending),
            tune_order=[r["run_id"] for r in ranked],
            training_performed=False,
            model_loaded=False,
        ),
    )


if __name__ == "__main__":
    main()
