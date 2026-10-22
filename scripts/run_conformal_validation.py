#!/usr/bin/env python3
"""Evaluation-only repaired conformal controls on immutable saved predictions.

Original calibration rows are split into gamma tuning and final calibration;
original fitting/model-tuning/test rows are never used for gamma selection.
Family spread is empirical matched-head dispersion, NOT theoretical uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from masked_stellar_autoencoder.reporting.theory import TheoryScaledScore


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_roles(data, pred):
    groups = []
    for role in ("fit", "tune", "calibration", "test"):
        rows = data[role + "_rows"]
        ids = data["source_id"][rows]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate object identities within role")
        if any(np.intersect1d(ids, other).size for other in groups):
            raise ValueError("object identities overlap roles")
        groups.append(ids)
        if role in ("calibration", "test"):
            if not np.array_equal(rows, pred[role + "_rows"]) or not np.array_equal(
                ids, pred[role + "_source_id"]
            ):
                raise ValueError("prediction role/source identity mismatch")
            for field in ("mean", "variance"):
                values = pred[role + "_" + field]
                if values.shape != (len(rows), 3) or not np.isfinite(values).all():
                    raise ValueError("invalid point moment shape or finiteness")
            if (pred[role + "_variance"] < 0).any():
                raise ValueError("negative predictive variance")


def evaluate(y, lo, hi, *, group, method, target, joint=False):
    mask = np.ones(len(y), dtype=bool) if group == "all" else y[:, 2] < -2
    covered = (
        ((y >= lo) & (y <= hi)).all(axis=1)
        if joint
        else ((y[:, target] >= lo[:, target]) & (y[:, target] <= hi[:, target]))
    )
    return {
        "method": method,
        "population": group,
        "target": "joint_bonferroni"
        if joint
        else ["Teff_K", "logg_dex", "FeH_dex"][target],
        "n": int(mask.sum()),
        "covered": int(covered[mask].sum()),
        "coverage": float(covered[mask].mean()) if mask.any() else None,
        "mean_width": None
        if joint or not mask.any()
        else float((hi[:, target] - lo[:, target])[mask].mean()),
    }


def run(case, base, output):
    source = base / case["source_run"]
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("source_sha") != case["source_sha"]
    ):
        raise ValueError("immutable source manifest mismatch")
    proof = {}
    for name in ("paired-cohort.npz", "predictions.npz"):
        actual = digest(source / name)
        if actual != case["sha256"][name]:
            raise ValueError("artifact SHA mismatch")
        proof[name] = actual
    with np.load(source / "paired-cohort.npz") as f:
        data = {
            k: f[k]
            for k in (
                "source_id",
                "value",
                "survey",
                "fit_rows",
                "tune_rows",
                "calibration_rows",
                "test_rows",
            )
        }
    with np.load(source / "predictions.npz") as f:
        pred = {
            k: f[k]
            for k in (
                "test_rows",
                "calibration_rows",
                "test_source_id",
                "calibration_source_id",
                "test_mean",
                "calibration_mean",
                "test_variance",
                "calibration_variance",
            )
        }
    validate_roles(data, pred)
    family = {
        role: np.zeros_like(pred[role + "_mean"]) for role in ("test", "calibration")
    }
    if case.get("family_members"):
        members = {role: [] for role in family}
        for member in case["family_members"]:
            path = base / member["run_id"] / "predictions.npz"
            if digest(path) != member["sha256"]:
                raise ValueError("family artifact SHA mismatch")
            with np.load(path) as f:
                for role in family:
                    if not np.array_equal(
                        f[role + "_source_id"], pred[role + "_source_id"]
                    ):
                        raise ValueError("family identity mismatch")
                    members[role].append(f[role + "_mean"])
        family = {role: np.std(members[role], axis=0, ddof=1) for role in family}
    ids = pred["calibration_source_id"]
    # Predeclared label-free split; source-ID order creates no test-label access.
    order = np.argsort(ids, kind="stable")
    tune_idx, cal_idx = order[::2], order[1::2]
    test_rows, cal_rows = data["test_rows"], data["calibration_rows"]
    y_test, y_cal = data["value"][test_rows], data["value"][cal_rows]
    sig = {role: np.maximum(np.sqrt(pred[role + "_variance"]), 1e-6) for role in family}
    rows, protocols = [], []
    test_survey, cal_survey = data["survey"][test_rows], data["survey"][cal_rows]
    for grouping in ("global", "mondrian_survey"):
        groups = [None] if grouping == "global" else np.unique(test_survey)
        for mode in (
            "fixed_gamma0",
            "calibration_selected_empirical",
            "independent_tuning",
        ):
            for alpha in (0.1, 0.1 / 3):
                lo, hi = (
                    np.zeros_like(y_test, dtype=float),
                    np.zeros_like(y_test, dtype=float),
                )
                for g in groups:
                    test_mask = (
                        np.ones(len(test_rows), dtype=bool)
                        if g is None
                        else test_survey == g
                    )
                    c = cal_idx if g is None else cal_idx[cal_survey[cal_idx] == g]
                    t = tune_idx if g is None else tune_idx[cal_survey[tune_idx] == g]
                    for label in range(3):
                        arrays = lambda indices: (
                            y_cal[indices, label],
                            pred["calibration_mean"][indices, label],
                            sig["calibration"][indices, label],
                            family["calibration"][indices, label],
                        )
                        score = TheoryScaledScore(
                            alpha=alpha,
                            fixed_gamma=0.0 if mode == "fixed_gamma0" else None,
                        )
                        if mode == "independent_tuning":
                            doc = score.fit_with_tuning(
                                arrays(t),
                                arrays(c),
                                tuning_ids=ids[t],
                                calibration_ids=ids[c],
                            )
                        else:
                            doc = score.fit(*arrays(c))
                        low, high = score.interval(
                            pred["test_mean"][test_mask, label],
                            sig["test"][test_mask, label],
                            family["test"][test_mask, label],
                        )
                        lo[test_mask, label], hi[test_mask, label] = low, high
                        protocols.append(
                            {
                                "grouping": grouping,
                                "group": None if g is None else str(g),
                                "mode": mode,
                                "alpha": alpha,
                                "label": label,
                                "gamma": doc["gamma"],
                                "threshold": doc["threshold"],
                                "n_cal": len(c),
                                "n_tune": len(t) if mode == "independent_tuning" else 0,
                                "finite_sample_exact": doc["finite_sample_exact"],
                                "coverage_guarantee": doc["coverage_guarantee"],
                            }
                        )
                method = grouping + "/" + mode
                for population in ("all", "FeH_below_minus2_posthoc"):
                    if alpha == 0.1:
                        rows.extend(
                            evaluate(
                                y_test,
                                lo,
                                hi,
                                group=population,
                                method=method,
                                target=j,
                            )
                            for j in range(3)
                        )
                    else:
                        rows.append(
                            evaluate(
                                y_test,
                                lo,
                                hi,
                                group=population,
                                method=method,
                                target=0,
                                joint=True,
                            )
                        )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "role-identities.npz",
        gamma_tune_ids=ids[tune_idx],
        final_calibration_ids=ids[cal_idx],
        test_ids=pred["test_source_id"],
    )
    report = {
        "status": "evaluation_only_complete",
        "run_id": case["run_id"],
        "source_run": case["source_run"],
        "artifact_sha256": proof,
        "family_members": case.get("family_members", []),
        "family_scale_kind": "matched_seed_head_sample_std_not_theory_error"
        if case.get("family_members")
        else "zero_family_scale_control",
        "prediction_scale_floor_native_units": 1e-6,
        "calibration_split": "source_id_sorted_alternating_even_tune_odd_finalcal",
        "strata": "preexisting_survey_only_no_label_based_calibration_groups",
        "training_updates": 0,
        "new_fit": False,
        "physical_truth_coverage_claim": False,
        "conformal_assumption": "exchangeable future objects and score/model independent of final calibration labels; NOT guaranteed under distribution shift",
        "protocols": protocols,
        "metrics": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/conformal-validation-wave-20261022.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    case = next(c for c in config["cases"] if c["run_id"] == args.run_id)
    run(case, args.base, args.output)


if __name__ == "__main__":
    main()
