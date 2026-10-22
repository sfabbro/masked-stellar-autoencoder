#!/usr/bin/env python3
"""Local readiness audit only: no acquisition, prediction, fitting or launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from masked_stellar_autoencoder.training.alpha_heads import ALPHA_GATES


def audit_bounds(frame):
    required = ["age16", "age50", "age84"]
    if not all(k in frame for k in required):
        return {"status": "blocked_missing_bounds"}
    values = frame[required].to_numpy(dtype=float)
    valid = (
        np.isfinite(values).all(axis=1)
        & (values[:, 0] > 0)
        & (values[:, 0] <= values[:, 1])
        & (values[:, 1] <= values[:, 2])
    )
    declared = (
        frame["age_bounds_valid"].to_numpy(dtype=bool)
        if "age_bounds_valid" in frame
        else valid
    )
    ids = (
        frame["source_id"].dropna()
        if "source_id" in frame
        else pd.Series(dtype="int64")
    )
    ids_valid = (
        len(ids) == len(frame) and bool((ids > 0).all()) and not ids.duplicated().any()
    )
    return {
        "rows": len(frame),
        "positive_ordered_bounds": int(valid.sum()),
        "invalid_or_missing_bounds": int((~valid).sum()),
        "declared_valid_but_invalid": int((declared & ~valid).sum()),
        "unique_positive_gaia_ids": ids_valid,
        "status": "usable_subset"
        if ids_valid and not (declared & ~valid).any()
        else "blocked_invalid_identity_or_bounds",
    }


def audit_cohort(path, anchor_ids):
    if not path.exists():
        return {"status": "blocked_missing_cohort", "path": str(path)}
    with h5py.File(path) as f:
        group = f["selected"]
        ids = group["source_id"][:]
        fit, tune = group["strict_fit"][:].astype(bool), group["tune"][:].astype(bool)
        refs = (
            f["external_reference/source_id"][:]
            if "external_reference" in f
            else f["ALL_reference_holdout_source_ids"][:]
        )
        benchmark = group["benchmark_partition"][:]
        anchor = np.isin(ids, anchor_ids)
        reference = np.isin(ids, refs)
        leakage = int(((fit | tune) & (reference | (benchmark == 1))).sum())
        return {
            "status": "existing_exclusions_passed_not_new_teacher_ready"
            if leakage == 0
            else "blocked_existing_reference_leakage",
            "path": str(path),
            "selected_rows": len(ids),
            "selected_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "strict_fit": int(fit.sum()),
            "tune": int(tune.sum()),
            "benchmark_train": int((benchmark == 0).sum()),
            "benchmark_validation": int((benchmark == 1).sum()),
            "apokasc_overlap": int(anchor.sum()),
            "apokasc_strict_fit_overlap": int((anchor & fit).sum()),
            "apokasc_tune_overlap": int((anchor & tune).sum()),
            "existing_reference_fit_tune_leaks": leakage,
            "fit_tune_overlap": int((fit & tune).sum()),
            "new_teacher_holdout_redesign_required": bool(
                (anchor & (fit | tune)).any()
            ),
        }


def audit(base):
    base = Path(base)
    ap = base / "age_teacher_anchors_v2/apokasc3.parquet"
    wi = base / "age_teacher_anchors_v2/willett.parquet"
    age = {
        "status": "blocked_not_integrated",
        "apokasc": {"status": "blocked_missing_anchor"},
        "willett": {"status": "blocked_missing_anchor"},
    }
    anchor_ids = np.array([], dtype="int64")
    if ap.exists():
        frame = pd.read_parquet(ap)
        age["apokasc"] = audit_bounds(frame)
        age["apokasc"]["parquet_sha256"] = hashlib.sha256(ap.read_bytes()).hexdigest()
        anchor_ids = frame["source_id"].dropna().to_numpy(dtype="int64")
    if wi.exists():
        frame = pd.read_parquet(wi)
        counts = (
            frame.groupby("source_file")["gaia_id_raw"].nunique(dropna=False).to_dict()
        )
        age["willett"] = {
            "status": "blocked_identity_crossmatch_required",
            "rows": len(frame),
            "unique_raw_gaia_ids_by_file": {str(k): int(v) for k, v in counts.items()},
            "files_with_constant_id": [str(k) for k, v in counts.items() if v == 1],
            "released_gaia_ids_usable": False,
            "parquet_sha256": hashlib.sha256(wi.read_bytes()).hexdigest(),
        }
    age["cohort"] = audit_cohort(
        base / "age_feature_cohort_v1/age-feature-cohort.h5", anchor_ids
    )
    # Known historical Hipparcos geometry is not evidence for a native Gaia
    # signed-parallax likelihood or for its correction/uncertainty provenance.
    distance = {
        "status": "blocked_native_parallax_provenance_and_runner_integration",
        "cohort": audit_cohort(
            base / "distance-feature-production-local-v2/distance-feature-cohort.h5", []
        ),
        "missing_evidence": [
            "raw_signed_gaia_parallax_mas_with_source_ids",
            "raw_gaia_parallax_error_mas",
            "zero_point_recipe_version_units_and_row_binding",
            "zero_point_uncertainty_recipe",
            "quadrature_convergence",
            "heldout_anchor_evaluation",
        ],
        "existing_hipparcos_reference_is_not_native_gaia_training_provenance": True,
    }
    alpha = {
        "status": "blocked_interface_only",
        "missing_evidence_gates": list(ALPHA_GATES),
        "gate_evidence_ingested": False,
    }
    return {
        "schema": "msa-target-integration-readiness-v1",
        "status": "blocked_not_ready_for_production_training",
        "scope": "bounded_local_artifacts_only_no_acquisition_no_training_no_integration_claim",
        "age": age,
        "alpha": alpha,
        "distance": distance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("experiments/xp/results-stellar-improvement")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.base)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "readiness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"status": result["status"], "report": str(args.output / "readiness.json")}
        )
    )


if __name__ == "__main__":
    main()
