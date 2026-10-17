"""Local readiness only: no original native-target projection or training claim."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from masked_stellar_autoencoder.training.joint_native_labels import assign_roles

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/xp/results-stellar-improvement"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    cohort = BASE / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    distance = BASE / "distance_feature_cohort_v1/distance-feature-cohort.h5"
    with h5py.File(distance) as f:
        refs = f["ALL_reference_holdout_source_ids"][:]
    with h5py.File(cohort) as f:
        ids = f["features/source_id"][:]
        available = f["features/feature_available"][:]
        partition = f["features/benchmark_partition"][:]
        roles = assign_roles(ids, partition, available, refs)
        names = json.loads(f.attrs["feature_columns_json"])
        native = f["raw_layers/native/measurements"][:]
        native_row = f["layer_index/native/feature_row"][:]
        np.testing.assert_array_equal(ids[native_row], native["source_id"])
        native_source = np.full(len(ids), "", dtype="U32")
        native_source[native_row] = native["survey"].astype(str)
        j = names.index("EBV")
        ebv = f["features/values"][:, j]
        finite = np.isfinite(ebv)
        observed = f["features/feature_observed"][:, j]
        error_col = json.loads(f.attrs["error_columns_json"])[j]
    counts = {}
    for name, mask in roles.items():
        if mask.dtype != bool:
            continue
        counts[name] = {
            "all_feature_ids": int(mask.sum()),
            "native_apogee_candidate": int((mask & (native_source == "apogee")).sum()),
            "native_galah_candidate": int((mask & (native_source == "galah")).sum()),
            "existing_li_vmps_not_joint_native": int(
                (mask & (native_source == "li_et_al_vmps")).sum()
            ),
        }
    paths = sorted({p.decode() for p in native["input_file"]})
    q = np.quantile(ebv[finite & observed], [0, 0.5, 0.9, 0.99, 0.999, 1])
    report = {
        "status": "actual_local_readiness_audit_complete_native_projection_pending",
        "cohort_sha256": sha(cohort),
        "reference_source_h5_sha256": sha(distance),
        "reference_union_count": len(refs),
        "reference_union_ordered_id_sha256": hashlib.sha256(
            np.asarray(refs, dtype="<i8").tobytes()
        ).hexdigest(),
        "counts_are_candidate_ids_not_valid_joint_target_counts": True,
        "role_counts": counts,
        "original_raw_files": [
            {"accepted_arc_path": p, "available_at_this_local_path": Path(p).is_file()}
            for p in paths
        ],
        "native_teff_logg_values_materialized_locally": False,
        "materializer_synthetic_fits_hdf5_integration_test_passed": True,
        "alpha_enabled": False,
        "age_enabled": False,
        "gaia_quality_zero_point_materialized": False,
        "distance_fit_eligible": False,
        "galah_logg_parallax_conditioned": True,
        "ebv": {
            "finite_observed_rows": int((finite & observed).sum()),
            "quantiles_0_50_90_99_99p9_100_raw": q.tolist(),
            "raw_unit": "not established by cohort metadata",
            "error_column": error_col,
            "map_provenance": "not established by cohort metadata",
            "maximum_source_id": int(ids[np.nanargmax(ebv)]),
            "safe_for_unqualified_physical_extinction_input": False,
        },
        "training_launch_ready": False,
        "remaining_gates": [
            "Run source-SHA-pinned original native FITS projection on the existing ARC mount before GPU allocation.",
            "Audit GALAH logg/Gaia dependence instead of an independent physical-observation likelihood claim.",
            "Audit astrometric quality and actual five/six-parameter Gaia zero-point inputs; no fabricated error floor or zero point.",
            "Keep EBV source/unit/extinction convention unresolved until actual raw source lineage is checked; no invented physical cut.",
        ],
    }
    output = BASE / "joint-continuation-preparation-v1/native-joint-readiness-v2.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
