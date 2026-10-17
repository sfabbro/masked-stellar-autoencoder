"""Audit existing native joint-input readiness; no acquisition or model fitting."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/xp/results-stellar-improvement"
cohort = BASE / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
manifest = json.loads((cohort.parent / "manifest.json").read_text())
with cohort.open("rb") as stream:
    cohort_sha = hashlib.file_digest(stream, "sha256").hexdigest()
assert cohort_sha == manifest["result"]["sha256"]
with h5py.File(BASE / "distance_feature_cohort_v1/distance-feature-cohort.h5") as f:
    external_ids = f["ALL_reference_holdout_source_ids"][:]
assert len(external_ids) == len(np.unique(external_ids)) == 99574
with h5py.File(cohort) as f:
    columns = json.loads(f.attrs["feature_columns_json"])
    errors = json.loads(f.attrs["error_columns_json"])
    j = columns.index("PARALLAX")
    assert errors[j] == "e_parallax"
    ids = f["canonical/source_id"][:]
    rows = f["canonical/feature_row"][:]
    native = f["canonical/layer"][:] == 0
    assert np.array_equal(ids, f["features/source_id"][:][rows])
    assert len(np.unique(ids)) == len(ids)
    raw = f["features/values"][:, j][rows]
    uncertainty = f["features/reported_feature_errors"][:, j][rows]
    finite = np.isfinite(raw)
    error_valid = np.isfinite(uncertainty) & (uncertainty > 0)
    assert np.array_equal(finite, f["features/feature_observed"][:, j][rows])
    assert np.array_equal(error_valid, f["features/error_observed"][:, j][rows])
    mathematical_valid = finite & error_valid
    partition = f["features/benchmark_partition"][:][rows]
    split = f["features/label_split"][:][rows]
    heldout_reference = np.isin(ids, external_ids)
    fit = f["canonical/strict_fit"][:] & native & ~heldout_reference
    broad = f["canonical/broader_fit"][:] & native & ~heldout_reference
    tune = native & (split == 1) & (partition != 2) & ~heldout_reference
    test = f["canonical/identity_test"][:] & native
    assert not np.any((fit | broad | tune) & ((partition == 2) | heldout_reference))
    assert not np.any((fit | broad | tune) & test)
    measurements = f["raw_layers/native/measurements"]
    native_fields = list(measurements.dtype.names)
    assert np.isin(ids[native], measurements["source_id"]).all()
    assert not {"teff", "logg"} & set(native_fields)
    counts = {}
    for name, mask in {
        "native_available": native,
        "native_strict_candidate": fit,
        "native_broad_candidate": broad,
        "native_tune_candidate": tune,
        "native_identity_test": test,
    }.items():
        counts[name] = {
            "ids": int(mask.sum()),
            "finite_raw_parallax": int((mask & finite).sum()),
            "finite_raw_parallax_and_positive_formal_error": int(
                (mask & mathematical_valid).sum()
            ),
            "nonpositive_parallax_with_valid_formal_error": int(
                (mask & mathematical_valid & (raw <= 0)).sum()
            ),
        }
report = {
    "status": "readiness_audit_complete_joint_training_not_ready",
    "cohort_sha256": cohort_sha,
    "native_measurement_fields": native_fields,
    "native_Teff_logg_values_errors_and_target_specific_flags_materialized": False,
    "counts": counts,
    "reference_holdout_ids": len(external_ids),
    "fit_tune_benchmark_validation_reference_and_identity_test_bans_checked": True,
    "new_surveys_acquired": False,
    "training_or_inference_performed": False,
    "limitations": [
        "Raw parallax mathematical validity does not certify Gaia total uncertainty, zero-point treatment, astrometric quality or unsupervised independence.",
        "Candidate counts retain nonpositive parallaxes; no model/teacher distance or sign/SNR cut is used.",
        "Existing native measurement sidecar preserves FeH only. Teff/logg require a separate audited projection from already acquired original native catalogues, including source rows/errors/quality/units; legacy imputed rows are not replacements.",
        "Preserve FeH as native iron abundance, distinct from APOGEE M_H; no cross-survey averaging.",
        "No publishable joint cohort, head budgets, supervised normalization, full parent integration or production launch gate is satisfied by this audit.",
    ],
}
out = BASE / "joint-continuation-preparation-v1/native-joint-input-readiness.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "output": str(out),
            "counts": counts,
            "joint_native_targets_materialized": False,
        }
    )
)
