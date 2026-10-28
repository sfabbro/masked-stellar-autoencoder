"""Audit actual Pristine/common-survey availability without fitting any model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/xp/results-stellar-improvement"
GROUPS = {
    "WISE": ["W1", "W2"],
    "Gaia": ["G", "BP", "RP"],
    "SMSS": ["U_SMSS", "V_SMSS", "G_SMSS", "R_SMSS", "I_SMSS", "Z_SMSS"],
    "SDSS": ["U_SDSS", "G_SDSS", "R_SDSS", "I_SDSS", "Z_SDSS"],
    "PS1": ["G_PS1", "R_PS1", "I_PS1", "Z_PS1", "Y_PS1"],
    "2MASS": ["J", "H", "KS"],
}
cohort = BASE / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
quality = BASE / "pristine_quality_view_v1/pristine-quality-view.h5"
prior = json.loads(
    (BASE / "pristine_quality_overlap_v1/coverage-audit.json").read_text()
)
with cohort.open("rb") as stream:
    cohort_sha = hashlib.file_digest(stream, "sha256").hexdigest()
assert cohort_sha == prior["source_artifacts"]["enriched_feature_cohort"]["sha256"]
with quality.open("rb") as stream:
    quality_sha = hashlib.file_digest(stream, "sha256").hexdigest()
assert quality_sha == prior["source_artifacts"]["project_quality_view"]["sha256"]
with h5py.File(quality) as f:
    pristine_ids = f["pristine/source_id"][:]
    pristine_eligible = f["pristine/input_eligible"][:]
with h5py.File(cohort) as f:
    ids = f["canonical/source_id"][:]
    rows = f["canonical/feature_row"][:]
    masks = {
        name: f["canonical/" + name][:]
        for name in ["strict_fit", "broader_fit", "identity_test"]
    }
    all_ids = f["features/source_id"][:]
    assert np.array_equal(all_ids[rows], ids)
    columns = json.loads(f.attrs["feature_columns_json"])
    error_columns = json.loads(f.attrs["error_columns_json"])
    group_masks = {name: np.zeros(len(all_ids), bool) for name in GROUPS}
    error_masks = {name: np.zeros(len(all_ids), bool) for name in GROUPS}
    for start in range(0, len(all_ids), 8192):
        stop = min(start + 8192, len(all_ids))
        values = f["features/values"][start:stop]
        errors = f["features/reported_feature_errors"][start:stop]
        observed = np.isfinite(values)
        eobserved = np.isfinite(errors) & (errors > 0)
        assert np.array_equal(observed, f["features/feature_observed"][start:stop])
        assert np.array_equal(eobserved, f["features/error_observed"][start:stop])
        for name, bands in GROUPS.items():
            indices = [columns.index(band) for band in bands]
            group_masks[name][start:stop] = observed[:, indices].all(1)
            error_masks[name][start:stop] = (
                observed[:, indices] & eobserved[:, indices]
            ).all(1)
    group_masks = {name: mask[rows] for name, mask in group_masks.items()}
    error_masks = {name: mask[rows] for name, mask in error_masks.items()}
assert len(np.unique(ids)) == len(ids)
common_ids, canonical_rows, pristine_rows = np.intersect1d(
    ids, pristine_ids, assume_unique=True, return_indices=True
)
matched = np.zeros(len(ids), bool)
matched[canonical_rows] = True
eligible = np.zeros(len(ids), bool)
eligible[canonical_rows] = pristine_eligible[pristine_rows]
assert len(common_ids) == prior["counts"]["all_canonical"]["exact_id_overlap"]
assert (
    int(eligible.sum()) == prior["counts"]["all_canonical"]["quality_eligible_overlap"]
)
raw_path = BASE / "pristine_full_fetch_v1_r2/pristine-dr1-full.fits"
with fits.open(raw_path, memmap=True) as hdus:
    raw = hdus[1].data
    assert np.array_equal(np.asarray(raw["GaiaDR3"]), pristine_ids)
    qrows = pristine_rows[pristine_eligible[pristine_rows]]
    cahk = np.asarray(raw["CaHKmag0"][qrows])
    cahk_error = np.asarray(raw["e_CaHKmag0"][qrows])
    assert np.isfinite(cahk).all() and np.isfinite(cahk_error).all()
    assert (cahk_error > 0).all()
    error_quantiles = {
        str(p): float(np.percentile(cahk_error.astype("f8"), p))
        for p in [0, 16, 50, 84, 95, 100]
    }
all_six = np.logical_and.reduce(list(group_masks.values()))
assert int(all_six.sum()) == 33800
cohorts = {"all_canonical": np.ones(len(ids), bool), **masks}
counts = {}
for key, mask in cohorts.items():
    counts[key] = {
        "n": int(mask.sum()),
        "pristine_quality_eligible": int((mask & eligible).sum()),
        "all_six_values": int((mask & all_six).sum()),
        "pristine_quality_and_all_six_values": int((mask & eligible & all_six).sum()),
        "by_group": {
            name: {
                "pristine_quality_and_all_bands_finite": int(
                    (mask & eligible & group_masks[name]).sum()
                ),
                "pristine_quality_and_all_bands_with_reported_positive_errors": int(
                    (mask & eligible & error_masks[name]).sum()
                )
                if all(error_columns[columns.index(b)] is not None for b in bands)
                else None,
                "configured_error_columns": [
                    error_columns[columns.index(b)] for b in bands
                ],
            }
            for name, bands in GROUPS.items()
        },
    }
report = {
    "audit": "Actual Pristine quality eligibility and common photometry footprint",
    "cohort_sha256": cohort_sha,
    "quality_view_sha256": quality_sha,
    "independent_intersection_matches_prior_raw_quality_audit": True,
    "all_756836_raw_feature_and_error_masks_independently_recomputed": True,
    "counts": counts,
    "quality_eligible_CaHK_reported_error_quantiles_mag": error_quantiles,
    "pairwise_quality_eligible_unique_counts": {
        a: {b: int((eligible & group_masks[a] & group_masks[b]).sum()) for b in GROUPS}
        for a in GROUPS
    },
    "limitations": [
        "Counts describe preserved artifact availability; finite magnitudes are not an all-survey quality screen.",
        "Missing configured errors are reported as null, never as zero physical uncertainty or a complete uncertainty certificate.",
        "Pristine CaHKmag0 is the released giant-solution dereddened quantity; no further extinction correction applied.",
        "No new survey acquired, no labels merged, no model trained, and no model-performance or all-sky claim.",
    ],
}
output = BASE / "pristine_quality_overlap_v1/common-footprint.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {
            "output": str(output),
            "counts": {
                k: {a: b for a, b in v.items() if a != "by_group"}
                for k, v in counts.items()
            },
        }
    )
)
