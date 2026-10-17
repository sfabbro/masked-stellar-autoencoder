from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/xp/results-stellar-improvement"
COHORT_DIR = RESULTS / "enriched_feature_cohort_v1"
PRISTINE_DIR = RESULTS / "pristine_full_fetch_v1_r2"
OUTPUT_DIR = RESULTS / "pristine_enriched_feature_overlap_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cohort_path = COHORT_DIR / "enriched-feature-cohort.h5"
cohort_manifest_path = COHORT_DIR / "manifest.json"
pristine_path = PRISTINE_DIR / "pristine-dr1-full.fits"
pristine_manifest_path = PRISTINE_DIR / "manifest.json"
cohort_manifest = json.loads(cohort_manifest_path.read_text())
pristine_manifest = json.loads(pristine_manifest_path.read_text())
assert cohort_manifest["status"] == "complete"
assert pristine_manifest["status"] == "complete"

with h5py.File(cohort_path, "r") as h5:
    feature_ids = h5["features/source_id"][:].astype(np.int64, copy=False)
    feature_available = h5["features/feature_available"][:].astype(bool, copy=False)
    canonical = h5["canonical"]
    canonical_feature_row = canonical["feature_row"][:].astype(np.int64, copy=False)
    canonical_source_id = canonical["source_id"][:].astype(np.int64, copy=False)
    canonical_layer = canonical["layer"][:].astype(np.int8, copy=False)
    canonical_layer_row = canonical["layer_row"][:].astype(np.int64, copy=False)
    masks = {
        name: canonical[name][:].astype(bool, copy=False)
        for name in ("strict_fit", "broader_fit", "identity_test")
    }
    layer_groups = {
        name: {
            field: group[field][:]
            for field in ("feature_row", "strict_fit", "broader_fit", "identity_test")
        }
        for name, group in h5["layer_index"].items()
    }

assert feature_ids.ndim == 1 and len(np.unique(feature_ids)) == len(feature_ids)
assert np.all((canonical_feature_row >= 0) & (canonical_feature_row < len(feature_ids)))
assert len(np.unique(canonical_feature_row)) == len(canonical_feature_row)
assert np.array_equal(canonical_source_id, feature_ids[canonical_feature_row])
assert np.all(feature_available[canonical_feature_row])

with fits.open(pristine_path, memmap=True) as hdus:
    pristine = hdus[1].data
    pristine_ids = np.asarray(pristine["GaiaDR3"], dtype=np.int64)
    assert len(np.unique(pristine_ids)) == len(pristine_ids)
    sort_order = np.argsort(pristine_ids)
    sorted_pristine_ids = pristine_ids[sort_order]

    def match(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(ids, dtype=np.int64)
        positions = np.searchsorted(sorted_pristine_ids, ids)
        safe_positions = np.minimum(positions, len(sorted_pristine_ids) - 1)
        found = (positions < len(sorted_pristine_ids)) & (
            sorted_pristine_ids[safe_positions] == ids
        )
        return found, sort_order[safe_positions]

    all_found, _ = match(feature_ids)
    # Independently check the sorted-search association using set intersection.
    independent_intersection = np.intersect1d(
        feature_ids, pristine_ids, assume_unique=True
    )
    assert int(all_found.sum()) == len(independent_intersection)
    assert np.array_equal(np.sort(feature_ids[all_found]), independent_intersection)

    def measurement_summary(
        found: np.ndarray, raw_rows: np.ndarray
    ) -> dict[str, object]:
        cahk = np.asarray(pristine["CaHKmag0"][raw_rows], dtype=np.float64)
        error = np.asarray(pristine["e_CaHKmag0"][raw_rows], dtype=np.float64)
        valid = found & np.isfinite(cahk) & np.isfinite(error) & (error > 0)
        precise = valid & (error <= 0.1)
        result: dict[str, object] = {
            "id_overlap": int(found.sum()),
            "finite_cahk_and_positive_reported_error": int(valid.sum()),
            "reported_error_le_0p1_mag": int(precise.sum()),
        }
        if np.any(valid):
            result["reported_error_mag_quantiles"] = {
                key: float(value)
                for key, value in zip(
                    ("p50", "p90", "p99"),
                    np.quantile(error[valid], (0.5, 0.9, 0.99)),
                    strict=True,
                )
            }
            result["raw_flag_counts_in_overlap"] = {
                str(int(flag)): int(count)
                for flag, count in zip(
                    *np.unique(pristine["Flag"][raw_rows[valid]], return_counts=True),
                    strict=True,
                )
            }
            result["pvar_finite"] = int(
                np.isfinite(pristine["Pvar"][raw_rows[valid]]).sum()
            )
            result["cstar_finite"] = int(
                np.isfinite(pristine["Cstar"][raw_rows[valid]]).sum()
            )
        return result

    canonical_ids = canonical_source_id
    canonical_found, canonical_raw_rows = match(canonical_ids)
    canonical_counts: dict[str, dict[str, object]] = {
        "all_canonical_feature_ids": measurement_summary(
            canonical_found, canonical_raw_rows
        )
    }

    canonical_counts["all_canonical_feature_ids"]["n"] = len(canonical_ids)
    for mask_name, mask in masks.items():
        canonical_counts[mask_name] = measurement_summary(
            canonical_found & mask, canonical_raw_rows
        )
        canonical_counts[mask_name]["n"] = int(mask.sum())

    # Resolve numeric source codes from the actual source-row maps and validate
    # every canonical row's source-row association before reporting by source.
    source_codes: dict[str, int] = {}
    for code in np.unique(canonical_layer):
        selected = canonical_layer == code
        candidates = []
        for source_name, arrays in layer_groups.items():
            source_rows = np.asarray(arrays["feature_row"], dtype=np.int64)
            local_rows = canonical_layer_row[selected]
            if np.all(local_rows < len(source_rows)) and np.array_equal(
                source_rows[local_rows], canonical_feature_row[selected]
            ):
                candidates.append(source_name)
        assert len(candidates) == 1, (int(code), candidates)
        source_codes[candidates[0]] = int(code)

    canonical_counts["by_canonical_source"] = {}
    for source_name, code in sorted(source_codes.items()):
        selected = canonical_layer == code
        source_summary = measurement_summary(
            canonical_found & selected, canonical_raw_rows
        )
        source_summary["n"] = int(selected.sum())
        source_summary["canonical_strict_fit"] = int(
            np.count_nonzero(canonical_found & selected & masks["strict_fit"])
        )
        source_summary["strict_fit_with_cahk_error_le_0p1"] = int(
            np.count_nonzero(
                canonical_found
                & selected
                & masks["strict_fit"]
                & np.isfinite(pristine["e_CaHKmag0"][canonical_raw_rows])
                & (pristine["e_CaHKmag0"][canonical_raw_rows] > 0)
                & (pristine["e_CaHKmag0"][canonical_raw_rows] <= 0.1)
            )
        )
        source_summary["identity_test_with_cahk_error_le_0p1"] = int(
            np.count_nonzero(
                canonical_found
                & selected
                & masks["identity_test"]
                & np.isfinite(pristine["e_CaHKmag0"][canonical_raw_rows])
                & (pristine["e_CaHKmag0"][canonical_raw_rows] > 0)
                & (pristine["e_CaHKmag0"][canonical_raw_rows] <= 0.1)
            )
        )
        canonical_counts["by_canonical_source"][source_name] = source_summary

    report = {
        "audit": "exact Gaia DR3 source_id intersection; coverage only",
        "source_artifacts": {
            "enriched_feature_cohort": {
                "run_id": cohort_manifest["run_id"],
                "sha256": cohort_manifest["result"]["sha256"],
                "content_id": cohort_manifest["result"]["content_id"],
                "bytes": cohort_manifest["result"]["bytes"],
                "canonical_unique_feature_ids": cohort_manifest["result"][
                    "canonical_selected"
                ],
            },
            "pristine_dr1": {
                "run_id": pristine_manifest["run_id"],
                "sha256": pristine_manifest["result"]["sha256"],
                "bytes": pristine_manifest["result"]["bytes"],
                "rows": len(pristine_ids),
                "source_id_sha256": pristine_manifest["result"]["audit"][
                    "source_id_sha256"
                ],
                "query": pristine_manifest["result"]["query"],
            },
        },
        "counts": {
            "feature_registry_rows": len(feature_ids),
            "feature_registry_unique_ids": len(np.unique(feature_ids)),
            "feature_available_ids": int(feature_available.sum()),
            "feature_unavailable_ids": int((~feature_available).sum()),
            "canonical_rows": len(canonical_ids),
            "pristine_unique_ids": len(pristine_ids),
            "registered_ids_with_pristine": int(all_found.sum()),
            "feature_available_ids_with_pristine": int(
                np.count_nonzero(all_found & feature_available)
            ),
        },
        "canonical_overlap": canonical_counts,
        "interpretation": [
            "This is exact DR3 identifier coverage in the preserved enriched feature cohort, not a positional crossmatch.",
            "Pristine CaHK columns are not part of the current six-survey enriched feature input; this audit measures candidate integration coverage.",
            "No Pristine quality, morphology, variability, extinction, or photometric-metallicity eligibility cut is asserted here.",
            "The reported 0.1 mag error threshold is descriptive availability only; Pristine photometric FeH is not spectroscopic truth.",
            "Benchmark validation remains held out, and correlated survey layers are not treated as independent stars.",
            "Coverage counts do not establish downstream accuracy benefit.",
        ],
    }
    out_json = OUTPUT_DIR / "coverage-audit.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    all_count = canonical_counts["all_canonical_feature_ids"]
    test_count = canonical_counts["identity_test"]
    strict_count = canonical_counts["strict_fit"]
    broad_count = canonical_counts["broader_fit"]
    lines = [
        "# Pristine overlap with the enriched XP feature cohort",
        "",
        "This audit uses exact Gaia DR3 identifier intersection between the preserved Pristine DR1 FITS and enriched feature HDF5. It reports data availability; it does not define a Pristine quality cut or measure model performance.",
        "",
        f"- Canonical complete-XP IDs: {len(canonical_ids):,}; Pristine matches: {all_count['id_overlap']:,}.",
        f"- Native/PIGS/Li identity-test IDs: {test_count['n']:,}; Pristine matches: {test_count['id_overlap']:,}.",
        f"- Strict-fit IDs: {strict_count['n']:,}; Pristine matches: {strict_count['id_overlap']:,}.",
        f"- Broader-fit IDs: {broad_count['n']:,}; Pristine matches: {broad_count['id_overlap']:,}.",
        "",
        "All matched records have a finite CaHK magnitude with a positive reported error; the report gives the measured error distribution and raw flags without interpreting flag semantics. Pristine CaHK is not included in the current six-survey input, so this is candidate integration coverage only.",
        "",
        "No inferred Pristine FeH value is counted as spectroscopic truth. Photometric FeH and any quality, variability, morphology, or extinction eligibility need separate review before an import.",
        "",
        f"Machine-readable counts and source artifact hashes: {out_json.name}.",
    ]
    out_md = OUTPUT_DIR / "coverage-audit.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "report_json": str(out_json),
                "report_markdown": str(out_md),
                "feature_available_ids_with_pristine": report["counts"][
                    "feature_available_ids_with_pristine"
                ],
                "canonical": {
                    "all": all_count,
                    "identity_test": test_count,
                    "strict_fit": strict_count,
                    "broader_fit": broad_count,
                    "by_source": canonical_counts["by_canonical_source"],
                },
            },
            indent=2,
        )
    )
