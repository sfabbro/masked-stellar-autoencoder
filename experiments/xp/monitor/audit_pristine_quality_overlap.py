from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/xp/results-stellar-improvement"
PRISTINE_DIR = RESULTS / "pristine_full_fetch_v1_r2"
COHORT_DIR = RESULTS / "enriched_feature_cohort_v1"
QUALITY_DIR = RESULTS / "pristine_quality_view_v1"
OUTPUT_DIR = RESULTS / "pristine_quality_overlap_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


pristine_manifest = json.loads((PRISTINE_DIR / "manifest.json").read_text())
cohort_manifest = json.loads((COHORT_DIR / "manifest.json").read_text())
quality_manifest = json.loads((QUALITY_DIR / "manifest.json").read_text())
assert pristine_manifest["status"] == "complete"
assert cohort_manifest["status"] == "complete"
assert quality_manifest["status"] == "complete"

pristine_path = PRISTINE_DIR / "pristine-dr1-full.fits"
cohort_path = COHORT_DIR / "enriched-feature-cohort.h5"
quality_path = QUALITY_DIR / "pristine-quality-view.h5"
pristine_sha = sha256_file(pristine_path)
cohort_sha = sha256_file(cohort_path)
assert pristine_sha == pristine_manifest["result"]["sha256"]
assert cohort_sha == cohort_manifest["result"]["sha256"]

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
    layer_index = h5["layer_index"]
    layer_names: dict[int, str] = {}
    for code in np.unique(canonical_layer):
        selected = canonical_layer == code
        local_rows = canonical_layer_row[selected]
        candidates = []
        for name, group in layer_index.items():
            source_rows = group["feature_row"][:].astype(np.int64, copy=False)
            if np.all(
                (local_rows >= 0) & (local_rows < len(source_rows))
            ) and np.array_equal(
                source_rows[local_rows], canonical_feature_row[selected]
            ):
                candidates.append(name)
        assert len(candidates) == 1, (int(code), candidates)
        layer_names[int(code)] = candidates[0]

assert len(np.unique(feature_ids)) == len(feature_ids)
assert np.array_equal(canonical_source_id, feature_ids[canonical_feature_row])
assert np.all(feature_available[canonical_feature_row])

with fits.open(pristine_path, memmap=True) as hdus:
    pristine = hdus[1].data
    pristine_ids = np.asarray(pristine["GaiaDR3"], dtype=np.int64)
    assert len(np.unique(pristine_ids)) == len(pristine_ids)
    order = np.argsort(pristine_ids)
    sorted_ids = pristine_ids[order]
    positions = np.searchsorted(sorted_ids, canonical_source_id)
    safe_positions = np.minimum(positions, len(sorted_ids) - 1)
    matched = (positions < len(sorted_ids)) & (
        sorted_ids[safe_positions] == canonical_source_id
    )
    pristine_rows = order[safe_positions]

    flag = np.asarray(pristine["Flag"][pristine_rows], dtype=np.int16)
    # Keep publisher dtypes for cut comparisons. Promoting the float32 fields
    # first changes boundary behavior for values stored as nominally 1.4, 0.1,
    # and similar survey thresholds.
    cahk_error = np.asarray(pristine["e_CaHKmag0"][pristine_rows])
    pvar = np.asarray(pristine["Pvar"][pristine_rows])
    ruwe = np.asarray(pristine["RUWE"][pristine_rows])
    cstar = np.asarray(pristine["Cstar"][pristine_rows])
    s_cstar = np.asarray(pristine["s_Cstar"][pristine_rows])
    ebv = np.asarray(pristine["E(B-V)"][pristine_rows])

    # Reproduce the project's conservative direct-input screen from raw FITS
    # fields. No photometric FeH or MC grid-membership predicate enters this mask.
    quality_eligible = (
        matched
        & np.isin(flag, (-1, -2))
        & np.isfinite(cahk_error)
        & (cahk_error > 0)
        & (cahk_error <= 0.1)
        & np.isfinite(pvar)
        & (pvar >= 0)
        & (pvar < 0.3)
        & np.isfinite(ruwe)
        & (ruwe > 0)
        & (ruwe < 1.4)
        & np.isfinite(cstar)
        & np.isfinite(s_cstar)
        & (s_cstar > 0)
        & (np.abs(cstar) < s_cstar)
        & np.isfinite(ebv)
        & (ebv >= 0)
        & (ebv < 0.3)
    )

    # Exact-ID comparison with the independently preserved quality-view mask.
    with h5py.File(quality_path, "r") as quality_h5:
        quality_ids = quality_h5["pristine/source_id"][:].astype(np.int64, copy=False)
        quality_mask = quality_h5["pristine/input_eligible"][:].astype(bool, copy=False)
        quality_order = np.argsort(quality_ids)
        sorted_quality_ids = quality_ids[quality_order]
        quality_positions = np.searchsorted(sorted_quality_ids, canonical_source_id)
        safe_quality_positions = np.minimum(
            quality_positions, len(sorted_quality_ids) - 1
        )
        quality_matched = (quality_positions < len(sorted_quality_ids)) & (
            sorted_quality_ids[safe_quality_positions] == canonical_source_id
        )
        quality_by_id = np.zeros(len(canonical_source_id), dtype=bool)
        quality_by_id[quality_matched] = quality_mask[
            quality_order[safe_quality_positions[quality_matched]]
        ]
    assert np.array_equal(quality_matched, matched)
    assert np.array_equal(quality_by_id, quality_eligible)

    counts: dict[str, object] = {
        "all_canonical": {
            "n": len(canonical_source_id),
            "exact_id_overlap": int(matched.sum()),
            "quality_eligible_overlap": int(quality_eligible.sum()),
        }
    }
    for name, mask in masks.items():
        counts[name] = {
            "n": int(mask.sum()),
            "exact_id_overlap": int(np.count_nonzero(mask & matched)),
            "quality_eligible_overlap": int(np.count_nonzero(mask & quality_eligible)),
            "quality_eligible_fraction": float(
                np.count_nonzero(mask & quality_eligible) / mask.sum()
            ),
        }

    by_layer: dict[str, object] = {}
    for code, source_name in layer_names.items():
        selected = canonical_layer == code
        row: dict[str, object] = {
            "canonical_rows": int(selected.sum()),
            "exact_id_overlap": int(np.count_nonzero(selected & matched)),
            "quality_eligible_overlap": int(
                np.count_nonzero(selected & quality_eligible)
            ),
        }
        for name, mask in masks.items():
            row[name] = {
                "n": int(np.count_nonzero(selected & mask)),
                "quality_eligible": int(
                    np.count_nonzero(selected & mask & quality_eligible)
                ),
            }
        by_layer[source_name] = row

    report = {
        "audit": "exact Gaia DR3 source_id intersection and raw-Pristine direct-input QC coverage",
        "source_artifacts": {
            "pristine_dr1": {
                "run_id": pristine_manifest["run_id"],
                "sha256": pristine_sha,
                "bytes": pristine_path.stat().st_size,
                "rows": len(pristine_ids),
                "source_id_sha256": pristine_manifest["result"]["audit"][
                    "source_id_sha256"
                ],
            },
            "enriched_feature_cohort": {
                "run_id": cohort_manifest["run_id"],
                "sha256": cohort_sha,
                "content_id": cohort_manifest["result"]["content_id"],
                "bytes": cohort_path.stat().st_size,
                "canonical_unique_feature_ids": len(canonical_source_id),
            },
            "project_quality_view": {
                "run_id": quality_manifest["run_id"],
                "sha256": quality_manifest["result"]["sha256"],
                "mask_exactly_reproduced_for_all_canonical_ids": True,
            },
        },
        "quality_profile": {
            "Flag": [-1, -2],
            "CaHK_error_mag": "finite, 0 < error <= 0.1",
            "Pvar": "finite, 0 <= Pvar < 0.3",
            "RUWE": "finite, 0 < RUWE < 1.4",
            "Cstar": "finite, s_Cstar > 0, abs(Cstar) < s_Cstar",
            "E(B-V)": "finite, 0 <= E(B-V) < 0.3",
            "photometric_FeH_or_MC_grid_cut": "not applied",
        },
        "counts": counts,
        "by_canonical_source": by_layer,
        "interpretation": [
            "Counts are exact source_id coverage in the preserved artifacts; no positional match is used.",
            "Quality eligibility reproduces the project's conservative direct-photometry input screen, not every cut used for the paper's photometric-metallicity catalogues.",
            "No Pristine photometric FeH or MC grid-membership value is used as an input-eligibility predicate or spectroscopic target.",
            "Benchmark validation remains excluded from fit masks. Layer counts are source-separated and are not independent-star counts when IDs recur across layers.",
            "Coverage indicates possible Pristine CaHK input availability only; it does not demonstrate accuracy or benefit from adding the feature.",
            "The paper's section 6.3 and section 7.3 describe/recommend MC fraction within the model grid and >0.8, while its Table 3 field description says outside the model color-color space; the ambiguity is retained and this audit avoids that predicate.",
        ],
    }
    out_json = OUTPUT_DIR / "coverage-audit.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Pristine quality-eligible coverage of the enriched XP cohort",
        "",
        "This exact Gaia DR3 source_id audit applies the project's conservative direct-photometry input screen to the preserved full Pristine DR1 FITS. It measures candidate CaHK availability, not model performance.",
        "",
        f"- Exact Pristine matches among 673,345 canonical complete-XP IDs: {int(matched.sum()):,}.",
        f"- Matches passing the direct-input screen: {int(quality_eligible.sum()):,} ({quality_eligible.sum() / matched.sum():.1%} of exact matches).",
        f"- Strict-fit IDs with eligible Pristine: {counts['strict_fit']['quality_eligible_overlap']:,} / {counts['strict_fit']['n']:,}.",
        f"- Identity-test IDs with eligible Pristine: {counts['identity_test']['quality_eligible_overlap']:,} / {counts['identity_test']['n']:,}.",
        "",
        "| Canonical source layer | Strict fit: eligible / rows | Broader fit: eligible / rows | Identity test: eligible / rows |",
        "|---|---:|---:|---:|",
    ]
    for name, row in by_layer.items():
        lines.append(
            f"| {name} | {row['strict_fit']['quality_eligible']:,} / {row['strict_fit']['n']:,} | "
            f"{row['broader_fit']['quality_eligible']:,} / {row['broader_fit']['n']:,} | "
            f"{row['identity_test']['quality_eligible']:,} / {row['identity_test']['n']:,} |"
        )
    lines.extend(
        [
            "",
            "The direct-input screen is: Flag −1/−2; finite positive CaHK uncertainty ≤0.1 mag; 0≤Pvar<0.3; 0<RUWE<1.4; |Cstar|<s_Cstar; and 0≤E(B−V)<0.3. The computed mask exactly matches the preserved project Pristine quality view for every canonical ID.",
            "",
            "No photometric FeH or MC fraction cut is applied. This avoids treating a photometric metallicity estimate as the spectroscopic target and avoids relying on a published MC-fraction description that conflicts between Table 3 and Sections 6.3/7.3. The paper also warns that grid-edge cuts can remove genuine ultra-metal-poor candidates.",
            "",
            "These are source-separated candidate input counts; recurrent IDs across layers are not independent stars, and availability alone does not show that adding CaHK improves predictions.",
            "",
            f"Source hashes and all counts: {out_json.name}.",
        ]
    )
    out_md = OUTPUT_DIR / "coverage-audit.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "report_json": str(out_json),
                "report_markdown": str(out_md),
                "counts": counts,
                "by_source": by_layer,
            },
            indent=2,
        )
    )
