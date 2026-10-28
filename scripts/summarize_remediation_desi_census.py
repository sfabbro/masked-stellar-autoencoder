"""Recompute source-only DESI counts and conservative non-overlap bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_MANIFEST_SHA = "9462c283cbe5e855d6db7bd142db03978a483c5493710065f38e8d8c15be5cdf"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(census_dir, base_manifest, output):
    census_dir, base_manifest, output = map(Path, (census_dir, base_manifest, output))
    receipt = json.loads((census_dir / "receipts.json").read_text())
    summary = json.loads((census_dir / "census/summary.json").read_text())
    identity = census_dir / "census/identities.npz"
    if (
        not receipt["producer_sha_verified"]
        or sha(identity) != receipt["identity_sha256"]
        or sha(identity) != summary["identities_sha256"]
    ):
        raise ValueError("Census identity producer SHA differs")
    if sha(base_manifest) != BASE_MANIFEST_SHA:
        raise ValueError("Preserved base manifest SHA differs")
    base = json.loads(base_manifest.read_text())
    if base["prepared_sha256"] != summary["base_sha256"]:
        raise ValueError("Census and base source hashes differ")
    with np.load(identity, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    ids = a["source_id"]
    if ids.dtype.kind not in "iu" or np.any(np.diff(ids) <= 0):
        raise ValueError("Census identities must be sorted exact unique integers")
    if (
        len(ids) != summary["unique_desi_ids"]
        or int(a["quality_rows"].sum()) != summary["quality_rows"]
    ):
        raise ValueError("Census row counts differ")
    np.testing.assert_array_equal(
        a["observed"].sum(axis=0), summary["observed_per_target_distinct_ids"]
    )
    if (
        int(a["base_match"].sum()) != summary["base_matched_ids"]
        or int(a["cache_complete_xp"].sum()) != summary["cache_complete_xp_ids"]
    ):
        raise ValueError("Census membership counts differ")
    results = {}
    for role in (
        "fit",
        "tune",
        "cal",
        "devtest",
        "excluded",
        "external",
        "not_in_base",
    ):
        select = a["base_role"] == role
        results[role] = {
            "stars": int(select.sum()),
            "primary_input": int((select & a["base_primary_input"]).sum()),
            "cached_complete_xp": int((select & a["cache_complete_xp"]).sum()),
            "multiple_desi_rows": int((select & (a["quality_rows"] > 1)).sum()),
            "known_supervised_ancestor": int((select & a["base_known_ancestor"]).sum()),
        }
        if role == "not_in_base":
            # The base has no source information for these stars; false masks
            # are placeholders, not evidence of absent ancestry or inputs.
            results[role]["primary_input"] = None
            results[role]["known_supervised_ancestor"] = None
        if results[role]["stars"] != summary["base_role_counts"].get(role, 0):
            raise ValueError("Role count differs")
    fit = (a["base_role"] == "fit") & a["base_primary_input"] & a["observed"][:, 1]
    single = fit & (a["quality_rows"] == 1)
    cached = single & a["cache_complete_xp"]
    existing = int(base["role_counts"]["fit"]["targets"]["logg"])
    bounds = {
        "prior_fit_logg_stars": existing,
        "desi_fit_logg_candidates": int(fit.sum()),
        "desi_fit_single_row_logg_candidates": int(single.sum()),
        "desi_fit_single_row_cached_xp_candidates": int(cached.sum()),
        "minimum_additional_single_row_candidates_from_cardinality": max(
            0, int(single.sum()) - existing
        ),
        "minimum_additional_single_row_cached_xp_from_cardinality": max(
            0, int(cached.sum()) - existing
        ),
    }
    result = {
        "source_sha256": summary["sidecar_sha256"],
        "base_sha256": summary["base_sha256"],
        "identity_sha256": receipt["identity_sha256"],
        "quality_rows": summary["quality_rows"],
        "unique_desi_ids": len(ids),
        "roles": results,
        "logg_cardinality_bounds": bounds,
        "scope": "Bounds are inferred from set sizes, not a direct prior-label overlap join; DESI values, uncertainties, cross-survey dependencies and scale are not yet certified as fit-ready. Nonmatches do not imply missing XP or absent features elsewhere.",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("census_dir", type=Path)
    p.add_argument("base_manifest", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            summarize(args.census_dir, args.base_manifest, args.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
