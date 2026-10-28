#!/usr/bin/env python3
"""Recover Kepler-only Gaia identities through the existing APOKASC3 KIC map.

No sky matching, raw released Willett Gaia IDs, non-Kepler ID namespaces,
acquisition, age transfer, fitting or inference is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def exact_positive_integer(value):
    """Parse integer identity without a floating-point round trip."""
    if isinstance(value, bool) or pd.isna(value):
        return None
    token = str(value).strip()
    if not re.fullmatch(r"[0-9]+", token):
        return None
    parsed = int(token)
    return parsed if 0 < parsed <= 2**63 - 1 else None


def crossmatch(apokasc, willett):
    mapping = {}
    reverse = {}
    for kic, source_id in zip(apokasc["kic"], apokasc["source_id"], strict=True):
        k, source = exact_positive_integer(kic), exact_positive_integer(source_id)
        if k is None or source is None:
            raise ValueError("APOKASC KIC and Gaia IDs must be exact positive integers")
        if k in mapping and mapping[k] != source:
            raise ValueError("conflicting APOKASC KIC identities")
        if source in reverse and reverse[source] != k:
            raise ValueError("APOKASC Gaia identity has multiple KICs")
        mapping[k], reverse[source] = source, k
    out = willett.copy()
    source_ids, statuses = [], []
    seen_kepler = set()
    for survey, raw_kic in zip(out["survey"], out["kic"], strict=True):
        if survey != "kepler":
            source_ids.append(pd.NA)
            statuses.append("unmatched_non_kepler_namespace")
            continue
        kic = exact_positive_integer(raw_kic)
        if kic is not None:
            if kic in seen_kepler:
                raise ValueError(
                    "duplicate Willett Kepler KIC requires explicit resolution"
                )
            seen_kepler.add(kic)
        source = mapping.get(kic)
        source_ids.append(source if source is not None else pd.NA)
        statuses.append(
            "matched_apokasc3_kic"
            if source is not None
            else "unmatched_missing_or_unknown_kic"
        )
    out["source_id"] = pd.array(source_ids, dtype="Int64")
    out["identity_status"] = statuses
    out["gaia_id_usable"] = out["source_id"].notna()
    out["identity_provenance"] = (
        "APOKASC3 GaiaDR3 via exact shared Kepler KIC; Willett gaia_id_raw ignored"
    )
    return out


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchors",
        type=Path,
        default=Path(
            "experiments/xp/results-stellar-improvement/age_teacher_anchors_v2"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ap, wi = args.anchors / "apokasc3.parquet", args.anchors / "willett.parquet"
    result = crossmatch(pd.read_parquet(ap), pd.read_parquet(wi))
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "willett-kepler-identities.parquet"
    result.to_parquet(output, index=False)
    recovered = pd.read_parquet(output)
    if not recovered["source_id"].equals(result["source_id"]):
        raise ValueError("64-bit Gaia identities failed parquet round trip")
    report = {
        "schema": "willett-kepler-local-identity-v1",
        "status": "partial_kepler_identity_only",
        "inputs": {str(ap): digest(ap), str(wi): digest(wi)},
        "output_sha256": digest(output),
        "rows": len(result),
        "identity_counts": {
            str(k): int(v) for k, v in result["identity_status"].value_counts().items()
        },
        "matched_by_survey": {
            str(k): int(v)
            for k, v in result.loc[result["source_id"].notna(), "survey"]
            .value_counts()
            .items()
        },
        "raw_willett_gaia_id_used": False,
        "age_values_transferred": False,
        "training_performed": False,
        "teacher_error_independence_established": False,
        "caveats": [
            "Identity crossmatch through shared Kepler KIC only; not an independent astrometric verification",
            "APOKASC/Willett share seismic inputs: neither independent truth nor independent teacher errors; correlated teacher-family treatment required",
            "Other surveys remain unmatched; disjoint teacher-family tuning/calibration/test roles still required",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
