"""Audit/recover expanded cache parallax coordinates; never activate a training profile."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
CACHE = ROOT + "residual-huber-e86-150m-cache-v1/"
FEATURE = ROOT + "multisurvey_feature_union_dr12_20261106_v1/new-star-features.h5"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if (
        sha(FEATURE)
        != "d4d8ea8e972625cb040c0fdcf4cc60675c03823b43826cef4b57d6d725e14f23"
    ):
        raise ValueError("Feature cohort differs")
    receipt = json.loads(Path(CACHE + "manifest.json").read_text())
    request = receipt["request"]
    if (
        receipt["request_sha256"]
        != "8d14aa3f741946e8fa220fd0d9f8d74bb44f5949e2f9d2ff0449d21cfa5e5b9c"
    ):
        raise ValueError("Cache binding differs")
    if sha(CACHE + "source_id.npy") != receipt["files"]["source_id.npy"]["sha256"]:
        raise ValueError("Cache IDs differ")
    names = request["features"]
    index = names.index("PARALLAX")
    cache_id = np.load(CACHE + "source_id.npy", mmap_mode="r")
    features = np.load(CACHE + "features.npy", mmap_mode="r")
    with h5py.File(FEATURE) as f:
        ids = f["source_id"][:]
        rows = f["cache_row"][:]
    hit = rows >= 0
    ii = np.flatnonzero(hit)
    order = np.argsort(rows[ii], kind="stable")
    ordered = ii[order]
    coordinate = np.full(len(ids), np.nan, dtype="f4")
    for start in range(0, len(ordered), 100000):
        sel = ordered[start : start + 100000]
        rr = rows[sel]
        np.testing.assert_array_equal(cache_id[rr], ids[sel])
        coordinate[sel] = features[rr, index]
    observed = hit & np.isfinite(coordinate) & (coordinate != -9999)
    with h5py.File(a.output / "cache-parallax-candidate.h5", "x") as f:
        f["source_id"] = ids
        f["cache_row"] = rows
        f["standardized_cache_coordinate"] = coordinate
        f["available_cache_coordinate"] = observed
        f["inverse_scaler_coordinate"] = np.where(
            observed,
            coordinate * request["scale"][index] + request["center"][index],
            np.nan,
        )
        f["formal_error_observed"] = np.zeros(len(ids), bool)
        f.attrs["scope"] = (
            "Exact cache ID association and retained PARALLAX coordinate only; inverse scaler is approximate. Native release/unit/error provenance and raw measurement association still require admission; not an activated measured-parallax input profile."
        )
    result = dict(
        status="expanded_cache_parallax_candidate_complete",
        training_performed=False,
        profile_activated=False,
        expanded_stars=len(ids),
        cache_matched=int(hit.sum()),
        available_cache_parallax_coordinate=int(observed.sum()),
        known_formal_errors=0,
        candidate_sha256=sha(a.output / "cache-parallax-candidate.h5"),
        scope="Recoverable cache coordinate, not certified measured Gaia parallax; final profile requires raw measurement/error/unit provenance",
    )
    (a.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
