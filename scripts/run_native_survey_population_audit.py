"""Count native DESI/LAMOST records and exact Gaia-ID unions before XP/quality cuts.

This source audit never assigns roles, admits labels, or executes a model.
"""

import argparse
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time

import h5py
import numpy as np
from astropy.io import fits


LAMOST = {
    "lamost_dr12_lrs": (
        "multisurvey_lamost_dr12_lrs_source_direct_v2/source.h5",
        "889c90553e71a681ddee05f97fc6dcffb7933a338253e3e57912460be8d58fb9",
    ),
    "lamost_dr12_mrs": (
        "multisurvey_lamost_dr12_mrs_source_direct_v2/source.h5",
        "293201d803cb569c79ce2c76f51b912e6dc8a69c27296450154a18946d60c8f6",
    ),
}


def positive_unique(ids):
    ids = np.asarray(ids)
    if ids.dtype.kind not in "iu":
        raise ValueError("Native Gaia identifiers must remain integer, never float")
    if ids.dtype.kind == "u" and (ids > np.iinfo(np.int64).max).any():
        raise ValueError("Gaia identifier exceeds signed 64-bit range")
    return np.unique(ids[ids > 0].astype("i8"))


def compare_sets(sources, parent):
    parent = positive_unique(parent)
    unique = {name: positive_unique(ids) for name, ids in sources.items()}
    union = np.unique(np.concatenate(list(unique.values())))
    missing = np.setdiff1d(union, parent, assume_unique=True)
    return {
        "unique_gaia_ids_by_source": {name: len(ids) for name, ids in unique.items()},
        "pairwise_unique_star_overlap": {
            left + "__" + right: len(np.intersect1d(unique[left], unique[right], assume_unique=True))
            for left, right in itertools.combinations(unique, 2)
        },
        "native_desi_lamost_union_unique_gaia_stars": len(union),
        "existing_candidate_parent_unique_gaia_stars": len(parent),
        "native_union_absent_from_candidate_parent": len(missing),
        "absent_by_source": {name: len(np.setdiff1d(ids, parent, assume_unique=True)) for name, ids in unique.items()},
        "expanded_candidate_parent_unique_gaia_stars": len(parent) + len(missing),
    }, missing, np.union1d(parent, union)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(root, output, save):
    result = {"native_sources": {}, "training_performed": False, "training_admitted": False}
    sources = {}
    for name, (relative, expected) in LAMOST.items():
        path = root / relative
        if sha(path) != expected:
            raise ValueError(f"Native all-row LAMOST sidecar changed: {name}")
        with h5py.File(path) as data:
            ids = data["observations/source_id"][:]
            sources[name] = positive_unique(ids)
            stored = data["stars/source_id"][:]
            if not np.array_equal(positive_unique(stored), sources[name]):
                raise ValueError(f"Observation/star identity sets differ: {name}")
        result["native_sources"][name] = {
            "path": str(path), "sha256": expected,
            "observation_rows": len(ids), "positive_gaia_id_rows": int((ids > 0).sum()),
            "unique_positive_gaia_stars": len(sources[name]),
            "id_set_sha256": hashlib.sha256(sources[name].astype("<i8").tobytes()).hexdigest(),
            "scope": "All staged native observation rows, before quality cuts and XP/feature joins",
        }
        save(result)
    path = root / "desi_collection_v1/mwsall-pix-iron.fits"
    with fits.open(path, memmap=True) as hdus:
        source = hdus["GAIA"]
        ids = np.asarray(source.data["SOURCE_ID"])
        sources["desi_dr1_coadds"] = positive_unique(ids)
        if len(ids) != hdus["RVTAB"].header["NAXIS2"]:
            raise ValueError("DESI Gaia/parameter row counts differ")
        result["native_sources"]["desi_dr1_coadds"] = {
            "path": str(path), "extension": "GAIA", "column": "SOURCE_ID",
            "observation_rows": len(ids), "positive_gaia_id_rows": int((ids > 0).sum()),
            "unique_positive_gaia_stars": len(sources["desi_dr1_coadds"]),
            "id_set_sha256": hashlib.sha256(sources["desi_dr1_coadds"].astype("<i8").tobytes()).hexdigest(),
            "file_bytes": path.stat().st_size,
            "previously_publisher_verified_sha256": "eae3b31807c58ac340c257e06f66935d01bb81698a08108e6ca57a372cab76b5",
            "whole_file_hash_recomputed": False,
            "scope": "Full native coadd Gaia-ID column; no PRIMARY/STAR/SNR/XP selection; single epochs are a distinct dependent product",
        }
    parent_path = root / "full_gaia_gaia_stage_20261108_v2/label-candidate-source-id.npy"
    parent = np.load(parent_path, mmap_mode="r")
    counts, missing, expanded = compare_sets(sources, parent)
    np.save(output / "native-stars-missing-from-candidate-parent.npy", missing)
    np.save(output / "expanded-native-candidate-source-id.npy", expanded)
    result.update(counts)
    result["native_observation_rows_total"] = sum(s["observation_rows"] for s in result["native_sources"].values())
    result["candidate_parent"] = {"path": str(parent_path), "sha256": sha(parent_path)}
    result["pending"] = [
        "Match missing native IDs against full Gaia without requiring XP",
        "Measure per-target quality/domain retention separately from raw population counts",
        "Retain repeated spectra and shared-input dependencies without double-counting independent evidence",
        "Discover and audit additional release/VAC products; these three files are not every possible DESI/LAMOST label product",
        "Preserve inherited protected roles; new native stars remain unassigned",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "started_utc": dt.datetime.now(dt.UTC).isoformat()}
    def save(extra=None):
        if extra:
            report.update(extra)
        temporary = args.output / "manifest.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "manifest.json")
    save()
    try:
        report.update(run(args.root, args.output, save), status="native_population_audit_complete")
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save()


if __name__ == "__main__":
    main()
