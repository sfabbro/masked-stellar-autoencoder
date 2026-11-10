"""Refresh full-Gaia inputs for the audited native DESI/LAMOST identity union.

No label admission, role assignment, normalization, or model execution occurs.
"""

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.audit) != args.audit_sha256:
        raise ValueError("Native population audit changed")
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "native_population_audit_complete":
        raise ValueError("Native population audit is not complete")
    parent = args.audit.parent / "expanded-native-candidate-source-id.npy"
    ids = np.load(parent, mmap_mode="r")
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids <= 0).any() or not np.all(ids[1:] > ids[:-1]):
        raise ValueError("Native parent requires sorted unique positive integer IDs")
    if len(ids) != audit["expanded_candidate_parent_unique_gaia_stars"]:
        raise ValueError("Native parent count differs from its audit")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "training_performed": False, "training_admitted": False,
        "role_assignment_performed": False, "normalization_performed": False,
        "audit": str(args.audit), "audit_sha256": args.audit_sha256,
        "candidate_stars": len(ids), "candidate_ids_sha256": sha(parent),
        "added_native_stars": audit["native_union_absent_from_candidate_parent"],
        "scope": "Full native identity parent before quality or XP/feature filters; matched source availability is not label admission",
        "steps": {},
    }
    def save():
        temporary = args.output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output / "manifest.json")
    save()
    jobs = [
        ("gaia", "scripts/run_full_gaia_correction_sources.py", ["--mode", "gaia", "--candidate-ids", str(parent)]),
        ("photometry", "scripts/run_full_gaia_correction_sources.py", ["--mode", "photometry", "--candidate-ids", str(parent)]),
        ("input_census", "scripts/run_full_gaia_input_census.py", [
            "--root", str(args.root), "--contract", "configs/full-gaia-observation-columns-20261108.json",
            "--gaia-stage", str(args.output / "gaia"), "--photometry-stage", str(args.output / "photometry"),
        ]),
    ]
    try:
        for name, script, arguments in jobs:
            manifest["current_step"] = name
            manifest["steps"][name] = {"status": "running", "output": str(args.output / name)}
            save()
            subprocess.run([sys.executable, "-u", script, *arguments, "--output", str(args.output / name)], check=True)
            result = json.loads((args.output / name / "manifest.json").read_text())
            if result.get("candidate_stars") != manifest["candidate_stars"] or result["status"] == "failed":
                raise ValueError(f"{name} parent/status differs")
            manifest["steps"][name]["status"] = result["status"]
            save()
        manifest["status"] = "native_parent_inputs_and_census_complete"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
        save()


if __name__ == "__main__":
    main()
