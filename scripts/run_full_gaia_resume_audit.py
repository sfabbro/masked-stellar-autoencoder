"""Read-only ARC artifact and runtime audit; no model execution or label admission."""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time


def audit(root):
    result = {
        "checked_utc": dt.datetime.now(dt.UTC).isoformat(),
        "python": sys.version,
        "executables": {name: shutil.which(name) for name in ["python", "pixi", "uv"]},
        "packages": {},
        "training_performed": False,
        "training_admitted": False,
    }
    for name in ["numpy", "pandas", "pyarrow", "astropy", "h5py", "torch"]:
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    manifest_path = root / "expanded_six_nsf_fullfit_20261107_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = manifest_path.parent / "selected-checkpoint.pth"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    result["comparison_checkpoint"] = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "sha256": digest.hexdigest(),
        "expected_sha256": manifest["selected_checkpoint_sha256"],
        "digest_matches": digest.hexdigest() == manifest["selected_checkpoint_sha256"],
        "selected_update": manifest["selected_update"],
        "selected_epoch": manifest["selected_epoch"],
    }
    result["stages"] = {}
    for name in [
        "full_gaia_gaia_stage_20261108_v2",
        "full_gaia_photometry_stage_20261108_v1",
        "full_gaia_alpha_stage_20261108_v1",
        "full_gaia_ages_stage_20261108_v1",
    ]:
        directory = root / name
        stage = json.loads((directory / "manifest.json").read_text())
        partitions = sorted(directory.rglob("*.parquet"))
        entry = {
            "status": stage["status"],
            "partition_count": len(partitions),
            "partition_bytes": sum(p.stat().st_size for p in partitions),
            "table_metadata": [],
        }
        if result["packages"]["pyarrow"] is not None:
            import pyarrow.parquet as pq

            groups = {}
            for path in partitions:
                meta = pq.ParquetFile(path)
                relative_parent = str(path.parent.relative_to(directory))
                group = groups.setdefault(relative_parent, {
                    "partitions": 0, "rows": 0,
                    "schema": str(meta.schema_arrow),
                })
                group["partitions"] += 1
                group["rows"] += meta.metadata.num_rows
            entry["table_metadata"] = groups
        result["stages"][name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"status": "running", "training_performed": False}
    target = args.output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        manifest.update(audit(args.root), status="artifact_audit_complete")
        if not manifest["comparison_checkpoint"]["digest_matches"]:
            raise ValueError("Selected comparison checkpoint digest mismatch")
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest.update(
            elapsed_seconds=time.monotonic() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(target)


if __name__ == "__main__":
    main()
