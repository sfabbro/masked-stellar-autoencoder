"""Execute one pinned read-only XP analysis in a headless session."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text())
    matches = [x for x in spec["analyses"] if x["run_id"] == args.run_id]
    if len(matches) != 1:
        parser.error("Require exactly one pinned analysis")
    settings = matches[0]
    if args.output.exists():
        parser.error("Analysis output already exists; preserve completed results")
    args.output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "settings": settings,
        "read_only_checkpoints_and_data": True,
    }

    def save():
        temporary = args.output / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output / "manifest.json")

    save()
    try:
        labels = spec["labels_fits"]
        if settings["mode"] == "probes":
            source = Path(labels)
            staging = Path("/scratch/msa-probe-data")
            staging.mkdir(parents=True, exist_ok=True)
            size = source.stat().st_size
            if shutil.disk_usage(staging).free < size + 5 * 1024**3:
                raise OSError("Insufficient scratch space to stage labelled FITS")
            destination = staging / source.name
            print(f"STAGING_LABELS {source} {size} bytes", flush=True)
            shutil.copyfile(source, destination.with_suffix(".copying"))
            destination.with_suffix(".copying").replace(destination)
            labels = str(destination)
            manifest["labels_staging"] = {
                "source": str(source),
                "scratch_path": labels,
                "size_bytes": size,
                "policy": "sequential ARC read; all FITS column access on private scratch",
            }
            save()
        splits = settings.get("splits", ["identity"])
        for split in splits:
            command = [
                sys.executable,
                "scripts/run_xp_representation_probes.py",
                "--mode",
                settings["mode"],
                "--checkpoint",
                settings["checkpoint"],
                "--benchmark",
                spec["benchmark"],
                "--output",
                str(args.output / split),
            ]
            if settings.get("checkpoint_sha256"):
                command += ["--checkpoint-sha256", settings["checkpoint_sha256"]]
            if settings["mode"] == "probes":
                command += [
                    "--labels-fits",
                    labels,
                    "--split",
                    split,
                    "--fits-xp-units",
                    "auto",
                    "--head-steps",
                    "1000",
                    "--finetune-steps",
                    "200",
                ]
            subprocess.run(command, check=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
