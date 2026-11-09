"""CPU-only source recovery and census; never submits or trains a model."""

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", choices=["artifact_audit", "ages", "alpha", "input_census"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "training_performed": False, "training_admitted": False, "steps": {},
        "scope": "Native-source extraction and availability census; no quality admission or production model claim",
    }
    def save():
        temporary = args.output / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output / "manifest.json")
    save()
    jobs = [
        ("artifact_audit", "scripts/run_full_gaia_resume_audit.py", ["--root", str(args.root)]),
        ("ages", "scripts/run_full_gaia_correction_sources.py", ["--mode", "ages"]),
        ("alpha", "scripts/run_full_gaia_correction_sources.py", ["--mode", "alpha"]),
        ("input_census", "scripts/run_full_gaia_input_census.py", ["--root", str(args.root), "--contract", "configs/full-gaia-observation-columns-20261108.json"]),
    ]
    if args.steps:
        jobs = [job for job in jobs if job[0] in args.steps]
    try:
        for name, script, arguments in jobs:
            manifest["current_step"] = name
            manifest["steps"][name] = {"status": "running", "output": str(args.output / name)}
            save()
            subprocess.run([sys.executable, "-u", script, *arguments, "--output", str(args.output / name)], check=True)
            result = json.loads((args.output / name / "manifest.json").read_text())
            errors = [s for s in result.get("sources", []) if s.get("error")]
            manifest["steps"][name].update(status=result["status"], source_errors=errors)
            save()
            if errors or result["status"] == "failed":
                raise RuntimeError(f"{name} reported extraction errors; inspect its manifest")
        manifest["status"] = "source_recovery_and_census_complete"
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["updated_utc"] = dt.datetime.now(dt.UTC).isoformat()
        save()


if __name__ == "__main__":
    main()
