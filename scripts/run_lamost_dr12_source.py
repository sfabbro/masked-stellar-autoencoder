"""One versioned LAMOST DR12 download, source projection and Gaia cache join."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from lamost_dr12_source import PRODUCTS, download, produce

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
BASE = ROOT + "remediation_source_v3/prepared.h5"
BASE_SHA = "112fb4b5708c6073ffd2388012a6a9dc7e5bd3eca540db15acbac0fe235f0a1c"
CACHE = ROOT + "residual-huber-e86-150m-cache-v1"
ORIGINAL = "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"
RUNS = {
    "multisurvey_lamost_dr12_lrs_source_v1": "lrs",
    "multisurvey_lamost_dr12_mrs_source_v1": "mrs",
}


def save(path, obj):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x") as out:
        json.dump(obj, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    os.replace(temp, path)


def run(run_id, output, source_commit, *, fetch=download, execute=produce):
    if (
        run_id not in RUNS
        or len(source_commit) != 40
        or any(c not in "0123456789abcdef" for c in source_commit)
    ):
        raise ValueError("Exact run and immutable source required")
    product = RUNS[run_id]
    spec = PRODUCTS[product]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "run_id": run_id,
        "status": "downloading",
        "source_commit": source_commit,
        "product": product,
        "source_url": spec["url"],
        "expected_source_bytes": spec["bytes"],
        "source_only": True,
        "model_loaded": False,
        "training_performed": False,
        "release": "LAMOST DR12 v2.0",
        "scope": "All raw records retained. Exact Gaia ID joins only; outside-base ancestry and absent-cache XP remain unknown. LASP/CNN share spectra and are not independent.",
    }
    save(output / "manifest.json", record)
    try:
        archive = output / ("dr12_" + product + "_stellar.fits.gz")
        digest = fetch(spec["url"], archive, spec["bytes"])
        record.update(
            status="crossmatching",
            archive_sha256=digest,
            archive_path=str(archive.resolve()),
        )
        save(output / "manifest.json", record)

        def progress(done, total):
            save(
                output / "progress.json",
                {
                    "run_id": run_id,
                    "phase": "cache_id_scan",
                    "rows_scanned": done,
                    "rows_total": total,
                },
            )

        result = execute(
            product=product,
            archive=archive,
            base=BASE,
            cache=CACHE,
            original=ORIGINAL,
            base_sha=BASE_SHA,
            output=output / "source.h5",
            progress=progress,
        )
        if (
            result["product"] != product
            or result["source_rows"] != spec["rows"]
            or result["output"] != str((output / "source.h5").resolve())
        ):
            raise ValueError("Source output contract differs")
        record.update(status="lamost_dr12_source_complete", result=result)
    except Exception as e:
        record.update(status="failed", error_type=type(e).__name__, error=str(e))
        save(output / "manifest.json", record)
        raise
    save(output / "manifest.json", record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1"
        or not Path("/arc/projects/k-pop").is_dir()
    ):
        raise RuntimeError("Requires CANFAR CPU headless")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(a.run_id, a.output, commit)


if __name__ == "__main__":
    main()
