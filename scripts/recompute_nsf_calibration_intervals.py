"""Correctly recompute CAL intervals from immutable saved marginal quantiles.

An empirical-CDF central mass of one gives an unbounded prediction set, not
sample extrema. This saved-array QA never changes a density or selects a head.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-empty-output", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        if not a.reuse_empty_output or any(a.output.iterdir()):
            raise FileExistsError(
                "Only a reconciled empty failed-report directory may be reused"
            )
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    from run_nsf_calibration_worker import interval_report

    source = ROOT / "nsf_confirmation_calibration_20261107_v1"
    m = dict(
        status="waiting_for_saved_arrays",
        model_loaded=False,
        training_performed=False,
        results={},
        correction="Empirical CDF mass 1 implies the full real line. Density parameters are unchanged; this is interval QA, not physical PDF certification.",
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    deadline = time.monotonic() + 36 * 3600
    save()
    while time.monotonic() < deadline:
        try:
            producer = json.loads((source / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            producer = {}
        for rid, item in producer.get("results", {}).items():
            if rid in m["results"]:
                continue
            if item.get("status") != "complete":
                m["results"][rid] = dict(
                    status="upstream_failed", error=item.get("error")
                )
                continue
            arrays = {}
            hashes = {}
            for role in ["cal", "devtest"]:
                arrays[role] = {}
                for kind in ["marginals", "seismic"]:
                    path = source / (rid + "-" + role + "-" + kind + ".npz")
                    hashes[path.name] = sha(path)
                    with np.load(path) as f:
                        arrays[role][kind] = {k: f[k] for k in f.files}
            m["results"][rid] = dict(
                status="complete",
                checkpoint_sha256=item["checkpoint_sha256"],
                input_sha256=hashes,
                catalogue=interval_report(
                    arrays["cal"]["marginals"],
                    arrays["devtest"]["marginals"],
                    ["teff", "logg", "fe_h", "alpha", "age", "distance"],
                ),
                seismic=interval_report(
                    arrays["cal"]["seismic"], arrays["devtest"]["seismic"], ["logg"]
                ),
            )
            save()
        if producer.get("status") == "complete":
            m["status"] = "corrected_interval_qa_complete"
            m["producer_manifest_sha256"] = sha(source / "manifest.json")
            save()
            return
        time.sleep(30)
    raise TimeoutError("Saved marginal calibration arrays did not complete in 36h")


if __name__ == "__main__":
    main()
