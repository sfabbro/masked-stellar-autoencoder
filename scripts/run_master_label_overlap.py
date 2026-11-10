"""Per-survey [Fe/H] overlap with the pooled high-res reference (CPU, no model).

Reads the master-label observation table (from ``run_master_label_census``)
and every staged ``native_observations`` sidecar under ``--existing-root``,
then writes ``overlap.json`` with binned offsets, floors, cross-validated map
complexity and metal-poor gains per survey. No roles, no training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masked_stellar_autoencoder.training import label_overlap as lo  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--existing-root", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{dt.datetime.now(dt.UTC):%H:%M:%S}] {msg}", flush=True)

    obs = pd.read_parquet(a.observations)
    log(f"observations: {len(obs)} rows from {a.observations}")
    sidecars, existing = [], {}
    for root in a.existing_root:
        for entry in lo.discover_sidecars(root, log=log):
            sidecars.append(entry)
            existing[entry["source"]] = lo.read_sidecar_feh(
                entry["path"], entry["source"]
            )
    report = lo.run(obs, existing, log=log)
    report.update(
        status="complete",
        utc=dt.datetime.now(dt.UTC).isoformat(),
        observations=str(a.observations),
        observations_sha256=lo_sha(a.observations),
        sidecars=sidecars,
        training_performed=False,
        roles_assigned=False,
    )
    tmp = a.output / "overlap.tmp"
    tmp.write_text(json.dumps(report, indent=2, default=float) + "\n")
    tmp.replace(a.output / "overlap.json")
    log("overlap.json written")


def lo_sha(path):
    import hashlib

    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


if __name__ == "__main__":
    main()
