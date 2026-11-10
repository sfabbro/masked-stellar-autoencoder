"""Write training sidecars for master-label sources and their map families.

Inputs: the census ``observations.parquet`` (``run_master_label_census``) and,
optionally, the overlap report (``run_master_label_overlap``). Outputs:
``master-label-sidecar.h5`` (``write_observation_sidecar`` format, one group
per training source), ``manifest.json`` with per-source counts, the
``additional_sources`` entries for a source-builder config, and per-source map
families. Nothing is activated or trained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masked_stellar_autoencoder.training import (  # noqa: E402
    map_policy,
)
from masked_stellar_autoencoder.training import (
    master_label_sidecars as mls,
)
from masked_stellar_autoencoder.training.multisurvey_overlay import (  # noqa: E402
    write_observation_sidecar,
)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--overlap", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--arc-path",
        help="path the sidecar will have on ARC (written into the config entries)",
    )
    p.add_argument("--sources", nargs="*", help="subset of training source names")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    obs = pd.read_parquet(a.observations)
    blocks, registry, counts = mls.build_all(obs, a.sources)
    sidecar = a.output / "master-label-sidecar.h5"
    receipt = write_observation_sidecar(
        sidecar,
        blocks,
        registry,
        provenance=dict(
            observations=str(a.observations),
            observations_sha256=sha(a.observations),
            builder="scripts/write_master_label_sidecars.py",
            module="training/master_label_sidecars.py",
            utc=dt.datetime.now(dt.UTC).isoformat(),
            training_enabled=False,
        ),
    )
    arc = a.arc_path or receipt["path"]
    kinds = None
    if a.overlap:
        report = json.loads(a.overlap.read_text())
        kinds = map_policy.source_map_kinds(
            report, list(blocks), aliases=mls.OVERLAP_ALIASES
        )
    manifest = dict(
        status="master_label_sidecars_complete",
        training_performed=False,
        roles_assigned=False,
        sidecar=dict(path=arc, sha256=receipt["sha256"]),
        observations_sha256=sha(a.observations),
        overlap=None
        if not a.overlap
        else dict(path=str(a.overlap), sha256=sha(a.overlap)),
        sources=counts,
        additional_sources=[
            dict(name=name, path=arc, sha256=receipt["sha256"]) for name in blocks
        ],
        source_map_kinds=None
        if kinds is None
        else {k: v["family"] for k, v in kinds.items()},
        source_map_reasons=kinds,
    )
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(dict(sources=counts, kinds=manifest["source_map_kinds"]), indent=1)
    )


if __name__ == "__main__":
    main()
