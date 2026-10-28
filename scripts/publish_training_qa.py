"""Publish a durable training CSV outbox to VOSpace without blocking the GPU.

Run in the existing monitor Pixi environment. No checkpoint/data export session
is needed; this process copies only validated bounded QA CSVs and their manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from masked_stellar_autoencoder.training.continuation_state import (
    atomic_json,
    exclusive_run,
    publish_qa,
)
from monitor.training_qa_io import remote_root, validate_csv, validate_manifest


def publish_once(output, remote, timeout=60):
    output = Path(output)
    remote = remote_root(remote)
    outbox = json.loads((output / "qa-outbox.json").read_text())
    if outbox["remote"] != remote:
        raise ValueError("Outbox belongs to a different VOS root")
    publication = outbox["publication_id"]
    if not publication.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid outbox publication")
    path = output / "qa" / publication / "manifest.json"
    manifest = validate_manifest(json.loads(path.read_text()))
    for name in manifest["files"]:
        validate_csv(name, path.parent / name, manifest["run_id"])
    published = output / "qa-published.json"
    if (
        published.exists()
        and json.loads(published.read_text()).get("publication_id") == publication
    ):
        return manifest["status"]
    publish_qa(output / "qa", manifest, remote, timeout=timeout)
    atomic_json(
        published,
        dict(
            publication_id=publication,
            remote=remote,
            at_utc=datetime.now(UTC).isoformat(),
            status="published",
        ),
    )
    return manifest["status"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--remote", required=True)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()
    if not 1 <= args.timeout <= 120:
        p.error("timeout must be1–120 seconds")
    remote = remote_root(args.remote)
    with exclusive_run(args.output / "publisher-lock"):
        while True:
            status = None
            try:
                status = publish_once(args.output, remote, args.timeout)
            except Exception as error:
                atomic_json(
                    args.output / "qa-publication-status.json",
                    dict(
                        status="unavailable",
                        error_type=type(error).__name__,
                        at_utc=datetime.now(UTC).isoformat(),
                        message="Local QA preserved; check CADC authentication and VOS availability. No training retry needed.",
                    ),
                )
                if not args.watch:
                    raise SystemExit(1) from None
            else:
                atomic_json(
                    args.output / "qa-publication-status.json",
                    dict(status="published", at_utc=datetime.now(UTC).isoformat()),
                )
            if not args.watch or status in {"complete", "failed", "stopped"}:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
