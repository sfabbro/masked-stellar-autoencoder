"""Fetch a verified training QA CSV snapshot, then open its local marimo view."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitor.training_qa_io import fetch_snapshot, remote_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote",
        default=os.environ.get(
            "MSA_QA_REMOTE", "vos:sfabbro/msa/joint_native_30e_v1/qa"
        ),
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "experiments/xp/training-qa-cache"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never contact VOSpace; verify and use cached files",
    )
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Check transfer/schema/checksums and print status without a server",
    )
    parser.add_argument("--port", type=int, default=2720)
    args = parser.parse_args(argv)
    remote = remote_root(args.remote)
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    try:
        snapshot = fetch_snapshot(remote, args.cache, offline=args.offline)
        print(
            json.dumps(
                {
                    "run_id": snapshot.manifest["run_id"],
                    "publication_id": snapshot.manifest["publication_id"],
                    "status": snapshot.manifest["status"],
                    "generated_at_utc": snapshot.manifest["generated_at_utc"],
                    "stale": snapshot.stale,
                    "warning": snapshot.warning,
                }
            ),
            flush=True,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        if args.fetch_only:
            return 1
    if args.fetch_only:
        return 0
    os.environ["MSA_QA_REMOTE"] = remote
    os.environ["MSA_QA_CACHE"] = str(args.cache.resolve())
    os.environ["MSA_QA_OFFLINE"] = "1" if args.offline else "0"
    # The monitor environment already includes marimo/pandas/matplotlib/vos.
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "marimo",
            "run",
            str(ROOT / "monitor/check_training.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
