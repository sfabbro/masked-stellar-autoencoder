"""Read-only bounded Gaia query batch timing on CANFAR."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from recover_esa_parallax_gaps import ROOT, query


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(status="querying_public_ids", training_performed=False)
    (a.output / "manifest.json").write_text(json.dumps(m))
    try:
        ids = pd.read_parquet(
            ROOT
            / "gaia_parallax_release_unit_audit_20261107_v2/paired-parallax-check.parquet"
        ).source_id.to_numpy(dtype="i8")[:512]
        start = time.monotonic()
        body, frame = query(ids)
        if set(map(int, frame.source_id)) != set(map(int, ids)):
            raise ValueError("Probe did not return every known published ID")
        (a.output / "esa.csv").write_bytes(body)
        m.update(
            status="complete",
            requested=len(ids),
            returned=len(frame),
            seconds=time.monotonic() - start,
            response_sha256=hashlib.sha256(body).hexdigest(),
            external_payload="Public Gaia IDs only",
        )
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        raise
    finally:
        (a.output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    main()
