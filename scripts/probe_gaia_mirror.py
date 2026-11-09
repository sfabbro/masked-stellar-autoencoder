"""Cross-check public Gaia DR3 astrometry from GAVO against saved ESA responses."""

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="querying_gaia_dr3_mirror", pass_gate=False, training_performed=False
    )

    def save():
        (a.output / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")

    save()
    try:
        ref = pd.read_parquet(
            ROOT
            / "gaia_parallax_release_unit_audit_20261107_v2/paired-parallax-check.parquet"
        ).iloc[:512]
        ids = ref.source_id.to_numpy(dtype="i8")
        adql = (
            "SELECT source_id,parallax,parallax_error FROM gaia.dr3lite WHERE source_id IN ("
            + ",".join(str(int(v)) for v in ids)
            + ")"
        )
        payload = urllib.parse.urlencode(
            dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=adql, MAXREC=1000)
        ).encode()
        start = time.monotonic()
        for retry in range(4):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(
                        "https://dc.zah.uni-heidelberg.de/tap/sync",
                        data=payload,
                        method="POST",
                    ),
                    timeout=60,
                ) as response:
                    body = response.read()
                break
            except Exception:
                if retry == 3:
                    raise
                time.sleep(5)
        got = pd.read_csv(io.BytesIO(body), dtype={"source_id": "Int64"})
        if (
            got.source_id.isna().any()
            or got.source_id.duplicated().any()
            or set(map(int, got.source_id)) != set(map(int, ids))
        ):
            raise ValueError("Mirror identity set differs")
        got = got.set_index("source_id").loc[ids]
        for key, reference in [
            ("parallax", "dr3_parallax"),
            ("parallax_error", "dr3_parallax_error"),
        ]:
            np.testing.assert_allclose(
                got[key].to_numpy(dtype="f8"),
                ref[reference].to_numpy(dtype="f8"),
                rtol=2e-6,
                atol=2e-6,
                equal_nan=True,
            )
        (a.output / "mirror.csv").write_bytes(body)
        audit = json.loads(
            (
                ROOT / "gaia_parallax_release_unit_audit_20261107_v2/manifest.json"
            ).read_text()
        )
        m.update(
            status="complete",
            pass_gate=True,
            stars=len(ids),
            seconds=time.monotonic() - start,
            response_sha256=hashlib.sha256(body).hexdigest(),
            parent_source_sha256=audit["source_sha256"],
            endpoint="https://dc.zah.uni-heidelberg.de/tap/sync",
            table="gaia.dr3lite",
            schema_reference="https://dc.zah.uni-heidelberg.de/tableinfo/gaia.dr3lite",
            external_payload="Public Gaia DR3 source IDs only; no survey labels or positions",
            tolerance="2e-6 relative + 2e-6 mas absolute; null availability also required to agree with ESA",
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
