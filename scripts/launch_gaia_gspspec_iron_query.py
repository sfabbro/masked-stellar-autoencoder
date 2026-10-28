"""Submit one bounded Gaia DR3 Fe/M TAP query, with an immutable receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

SERVICE = "https://gaia.ari.uni-heidelberg.de/tap/async"
QUERY = (
    "SELECT source_id, mh_gspspec, fem_gspspec, fem_gspspec_lower, "
    "fem_gspspec_upper, fem_gspspec_nlines, fem_gspspec_linescatter, "
    "flags_gspspec FROM gaiadr3.astrophysical_parameters "
    "WHERE fem_gspspec IS NOT NULL"
)


def _write(path, value, *, create=False):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if create:
        with path.open("x") as stream:
            stream.write(payload)
    else:
        temp = path.with_name(path.name + ".tmp")
        with temp.open("x") as stream:
            stream.write(payload)
        temp.replace(path)


def submit(record_path, *, source_sha, client):
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise ValueError("A full immutable MSA revision is required")
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "gaia-gspspec-fem-tap-submission-v1",
        "status": "submitting",
        "source_sha": source_sha,
        "service": SERVICE,
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "maxrec": 10000000,
        "executionduration_seconds": 14400,
        "scope": "All non-null Gaia GSP-Spec Fe I/M source rows; no model, no label calibration, no FIT or feature claim",
        "attempts": 1,
        "submitted_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    _write(record_path, record, create=True)
    try:
        response = client.post(
            SERVICE,
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "csv",
                "MAXREC": "10000000",
                "EXECUTIONDURATION": "14400",
                "PHASE": "RUN",
                "QUERY": QUERY,
            },
            allow_redirects=False,
            timeout=60,
        )
    except Exception as error:
        record.update(
            status="submission_ambiguous",
            error_type=type(error).__name__,
            error=str(error)[:1000],
        )
        _write(record_path, record)
        raise
    location = response.headers.get("Location", "")
    absolute = urljoin(SERVICE + "/", location)
    parsed = urlparse(absolute)
    valid = (
        response.status_code == 303
        and parsed.scheme == "https"
        and parsed.netloc == "gaia.ari.uni-heidelberg.de"
        and re.fullmatch(r"/tap/async/[0-9]+", parsed.path)
        and not parsed.query
        and not parsed.fragment
    )
    if not valid:
        status = (
            "rejected" if 400 <= response.status_code < 500 else "submission_ambiguous"
        )
        record.update(
            status=status,
            http_status=response.status_code,
            response_body=response.text[:4096],
            location=location,
        )
        _write(record_path, record)
        raise RuntimeError(
            "Gaia TAP query was rejected or its job identity is ambiguous"
        )
    record.update(
        status="submitted",
        http_status=response.status_code,
        job_url=absolute,
        job_id=parsed.path.rsplit("/", 1)[1],
    )
    _write(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    result = submit(args.record, source_sha=args.source_sha, client=requests.Session())
    print(
        json.dumps(
            {
                "status": result["status"],
                "job_id": result["job_id"],
                "job_url": result["job_url"],
            }
        )
    )


if __name__ == "__main__":
    main()
