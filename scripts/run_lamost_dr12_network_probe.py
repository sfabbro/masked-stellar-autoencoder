"""Bounded headless source connectivity diagnostics; never downloads a catalogue."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

RUN = "multisurvey_lamost_dr12_network_probe_v1"
HTTPS = "https://www.lamost.org/dr12/v2.0/catdl?name=dr12_v2.0_LRS_stellar.fits.gz"
HTTP = "http://www.lamost.org/dr12/v2.0/catdl?name=dr12_v2.0_LRS_stellar.fits.gz"


def probe(url, *, direct=False):
    handler = (
        urllib.request.ProxyHandler({}) if direct else urllib.request.ProxyHandler()
    )
    req = urllib.request.Request(
        url, headers={"Range": "bytes=0-4095", "User-Agent": "msa-lamost-probe/1.0"}
    )
    try:
        with urllib.request.build_opener(handler).open(req, timeout=20) as r:
            data = r.read(4096)
            return {
                "status": "response",
                "http_status": r.status,
                "content_type": r.headers.get("Content-Type"),
                "content_range": r.headers.get("Content-Range"),
                "bytes_read": len(data),
                "gzip_magic": data[:2] == b"\x1f\x8b",
            }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)[:350],
        }


def run(output, commit, *, check=probe):
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Immutable source commit required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    result = {
        "run_id": RUN,
        "source_commit": commit,
        "source_only": True,
        "training_performed": False,
        "model_loaded": False,
        "status": "probing",
        "tests": {},
    }

    def save():
        tmp = out / "manifest.tmp"
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(tmp, out / "manifest.json")

    save()
    for name, url, direct in [
        ("https_default_proxy", HTTPS, False),
        ("https_no_proxy", HTTPS, True),
        ("http_default_proxy", HTTP, False),
        ("http_no_proxy", HTTP, True),
    ]:
        result["tests"][name] = check(url, direct=direct)
        save()
    result["status"] = "lamost_network_probe_complete"
    save()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1"
        or not Path("/arc/projects/k-pop").is_dir()
    ):
        raise RuntimeError("CANFAR CPU headless required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(a.output, commit)


if __name__ == "__main__":
    main()
