"""Bounded asynchronous TAP acquisition of the real Pristine DR1 catalogue."""

import hashlib
import shutil
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from astropy.io import fits

BASE = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/async"
QUERY = 'SELECT * FROM "J/A+A/692/A115/pristine_dr1"'
EXPECTED_ROWS = 6380652


def verify_catalogue(path, expected_rows=EXPECTED_ROWS):
    from stellar_catalog_jobs import verify_expected_fits_rows

    with fits.open(path, memmap=True) as hdul:
        verify_expected_fits_rows(hdul, expected_rows)
        table = next(h for h in hdul if isinstance(h, fits.BinTableHDU))
        required = {
            "GaiaDR3",
            "RAICRS",
            "DEICRS",
            "CaHKmag0",
            "e_CaHKmag0",
            "[Fe/H]phP",
            "b_[Fe/H]phP",
            "B_[Fe/H]phP",
            "Flag",
        }
        if not required <= set(table.columns.names):
            raise ValueError("Unexpected real Pristine DR1 schema")
        ids = np.asarray(table.data["GaiaDR3"], dtype="i8")
        if np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("Pristine DR3 identifiers are invalid or duplicated")
        return {
            "rows": len(ids),
            "unique_positive_DR3_IDs": len(ids),
            "source_id_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            "columns": table.columns.names,
            "units": {c.name: c.unit for c in table.columns if c.unit},
        }


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def build_catalogue(output, scratch):
    # One submission only; no retry of an ambiguous POST or an existing output.
    receipt = output / "tap-job-url.txt"
    if receipt.exists():
        raise FileExistsError(receipt)
    with requests.Session() as session:
        response = session.post(
            BASE,
            data={
                "REQUEST": "doQuery",
                "LANG": "ADQL",
                "FORMAT": "fits",
                "QUERY": QUERY,
                "MAXREC": "7000000",
                "PHASE": "RUN",
            },
            allow_redirects=False,
            timeout=60,
        )
        if response.status_code != 303:
            raise ValueError(f"TAP creation returned {response.status_code}")
        job = urljoin(BASE + "/", response.headers["Location"])
        if (
            urlparse(job).scheme != "https"
            or urlparse(job).hostname != urlparse(BASE).hostname
        ):
            raise ValueError("Unexpected TAP job location")
        receipt.write_text(job + "\n")
        deadline = time.monotonic() + 1800
        while True:
            response = session.get(job, timeout=60)
            response.raise_for_status()
            (output / "tap-job.xml").write_text(response.text)
            phase = session.get(job + "/phase", timeout=60)
            phase.raise_for_status()
            state = phase.text.strip()
            if state == "COMPLETED":
                break
            if state in {"ERROR", "ABORTED"}:
                raise ValueError(f"TAP export ended {state}; inspect tap-job.xml")
            if state not in {"QUEUED", "EXECUTING", "PENDING", "HELD"}:
                raise ValueError(f"Unexpected TAP phase {state!r}")
            if time.monotonic() >= deadline:
                session.post(job + "/phase", data={"PHASE": "ABORT"}, timeout=30)
                raise TimeoutError("TAP export exceeded bounded 1800seconds")
            time.sleep(5)
        staged = scratch / "pristine-dr1-full.fits"
        size = 0
        with session.get(job + "/results/result", stream=True, timeout=180) as result:
            result.raise_for_status()
            with staged.open("xb") as dest:
                for block in result.iter_content(8 * 1024 * 1024):
                    size += len(block)
                    if size > 2500000000:
                        raise ValueError("Pristine export exceeds bounded2.5GB")
                    dest.write(block)
    audit = verify_catalogue(staged)
    sha = digest(staged)
    destination = output / staged.name
    partial = output / (staged.name + ".copying")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    shutil.copyfile(staged, partial)
    if digest(partial) != sha:
        raise ValueError("Pristine publication checksum mismatch")
    partial.replace(destination)
    return {
        "path": str(destination),
        "sha256": sha,
        "bytes": size,
        "tap_job": job,
        "query": QUERY,
        "audit": audit,
        "scientific_role": "Real CaHK photometry; photometric FeH candidate labels, not independent spectroscopic truth",
        "quality": "All native columns/flags/errors retained; downstream eligibility not yet applied",
    }
