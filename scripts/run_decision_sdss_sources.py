"""Acquire DR20 BOSSNet and final legacy SSPP products, retaining native schemas/checksums."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits

FILES = {
    "dr20_bossnet": dict(
        url="https://data.sdss.org/sas/dr20/spectro/astra/0.8.1/summary/astraAllStarBossNet-0.8.1.fits.gz",
        bytes=849231687,
    ),
    "legacy_sspp": dict(url="https://data.sdss.org/sas/dr9/sdss/sspp/ssppOut-dr9.fits"),
}
CHECKSUM_URL = "https://data.sdss.org/sas/dr20/spectro/astra/0.8.1/summary/spectro_astra_0.8.1_summary.sha1sum"
MRS = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/multisurvey_lamost_dr12_mrs_source_direct_v2/dr12_mrs_stellar.fits.gz"


def digest(path, algorithm="sha256"):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def download(name, spec, out):
    path = out / (name + (".fits.gz" if name == "dr20_bossnet" else ".fits"))
    with urllib.request.urlopen(spec["url"], timeout=120) as r, path.open("xb") as f:
        if r.status != 200:
            raise ValueError("Expected full native catalogue")
        expected = spec.get("bytes", int(r.headers.get("Content-Length", 0)))
        total = 0
        while block := r.read(8 * 1024 * 1024):
            f.write(block)
            total += len(block)
        if expected and total != expected:
            raise ValueError("Truncated source")
    return path


def audit(path):
    with fits.open(path, memmap=False) as f:
        tables = []
        for h in f:
            if not isinstance(h, fits.BinTableHDU):
                continue
            columns = {c.name.lower(): c.name for c in h.columns}
            target = {}
            for key in [
                "teff",
                "logg",
                "fe_h",
                "feh",
                "teff_adop",
                "logg_adop",
                "feh_adop",
                "teff_lasp",
                "logg_lasp",
                "feh_lasp",
                "alpha_m_lasp",
            ]:
                if key in columns:
                    value = np.asarray(h.data[columns[key]], dtype="f8")
                    target[key] = dict(
                        finite=int((np.isfinite(value) & (value != -9999)).sum())
                    )
            tables.append(
                dict(
                    hdu=h.name,
                    rows=int(h.header["NAXIS2"]),
                    columns=[
                        dict(name=c.name, format=c.format, unit=c.unit)
                        for c in h.columns
                    ],
                    native_target_availability=target,
                )
            )
        return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.output).free < 100_000_000_000:
        raise RuntimeError("Need 100GB storage reserve")
    result = dict(
        status="running",
        training_performed=False,
        model_loaded=False,
        sources={},
        scope="Native products and schema audit only; no atmospheric-label quality admission, Gaia association or training activation. DR20 redirects legacy survey products to final older releases.",
    )

    def save():
        (a.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    with urllib.request.urlopen(CHECKSUM_URL, timeout=60) as response:
        checksums = response.read().decode()
    (a.output / "dr20-publisher.sha1sum").write_text(checksums)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(download, name, spec, a.output): name
            for name, spec in FILES.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                path = future.result()
                entry = dict(
                    path=str(path),
                    sha256=digest(path),
                    bytes=path.stat().st_size,
                    url=FILES[name]["url"],
                )
                if name == "dr20_bossnet":
                    expected = next(
                        line.split()[0]
                        for line in checksums.splitlines()
                        if line.split()[-1] == "astraAllStarBossNet-0.8.1.fits.gz"
                    )
                    if digest(path, "sha1") != expected:
                        raise ValueError("DR20 publisher checksum differs")
                    entry["publisher_sha1_verified"] = True
                    raw = Path("/scratch/dr20-bossnet.fits")
                    with gzip.open(path, "rb") as x, raw.open("xb") as y:
                        shutil.copyfileobj(x, y, 8 * 1024 * 1024)
                else:
                    raw = path
                entry.update(status="native_source_schema_complete", schema=audit(raw))
                result["sources"][name] = entry
            except Exception as e:
                result["sources"][name] = dict(status="failed", error=repr(e))
            save()
            print(
                json.dumps(dict(source=name, status=result["sources"][name]["status"])),
                flush=True,
            )
    if (
        digest(MRS)
        != "e10551be296cddcacf0ceb9c9b2827884d10e550b14850adb01348435be12369"
    ):
        raise ValueError("DR12 MRS archive differs")
    raw = Path("/scratch/lamost-dr12-mrs.fits")
    with gzip.open(MRS, "rb") as x, raw.open("xb") as y:
        shutil.copyfileobj(x, y, 8 * 1024 * 1024)
    result["sources"]["lamost_dr12_mrs"] = dict(
        status="native_source_schema_complete",
        schema=audit(raw),
        archive_sha256=digest(MRS),
    )
    result["status"] = (
        "native_source_audit_complete"
        if all(
            v["status"] == "native_source_schema_complete"
            for v in result["sources"].values()
        )
        else "partial_failure"
    )
    save()


if __name__ == "__main__":
    main()
