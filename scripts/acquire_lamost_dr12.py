"""Acquire publisher DR12 v2 LRS catalogue on ARC; verify gzip and native schema."""

import argparse
import gzip
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

from astropy.io import fits

URL = "https://www.lamost.org/dr12/v2.0/catdl?name=dr12_v2.0_LRS_stellar.fits.gz"
EXPECTED_ROWS = 8376135
REQUIRED = [
    "gaia_source_id",
    "snrg",
    "teff",
    "teff_err",
    "logg",
    "logg_err",
    "feh",
    "feh_err",
]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--existing-archive", type=Path)
    p.add_argument("--expected-archive-sha256")
    args = p.parse_args()
    if bool(args.existing_archive) != bool(args.expected_archive_sha256):
        p.error("Reuse requires both existing archive and its recorded SHA256")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 30_000_000_000:
        raise RuntimeError("Need 30GB filesystem reserve")
    compressed = out / "dr12_v2.0_LRS_stellar.fits.gz"
    manifest = dict(
        url=URL,
        training_performed=False,
        status="downloading",
        publisher_checksum_verified=False,
    )

    def save():
        (out / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    if args.existing_archive:
        compressed = args.existing_archive
        total = compressed.stat().st_size
        if total != 1221786815 or sha(compressed) != args.expected_archive_sha256:
            raise ValueError("Existing archive differs from recorded DR12 source")
        manifest.update(
            reused_archive=str(compressed), recorded_archive_sha256_verified=True
        )
    else:
        with (
            urllib.request.urlopen(URL, timeout=120) as response,
            compressed.open("xb") as target,
        ):
            if response.status != 200 or "fits" not in response.headers.get(
                "Content-Type", ""
            ):
                raise ValueError(
                    "Expected full public FITS download, not authentication HTML"
                )
            manifest.update(final_url=response.url, http_headers=dict(response.headers))
            expected = int(response.headers["Content-Length"])
            total = 0
            last = time.monotonic()
            while block := response.read(4 * 1024 * 1024):
                target.write(block)
                total += len(block)
                if time.monotonic() - last > 30:
                    print(
                        json.dumps(
                            dict(phase="download", bytes=total, expected=expected)
                        ),
                        flush=True,
                    )
                    last = time.monotonic()
            if total != expected:
                raise ValueError("Incomplete source transfer")
    manifest.update(
        status="decompressing",
        compressed_bytes=total,
        compressed_sha256=sha(compressed),
    )
    save()
    native = out / "dr12_v2.0_LRS_stellar.fits"
    with gzip.open(compressed, "rb") as source, native.open("xb") as target:
        shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    with fits.open(native, memmap=True) as hdus:
        table = hdus[1]
        if table.header["NAXIS2"] != EXPECTED_ROWS:
            raise ValueError("DR12 publisher row count differs")
        if set(REQUIRED) - set(table.columns.names):
            raise ValueError("Required native label/identity columns missing")
        # IDs may not pass through floating-point conversion.
        if table.data["gaia_source_id"].dtype.kind not in "iuSU":
            raise ValueError(
                "Native Gaia IDs must be exact integers or decimal strings"
            )
        manifest.update(
            rows=EXPECTED_ROWS,
            columns=[
                dict(name=c.name, format=c.format, unit=c.unit) for c in table.columns
            ],
        )
    manifest.update(
        status="verified_native_source",
        native_bytes=native.stat().st_size,
        native_sha256=sha(native),
        native_path=str(native),
        gzip_crc_verified=True,
    )
    save()
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
