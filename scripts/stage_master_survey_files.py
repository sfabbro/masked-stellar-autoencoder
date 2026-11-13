"""Download the public survey catalogues the master spectroscopic table reads.

Writes one folder with fixed file names (what build_master_spectroscopic_labels
expects under ``--stage``) and a manifest with URL, bytes and sha256 per file.
GALAH DR4 and Gaia GSP-Spec are read from /arc/projects/k-pop/catalogues and
are not downloaded here. Existing complete files are kept (resumable).
"""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

LAMOST = "https://www.lamost.org/dr12/v2.0/"
SAS = "https://data.sdss.org/sas/"
RAVE = "https://www.rave-survey.org/files/csv/"
RAVE_XMATCH = "SELECT TOP 600000 rave_obs_id,source_id,dist FROM ravedr6.dr6_x_gaiaedr3"
FILES = {
    "dr12_lrs_stellar.fits.gz": LAMOST + "catdl?name=dr12_v2.0_LRS_stellar.fits.gz",
    "dr12_mrs_stellar.fits.gz": LAMOST
    + "medcas/catdl?name=dr12_v2.0_MRS_stellar.fits.gz",
    "mwsall-pix-iron.fits": "https://data.desi.lbl.gov/public/dr1/vac/dr1/mws/iron/"
    "v1.0/mwsall-pix-iron.fits",
    "allStar-dr17-synspec_rev1.fits": SAS
    + "dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits",
    "dr20_bossnet.fits.gz": SAS
    + "dr20/spectro/astra/0.8.1/summary/astraAllStarBossNet-0.8.1.fits.gz",
    "madera.csv.gz": RAVE + "ravedr6_dr6_madera.csv.gz",
    "iron.csv.gz": RAVE + "ravedr6_dr6_gauguin_madera.csv.gz",
    "crossmatch.xml": "https://www.rave-survey.org/tap/sync?"
    + urllib.parse.urlencode(dict(REQUEST="doQuery", LANG="ADQL", QUERY=RAVE_XMATCH)),
}
# Sizes published by LAMOST (lamost_dr12_source.PRODUCTS); others are checked
# against the Content-Length the server sends.
EXPECTED_BYTES = {
    "dr12_lrs_stellar.fits.gz": 1221786815,
    "dr12_mrs_stellar.fits.gz": 725541580,
}


def fetch(url, path, expected=None, attempts=4):
    partial = path.with_name(path.name + ".partial")
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "msa-master-stage/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as src:
                announced = int(src.headers.get("Content-Length", "-1"))
                if expected is not None and announced not in (-1, expected):
                    raise ValueError(
                        f"{path.name}: server size {announced} != {expected}"
                    )
                digest, total = hashlib.sha256(), 0
                with partial.open("wb") as out:
                    while block := src.read(8 * 2**20):
                        out.write(block)
                        digest.update(block)
                        total += len(block)
            if (expected is not None and total != expected) or (
                announced > 0 and total != announced
            ):
                raise ValueError(f"{path.name}: truncated ({total} bytes)")
            partial.replace(path)
            return dict(url=url, bytes=total, sha256=digest.hexdigest())
        except Exception as exc:  # retry network failures with backoff
            print(json.dumps(dict(file=path.name, attempt=attempt, error=repr(exc))))
            if attempt == attempts:
                raise
            time.sleep(2 ** (attempt + 2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    path = a.output / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else dict(files={})
    m["status"] = "staging"
    for name, url in FILES.items():
        if a.only and name not in a.only:
            continue
        done = m["files"].get(name)
        if (
            done
            and (a.output / name).exists()
            and ((a.output / name).stat().st_size == done["bytes"])
        ):
            continue
        m["files"][name] = fetch(url, a.output / name, EXPECTED_BYTES.get(name))
        path.write_text(json.dumps(m, indent=2) + "\n")
        print(json.dumps(dict(staged=name, **m["files"][name])), flush=True)
    m["status"] = "complete"
    path.write_text(json.dumps(m, indent=2) + "\n")


if __name__ == "__main__":
    main()
