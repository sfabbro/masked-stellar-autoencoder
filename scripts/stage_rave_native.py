"""Persist the audited RAVE candidate on CANFAR using publisher source files."""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from astropy.table import Table

from masked_stellar_autoencoder.training.survey_expansion_sources import (
    write_rave_candidate,
)

FILES = {
    "madera.csv.gz": (
        "https://www.rave-survey.org/files/csv/ravedr6_dr6_madera.csv.gz",
        "53e11b82aea7302a614bd9872a3622f51340efa68b75f0f3bf115035a501b0a6",
    ),
    "iron.csv.gz": (
        "https://www.rave-survey.org/files/csv/ravedr6_dr6_gauguin_madera.csv.gz",
        "bb903b093ca397c9b2c7f4f80820153236a1df6712ed150175db853eb9431255",
    ),
}


def fetch(url, path, expected=None):
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=30) as response, path.open("xb") as stream:
        while chunk := response.read(2**20):
            size += len(chunk)
            if size > 200_000_000:
                raise ValueError("Source download exceeds 200MB bound")
            stream.write(chunk)
            digest.update(chunk)
    if expected and digest.hexdigest() != expected:
        raise ValueError("Publisher-file content differs from audited snapshot")
    return dict(url=url, bytes=size, sha256=digest.hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not Path("/arc/projects/k-pop").is_dir():
        raise RuntimeError("Run native staging on CANFAR")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    ids = np.asarray(json.loads(args.benchmarks.read_text()), dtype="i8")
    manifest = dict(
        status="downloading", training_performed=False, model_loaded=False, sources={}
    )

    def save():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    for name, (url, expected) in FILES.items():
        manifest["sources"][name] = fetch(url, out / name, expected)
        save()
        print(json.dumps(dict(downloaded=name)), flush=True)
    query = "SELECT TOP 600000 rave_obs_id,source_id,dist FROM ravedr6.dr6_x_gaiaedr3"
    url = "https://www.rave-survey.org/tap/sync?" + urllib.parse.urlencode(
        dict(REQUEST="doQuery", LANG="ADQL", QUERY=query)
    )
    manifest["sources"]["crossmatch.xml"] = fetch(url, out / "crossmatch.xml")
    table = Table.read(out / "crossmatch.xml", format="votable")
    if len(table) != 517357:
        raise ValueError("RAVE crossmatch changed or was truncated")
    manifest["sources"]["crossmatch.xml"]["rows"] = len(table)
    save()
    block, receipt = write_rave_candidate(
        out / "madera.csv.gz",
        out / "iron.csv.gz",
        out / "crossmatch.xml",
        out / "rave-candidate.h5",
        ids,
    )
    manifest.update(
        status="native_rave_candidate_complete",
        receipt=receipt,
        selected_unique_stars=len(block["source_id"]),
        selected_per_target=block["observed"].sum(0).tolist(),
        benchmark_reserved=int(block["benchmark_reserved"].sum()),
        training_enabled=False,
    )
    save()
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
