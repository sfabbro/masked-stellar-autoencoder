"""Header-only source inventory for full-Gaia inputs, alpha and multi-source ages."""

import argparse
import csv
import gzip
import json
import os
import re
import time
from pathlib import Path

import h5py
import pyarrow.parquet as pq
from astropy.io import fits

ROOT = Path("/arc/projects/k-pop")
RUN = ROOT / "msa_runs/stellar-improvement-20261015"
INTEREST = (
    "gaia",
    "apogee",
    "galah",
    "lamost",
    "rave",
    "ambre",
    "desi",
    "boss",
    "segue",
    "mastar",
    "ges_",
    "gaiaeso",
    "gaia-eso",
    "age",
    "alpha",
    "phot",
    "starhorse",
    "skymapper",
    "panstarr",
    "sdss",
    "tmass",
    "wise",
)


def classify(columns):
    names = [str(c) for c in columns]
    return dict(
        identity=[
            c
            for c in names
            if any(
                k in c.lower()
                for k in [
                    "source_id",
                    "gaia",
                    "apogee_id",
                    "sobject_id",
                    "rave_obs_id",
                    "obsid",
                    "designation",
                ]
            )
        ],
        ages=[
            c
            for c in names
            if re.search(r"(^|_)(age|ages|logage|log_age|lnage|tau)(_|$)", c.lower())
        ],
        alpha_and_elements=[
            c
            for c in names
            if any(
                k in c.lower()
                for k in [
                    "alpha",
                    "mg_fe",
                    "si_fe",
                    "ca_fe",
                    "ti_fe",
                    "mg_h",
                    "si_h",
                    "ca_h",
                    "ti_h",
                    "mgfe",
                    "sife",
                ]
            )
        ],
        photometry=[
            c
            for c in names
            if any(
                k in c.lower()
                for k in [
                    "phot_",
                    "mag",
                    "smss",
                    "sdss",
                    "ps1",
                    "2mass",
                    "tmass",
                    "wise",
                ]
            )
            or c in ["G", "BP", "RP", "W1", "W2", "J", "H", "KS"]
        ],
        auxiliary=[
            c
            for c in names
            if c.lower()
            in [
                "ra",
                "dec",
                "ra_icrs",
                "de_icrs",
                "pmra",
                "pmdec",
                "parallax",
                "e_parallax",
                "parallax_error",
                "ebv",
                "e_ebv",
                "pmra_error",
                "pmdec_error",
            ]
            or "corr" in c.lower()
        ],
    )


def schema(path):
    lower = path.name.lower()
    if lower.endswith((".fits", ".fit", ".fits.gz")):
        tables = []
        with fits.open(
            path, memmap=not lower.endswith(".gz"), lazy_load_hdus=True
        ) as hdul:
            for h in hdul:
                if isinstance(h, fits.BinTableHDU | fits.TableHDU):
                    cols = h.columns.names
                    tables.append(
                        dict(
                            hdu=h.name,
                            rows=int(h.header.get("NAXIS2", 0)),
                            columns=cols,
                            units={c.name: c.unit for c in h.columns if c.unit},
                            candidate_fields=classify(cols),
                        )
                    )
        return dict(format="fits", tables=tables, header_only=True)
    if lower.endswith(".parquet") or lower in ["_metadata", "_common_metadata"]:
        f = pq.ParquetFile(path)
        cols = f.schema_arrow.names
        return dict(
            format="parquet",
            rows=f.metadata.num_rows,
            columns=cols,
            candidate_fields=classify(cols),
            header_only=True,
        )
    if lower.endswith((".h5", ".hdf5")):
        datasets = []
        with h5py.File(path) as f:
            keys = list(f)

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and len(datasets) < 24:
                    cols = list(obj.dtype.names or [])
                    datasets.append(
                        dict(
                            name=name,
                            shape=list(obj.shape),
                            dtype=str(obj.dtype)[:300],
                            fields=cols,
                            candidate_fields=classify(cols),
                        )
                    )

            f.visititems(visit)
        return dict(
            format="hdf5",
            root_keys=keys,
            datasets=datasets,
            header_only=True,
            dataset_limit=24,
        )
    if lower.endswith((".csv", ".csv.gz")):
        op = gzip.open if lower.endswith(".gz") else open
        with op(path, "rt") as f:
            text = f.readline(1024 * 1024)
        cols = next(csv.reader([text]))
        return dict(
            format="csv",
            columns=cols,
            candidate_fields=classify(cols),
            header_only=True,
            rows=None,
        )
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="inventory_running",
        training_performed=False,
        model_loaded=False,
        sources=[],
        scope="Header/schema discovery only. Candidate age/alpha fields require semantic, unit, quality and provenance audits before training. No XP criterion; no underlying source files modified.",
        roots=[],
        limits=dict(files=600, max_depth=4, max_seconds=1200),
        walk_complete=True,
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(m, indent=2) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    start = time.monotonic()
    paths = set()
    for root in [ROOT / "gaia", ROOT / "catalogues"]:
        m["roots"].append(str(root))
        if not root.exists():
            continue
        for base, dirs, files in os.walk(root):
            depth = len(Path(base).relative_to(root).parts)
            dirs[:] = (
                sorted(
                    d
                    for d in dirs
                    if not d.startswith(".") and d not in ["spectra", "xp_continuous"]
                )
                if depth < 4
                else []
            )
            for name in files:
                path = Path(base) / name
                if name.startswith(".") or not any(
                    k in str(path).lower() for k in INTEREST
                ):
                    continue
                if name.lower().endswith(
                    (
                        ".fits",
                        ".fit",
                        ".fits.gz",
                        ".parquet",
                        ".h5",
                        ".hdf5",
                        ".csv",
                        ".csv.gz",
                    )
                ) or name in ["_metadata", "_common_metadata"]:
                    paths.add(path)
            if len(paths) > 600:
                m["walk_complete"] = False
                break
    # Add known native files outside the catalogue root, avoiding model/checkpoint trees.
    for relative in [
        "apogee_native_fetch_v1/allStar-dr17-synspec_rev1.fits",
        "desi_collection_v1/mwsall-pix-iron.fits",
        "apokasc3_collection_v1/apokasc3-table4.dat.gz",
    ]:
        path = RUN / relative
        if path.exists() and ".dat" not in path.name:
            paths.add(path)
    for path in ROOT.glob("*.parquet"):
        paths.add(path)
    paths = sorted(
        paths,
        key=lambda p: (
            0
            if any(
                k in p.name.lower()
                for k in ["age", "allstar", "bdasp", "gaia_source", "nn_latent"]
            )
            else 1,
            str(p),
        ),
    )
    m["discovered_files"] = len(paths)
    save()
    for path in paths[:600]:
        if time.monotonic() - start > 1200:
            m["walk_complete"] = False
            break
        item = dict(path=str(path), bytes=path.stat().st_size)
        try:
            item.update(schema(path) or {})
        except Exception as e:
            item["schema_error"] = str(e)[:500]
        m["sources"].append(item)
        if len(m["sources"]) % 10 == 0:
            save()
    m.update(
        status="full_gaia_source_inventory_complete",
        inspected_files=len(m["sources"]),
        complete=m["walk_complete"] and len(paths) <= 600,
    )
    save()


if __name__ == "__main__":
    main()
