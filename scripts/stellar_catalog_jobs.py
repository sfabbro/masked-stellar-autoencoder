"""CPU catalogue inventory, source-aware label registry and experiment split plans.

All output is new/versioned; input catalogues and benchmark are read-only.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy.io import fits


def atomic_json(path, value):
    tmp = path.with_suffix(".building")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def verify_expected_fits_rows(hdul, expected_rows):
    """Reject truncated TAP exports before publishing them as full catalogues."""
    if expected_rows is None:
        return
    tables = [h for h in hdul if isinstance(h, fits.BinTableHDU)]
    if len(tables) != 1 or len(tables[0].data) != int(expected_rows):
        raise ValueError("FITS catalogue row count differs from expected full release")


def identity_split(ids):
    x = np.asarray(ids, dtype=np.uint64) ^ np.uint64(42)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return np.where(x % 1000 < 700, 0, np.where(x % 1000 < 850, 1, 2)).astype("u1")


def inventory(root):
    found = []
    for base, dirs, files in os.walk(root):
        depth = len(Path(base).relative_to(root).parts)
        dirs[:] = sorted(dirs) if depth < 3 else []
        for name in sorted(files):
            if any(
                s in name.lower()
                for s in [
                    "galah",
                    "apogee",
                    "allstar",
                    "astronn",
                    "li_et",
                    "pigs",
                    "pristine",
                    "mwsall",
                    "ftset_spec",
                ]
            ):
                p = Path(base) / name
                row = {"path": str(p), "bytes": p.stat().st_size}
                if name.endswith((".fits", ".fits.gz", ".fit")):
                    try:
                        with fits.open(p, memmap=True, lazy_load_hdus=True) as f:
                            table = next(
                                x for x in f if isinstance(x, fits.BinTableHDU)
                            )
                            row.update(
                                rows=table.header["NAXIS2"],
                                columns=table.columns.names,
                                units={c.name: c.unit for c in table.columns if c.unit},
                            )
                    except Exception as error:
                        row["schema_error"] = str(error)
                found.append(row)
    return found


def selected_native(table, source):
    names = {n.lower(): n for n in table.names}

    def col(aliases, default=np.nan):
        n = next((names[a.lower()] for a in aliases if a.lower() in names), None)
        return np.asarray(table[n]) if n else np.full(len(table), default)

    ids = col(["source_id", "gaiadr3_source_id", "gaiaedr3_source_id"], 0).astype("i8")
    value = col(["fe_h", "feh", "fe_h_corr"]).astype("f8")
    error = col(["e_fe_h", "fe_h_err", "e_feh"]).astype("f8")
    eligible = (
        (ids > 0)
        & np.isfinite(value)
        & np.isfinite(error)
        & (error > 0)
        & (error < 0.5)
    )
    if source == "galah":
        if not {"flag_sp", "flag_fe_h"} <= names.keys():
            eligible[:] = False
        else:
            eligible &= (col(["flag_sp"]) == 0) & (col(["flag_fe_h"]) == 0)
    elif source == "apogee":
        # Native FE_H only, never M_H. Require flags; preserve rejected records too.
        if not {"aspcapflag", "starflag", "fe_h_flag"} <= names.keys():
            eligible[:] = False
        else:
            eligible &= (
                (col(["aspcapflag"]) == 0)
                & (col(["starflag"]) == 0)
                & (col(["fe_h_flag"]) == 0)
            )
    return ids, value, error, eligible


def build_registry(args, inv, scratch):
    # Retain every source measurement; do not pick the first catalogue row.
    dtype = np.dtype(
        [
            ("source_id", "<i8"),
            ("value", "<f8"),
            ("uncertainty", "<f8"),
            ("quantity", "S8"),
            ("survey", "S32"),
            ("input_file", "S512"),
            ("source_row", "<i8"),
            ("quality_pass", "?"),
            ("split", "u1"),
        ]
    )
    staged = scratch / "labels.h5.building"
    totals = {}
    native = []
    legacy = Path(args.legacy)
    with h5py.File(staged, "x") as out:
        ds = out.create_dataset(
            "measurements",
            shape=(0,),
            maxshape=(None,),
            dtype=dtype,
            chunks=(16384,),
            compression="gzip",
            compression_opts=1,
        )
        files = [(legacy, "legacy")]
        for item in inv:
            path = Path(item["path"])
            low = path.name.lower()
            if path == legacy or not low.endswith(".fits"):
                continue
            survey = (
                "galah"
                if "galah_dr4" in low
                else "apogee"
                if "sdss-apogee" in low or low.startswith("allstar")
                else None
            )
            if survey:
                files.append((path, survey))
        for index, (path, survey) in enumerate(files):
            local = scratch / f"input-{index}.fits"
            shutil.copyfile(path, local)
            with fits.open(local, memmap=True) as f:
                table = next(x for x in f if isinstance(x, fits.BinTableHDU)).data
                for start in range(0, len(table), 65536):
                    block = table[start : start + 65536]
                    if survey == "legacy":
                        ids = np.asarray(block["source_id"], dtype="i8")
                        value = np.asarray(block["fe_h"], dtype="f8")
                        error = np.asarray(block["e_fe_h"], dtype="f8")
                        tags = np.char.strip(
                            np.asarray(block["spec_source"]).astype(str)
                        )
                        quantity = np.where(tags == "apogee-latentage", "MH", "FeH")
                        eligible = (
                            (tags == "li_et_al_vmps")
                            & (ids > 0)
                            & np.isfinite(value)
                            & np.isfinite(error)
                            & (error > 0)
                            & (error < 0.5)
                        )
                        # Legacy GALAH quality flags were discarded; never claim quality controlled.
                    else:
                        ids, value, error, eligible = selected_native(block, survey)
                        tags = np.full(len(block), survey)
                        quantity = np.full(len(block), "FeH")
                    records = np.zeros(len(block), dtype=dtype)
                    for key, v in [
                        ("source_id", ids),
                        ("value", value),
                        ("uncertainty", error),
                        ("quantity", quantity.astype("S8")),
                        ("survey", tags.astype("S32")),
                        ("input_file", str(path).encode()),
                        ("source_row", np.arange(start, start + len(block))),
                        ("quality_pass", eligible),
                        ("split", identity_split(ids)),
                    ]:
                        records[key] = v
                    n = len(ds)
                    ds.resize(n + len(records), axis=0)
                    ds[n:] = records
                    for tag in np.unique(tags):
                        select = tags == tag
                        t = totals.setdefault(
                            str(tag), {"measurements": 0, "quality_pass": 0}
                        )
                        t["measurements"] += int(select.sum())
                        t["quality_pass"] += int((select & eligible).sum())
                native.append({"path": str(path), "source": survey, "rows": len(table)})
            local.unlink()
        out.attrs["chemical_definition"] = (
            "FeH and MH retained separately; survey systematics not silently corrected"
        )
        out.attrs["age_policy"] = (
            "Legacy synthetic sqrt(age) errors are excluded; age target assembly pending native quality audit"
        )
        out.attrs["distance_policy"] = (
            "No inverse-parallax distance labels; probabilistic distance targets require independent quality metadata"
        )
        # Unique selected FeH stars, priority high-resolution Li then flagged native surveys.
        candidates = ds[:]
        ok = candidates["quality_pass"] & (candidates["quantity"] == b"FeH")
        good = candidates[ok]
        rank = np.where(
            good["survey"] == b"li_et_al_vmps",
            0,
            np.where(good["survey"] == b"galah", 1, 2),
        )
        order = np.lexsort((good["uncertainty"], rank, good["source_id"]))
        good = good[order]
        keep = (
            np.r_[True, good["source_id"][1:] != good["source_id"][:-1]]
            if len(good)
            else np.array([], dtype=bool)
        )
        out.create_dataset("selected/labels", data=good[keep], compression="gzip")
        out.attrs["selected_rows"] = int(keep.sum())
    destination = args.output / "labels.h5"
    copying = args.output / "labels.h5.copying"
    if destination.exists() or copying.exists():
        raise FileExistsError(destination)
    shutil.copyfile(staged, copying)
    with h5py.File(copying, "r") as f:
        assert len(f["measurements"]) == sum(v["measurements"] for v in totals.values())
        selected = len(f["selected/labels"])
        assert len(np.unique(f["selected/labels"]["source_id"])) == selected
    copying.replace(destination)
    return {
        "label_file": str(destination),
        "bytes": destination.stat().st_size,
        "sources": totals,
        "inputs": native,
        "selected_FeH_stars": selected,
        "native_galah_apogee_complete": all(
            any(x["source"] == s for x in native) for s in ["galah", "apogee"]
        ),
        "age_distance_status": "native labels and independent validation pending; no fabricated targets",
    }


def collect(args):
    spec = yaml.safe_load(Path(args.sources).read_text())
    results = []
    selected = set(filter(None, os.environ.get("MSA_CATALOG_SOURCES", "").split(",")))
    known = {item["name"] for item in spec["sources"]}
    if selected - known:
        raise ValueError(f"Unknown catalogue sources: {sorted(selected - known)}")
    for item in spec["sources"]:
        if selected and item["name"] not in selected:
            continue
        result = dict(item)
        try:
            limit = int(item.get("max_bytes", args.max_bytes))
            request = urllib.request.Request(
                item["url"],
                method="HEAD",
                headers={"User-Agent": "MSA-science-catalogue-audit/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=40) as response:
                    result.update(
                        http_status=response.status,
                        content_length=response.headers.get("Content-Length"),
                        content_type=response.headers.get("Content-Type"),
                    )
            except Exception as error:
                result["head_error"] = str(error)
            if item.get("download") and int(result.get("content_length") or 0) <= limit:
                published = args.output / item["filename"]
                target = (
                    Path(
                        os.environ.get(
                            "MSA_LABEL_SCRATCH", "/scratch/msa-label-staging"
                        )
                    )
                    / args.run_id
                    / item["filename"]
                )
                partial = target.with_suffix(target.suffix + ".copying")
                if published.exists():
                    raise FileExistsError(published)
                if target.exists() or partial.exists():
                    raise FileExistsError(target)
                with (
                    urllib.request.urlopen(item["url"], timeout=180) as source,
                    partial.open("xb") as dest,
                ):
                    digest = hashlib.sha256()
                    size = 0
                    while chunk := source.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(
                                "Download exceeds bounded per-product limit"
                            )
                        digest.update(chunk)
                        dest.write(chunk)
                if (
                    item.get("expected_sha256")
                    and digest.hexdigest() != item["expected_sha256"]
                ):
                    raise ValueError("Catalogue checksum mismatch")
                partial.replace(target)
                result.update(
                    file=str(published), sha256=digest.hexdigest(), actual_bytes=size
                )
                if target.name.endswith(".fits"):
                    with fits.open(target, memmap=True) as hdul:
                        verify_expected_fits_rows(hdul, item.get("expected_fits_rows"))
                        result["fits_tables"] = [
                            {
                                "name": h.name,
                                "rows": h.header.get("NAXIS2"),
                                "columns": h.columns.names,
                                "units": {c.name: c.unit for c in h.columns if c.unit},
                            }
                            for h in hdul
                            if isinstance(h, fits.BinTableHDU)
                        ]
                        # Candidate discovery only: keep raw uncertainties/flags and original row.
                        for h in hdul:
                            if not isinstance(h, fits.BinTableHDU):
                                continue
                            columns = {c.lower(): c for c in h.columns.names}
                            metal = next(
                                (
                                    columns[k]
                                    for k in ["feh", "fe_h", "[fe/h]"]
                                    if k in columns
                                ),
                                None,
                            )
                            if metal is None:
                                continue
                            value = np.asarray(h.data[metal])
                            rows = np.flatnonzero(
                                np.isfinite(value) & (value < -2) & (value > -6)
                            )
                            rng = np.random.default_rng(42)
                            rng.shuffle(rows)
                            rows = rows[:50000]
                            rows.sort()
                            if len(rows):
                                candidate = target.with_name(
                                    target.stem + "-candidate-sample.fits"
                                )
                                fits.BinTableHDU(h.data[rows], name=h.name).writeto(
                                    candidate, overwrite=False
                                )
                                result["candidate_sample"] = {
                                    "path": str(candidate),
                                    "rows": len(rows),
                                    "original_row_indices_file": str(candidate)
                                    + ".rows.npy",
                                    "selection": "seed42 uniform among nominal raw metallicity<-2; uncorrected values, retain original uncertainties/flags; not independent XP truth",
                                }
                                np.save(str(candidate) + ".rows.npy", rows)
                            break
                raw_copy = published.with_suffix(published.suffix + ".copying")
                if raw_copy.exists():
                    raise FileExistsError(raw_copy)
                shutil.copyfile(target, raw_copy)
                raw_copy.replace(published)
                if result.get("candidate_sample"):
                    candidate = Path(result["candidate_sample"]["path"])
                    sample_output = args.output / candidate.name
                    shutil.copyfile(candidate, sample_output)
                    shutil.copyfile(
                        str(candidate) + ".rows.npy", str(sample_output) + ".rows.npy"
                    )
                    result["candidate_sample"].update(
                        path=str(sample_output),
                        original_row_indices_file=str(sample_output) + ".rows.npy",
                    )
            else:
                result["download_status"] = "metadata_only_or_exceeds_limit"
        except Exception as error:
            result["error"] = str(error)
        results.append(result)
        atomic_json(args.output / "collection.json", results)
    return {
        "sources": results,
        "scientific_ingestion_complete": False,
        "note": "Raw candidate samples require verified survey-specific ID joins, quality flags, uncertainties and correction application before scientific labels are eligible",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=[
            "inventory",
            "registry",
            "collect",
            "datasets",
            "posterior-labels",
            "dataset-view",
            "desi-labels",
            "pigs-labels",
            "pristine-async",
            "survey-overlap",
            "pristine-quality-view",
            "enriched-label-view",
        ],
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", default="catalog")
    p.add_argument("--root", type=Path, default=Path("/arc/projects/k-pop/catalogues"))
    p.add_argument(
        "--legacy",
        default="/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits",
    )
    p.add_argument("--sources", default="configs/stellar-catalog-sources.yaml")
    p.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = (
        Path(os.environ.get("MSA_LABEL_SCRATCH", "/scratch/msa-label-staging"))
        / args.run_id
    )
    scratch.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "mode": args.mode,
        "run_id": args.run_id,
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    atomic_json(args.output / "manifest.json", manifest)
    try:
        if args.mode == "collect":
            result = collect(args)
        elif args.mode == "posterior-labels":
            from stellar_dataset_views import posterior_tables

            result = posterior_tables(
                args.output,
                scratch,
                [
                    "/arc/projects/k-pop/catalogues/starhorse_ages/apogee17.fits",
                    "/arc/projects/k-pop/catalogues/starhorse_ages/galahdr3.fits",
                ],
            )
        elif args.mode == "enriched-label-view":
            from stellar_enriched_views import build_view

            result = build_view(args.output, scratch)
        elif args.mode == "pristine-quality-view":
            from pristine_quality_views import build_view

            result = build_view(args.output, scratch)
        elif args.mode == "survey-overlap":
            from stellar_survey_overlap import build_audit

            result = build_audit(args.output, scratch)
        elif args.mode == "pristine-async":
            from pristine_async_fetch import build_catalogue

            result = build_catalogue(args.output, scratch)
        elif args.mode == "pigs-labels":
            from pigs_stellar_labels import build_labels

            result = build_labels(args.output, scratch)
        elif args.mode == "desi-labels":
            from desi_stellar_labels import build_labels

            result = build_labels(args.output, scratch)
        elif args.mode == "dataset-view":
            from stellar_dataset_views import dataset_view

            result = dataset_view(
                args.output, scratch, os.environ["MSA_LABEL_REGISTRY"]
            )
        else:
            inv = inventory(args.root)
            for extra in filter(
                None, os.environ.get("MSA_NATIVE_LABEL_FILES", "").split(",")
            ):
                path = Path(extra)
                if not path.is_file():
                    raise FileNotFoundError(path)
                inv.append({"path": str(path), "bytes": path.stat().st_size})
            atomic_json(args.output / "inventory.json", inv)
            result = {"files": len(inv)}
            if args.mode == "registry":
                result = build_registry(args, inv, scratch)
            if args.mode == "datasets":
                result = {
                    "parent_benchmark": "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5",
                    "train_stars": 9000000,
                    "validation_stars": 1000000,
                    "presentations": 20000000,
                    "split_policy": "source-ID hash, DEC sky AND survey holdouts, preserve all rare training stars before caps",
                    "metallicity": "FeH and MH separate; quality flags and measured uncertainties required",
                    "age": "independent quality-controlled ages; exclude synthetic errors and proxy label leakage",
                    "distance": "hide parallax and aliases for spectrophotometric task; separate parallax-assisted task; parallax likelihood rather than inverse parallax",
                    "candidate_policy": "XP/photometric metallicities are selection/weak labels, not independent XP evaluation truth",
                    "status": "design_only_not_materialized",
                }
                atomic_json(args.output / "dataset-design.json", result)
        manifest.update(
            status="complete",
            result=result,
            finished_utc=dt.datetime.now(dt.UTC).isoformat(),
        )
        atomic_json(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        atomic_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
