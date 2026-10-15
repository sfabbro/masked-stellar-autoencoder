"""PIGS spectroscopy and derived distance posteriors with native flags/errors."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from astropy.table import Table

ROOT = Path(
    "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/candidate_collection_v2"
)


def numeric(table, name, integer=False):
    column = np.ma.asarray(table[name])
    return np.asarray(
        column.filled(0 if integer else np.nan), dtype="i8" if integer else "f8"
    )


def text(table, name):
    return np.char.strip(np.asarray(np.ma.asarray(table[name]).filled("")).astype(str))


def raw_table(table):
    """Preserve every CDS field and its missingness; HDF5 strings are ASCII here."""
    fields = []
    for name in table.colnames:
        dtype = table[name].dtype
        fields.append((name, f"S{dtype.itemsize // 4}" if dtype.kind == "U" else dtype))
    values = np.empty(len(table), dtype=fields)
    missing = np.zeros((len(table), len(table.colnames)), bool)
    for index, name in enumerate(table.colnames):
        column = np.ma.asarray(table[name])
        missing[:, index] = np.ma.getmaskarray(column)
        fill = (
            ""
            if column.dtype.kind in "SU"
            else np.nan
            if column.dtype.kind == "f"
            else 0
        )
        values[name] = column.filled(fill)
    return values, missing


def spectroscopy_measurements(table):
    dtype = [
        ("source_id", "<i8"),
        ("value", "<f8"),
        ("uncertainty", "<f8"),
        ("quantity", "S8"),
        ("survey", "S16"),
        ("pipeline", "S16"),
        ("source_row", "<i8"),
        ("quality_pass", "?"),
        ("split", "u1"),
    ]
    from stellar_catalog_jobs import identity_split

    ids = numeric(table, "GaiaDR3", integer=True)
    single = np.char.lower(text(table, "double")) == "false"
    records = []
    for suffix, pipeline in [("F", "FERRE"), ("U", "ULySS")]:
        value = numeric(table, f"[Fe/H]{suffix}")
        error = numeric(table, f"e_[Fe/H]{suffix}tot")
        good = np.char.lower(text(table, "good" + suffix)) == "true"
        good &= (
            single
            & (ids > 0)
            & np.isfinite(value)
            & np.isfinite(error)
            & (error > 0)
            & (error < 0.5)
        )
        block = np.zeros(len(table), dtype=dtype)
        for name, data in dict(
            source_id=ids,
            value=value,
            uncertainty=error,
            quantity=b"FeH",
            survey=b"PIGS",
            pipeline=pipeline.encode(),
            source_row=np.arange(len(table)),
            quality_pass=good,
            split=identity_split(ids),
        ).items():
            block[name] = data
        records.append(block)
    return np.concatenate(records)


def join_distances(spectroscopy, distances):
    """Require an unambiguous name-to-Gaia-ID mapping and a bounded repeated-FeH agreement audit."""
    mapping = {}
    ids = numeric(spectroscopy, "GaiaDR3", integer=True)
    feh = numeric(spectroscopy, "[Fe/H]F")
    for row, name in enumerate(text(spectroscopy, "Star")):
        mapping.setdefault(name, []).append(row)
    target_ids = np.zeros(len(distances), dtype="i8")
    spec_rows = np.full(len(distances), -1, dtype="i8")
    repeated_feh = numeric(distances, "[Fe/H]")
    for row, name in enumerate(text(distances, "Star")):
        candidates = np.asarray(mapping.get(name, []), dtype=int)
        unique = np.unique(ids[candidates])
        if len(unique) != 1 or unique[0] <= 0:
            continue
        consistent = candidates[
            np.isclose(feh[candidates], repeated_feh[row], rtol=0, atol=0.002)
        ]
        if len(consistent):
            target_ids[row] = unique[0]
            spec_rows[row] = consistent[0]
    percentiles = np.column_stack(
        [numeric(distances, "dist" + s) for s in ["16", "50", "84"]]
    )
    valid = (
        (target_ids > 0) & np.isfinite(percentiles).all(1) & (percentiles > 0).all(1)
    )
    valid &= (np.diff(percentiles, axis=1) >= 0).all(1)
    return target_ids, spec_rows, percentiles, valid


def build_labels(output, scratch, root=ROOT):
    from stellar_dataset_views import publish

    root = Path(root)
    collection = json.loads((root / "collection.json").read_text())
    inputs = {}
    for name in ["pigs-tabled2.dat.gz", "pigs-tabled3.dat.gz", "pigs-ReadMe"]:
        source = root / name
        expected = next(
            item["sha256"] for item in collection if item.get("filename") == name
        )
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"PIGS collected-file digest mismatch: {name}")
        inputs[name] = expected
    shutil.copyfile(root / "pigs-ReadMe", scratch / "ReadMe")
    tables = []
    for number in [2, 3]:
        local = scratch / f"tabled{number}.dat"
        with (
            gzip.open(root / f"pigs-tabled{number}.dat.gz", "rb") as source,
            local.open("xb") as target,
        ):
            shutil.copyfileobj(source, target)
        tables.append(
            Table.read(local, format="ascii.cds", readme=str(scratch / "ReadMe"))
        )
    spec, distance = tables
    for name in ["dist16", "dist50", "dist84"]:
        if str(distance[name].unit) != "kpc":
            raise ValueError(f"Unexpected PIGS distance units: {name}")
    records = spectroscopy_measurements(spec)
    good = records[records["quality_pass"]]
    rank = np.where(good["pipeline"] == b"FERRE", 0, 1)
    good = good[np.lexsort((good["uncertainty"], rank, good["source_id"]))]
    keep = (
        np.r_[True, np.diff(good["source_id"]) != 0]
        if len(good)
        else np.array([], bool)
    )
    selected = good[keep]
    ids, spec_rows, percentiles, valid = join_distances(spec, distance)
    if len(distance) and not (ids > 0).any():
        raise ValueError("PIGS distance-name/Gaia/metallicity join produced no matches")
    repeated_feh_delta = np.where(
        spec_rows >= 0,
        numeric(distance, "[Fe/H]")
        - numeric(spec, "[Fe/H]F")[np.maximum(spec_rows, 0)],
        np.nan,
    )
    staged = scratch / "pigs-labels.h5.building"
    with h5py.File(staged, "x") as out:
        out.create_dataset("measurements", data=records, compression="gzip")
        out.create_dataset("selected/labels", data=selected, compression="gzip")
        for group, table in [("raw/spectroscopy", spec), ("raw/distances", distance)]:
            values, missing = raw_table(table)
            ds = out.create_dataset(group + "/records", data=values, compression="gzip")
            ds.attrs["columns"] = json.dumps(table.colnames)
            ds.attrs["units"] = json.dumps(
                {
                    n: str(table[n].unit)
                    for n in table.colnames
                    if table[n].unit is not None
                }
            )
            out.create_dataset(group + "/missing", data=missing, compression="gzip")
        for name, data in [
            ("source_id", ids),
            ("source_spectroscopy_row", spec_rows),
            ("repeated_feh_delta_dex", repeated_feh_delta),
            ("percentiles", percentiles),
            ("posterior_valid_and_joined", valid),
        ]:
            out.create_dataset(
                "distance_posteriors/" + name, data=data, compression="gzip"
            )
        out["distance_posteriors/percentiles"].attrs.update(
            unit="kpc", percentiles=[16, 50, 84]
        )
        out.attrs.update(
            complete=True,
            content_id=inputs["pigs-tabled2.dat.gz"],
            input_files_sha256=json.dumps(inputs),
            provenance="CDS J/MNRAS/530/3391 tablesD2/D3, all native fields/masks retained",
            uncertainty_definition="Published FERRE/ULySS total FeH errors already include0.13dex floor; do NOT add the floor again. Pipelines are correlated and are not averaged.",
            quality_policy="Native goodF/goodU=true; double=false; positive GaiaDR3 ID; finite positive FeH error<0.5dex. FERRE preferred before ULySS; duplicate GaiaIDs resolved by pipeline then smaller error.",
            distance_caveat="StarHorse model-dependent posteriors, not independent geometric truth. Exact Star-name join also checks GaiaID uniqueness and FERRE metallicity agreement within0.002dex. Actual D3-minus-D2 offset~0.0008903dex is preserved per row; values are not shifted. Posterior validity is not a complete distance quality audit.",
        )
    destination = output / "pigs-labels.h5"
    publish(staged, destination)
    return dict(
        file=str(destination),
        bytes=destination.stat().st_size,
        input_files_sha256=inputs,
        spectroscopy_rows=len(spec),
        retained_measurements=len(records),
        selected_unique_FeH_stars=len(selected),
        selected_below_minus2=int((selected["value"] < -2).sum()),
        selected_below_minus3=int((selected["value"] < -3).sum()),
        distance_rows=len(distance),
        distance_ID_joined=int((ids > 0).sum()),
        distance_valid_posteriors=int(valid.sum()),
        repeated_feh_delta_dex_range=[
            float(np.nanmin(repeated_feh_delta)),
            float(np.nanmax(repeated_feh_delta)),
        ],
        independent_distance_truth=False,
        publication_roundtrip=True,
    )
