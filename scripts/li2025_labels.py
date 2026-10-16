"""Versioned CaT silver-label ingestion; never merges into the native gold registry."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import ascii
from stellar_catalog_jobs import atomic_json, identity_split

PAPER = "https://arxiv.org/html/2506.09705v1"
DTYPE = [
    ("source_id", "<i8"),
    ("source_row", "<i8"),
    ("obsid", "<i8"),
    ("ra_deg", "<f8"),
    ("dec_deg", "<f8"),
    ("snr", "<f8"),
    ("value", "<f8"),
    ("reported_uncertainty", "<f8"),
    ("published_ca_fe_allowance_dex", "<f8"),
    ("method", "S5"),
    ("cat_index_flag", "i1"),
    ("silver_eligible", "?"),
    ("gold_nll_eligible", "?"),
    ("split", "u1"),
]


def records_from_table(table):
    """Both CaT index definitions are valid; SNR>20 is an explicit local cut."""
    ids = np.asarray(table["Gaia"])
    if ids.dtype.kind not in "iu" or (ids > np.iinfo("i8").max).any():
        raise ValueError("Gaia DR3 IDs must be exact signed-range integers")
    out = np.zeros(len(table), dtype=DTYPE)
    for target, source in [
        ("source_id", "Gaia"),
        ("obsid", "obsid"),
        ("ra_deg", "RAdeg"),
        ("dec_deg", "DEdeg"),
        ("snr", "snr"),
        ("value", "[Fe/H]"),
        ("reported_uncertainty", "e_[Fe/H]"),
        ("method", "n_[Fe/H]"),
        ("cat_index_flag", "q_EW1"),
    ]:
        out[target] = (
            np.ma.asarray(table[source]).filled(np.nan)
            if out[target].dtype.kind == "f"
            else table[source]
        )
    out["source_row"] = np.arange(len(out))
    out["published_ca_fe_allowance_dex"] = 0.2
    out["split"] = identity_split(out["source_id"])
    valid = (out["source_id"] > 0) & np.isin(out["method"], [b"mag", b"color"])
    valid &= np.isin(out["cat_index_flag"], [0, 1])
    valid &= np.isfinite(out["value"]) & np.isfinite(out["reported_uncertainty"])
    valid &= out["reported_uncertainty"] > 0
    valid &= np.isfinite(out["snr"]) & (out["snr"] > 20)
    valid &= np.isfinite(out["ra_deg"]) & (out["ra_deg"] >= 0) & (out["ra_deg"] < 360)
    valid &= np.isfinite(out["dec_deg"]) & (abs(out["dec_deg"]) <= 90)
    out["silver_eligible"] = valid
    # No total calibrated sigma is asserted: inclusion of the paper's Ca/Fe
    # allowance in the electronic-table error remains unresolved.
    out["gold_nll_eligible"] = False
    return out


def canonical_rows(records):
    """Highest-SNR eligible observation per exact ID, independent of FeH/errors."""
    eligible = np.flatnonzero(records["silver_eligible"])
    r = records[eligible]
    order = np.lexsort((r["source_row"], -r["snr"], r["source_id"]))
    eligible = eligible[order]
    ids = records["source_id"][eligible]
    return eligible[np.r_[True, ids[1:] != ids[:-1]]] if len(ids) else eligible


def build(table_path, readme_path, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        str(p): dict(
            bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
        )
        for p in [table_path, readme_path]
    }
    table = ascii.read(table_path, format="cds", readme=str(readme_path))
    rows = records_from_table(table)
    selected = canonical_rows(rows)
    ids, counts = np.unique(rows["source_id"], return_counts=True)
    duplicate_rows = rows[np.isin(rows["source_id"], ids[counts > 1])]
    policy = dict(
        version="li2025-cat-silver-v1",
        paper=PAPER,
        source="CDS J/ApJS/279/53/table2",
        quantity="FeH",
        units="dex",
        eligible="finite measured FeH/error, exact DR3 ID, native method/flag, SNR>20, valid sky",
        flag_semantics={
            "0": "sum of three CaT lines",
            "1": "1.24 times two redder CaT lines",
        },
        duplicate_policy="highest SNR eligible observation; original source row breaks ties; no averaging",
        uncertainty="raw reported error and published 0.2 dex Ca/Fe allowance kept separate; inclusion/covariance unresolved; no total sigma or gold NLL eligibility",
        selection="published FeH-selected candidate sample, not population-complete validation truth",
        independence="mag depends on Gaia geometric distance; color calibration trained on mag; not independent HRS truth",
        benchmark="no features fitted here; downstream fitting must independently exclude benchmark validation IDs",
    )
    target = output / "li2025-silver-labels.h5"
    with h5py.File(target, "w") as f:
        f.attrs["policy_json"] = json.dumps(policy)
        f.attrs["input_provenance_json"] = json.dumps(inputs)
        f.create_dataset("measurements", data=rows, compression="gzip")
        f.create_dataset(
            "selected/measurements", data=rows[selected], compression="gzip"
        )
        f.create_dataset("selected/source_row", data=selected)
        f.create_dataset("duplicate_measurements", data=duplicate_rows)
    with h5py.File(target) as f:
        np.testing.assert_array_equal(f["measurements"][:], rows)
        np.testing.assert_array_equal(f["selected/measurements"][:], rows[selected])
    report = dict(
        policy=policy,
        inputs=inputs,
        measurement_rows=len(rows),
        unique_ids=len(ids),
        duplicate_ids=int(sum(counts > 1)),
        snr_le_20=int(sum(rows["snr"] <= 20)),
        silver_measurements=int(sum(rows["silver_eligible"])),
        selected_unique_silver_ids=len(selected),
        gold_nll_eligible=0,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        publication_roundtrip="exact structured arrays",
    )
    atomic_json(output / "ingestion-report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.table, args.readme, args.output), indent=2))
