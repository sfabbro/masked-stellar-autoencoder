"""Fine-tune table for the stopgap run: master-table labels + raw MSA inputs.

Picks one label set per Gaia star from the master spectroscopic table
(training.stopgap_labels), keeps every high-res / APOGEE / GALAH /
metal-poor star plus a random fill up to --max-stars, and copies each star's
feature and error columns (the finetune config's feature_cols/error_cols)
from the raw pretraining HDF5. Writes ``finetune.fits`` in the column layout
finetune_msa reads (teff, e_teff, logg, e_logg, fe_h, e_fe_h, alpha, e_alpha,
age, e_age, features, errors) plus source_id and label provenance.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAW = "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_shard(args):
    """Rows of one raw shard whose source_id is wanted: (ids, matrix)."""
    path, key, wanted, fields, chunk = args
    import h5py

    from masked_stellar_autoencoder.training.hdf5_io import _clean_column

    ids, blocks = [], []
    with h5py.File(path, "r") as f:
        d = f[key]
        for start in range(0, d.shape[0], chunk):
            sid = d.fields(["source_id"])[start : start + chunk]["source_id"]
            sid = sid.astype("i8")
            pos = np.searchsorted(wanted, sid).clip(max=len(wanted) - 1)
            hit = wanted[pos] == sid
            if not hit.any():
                continue
            rec = d.fields(fields)[start : start + chunk][hit]
            ids.append(sid[hit])
            blocks.append(np.column_stack([_clean_column(n, rec[n]) for n in fields]))
    if not ids:
        return key, np.empty(0, "i8"), np.empty((0, len(fields)), "f4")
    return key, np.concatenate(ids), np.concatenate(blocks).astype("f4")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/finetune.canfar.yaml"))
    p.add_argument("--raw", default=RAW)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-stars", type=int, default=3_000_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=262_144)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    m = dict(status="building", master=str(a.master), raw=a.raw)

    def save():
        (a.output / "manifest.json").write_text(json.dumps(m, indent=2, default=str))

    save()
    from masked_stellar_autoencoder.training.stopgap_labels import (
        PARAMETERS,
        choose_stars,
        select_labels,
    )

    try:
        cols = [
            "survey", "tier", "source_id", "fit_ok", "fe_h_kind", "fe_h_corrected",
            "alpha_kind",
            *[c for p_ in PARAMETERS for c in (p_, "e_" + p_)],
        ]  # fmt: skip
        records = pd.read_parquet(a.master / "records", columns=cols)
        log(f"{len(records)} master records")
        labels = select_labels(records)
        del records
        stars = choose_stars(labels, a.max_stars)
        labels = labels.loc[stars]
        m.update(labelled_gaia_stars=int(len(stars)))
        log(f"{len(labels)} stars chosen")

        cfg = yaml.safe_load(a.config.read_text())["data"]
        features = cfg["feature_cols"]
        errors = [e for e in cfg["error_cols"] if e is not None]
        fields = list(dict.fromkeys([*features, *errors]))
        wanted = np.sort(labels.index.to_numpy("i8"))
        import h5py

        with h5py.File(a.raw, "r") as f:
            keys = sorted(f.keys())
        log(f"scanning {len(keys)} raw shards for {len(wanted)} stars")
        found_ids, found = [], []
        jobs = [(a.raw, k, wanted, fields, a.chunk) for k in keys]
        with ProcessPoolExecutor(a.workers) as pool:
            for i, (key, ids, block) in enumerate(pool.map(scan_shard, jobs), 1):
                found_ids.append(ids)
                found.append(block)
                if i % 10 == 0 or i == len(keys):
                    n = sum(len(x) for x in found_ids)
                    log(f"{i}/{len(keys)} shards, {n} rows matched")
                    m.update(shards_scanned=i, rows_matched=n)
                    save()
        ids = np.concatenate(found_ids)
        matrix = np.concatenate(found)
        ids, first = np.unique(ids, return_index=True)  # duplicate raw rows
        matrix = matrix[first]
        labels = labels.loc[ids]
        m.update(stars_with_inputs=int(len(ids)), stars_without_inputs=int(
            len(wanted) - len(ids)))  # fmt: skip

        from astropy.table import Table

        t = Table()
        t["source_id"] = ids
        for name in PARAMETERS:
            t[name] = labels[name].to_numpy("f4")
            t["e_" + name] = labels["e_" + name].to_numpy("f4")
        t["age"] = np.full(len(ids), np.nan, "f4")  # no age labels in the stopgap
        t["e_age"] = np.full(len(ids), np.nan, "f4")
        for j, name in enumerate(fields):
            t[name] = matrix[:, j]
        for name in PARAMETERS:
            t["src_" + name] = np.asarray(
                labels["src_" + name].fillna("").astype(str).tolist(), dtype="U40"
            )
        t.write(a.output / "finetune.fits", overwrite=True)
        counts = {
            name: labels["src_" + name].value_counts().to_dict() for name in PARAMETERS
        }
        feh = labels["fe_h"].to_numpy("f8")
        m.update(
            status="complete",
            label_sources=counts,
            finite_labels={n: int(np.isfinite(labels[n]).sum()) for n in PARAMETERS},
            metal_poor={str(c): int((feh < c).sum()) for c in (-1, -2, -3)},
        )
        save()
        log("done")
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
