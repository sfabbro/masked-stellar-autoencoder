"""Fine-tune table for the stopgap run: master-table labels + raw MSA inputs.

Picks one label set per Gaia star from the master spectroscopic table
(training.stopgap_labels), keeps every high-res / APOGEE / GALAH /
metal-poor star plus a random fill up to --max-stars, and copies each star's
feature and error columns (the finetune config's feature_cols/error_cols)
from the raw pretraining HDF5. Options for the semi-supervised run (v5):
``--irfm-anchors`` adds photometric IRFM-scale Teff records
(training.irfm_teff) as an anchor survey, ``--unlabelled-fraction`` keeps a
random sample of raw stars outside the table (``unlabelled.npz``, same
columns) for reconstruction/consistency, and ``--exclude-test-of`` drops an
earlier table's test stars so models can be compared on them. Writes ``finetune.fits`` in the column layout
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


def _isin_sorted(sorted_ids, sid):
    if not len(sorted_ids):
        return np.zeros(len(sid), bool)
    pos = np.searchsorted(sorted_ids, sid).clip(max=len(sorted_ids) - 1)
    return sorted_ids[pos] == sid


def scan_shard(args):
    """Rows of one raw shard whose source_id is wanted: (ids, matrix).

    With ``fraction`` > 0 it also returns a random sample of the shard's
    other rows, minus ``avoid`` (ids, matrix), drawn with a per-shard seed.
    """
    path, key, wanted, fields, chunk, fraction, avoid, seed = args
    import h5py

    from masked_stellar_autoencoder.training.hdf5_io import _clean_column

    rng = np.random.default_rng(seed)
    ids, blocks, u_ids, u_blocks = [], [], [], []

    def read(d, start, keep):
        rec = d.fields(fields)[start : start + chunk][keep]
        return np.column_stack([_clean_column(n, rec[n]) for n in fields])

    with h5py.File(path, "r") as f:
        d = f[key]
        for start in range(0, d.shape[0], chunk):
            sid = d.fields(["source_id"])[start : start + chunk]["source_id"]
            sid = sid.astype("i8")
            hit = _isin_sorted(wanted, sid)
            if hit.any():
                ids.append(sid[hit])
                blocks.append(read(d, start, hit))
            if fraction > 0:
                take = (rng.random(len(sid)) < fraction) & ~hit
                take &= ~_isin_sorted(avoid, sid) & (sid > 0)
                if take.any():
                    u_ids.append(sid[take])
                    u_blocks.append(read(d, start, take))

    def stack(i, b):
        if not i:
            return np.empty(0, "i8"), np.empty((0, len(fields)), "f4")
        return np.concatenate(i), np.concatenate(b).astype("f4")

    return key, *stack(ids, blocks), *stack(u_ids, u_blocks)


def test_ids_of(table_dir, config):
    """source_ids of an earlier table's test split (finetune_data's split)."""
    from astropy.io import fits

    from masked_stellar_autoencoder.training.finetune_data import (
        kept_rows,
        split_row_positions,
    )

    path = str(Path(table_dir) / "finetune.fits")
    rows = kept_rows(path, config)
    mp = config["finetuning"].get("metal_poor") or {}
    _, _, test = split_row_positions(len(rows), mp)
    with fits.open(path, memmap=True) as h:
        sid = np.asarray(h[1].data["source_id"], dtype="i8")
    return np.unique(sid[rows[test]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/finetune.canfar.yaml"))
    p.add_argument("--raw", default=RAW)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-stars", type=int, default=3_000_000)
    p.add_argument(
        "--references",
        type=Path,
        help="catalogue folder for seismic anchors and Gaia benchmark stars "
        "(training.anchor_labels); anchors are forced into the table and written "
        "to records.npz, benchmarks go to validation.fits only",
    )
    p.add_argument(
        "--irfm-anchors",
        action="store_true",
        help="add photometric IRFM-scale Teff anchor records (needs --references)",
    )
    p.add_argument(
        "--unlabelled-fraction",
        type=float,
        default=0.0,
        help="fraction of raw stars outside the table kept in unlabelled.npz",
    )
    p.add_argument(
        "--exclude-test-of",
        type=Path,
        help="earlier table folder whose test-split stars are left out entirely",
    )
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
        survey_records,
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
        cfg_all = yaml.safe_load(a.config.read_text())
        excluded = np.empty(0, "i8")
        if a.exclude_test_of:
            excluded = test_ids_of(a.exclude_test_of, cfg_all)
            labels = labels.drop(labels.index.intersection(excluded))
            m.update(excluded_test_stars=int(len(excluded)))
            log(f"left out {len(excluded)} test stars of {a.exclude_test_of}")
        stars = choose_stars(labels, a.max_stars)
        benchmark_ids = np.empty(0, "i8")
        references = None
        if a.references:
            from masked_stellar_autoencoder.training.anchor_labels import (
                load_reference_labels,
            )

            references = load_reference_labels(a.references)
            anchor_ids = references.loc[references.role == "anchor", "source_id"]
            benchmark_ids = np.unique(
                references.loc[references.role == "validation", "source_id"]
            )
            stars = (
                stars.union(pd.Index(anchor_ids.unique()))
                .difference(pd.Index(benchmark_ids))
                .difference(pd.Index(excluded))
            )
            m.update(
                reference_records=references.groupby("survey").size().to_dict(),
                benchmark_stars=int(len(benchmark_ids)),
            )
        else:
            del records
        labels = labels.reindex(stars)
        m.update(labelled_gaia_stars=int(len(stars)))
        log(f"{len(labels)} stars chosen")

        cfg = cfg_all["data"]
        features = cfg["feature_cols"]
        errors = [e for e in cfg["error_cols"] if e is not None]
        fields = list(dict.fromkeys([*features, *errors]))
        wanted = np.union1d(labels.index.to_numpy("i8"), benchmark_ids)
        import h5py

        with h5py.File(a.raw, "r") as f:
            keys = sorted(f.keys())
        log(f"scanning {len(keys)} raw shards for {len(wanted)} stars")
        found_ids, found, unl_ids, unl = [], [], [], []
        avoid = np.union1d(excluded, benchmark_ids)
        jobs = [
            (a.raw, k, wanted, fields, a.chunk, a.unlabelled_fraction, avoid, 1000 + i)
            for i, k in enumerate(keys)
        ]
        with ProcessPoolExecutor(a.workers) as pool:
            for i, (key, ids, block, u_ids, u_block) in enumerate(
                pool.map(scan_shard, jobs), 1
            ):
                found_ids.append(ids)
                found.append(block)
                unl_ids.append(u_ids)
                unl.append(u_block)
                if i % 10 == 0 or i == len(keys):
                    n = sum(len(x) for x in found_ids)
                    log(f"{i}/{len(keys)} shards, {n} rows matched")
                    m.update(shards_scanned=i, rows_matched=n)
                    save()
        ids = np.concatenate(found_ids)
        matrix = np.concatenate(found)
        ids, first = np.unique(ids, return_index=True)  # duplicate raw rows
        matrix = matrix[first]
        bench = np.isin(ids, benchmark_ids)
        bench_ids, bench_matrix = ids[bench], matrix[bench]
        ids, matrix = ids[~bench], matrix[~bench]
        labels = labels.loc[ids]
        if a.unlabelled_fraction > 0:
            u_ids = np.concatenate(unl_ids)
            u_ids, first = np.unique(u_ids, return_index=True)
            np.savez(
                a.output / "unlabelled.npz",
                source_id=u_ids,
                features=np.concatenate(unl)[first],
                fields=np.array(fields, dtype="U40"),
            )
            m.update(unlabelled_stars=int(len(u_ids)))
            log(f"{len(u_ids)} unlabelled stars")
        del unl
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
        if references is not None and a.irfm_anchors:
            from masked_stellar_autoencoder.training.anchor_labels import _frame
            from masked_stellar_autoencoder.training.irfm_teff import (
                photometric_teff,
            )

            col = {n: matrix[:, fields.index(n)] for n in
                   ("G", "BP", "RP", "J", "H", "KS", "EBV")}  # fmt: skip
            teff, e_teff = photometric_teff(
                col["G"], col["BP"], col["RP"], col["J"], col["H"], col["KS"],
                col["EBV"], labels["logg"].to_numpy("f8"), labels["fe_h"].to_numpy("f8"),
            )  # fmt: skip
            ok = np.isfinite(teff)
            irfm = _frame("anchor_irfm_colte", "anchor", ids[ok],
                          teff=teff[ok], e_teff=e_teff[ok])  # fmt: skip
            references = pd.concat([references, irfm], ignore_index=True)
            m.update(irfm_anchor_stars=int(ok.sum()))
            log(f"{int(ok.sum())} IRFM Teff anchors")
        if references is not None:
            rec = survey_records(records, references, ids)
            np.savez_compressed(a.output / "records.npz", **rec)
            names = rec["survey_names"]
            m.update(
                records=int(len(rec["row"])),
                records_per_survey={
                    str(n): int(c)
                    for n, c in zip(
                        names,
                        np.bincount(rec["survey"], minlength=len(names)),
                        strict=True,
                    )
                },
                anchor_surveys=[str(n) for n in names[rec["survey_is_anchor"]]],
            )
            b = (
                references[references.role == "validation"]
                .drop_duplicates("source_id")
                .set_index("source_id")
                .loc[bench_ids]
            )
            v = Table()
            v["source_id"] = bench_ids
            for name in PARAMETERS:
                v[name] = b[name].to_numpy("f4")
                v["e_" + name] = b["e_" + name].to_numpy("f4")
            v["age"] = np.full(len(bench_ids), np.nan, "f4")
            v["e_age"] = np.full(len(bench_ids), np.nan, "f4")
            for j, name in enumerate(fields):
                v[name] = bench_matrix[:, j]
            v.write(a.output / "validation.fits", overwrite=True)
            m.update(benchmark_stars_with_inputs=int(len(bench_ids)))
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
