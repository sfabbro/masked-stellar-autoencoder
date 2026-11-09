"""Durable source preparation for all-Gaia corrections; never changes production inputs.

Each mode writes native records/provenance, not an automatic training admission.
"""

import argparse
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

ROOT = Path("/arc/projects/k-pop")
RUN = ROOT / "msa_runs/stellar-improvement-20261015"
CAT = ROOT / "catalogues"
BASE = RUN / "expanded_six_source_20261107_v5/prepared.h5"


def match_positions(ids, query):
    """Exact integer join; no XP membership predicate and no float ID coercion."""
    q = np.asarray(query)
    if q.dtype.kind not in "iu":
        raise ValueError("Exact integer Gaia identifiers required")
    p = np.searchsorted(ids, q)
    hit = (p < len(ids)) & (q > 0)
    hit[hit] &= ids[p[hit]] == q[hit]
    return np.flatnonzero(hit), p[hit]


def candidates():
    arrays, provenance = [], []
    with h5py.File(BASE) as f:
        arrays.append(f["source_id"][:])
        provenance.append(
            dict(
                path=str(BASE),
                key="source_id",
                rows=len(arrays[-1]),
                scope="existing registry, including ineligible rows",
            )
        )
    config = json.loads(Path("configs/expanded-six-source-20261107.json").read_text())
    for spec in config.get("additional_sources", []):
        with h5py.File(spec["path"]) as f:
            g = f["native_observations/" + spec["name"]]
            a = g["source_id"][:]
            if a.dtype.kind not in "iu":
                raise ValueError("Noninteger sidecar IDs")
            arrays.append(a[a > 0])
            provenance.append(
                dict(
                    path=spec["path"],
                    source=spec["name"],
                    rows=len(a),
                    scope="native sidecar before feature intersection",
                )
            )
    # Original native releases recover stars absent from the existing feature registry.
    for path in [
        CAT / "galah_dr4_allstar_240705.fits",
        RUN / "apogee_native_fetch_v1/allStar-dr17-synspec_rev1.fits",
    ]:
        if not path.exists():
            continue
        with fits.open(path, memmap=True) as hd:
            for h in hd:
                cols = list(getattr(getattr(h, "columns", None), "names", []) or [])
                for c in cols:
                    if c.lower() in [
                        "gaiadr3_source_id",
                        "gaiaedr3_source_id",
                        "gaia_dr3_source_id",
                    ]:
                        a = np.asarray(h.data[c])
                        if a.dtype.kind not in "iu":
                            provenance.append(
                                dict(
                                    path=str(path),
                                    column=c,
                                    rejected="noninteger identifier storage requires explicit recovery",
                                )
                            )
                            continue
                        arrays.append(a[a > 0].astype("i8"))
                        provenance.append(
                            dict(
                                path=str(path),
                                column=c,
                                rows=len(a),
                                scope="native release, no XP selection",
                            )
                        )
    return np.unique(np.concatenate(arrays)).astype("i8"), provenance


def native_fields(cols):
    return [
        c
        for c in cols
        if re.search(
            r"(^|_)(age|ages|logage|log_age|lnage|tau)(_|$)|agergb|agerc|alpha|mg.?fe|si.?fe|ca.?fe|ti.?fe",
            c.lower(),
        )
        and not c.lower().startswith("tau_gran")
    ]


def table_frame(data, cols, lo, hi):
    out = {}
    for c in cols:
        a = np.asarray(data[c][lo:hi])
        if a.ndim != 1:
            continue  # vector fields need an explicit element response, never flatten implicitly
        if a.dtype.kind == "S":
            a = np.char.decode(a, "utf-8", errors="replace")
        elif a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("="))
        out[c] = a
    return pd.DataFrame(out)


def ascii_table_frame(path, hdu, cols, lo, hi, *, extra_null_tokens=()):
    """Read native ASCII fields without Astropy's zero-filled numeric nulls.

    Parse integer fields directly from text, never through floating point.
    Additional null tokens are explicit per-product representation repairs.
    """
    rows = min(hi, int(hdu.header["NAXIS2"])) - lo
    width = int(hdu.header["NAXIS1"])
    selected = [hdu.columns[c] for c in cols]
    formats = [re.fullmatch(r"([AIFED])(\d+)(?:\.\d+)?", str(c.format)) for c in selected]
    if any(f is None for f in formats):
        raise ValueError("Unsupported ASCII field format")
    dtype = np.dtype({
        "names": cols,
        "formats": [f"S{int(f.group(2))}" for f in formats],
        "offsets": [int(c.start) - 1 for c in selected],
        "itemsize": width,
    })
    with Path(path).open("rb") as stream:
        stream.seek(hdu.fileinfo()["datLoc"] + lo * width)
        payload = stream.read(rows * width)
    if len(payload) != rows * width:
        raise ValueError("Truncated ASCII table records")
    raw = np.frombuffer(payload, dtype=dtype, count=rows)
    frame = {}
    for column, fmt in zip(selected, formats, strict=True):
        text = np.char.strip(np.char.decode(raw[column.name], "ascii"))
        if fmt.group(1) == "A":
            frame[column.name] = text
            continue
        nulls = {"", *extra_null_tokens}
        if column.null is not None:
            nulls.add(str(column.null).strip())
        missing = np.isin(text, list(nulls))
        if fmt.group(1) == "I":
            frame[column.name] = pd.array(
                [pd.NA if absent else int(value) for value, absent in zip(text, missing, strict=True)],
                dtype="Int64",
            )
        else:
            values = [None if absent else value.replace("D", "E").replace("d", "e") for value, absent in zip(text, missing, strict=True)]
            frame[column.name] = pd.to_numeric(pd.Series(values), errors="raise")
    return pd.DataFrame(frame)


def stage_labels(mode, out, m, save):
    age_paths = [
        "ambre-ages.fits",
        "lamost-kepler-ages.fits",
        "rave_dr6_bdasp.fits",
        "galah_dr4_allstar_240705.fits",
        "apokasc-3-2024.fits",
        "apokasc-2-2018.fits",
        "apo_k2_2024.fits",
        "apo_k2_2024_recal.fits",
    ]
    alpha_paths = [
        "galah_dr4_allstar_240705.fits",
        "rave_dr6_cnn.fits",
        "rave_dr6_madera.fits",
        "rave_dr6_gaia_bdasp.fits",
        "gaia_rvs_lite_flags.fits.gz",
        "ges_dr5.1.fits",
        "astraMWMLite-0.6.0.fits.gz",
        "mwsall-pix-iron.fits",
    ]
    paths = [CAT / x for x in (age_paths if mode == "ages" else alpha_paths)]
    if mode == "alpha":
        paths += [RUN / "apogee_native_fetch_v1/allStar-dr17-synspec_rev1.fits"]
        # LAMOST releases may store extra alpha columns in native tables outside the narrow sidecars.
        for folder in [
            CAT / "lamost-param",
            RUN / "lamost_dr12_lrs_collection_v1",
            RUN / "lamost_dr12_mrs_collection_v1",
        ]:
            if folder.exists():
                paths.extend(sorted(folder.rglob("*.fits")))
    for n, path in enumerate(dict.fromkeys(paths)):
        item = dict(path=str(path), exists=path.exists(), tables=[])
        m["sources"].append(item)
        save()
        if not path.exists():
            continue
        try:
            with fits.open(path, memmap=not path.name.endswith(".gz")) as hd:
                for k, h in enumerate(hd):
                    if not isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                        continue
                    cols = list(h.columns.names)
                    selected = native_fields(cols)
                    selected = [
                        c
                        for c in selected
                        if (bool(re.search("age|tau", c, re.I)) == (mode == "ages"))
                    ]
                    ids = [
                        c
                        for c in cols
                        if re.search(
                            "gaia|source.?id|apogee.?id|sobject|rave.*id|obsid|designation|^ra$|^dec$|radeg|dedeg|^status$|flag|snr",
                            c,
                            re.I,
                        )
                    ]
                    fields = list(dict.fromkeys(ids + selected))
                    entry = dict(
                        hdu=h.name,
                        hdu_index=k,
                        hdu_type=type(h).__name__,
                        rows=int(h.header["NAXIS2"]),
                        selected_fields=selected,
                        identity_and_quality_fields=ids,
                        units={c.name: c.unit for c in h.columns if c.unit},
                        training_admitted=False,
                    )
                    extra_nulls = ("---",) if path.name == "lamost-kepler-ages.fits" else ()
                    if isinstance(h, fits.TableHDU):
                        entry["ascii_null_policy"] = dict(
                            blank_is_missing=True,
                            header_null_tokens={c.name: c.null for c in h.columns if c.null is not None},
                            additional_numeric_null_tokens=list(extra_nulls),
                            additional_token_reason="Observed undeclared native missing marker" if extra_nulls else None,
                            integer_parser="direct text-to-integer; nullable Int64",
                        )
                    item["tables"].append(entry)
                    if not selected:
                        continue
                    target = out / f"{n:03}-{k:02}"
                    target.mkdir()
                    finite = {c: 0 for c in selected}
                    for lo in range(0, entry["rows"], 100000):
                        if isinstance(h, fits.TableHDU):
                            df = ascii_table_frame(path, h, fields, lo, lo + 100000, extra_null_tokens=extra_nulls)
                        else:
                            df = table_frame(h.data, fields, lo, lo + 100000)
                        if "source_row" in df:
                            raise ValueError("Reserved source_row column already present")
                        df["source_row"] = np.arange(lo, lo + len(df), dtype="i8")
                        pq.write_table(
                            pa.Table.from_pandas(df, preserve_index=False),
                            target / f"part-{lo:010}.parquet",
                        )
                        for c in selected:
                            if c in df and pd.api.types.is_numeric_dtype(df[c]):
                                a = df[c].to_numpy(dtype="f8", na_value=np.nan)
                                finite[c] += int((np.isfinite(a) & (a != -9999)).sum())
                    entry.update(output=str(target), finite_not_quality_selected=finite)
                    save()
        except Exception as e:
            item["error"] = repr(e)
            save()
    if mode == "ages":
        path = CAT / "nn_latent_age_dr17.csv.gz"
        if path.exists():
            target = out / "apogee-latent-age"
            target.mkdir()
            count = 0
            finite = 0
            for k, df in enumerate(
                pd.read_csv(path, chunksize=100000, dtype={"APOGEE_ID": "string"})
            ):
                if "source_row" in df:
                    raise ValueError("Reserved source_row column already present")
                df["source_row"] = np.arange(count, count + len(df), dtype="i8")
                keep = [
                    c
                    for c in df
                    if c
                    in [
                        "APOGEE_ID",
                        "source_row",
                        "TELESCOPE",
                        "FIELD",
                        "STARFLAG",
                        "ASPCAPFLAG",
                        "Age",
                        "Age_Error",
                        "LogAge",
                        "LogAge_Error",
                    ]
                ]
                pq.write_table(
                    pa.Table.from_pandas(df[keep], preserve_index=False),
                    target / f"part-{k:05}.parquet",
                )
                count += len(df)
                finite += int(
                    np.isfinite(pd.to_numeric(df["Age"], errors="coerce")).sum()
                )
            m["sources"].append(
                dict(
                    path=str(path),
                    output=str(target),
                    rows=count,
                    finite_age_before_quality=finite,
                    training_admitted=False,
                    age_units="must verify publication; CSV has no unit header",
                )
            )
    m["semantic_admission_pending"] = [
        "native quantity definitions and solar conventions",
        "quality/domain cuts",
        "publication/release identity and priors",
        "shared-calibrator/dependency groups",
        "exact Gaia ID association; no duplication across surveys",
    ]


def stage_gaia(out, m, save):
    ids, prov = candidates()
    np.save(out / "label-candidate-source-id.npy", ids)
    m.update(
        candidate_stars=len(ids),
        candidate_provenance=prov,
        scope="Full Gaia shard join for native-label candidate union, independent of XP; does not certify discovery of every possible survey catalogue",
    )
    save()
    index = ROOT / "gaia/source_ids_x_file_names.h5"
    if index.exists():
        with h5py.File(index) as f:
            info = []

            def visit(n, o):
                if isinstance(o, h5py.Dataset):
                    info.append(
                        dict(name=n, shape=list(o.shape), dtype=str(o.dtype)[:500])
                    )

            f.visititems(visit)
            m["gaia_index_schema"] = info[:30]
            save()
    paths = sorted(
        p
        for p in (ROOT / "gaia/GaiaSource").rglob("*")
        if p.suffix.lower() in [".h5", ".hdf5"]
    )
    m.update(gaia_shards=len(paths), matched_rows=0, all_gaia_rows_scanned=0)
    if not paths:
        raise RuntimeError(
            "No GaiaSource HDF5 shards discovered; inspect index/schema before adapting reader"
        )
    matched = out / "labelled-gaia"
    matched.mkdir()
    reservoir = out / "ssl-reservoir"
    reservoir.mkdir()
    seen = np.zeros(len(ids), bool)
    for i, path in enumerate(paths):
        with h5py.File(path) as f:
            # GaiaSource stores column arrays at root, unlike the legacy compound XP catalogue.
            if "source_id" not in f or not isinstance(f["source_id"], h5py.Dataset):
                m.setdefault("unsupported_shards", []).append(str(path))
                save()
                continue
            fields = [
                c
                for c in [
                    "source_id",
                    "ra",
                    "dec",
                    "ra_error",
                    "dec_error",
                    "parallax",
                    "parallax_error",
                    "pmra",
                    "pmdec",
                    "pmra_error",
                    "pmdec_error",
                    "ra_dec_corr",
                    "parallax_pmra_corr",
                    "parallax_pmdec_corr",
                    "pmra_pmdec_corr",
                    "phot_g_mean_mag",
                    "phot_bp_mean_mag",
                    "phot_rp_mean_mag",
                    "phot_g_mean_flux",
                    "phot_g_mean_flux_error",
                    "phot_bp_mean_flux",
                    "phot_bp_mean_flux_error",
                    "phot_rp_mean_flux",
                    "phot_rp_mean_flux_error",
                    "has_xp_continuous",
                    "ruwe",
                ]
                if c in f
            ]
            n = len(f["source_id"])
            sample = []
            for lo in range(0, n, 262144):
                q = f["source_id"][lo : lo + 262144]
                rr, pos = match_positions(ids, q)
                if len(rr):
                    df = pd.DataFrame({c: f[c][lo : lo + len(q)][rr] for c in fields})
                    df["candidate_row"] = pos
                    pq.write_table(
                        pa.Table.from_pandas(df, preserve_index=False),
                        matched / f"{i:05}-{lo:010}.parquet",
                    )
                    seen[pos] = True
                    m["matched_rows"] += len(rr)
                # Fixed, explicitly stratified reservoir, including actual has_xp_continuous=False stars.
                flags = (
                    f["has_xp_continuous"][lo : lo + len(q)].astype(bool)
                    if "has_xp_continuous" in f
                    else np.zeros(len(q), bool)
                )
                take = np.concatenate(
                    [np.flatnonzero(flags)[:32], np.flatnonzero(~flags)[:32]]
                )
                if len(take):
                    sample.append(
                        pd.DataFrame({c: f[c][lo : lo + len(q)][take] for c in fields})
                    )
                m["all_gaia_rows_scanned"] += len(q)
            if sample:
                pq.write_table(
                    pa.Table.from_pandas(
                        pd.concat(sample, ignore_index=True), preserve_index=False
                    ),
                    reservoir / f"{i:05}.parquet",
                )
        m["completed_gaia_shards"] = i + 1
        m["matched_unique_candidates"] = int(seen.sum())
        save()
    np.save(out / "candidate-found-in-gaia.npy", seen)
    np.save(out / "unmatched-candidate-source-id.npy", ids[~seen])
    m["ssl_reservoir_sampling"] = (
        "bounded per-shard/per-chunk XP-availability stratification, not population-uniform; native source IDs retained for role assignment"
    )
    m["roles"] = (
        "existing roles must be inherited by Gaia ID; new stars require protected benchmark/cluster and dependency-group assignment before training"
    )


def stage_photometry(out, m, save):
    ids, prov = candidates()
    np.save(out / "label-candidate-source-id.npy", ids)
    contract = json.loads(
        Path("configs/full-gaia-observation-columns-20261108.json").read_text()
    )
    names = contract["feature_names"]
    errors = contract["error_names"]
    original = CAT / "andrae2023/sslset-realmags-full-052725.h5"
    m.update(
        candidate_stars=len(ids),
        candidate_provenance=prov,
        source=str(original),
        scope="Full original source scan for all declared bands/auxiliaries, exact native-ID join, no XP requirement; availability is not a quality certificate",
        sources=[],
    )
    target = out / "raw-observations"
    target.mkdir()
    with h5py.File(original) as f:
        for k, key in enumerate(f):
            d = f[key]
            cols = list(d.dtype.names or [])
            if "source_id" not in cols:
                m["sources"].append(dict(shard=key, unsupported_schema=True))
                save()
                continue
            fields = list(
                dict.fromkeys(
                    ["source_id"] + [c for c in names + errors if c and c in cols]
                )
            )
            counts = {c: 0 for c in names if c in cols}
            count = 0
            noxp = 0
            for lo in range(0, len(d), 131072):
                q = d.fields("source_id")[lo : lo + 131072]
                rr, pos = match_positions(ids, q)
                if not len(rr):
                    continue
                # Read source fields in row order, then preserve every declared available channel.
                records = d.fields(fields)[lo : lo + len(q)][rr]
                df = table_frame(records, fields, 0, len(records))
                df["candidate_row"] = pos
                xp = [c for c in names if re.match("^[br]p_\\d+$", c) and c in df]
                present = np.zeros(len(df), bool)
                for c in xp:
                    present |= np.isfinite(df[c].to_numpy()) & (
                        df[c].to_numpy() != -9999
                    )
                df["raw_any_xp_present"] = present
                pq.write_table(
                    pa.Table.from_pandas(df, preserve_index=False),
                    target / f"{k:04}-{lo:010}.parquet",
                )
                count += len(df)
                noxp += int((~present).sum())
                for c in counts:
                    a = df[c].to_numpy()
                    counts[c] += int((np.isfinite(a) & (a != -9999)).sum())
                m.update(current_shard=key, current_shard_matched_rows=count)
                save()
            m["sources"].append(
                dict(
                    shard=key,
                    rows=len(d),
                    matched_rows=count,
                    matched_no_xp=noxp,
                    available_feature_columns=[c for c in names if c in cols],
                    missing_columns=[c for c in names if c not in cols],
                    visible_counts_before_quality=counts,
                )
            )
            save()
    m["pending"] = (
        "Join additional native photometric catalogues wherever legacy source lacks coverage; inspect native bandpass, AB/Vega, quality and uncertainty semantics before normalization. FIT-only normalization and head integration follow source staging."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=["gaia", "photometry", "alpha", "ages"], required=True
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="running",
        mode=a.mode,
        training_performed=False,
        sources=[],
        started_unix=time.time(),
    )

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(m, indent=2, default=str) + "\n")
        tmp.replace(a.output / "manifest.json")

    save()
    try:
        if a.mode in ["ages", "alpha"]:
            stage_labels(a.mode, a.output, m, save)
        elif a.mode == "gaia":
            stage_gaia(a.output, m, save)
        else:
            stage_photometry(a.output, m, save)
        m.update(
            status="source_stage_complete",
            training_admitted=False,
            finished_unix=time.time(),
        )
        save()
    except Exception as e:
        m.update(status="failed", error=repr(e))
        save()
        raise


if __name__ == "__main__":
    main()
