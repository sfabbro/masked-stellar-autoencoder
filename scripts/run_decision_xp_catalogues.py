"""Unattended publisher-verified XP catalogue acquisition and paired frozen-model QA."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
PRED = ROOT + "multisurvey_joint_fullfit_20261105_v1/"
MAPPED = ROOT + "frozen_survey_qa_20261105_v2/"
BASE = ROOT + "remediation_source_v3/prepared.h5"
SPECS = {
    "andrae": dict(
        url="https://zenodo.org/records/7945154/files/table_1_catwise.fits.gz?download=1",
        name="table_1_catwise.fits.gz",
        bytes=3591656059,
        md5="64c33b28fca90fcea7d59272a970b588",
        rows=174922161,
    ),
    "gaia_net": dict(
        url="https://zenodo.org/records/15048995/files/gaia%20xp.parquet?download=1",
        name="gaia-xp.parquet",
        bytes=9049244346,
        md5="4cc09524fc1b4fe774dbeeb9dc0d50c2",
        rows=219196764,
    ),
}


def sha(path, algorithm="sha256"):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def exact_match(sorted_ids, other):
    other = np.asarray(other)
    if other.dtype.kind not in "iu":
        raise ValueError("Gaia IDs may not pass through floats")
    pos = np.searchsorted(sorted_ids, other)
    good = pos < len(sorted_ids)
    good[good] &= sorted_ids[pos[good]] == other[good]
    return np.flatnonzero(good)


def stats(pred, ref):
    d = np.asarray(pred) - np.asarray(ref)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0)
    median = float(np.median(d))
    return dict(
        n=len(d),
        bias_median=median,
        mae=float(abs(d).mean()),
        nmad=float(1.4826 * np.median(abs(d - median))),
        rmse=float(np.sqrt(np.mean(d * d))),
    )


def download(name, spec, out):
    path = out / spec["name"]
    print(
        json.dumps(dict(phase="downloading", catalogue=name, bytes=spec["bytes"])),
        flush=True,
    )
    h = hashlib.md5()
    h256 = hashlib.sha256()
    count = 0
    with (
        urllib.request.urlopen(spec["url"], timeout=120) as response,
        path.open("xb") as stream,
    ):
        if response.status != 200:
            raise ValueError("Expected complete public catalogue")
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
            h.update(block)
            h256.update(block)
            count += len(block)
            if count // (256 * 1024 * 1024) != (count - len(block)) // (
                256 * 1024 * 1024
            ):
                atomic(
                    out / (name + "-download.json"),
                    dict(bytes=count, total=spec["bytes"]),
                )
        if count != spec["bytes"] or h.hexdigest() != spec["md5"]:
            raise ValueError("Publisher catalogue size/checksum differs")
    result = dict(**spec, sha256=h256.hexdigest(), publisher_md5_verified=True)
    atomic(out / (name + "-source.json"), result)
    return path


def extract(name, path, ids, out):
    pieces = []
    seen = 0
    if name == "andrae":
        raw = Path("/scratch/andrae175m.fits")
        with gzip.open(path, "rb") as a, raw.open("xb") as b:
            shutil.copyfileobj(a, b, 8 * 1024 * 1024)
        with fits.open(raw, memmap=True) as f:
            if len(f[1].data) != SPECS[name]["rows"]:
                raise ValueError("Andrae row count differs")
            for start in range(0, len(f[1].data), 1000000):
                a = f[1].data[start : start + 1000000]
                hit = exact_match(ids, a["source_id"])
                seen += len(a)
                if len(hit):
                    pieces.append(
                        (
                            np.asarray(a["source_id"][hit], dtype="i8"),
                            np.column_stack(
                                [
                                    a[k][hit]
                                    for k in [
                                        "teff_xgboost",
                                        "logg_xgboost",
                                        "mh_xgboost",
                                    ]
                                ]
                            ),
                            np.asarray(a["in_training_sample"][hit], dtype=bool),
                        )
                    )
    else:
        source = pq.ParquetFile(path)
        if source.metadata.num_rows != SPECS[name]["rows"]:
            raise ValueError("Gaia Net row count differs")
        for batch in source.iter_batches(
            batch_size=500000, columns=["source_id", "logTeff", "logg", "FeH"]
        ):
            a = batch.to_pydict()
            sid = np.asarray(a["source_id"], dtype="i8")
            hit = exact_match(ids, sid)
            seen += len(sid)
            if len(hit):
                value = np.column_stack(
                    [
                        10.0 ** np.asarray(a["logTeff"])[hit],
                        np.asarray(a["logg"])[hit],
                        np.asarray(a["FeH"])[hit],
                    ]
                )
                pieces.append((sid[hit], value, np.full(len(hit), False)))
    sid = np.concatenate([x[0] for x in pieces])
    order = np.argsort(sid)
    sid = sid[order]
    if np.any(np.diff(sid) <= 0):
        raise ValueError("Comparator has duplicate IDs")
    with h5py.File(out / (name + "-matched.h5"), "x") as f:
        f["source_id"] = sid
        f["value"] = np.concatenate([x[1] for x in pieces])[order]
        if name == "andrae":
            f["in_training_sample"] = np.concatenate([x[2] for x in pieces])[order]
        f.attrs["scope"] = (
            "XP comparator only; metallicity definitions and input/teacher dependencies retained; not independent truth"
        )
    return dict(
        raw_rows=seen,
        matched_unique_stars=len(sid),
        sha256=sha(out / (name + "-matched.h5")),
    )


def score(name, out, bindings):
    result = {}
    with h5py.File(out / (name + "-matched.h5")) as c:
        cid = c["source_id"][:]
        cv = c["value"][:]
        train = c["in_training_sample"][:] if "in_training_sample" in c else None
        for role in ["fit", "tune", "cal", "devtest", "external"]:
            path = PRED + "predictions-" + role + ".h5"
            if sha(path) != bindings["predictions"][role]:
                raise ValueError("Frozen prediction SHA differs")
            with h5py.File(path) as p:
                pid = p["source_id"][:]
                ri = exact_match(cid, pid)
                ci = np.searchsorted(cid, pid[ri])
                ours = p["mean"][:][ri]
                other = cv[ci]
                subsets = {"all": np.ones(len(ri), bool)}
                if train is not None:
                    subsets.update(
                        comparator_training=train[ci], comparator_nontraining=~train[ci]
                    )
                entry = dict(
                    model_stars=len(pid),
                    matched_stars=len(ri),
                    agreement={
                        key: {
                            t: stats(ours[mask, j], other[mask, j])
                            for j, t in enumerate(
                                [
                                    "teff",
                                    "logg",
                                    "feh_vs_mh_proxy" if name == "andrae" else "feh",
                                ]
                            )
                        }
                        for key, mask in subsets.items()
                    },
                )
                if role in ["devtest", "external"]:
                    mp = MAPPED + "mapped-" + role + ".h5"
                    if sha(mp) != bindings["mapped"][role]:
                        raise ValueError("Frozen mapped prediction SHA differs")
                    with h5py.File(mp) as m:
                        np.testing.assert_array_equal(m["source_id"][:], pid)
                        w = m["weights"][:][ri]
                        mu = m["component_mean"][:][ri]
                        mapped = (mu * w[:, None, :, None]).sum(2)
                    val = p["native_value"][:][ri]
                    observed = p["native_observed"][:][ri].astype(bool)
                    entry["paired_native_reference"] = {}
                    for k, survey in enumerate(["APOGEE", "DESI", "GALAH"]):
                        entry["paired_native_reference"][survey] = {}
                        for subset, ss in subsets.items():
                            target = {}
                            for j, t in enumerate(
                                [
                                    "teff",
                                    "logg",
                                    "metallicity_proxy" if name == "andrae" else "feh",
                                ]
                            ):
                                keep = (
                                    ss
                                    & observed[:, k, j]
                                    & np.isfinite(other[:, j])
                                    & np.isfinite(mapped[:, k, j])
                                )
                                target[t] = dict(
                                    ours=stats(mapped[keep, k, j], val[keep, k, j]),
                                    comparator=stats(other[keep, j], val[keep, k, j]),
                                    paired_mae_difference=float(
                                        np.mean(
                                            abs(mapped[keep, k, j] - val[keep, k, j])
                                            - abs(other[keep, j] - val[keep, k, j])
                                        )
                                    )
                                    if keep.any()
                                    else None,
                                )
                            entry["paired_native_reference"][survey][subset] = target
                    np.savez_compressed(
                        out / (name + "-" + role + "-paired.npz"),
                        source_id=pid[ri],
                        common_mean=ours,
                        mapped_mean=mapped,
                        comparator=other,
                        reference=val,
                        reference_observed=observed,
                        comparator_training=train[ci]
                        if train is not None
                        else np.full(len(ci), -1, dtype="i1"),
                    )
                result[role] = entry
    atomic(
        out / (name + "-qa.json"),
        dict(
            scope="Agreement and native-reference compatibility; native catalogue metallicity scales differ; historical inspected development roles, not independent physical accuracy",
            metrics=result,
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not Path(BASE).exists() or shutil.disk_usage(args.output).free < 100_000_000_000:
        raise RuntimeError("Native source mounts and 100GB reserve required")
    bindings = json.loads(args.bindings.read_text())
    manifest = dict(
        status="running", training_performed=False, model_loaded=False, catalogues={}
    )
    atomic(args.output / "manifest.json", manifest)
    with h5py.File(BASE) as f:
        ids = f["source_id"][:]
    downloads = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(download, name, spec, args.output): name
            for name, spec in SPECS.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                downloads[name] = future.result()
            except Exception as e:
                manifest["catalogues"][name] = dict(
                    status="failed_download", error=repr(e)
                )
    for name, path in downloads.items():
        try:
            manifest["catalogues"][name] = dict(status="extracting")
            atomic(args.output / "manifest.json", manifest)
            receipt = extract(name, path, ids, args.output)
            score(name, args.output, bindings)
            manifest["catalogues"][name] = dict(status="qa_complete", **receipt)
        except Exception as e:
            manifest["catalogues"][name] = dict(status="failed_qa", error=repr(e))
        atomic(args.output / "manifest.json", manifest)
    manifest["status"] = (
        "xp_catalogue_qa_complete"
        if all(x["status"] == "qa_complete" for x in manifest["catalogues"].values())
        else "partial_failure"
    )
    atomic(args.output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
