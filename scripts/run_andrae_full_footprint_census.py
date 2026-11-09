"""Exact-ID census of the full Andrae release against raw input and inference cache.

No model execution, label admission or role changes. Records raw row locations
for subsequent catalogue-wide inference; raw coverage is not prediction coverage.
"""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
SOURCE = ROOT + "decision_xp_catalogue_qa_20261106_v1/table_1_catwise.fits.gz"
RAW = "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"
CACHE = ROOT + "residual-huber-e86-150m-cache-v1/source_id.npy"


def match(sorted_ids, ids):
    ids = np.asarray(ids)
    if ids.dtype.kind not in "iu":
        raise ValueError("Exact integer Gaia IDs required")
    pos = np.searchsorted(sorted_ids, ids)
    good = pos < len(sorted_ids)
    good[good] &= sorted_ids[pos[good]] == ids[good]
    return np.flatnonzero(good), pos[good]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="verifying_full_release",
        training_performed=False,
        model_loaded=False,
        reference_truth_claim=False,
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(m, indent=2) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    try:
        with open(SOURCE, "rb") as f:
            if (
                hashlib.file_digest(f, "md5").hexdigest()
                != "64c33b28fca90fcea7d59272a970b588"
            ):
                raise ValueError("Andrae publisher checksum differs")
        rawfits = Path("/scratch/andrae-full.fits")
        with gzip.open(SOURCE, "rb") as x, rawfits.open("xb") as y:
            shutil.copyfileobj(x, y, 8 * 1024 * 1024)
        with fits.open(rawfits, memmap=True) as f:
            data = f[1].data
            if len(data) != 174922161:
                raise ValueError("Andrae full release row count differs")
            ids = np.asarray(data["source_id"], dtype="i8")
            order = np.argsort(ids)
            ids = ids[order]
            if np.any(np.diff(ids) <= 0):
                raise ValueError("Andrae IDs must be unique")
            n = len(ids)
            locations = np.full(n, -1, dtype="i8")
            shard_index = np.full(n, -1, dtype="i2")
            seen_cache = np.zeros(n, bool)
            m.update(
                status="scanning_raw_input",
                andrae_unique_stars=n,
                raw_rows_scanned=0,
                raw_shards={},
            )
            save()
            before = Path(RAW).stat()
            with h5py.File(RAW) as h:
                keys = [k for k in h if isinstance(h[k], h5py.Dataset)]
                m["raw_rows_total"] = sum(len(h[k]) for k in keys)
                for j, key in enumerate(keys):
                    d = h[key]
                    count = 0
                    for start in range(0, len(d), 500000):
                        sid = d.fields("source_id")[start : start + 500000]
                        ii, ci = match(ids, sid)
                        conflict = locations[ci] >= 0
                        if conflict.any():
                            raise ValueError("Duplicate matching raw Gaia IDs")
                        locations[ci] = start + ii
                        shard_index[ci] = j
                        count += len(ii)
                        m["raw_rows_scanned"] += len(sid)
                    m["raw_shards"][key] = dict(
                        index=j, rows=len(d), andrae_matches=count
                    )
                    m["raw_matches_so_far"] = int((locations >= 0).sum())
                    save()
            after = Path(RAW).stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Raw input changed during census")
            m["raw_identity"] = dict(
                path=RAW, bytes=before.st_size, mtime_ns=before.st_mtime_ns
            )
            m["status"] = "scanning_150m_cache"
            save()
            cache = np.load(CACHE, mmap_mode="r")
            for start in range(0, len(cache), 1000000):
                _, ci = match(ids, cache[start : start + 1000000])
                seen_cache[ci] = True
            with h5py.File(a.output / "andrae-full-inference-index.h5", "x") as o:
                o["source_id"] = ids
                o["publisher_row"] = order
                o["raw_shard_index"] = shard_index
                o["raw_row"] = locations
                o["in_150m_cache"] = seen_cache
                for name in [
                    "teff_xgboost",
                    "logg_xgboost",
                    "mh_xgboost",
                    "in_training_sample",
                ]:
                    o.create_dataset(
                        name, data=np.asarray(data[name])[order], compression="gzip"
                    )
                o.attrs["raw_shards_json"] = json.dumps(keys)
                o.attrs["scope"] = (
                    "Input locations and comparator estimates, not our predictions or independent physical truth"
                )
            m.update(
                status="complete",
                raw_input_overlap=int((locations >= 0).sum()),
                absent_from_raw_input=int((locations < 0).sum()),
                cache_overlap=int(seen_cache.sum()),
                cache_rows=len(cache),
                raw_present_outside_cache=int(((locations >= 0) & ~seen_cache).sum()),
                cache_present_absent_raw=int(((locations < 0) & seen_cache).sum()),
                scope="Full released Andrae catalogue census; catalogue-wide MSA prediction still requires inference; spectroscopy/seismology accuracy only on their exact matched subset",
            )
        with open(a.output / "andrae-full-inference-index.h5", "rb") as f:
            m["index_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        save()
    except Exception as e:
        m.update(status="failed", error=repr(e))
        save()
        raise


if __name__ == "__main__":
    main()
