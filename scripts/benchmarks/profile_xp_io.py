"""Separate XP CSV read/parse and representative random HDF5 write costs."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    local = Path("/scratch/msa-xp-profile")
    local.mkdir(exist_ok=True)
    source = Path(args.csv)
    staged = local / source.name
    results = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "timings": [],
    }

    def record(name, started, **extra):
        entry = {"phase": name, "seconds": time.perf_counter() - started, **extra}
        results["timings"].append(entry)
        (root / "profile.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(entry), flush=True)

    start = time.perf_counter()
    shutil.copyfile(source, staged)
    record("copy_gzip_arc_to_scratch", start)
    cols = ["source_id", "bp_coefficient_correlations", "rp_coefficient_correlations"]
    reference = None
    for repeat in range(2):
        for location, path in [("ARC", source), ("scratch", staged)]:
            start = time.perf_counter()
            chunks = []
            count = 0
            for chunk in pd.read_csv(path, comment="#", usecols=cols, chunksize=10000):
                count += len(chunk)
                chunks.append(chunk.loc[chunk.source_id % 23 == 0])
            selected = pd.concat(chunks, ignore_index=True)
            record(
                "pandas_read_" + location,
                start,
                repeat=repeat,
                rows=count,
                selected=len(selected),
            )
            start = time.perf_counter()
            values = np.asarray(
                [
                    [
                        np.fromstring(
                            str(row[b + "_coefficient_correlations"]).strip("[]() "),
                            sep=",",
                            dtype=np.float32,
                        )
                        for b in ("bp", "rp")
                    ]
                    for row in selected.to_dict("records")
                ]
            )
            ids = selected.source_id.to_numpy()
            if values.shape != (len(selected), 2, 1485):
                raise ValueError(values.shape)
            digest = hashlib.sha256(ids.tobytes() + values.tobytes()).hexdigest()
            if reference is None:
                reference = digest
            assert digest == reference
            record(
                "selected_float_parse",
                start,
                repeat=repeat,
                location=location,
                sha256=digest,
            )
    start = time.perf_counter()
    parsed = []
    streamed_ids = []
    with gzip.open(staged, "rt") as stream:
        for row in csv.DictReader(line for line in stream if not line.startswith("#")):
            if int(row["source_id"]) % 23:
                continue
            streamed_ids.append(int(row["source_id"]))
            parsed.append(
                [
                    np.fromstring(
                        row[b + "_coefficient_correlations"].strip("[]() "),
                        sep=",",
                        dtype=np.float32,
                    )
                    for b in ("bp", "rp")
                ]
            )
    assert (
        hashlib.sha256(
            np.asarray(streamed_ids, dtype=np.int64).tobytes()
            + np.asarray(parsed, dtype=np.float32).tobytes()
        ).hexdigest()
        == reference
    )
    record("stdlib_csv_scratch_with_parse", start, selected=len(parsed))
    # Model the current 9M-row scatter pattern without paying initial full fill.
    n = min(2048, len(values))
    v = values[:n]
    positions = np.sort(np.random.default_rng(42).choice(9000000, n, replace=False))
    with h5py.File(local / "writes.h5", "w") as f:

        def create(name):
            space = h5py.h5s.create_simple((9000000, 2, 1485))
            props = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            props.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
            ident = h5py.h5d.create(
                f.id, name.encode(), h5py.h5t.IEEE_F32LE, space, dcpl=props
            )
            return h5py.Dataset(ident)

        for mode in ["fancy512", "row_slices"]:
            d = create(mode)
            start = time.perf_counter()
            for begin in range(0, n, 512):
                ix = positions[begin : begin + 512]
                block = v[begin : begin + 512]
                t = time.perf_counter()
                if mode == "fancy512":
                    d[ix] = block
                else:
                    for i, row in zip(ix, block):
                        d[int(i) : int(i) + 1] = row[None]
                print(
                    json.dumps(
                        {
                            "write_mode": mode,
                            "written_rows": begin + len(ix),
                            "batch_seconds": time.perf_counter() - t,
                        }
                    ),
                    flush=True,
                )
            f.flush()
            record("correlation_write_" + mode, start, rows=n)
            for j in range(n):
                assert d[int(positions[j])].tobytes() == v[j].tobytes()
        for mode in ["fancy512", "row_slices"]:
            d = f.create_dataset(
                "metadata_" + mode,
                (9000000, 2, 4),
                dtype="f8",
                chunks=(1024, 2, 4),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            m = np.ones((n, 2, 4))
            start = time.perf_counter()
            for begin in range(0, n, 512):
                ix = positions[begin : begin + 512]
                block = m[begin : begin + 512]
                if mode == "fancy512":
                    d[ix] = block
                else:
                    for i, row in zip(ix, block):
                        d[int(i) : int(i) + 1] = row[None]
            f.flush()
            record("metadata_write_" + mode, start, rows=n)
    results["status"] = "complete"
    (root / "profile.json").write_text(json.dumps(results, indent=2))
    print("RESULT " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
