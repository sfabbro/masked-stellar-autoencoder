"""Compare full CSV tokenization with selected-line parsing, threads and processes."""

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def digest_rows(ids, arrays):
    return hashlib.sha256(
        np.asarray(ids, dtype=np.int64).tobytes()
        + np.asarray(arrays, dtype=np.float32).tobytes()
    ).hexdigest()


def read(path, mode):
    ids = []
    arrays = []
    if mode == "pandas":
        columns = [
            "source_id",
            "bp_coefficient_correlations",
            "rp_coefficient_correlations",
        ]
        for chunk in pd.read_csv(path, comment="#", usecols=columns, chunksize=10000):
            for row in chunk.loc[chunk.source_id % 23 == 0].to_dict("records"):
                ids.append(int(row["source_id"]))
                arrays.append(
                    [
                        np.fromstring(
                            row[b + "_coefficient_correlations"].strip("[]() "),
                            sep=",",
                            dtype=np.float32,
                        )
                        for b in ("bp", "rp")
                    ]
                )
    else:
        with gzip.open(path, "rb") as f:
            header = None
            for line in f:
                if line.startswith(b"#"):
                    continue
                if header is None:
                    header = next(csv.reader([line.decode()]))
                    assert header[0] == "source_id"
                    indices = [
                        header.index(b + "_coefficient_correlations")
                        for b in ("bp", "rp")
                    ]
                    continue
                source_id = int(line.split(b",", 1)[0].strip(b'"'))
                if source_id % 23:
                    continue
                row = next(csv.reader([line.decode()]))
                ids.append(source_id)
                arrays.append(
                    [
                        np.fromstring(row[i].strip("[]() "), sep=",", dtype=np.float32)
                        for i in indices
                    ]
                )
    return {"rows": len(ids), "sha256": digest_rows(ids, arrays)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output")
    p.add_argument("--worker", choices=["pandas", "prefilter"])
    args = p.parse_args()
    if args.worker:
        print(json.dumps(read(args.csv, args.worker)))
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    stage = Path("/scratch/xp-readers")
    stage.mkdir(exist_ok=True)
    path = stage / Path(args.csv).name
    start = time.perf_counter()
    shutil.copyfile(args.csv, path)
    results = {
        "copy_seconds": time.perf_counter() - start,
        "source_bytes": path.stat().st_size,
        "timings": [],
    }
    reference = None

    def record(method, start, values):
        nonlocal reference
        if reference is None:
            reference = values[0]
        assert all(v == reference for v in values)
        entry = {
            "method": method,
            "seconds": time.perf_counter() - start,
            "copies": len(values),
            **reference,
        }
        results["timings"].append(entry)
        (root / "readers.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(entry), flush=True)

    for repeat in range(2):
        for mode in ("pandas", "prefilter"):
            start = time.perf_counter()
            record(mode + "_" + str(repeat), start, [read(path, mode)])
    for mode in ("pandas", "prefilter"):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            values = list(executor.map(lambda _: read(path, mode), range(4)))
        record(mode + "_threads4", start, values)
        start = time.perf_counter()
        jobs = [
            subprocess.Popen(
                [sys.executable, __file__, "--csv", str(path), "--worker", mode],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(4)
        ]
        values = []
        for job in jobs:
            stdout, stderr = job.communicate()
            if job.returncode:
                raise RuntimeError(stderr)
            values.append(json.loads(stdout))
        record(mode + "_processes4", start, values)
    results["status"] = "complete"
    (root / "readers.json").write_text(json.dumps(results, indent=2))
    print("RESULT " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
