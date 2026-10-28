"""Compare one Flatiron XP binary shard against the corresponding CSV."""

import hashlib
import json
import os
import shutil
import socket
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main():
    # Preserve HTTPS hostname/certificate validation while bypassing worker DNS failure.
    source_host = "sdsc-users.flatironinstitute.org"
    resolved_ip = os.environ.get("XP_SOURCE_IP")
    if resolved_ip:
        original_resolver = socket.getaddrinfo

        def resolve(host, port, *args, **kwargs):
            return original_resolver(
                resolved_ip if host == source_host else host, port, *args, **kwargs
            )

        socket.getaddrinfo = resolve
    root = Path("/arc/projects/k-pop/msa_runs/xp-hdf5-comparison-20261012")
    root.mkdir(exist_ok=True)
    name = "XpContinuousMeanSpectrum_000000-003111"
    source = root / (name + ".hdf5")
    url = (
        "https://sdsc-users.flatironinstitute.org/~gaia/dr3/hdf5/XpContinuousMeanSpectrum/"
        + source.name
    )
    results = {"url": url, "timings": []}

    def record(method, start, **values):
        item = {"method": method, "seconds": time.perf_counter() - start, **values}
        results["timings"].append(item)
        (root / "comparison.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(item), flush=True)

    start = time.perf_counter()
    if not source.exists():
        tmp = Path(str(source) + ".download")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(source)
    record("download_one_hdf5", start, bytes=source.stat().st_size)
    expected = "73b0696ce4d3c163de136ff31df485105752313547fd4ac7321cb1826fa7bf48"
    for repeat in range(2):
        for method in ("selected", "bulk"):
            start = time.perf_counter()
            with h5py.File(source) as f:
                ids = f["source_id"][:]
                indices = np.flatnonzero(ids % 23 == 0)
                arrays = np.stack(
                    [
                        f[b + "_coefficient_correlations"][indices]
                        if method == "selected"
                        else f[b + "_coefficient_correlations"][:][indices]
                        for b in ("bp", "rp")
                    ],
                    axis=1,
                )
            digest = hashlib.sha256(
                ids[indices].tobytes() + arrays.astype("f4").tobytes()
            ).hexdigest()
            assert digest == expected
            record(
                "hdf5_arc_" + method,
                start,
                repeat=repeat,
                rows=len(indices),
                sha256=digest,
            )
    local = Path("/scratch") / source.name
    start = time.perf_counter()
    shutil.copyfile(source, local)
    record("copy_hdf5_to_scratch", start)
    start = time.perf_counter()
    with h5py.File(local) as f:
        ids = f["source_id"][:]
        indices = np.flatnonzero(ids % 23 == 0)
        arrays = np.stack(
            [f[b + "_coefficient_correlations"][indices] for b in ("bp", "rp")], axis=1
        )
    digest = hashlib.sha256(
        ids[indices].tobytes() + arrays.astype("f4").tobytes()
    ).hexdigest()
    assert digest == expected
    record("hdf5_scratch_selected", start, sha256=digest)
    csv = Path("/arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_mean_spectrum") / (
        name + ".csv.gz"
    )
    for repeat in range(2):
        start = time.perf_counter()
        values = []
        matched = []
        for chunk in pd.read_csv(
            csv,
            comment="#",
            usecols=[
                "source_id",
                "bp_coefficient_correlations",
                "rp_coefficient_correlations",
            ],
            chunksize=10000,
        ):
            for row in chunk.loc[chunk.source_id % 23 == 0].to_dict("records"):
                matched.append(int(row["source_id"]))
                values.append(
                    [
                        np.fromstring(
                            row[b + "_coefficient_correlations"].strip("[]() "),
                            sep=",",
                            dtype=np.float32,
                        )
                        for b in ("bp", "rp")
                    ]
                )
        digest = hashlib.sha256(
            np.asarray(matched, dtype="i8").tobytes()
            + np.asarray(values, dtype="f4").tobytes()
        ).hexdigest()
        assert digest == expected
        record("csv_arc", start, repeat=repeat, sha256=digest)
    results["status"] = "complete"
    (root / "comparison.json").write_text(json.dumps(results, indent=2))
    print("RESULT " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
