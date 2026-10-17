"""Measure a Gaia XP correlation sidecar on one original CSV shard, aligned to HDF5 IDs."""

import csv
import gzip
import json
import time
from pathlib import Path

import h5py
import numpy as np


def main():
    start = time.perf_counter()
    root = Path("/arc/projects/k-pop/msa_runs/xp-covariance-benchmark")
    root.mkdir(exist_ok=True)
    csvroot = Path("/arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_mean_spectrum")
    source = sorted(csvroot.glob("XpContinuousMeanSpectrum_*.csv.gz"))[0]
    output = root / (source.stem.replace(".csv", "") + "-correlations.h5")
    if output.exists():
        raise FileExistsError(output)
    rows = 0
    parse_seconds = 0.0
    write_seconds = 0.0
    buffers = []
    with h5py.File(output, "w") as out:
        out.attrs["source_csv"] = str(source)
        out.attrs["representation"] = (
            "Gaia original packed BP/RP correlations; strict lower triangle; diagonal implicit 1. Covariance reconstructed with errors in training HDF5."
        )
        ids = out.create_dataset(
            "source_id",
            (0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(1024,),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        corr = out.create_dataset(
            "correlations",
            (0, 2, 1485),
            maxshape=(None, 2, 1485),
            dtype="f4",
            chunks=(128, 1, 1485),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        meta = out.create_dataset(
            "band_metadata",
            (0, 2, 4),
            maxshape=(None, 2, 4),
            dtype="f8",
            chunks=(1024, 2, 4),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        meta.attrs["columns"] = (
            "basis_function_id,n_parameters,n_relevant_bases,standard_deviation"
        )

        def flush():
            nonlocal rows, write_seconds
            if not buffers:
                return
            t = time.perf_counter()
            n = len(buffers)
            ids.resize(rows + n, axis=0)
            corr.resize(rows + n, axis=0)
            meta.resize(rows + n, axis=0)
            ids[rows:] = [x[0] for x in buffers]
            corr[rows:] = np.stack([x[1] for x in buffers])
            meta[rows:] = np.stack([x[2] for x in buffers])
            rows += n
            buffers.clear()
            out.flush()
            write_seconds += time.perf_counter() - t
            if rows % 10240 == 0:
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "elapsed_seconds": time.perf_counter() - start,
                            "bytes": output.stat().st_size,
                        }
                    ),
                    flush=True,
                )

        with gzip.open(source, "rt") as f:
            reader = csv.DictReader(line for line in f if not line.startswith("#"))
            for row in reader:
                t = time.perf_counter()
                bands = []
                metadata = []
                for band in ("bp", "rp"):
                    values = np.fromstring(
                        row[band + "_coefficient_correlations"].strip("[]() "),
                        sep=",",
                        dtype=np.float32,
                    )
                    if len(values) != 1485:
                        raise ValueError((row["source_id"], band, len(values)))
                    bands.append(values)
                    metadata.append(
                        [
                            float(row[band + "_" + key])
                            for key in (
                                "basis_function_id",
                                "n_parameters",
                                "n_relevant_bases",
                                "standard_deviation",
                            )
                        ]
                    )
                buffers.append(
                    (int(row["source_id"]), np.stack(bands), np.array(metadata))
                )
                parse_seconds += time.perf_counter() - t
                if len(buffers) >= 1024:
                    flush()
            flush()
    scanstart = time.perf_counter()
    matches = {}
    side_ids = None
    with h5py.File(output) as f:
        side_ids = f["source_id"][:]
    with h5py.File(
        "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"
    ) as f:
        for key in f:
            ds = f[key]
            if not hasattr(ds, "dtype") or "source_id" not in (ds.dtype.names or ()):
                continue
            count = int(np.isin(ds.fields("source_id")[:], side_ids).sum())
            if count:
                matches[key] = count
    result = {
        "source_csv": str(source),
        "source_compressed_bytes": source.stat().st_size,
        "output": str(output),
        "rows": rows,
        "output_bytes": output.stat().st_size,
        "bytes_per_star": output.stat().st_size / rows,
        "parse_seconds": parse_seconds,
        "write_seconds": write_seconds,
        "build_seconds": scanstart - start,
        "alignment_scan_seconds": time.perf_counter() - scanstart,
        "training_matches": matches,
        "projected_bytes_10M": output.stat().st_size / rows * 10_000_000,
        "projected_bytes_218M": output.stat().st_size / rows * 218_000_000,
        "uncompressed_float32_packed_bytes_per_star": 11880,
        "full_two_55_by_55_matrices_bytes_per_star": 24200,
    }
    (root / "benchmark.json").write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
