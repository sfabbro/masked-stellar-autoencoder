"""Build a seeded, split-preserving HDF5 benchmark with packed XP correlations."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def allocations(sizes, total):
    if total > sum(sizes):
        raise ValueError("Requested sample exceeds available rows")
    exact = np.asarray(sizes, dtype=np.float64) * total / sum(sizes)
    result = np.floor(exact).astype(int)
    order = np.argsort(-(exact - result), kind="stable")
    result[order[: total - int(result.sum())]] += 1
    return result


def read_correlations(path, sorted_ids):
    columns = ["source_id"] + [
        f"{b}_{k}"
        for b in ("bp", "rp")
        for k in (
            "coefficient_correlations",
            "basis_function_id",
            "n_parameters",
            "n_relevant_bases",
            "standard_deviation",
        )
    ]
    positions = []
    values = []
    metadata = []
    for chunk in pd.read_csv(path, comment="#", usecols=columns, chunksize=10000):
        ids = chunk.source_id.to_numpy(dtype=np.int64)
        pos = np.searchsorted(sorted_ids, ids)
        keep = (pos < len(sorted_ids)) & (
            sorted_ids[np.minimum(pos, len(sorted_ids) - 1)] == ids
        )
        selected = chunk.loc[keep]
        for row, p in zip(selected.itertuples(index=False, name=None), pos[keep]):
            record = dict(zip(selected.columns, row))
            bands = []
            meta = []
            for b in ("bp", "rp"):
                v = np.fromstring(
                    str(record[b + "_coefficient_correlations"]).strip("[]() "),
                    sep=",",
                    dtype=np.float32,
                )
                if (
                    len(v) != 1485
                    or not np.isfinite(v).all()
                    or np.any(abs(v) > 1.00001)
                ):
                    raise ValueError(
                        f"Invalid {b} correlations for source {record['source_id']}"
                    )
                bands.append(v)
                meta.append(
                    [
                        record[b + "_" + k]
                        for k in (
                            "basis_function_id",
                            "n_parameters",
                            "n_relevant_bases",
                            "standard_deviation",
                        )
                    ]
                )
            positions.append(p)
            values.append(bands)
            metadata.append(meta)
    return (
        str(path),
        np.asarray(positions, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        np.asarray(metadata, dtype=np.float64),
    )


def publish_hdf5(staged, output):
    """Sequentially compress a closed scratch file into an atomic ARC artifact."""
    copying = Path(str(output) + ".copying")
    if output.exists() or copying.exists():
        raise FileExistsError(output)
    with h5py.File(staged, "r") as source, h5py.File(copying, "x") as destination:
        for key, value in source.attrs.items():
            destination.attrs[key] = value

        def copy_group(src, dst):
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for name, obj in src.items():
                if isinstance(obj, h5py.Group):
                    copy_group(obj, dst.create_group(name))
                    continue
                if obj.ndim == 0:
                    target = dst.create_dataset(name, data=obj[()])
                else:
                    block = min(
                        128 if name == "xp_correlations" else 2048, obj.shape[0]
                    )
                    target = dst.create_dataset(
                        name,
                        obj.shape,
                        dtype=obj.dtype,
                        chunks=(block, *obj.shape[1:]),
                        compression="gzip",
                        compression_opts=1,
                        shuffle=True,
                    )
                    for start in range(0, obj.shape[0], block):
                        values = obj[start : start + block]
                        if name == "xp_correlations" and not np.isfinite(values).all():
                            raise ValueError("Incomplete correlation staging file")
                        target[start : start + block] = values
                for key, value in obj.attrs.items():
                    target.attrs[key] = value

        copy_group(source, destination)
        destination.flush()
    with h5py.File(staged, "r") as source, h5py.File(copying, "r") as destination:
        for split in ("train", "validation"):
            for name in source[split]:
                a, b = source[split][name], destination[split][name]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError("Published schema mismatch")
                for begin in (0, max(0, len(a) // 2 - 2), max(0, len(a) - 4)):
                    if a[begin : begin + 4].tobytes() != b[begin : begin + 4].tobytes():
                        raise ValueError("Published round-trip mismatch")
    os.replace(copying, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-keys", nargs="+", required=True)
    parser.add_argument("--train-rows", type=int, default=9000000)
    parser.add_argument("--validation-rows", type=int, default=1000000)
    parser.add_argument("--parent-checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scratch-dir", type=Path)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_root = args.scratch_dir or output.parent
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / (output.name + ".building")
    if output.exists() or temporary.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    manifest = vars(args).copy()
    manifest["scratch_dir"] = str(args.scratch_dir) if args.scratch_dir else None
    manifest.update(
        status="building",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        covariance_policy="Original Gaia packed correlations; diagonal implicit one; existing G-normalized standard errors retained unchanged. No BP/RP cross covariance or propagated G uncertainty. Scale covariance at training time.",
        sample_policy="Seeded uniform sampling without replacement within every shard; counts proportional to shard size; original validation membership preserved.",
    )

    def report(**updates):
        manifest.update(updates)
        manifest["elapsed_seconds"] = time.perf_counter() - started
        Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(updates), flush=True)

    with h5py.File(args.source) as source, h5py.File(temporary, "x") as out:
        keys = sorted(
            k
            for k in source
            if isinstance(source[k], h5py.Dataset) and source[k].dtype.names
        )
        if not set(args.validation_keys) <= set(keys):
            raise ValueError("Validation shard missing")
        dtype = source[keys[0]].dtype
        if any(source[k].dtype != dtype for k in keys):
            raise ValueError("Mixed record schemas")
        out.attrs["parent_checkpoint"] = args.parent_checkpoint
        if args.parent_checkpoint:
            import torch

            checkpoint = torch.load(
                args.parent_checkpoint, map_location="cpu", weights_only=False
            )
            signature = checkpoint["run_signature"]
            scaler = out.create_group("reference_scaler")
            scaler.create_dataset("center", data=signature["scaler_center"])
            scaler.create_dataset("scale", data=signature["scaler_scale"])
            scaler.attrs["feature_cols"] = json.dumps(signature["feature_cols"])
            scaler.attrs["recon_cols"] = json.dumps(signature["recon_cols"])
            scaler.attrs["checkpoint_epoch"] = checkpoint["epoch"]
            digest = hashlib.sha256()
            with open(args.parent_checkpoint, "rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            scaler.attrs["checkpoint_sha256"] = digest.hexdigest()
            del checkpoint
        out.attrs["schema_version"] = 1
        out.attrs["record_policy"] = (
            "All original HDF5 fields preserved, without refitting or applying a scaler."
        )
        all_ids = []
        all_splits = []
        all_rows = []
        counts = {}
        for split, total in [
            ("train", args.train_rows),
            ("validation", args.validation_rows),
        ]:
            selected = [
                k
                for k in keys
                if (k in args.validation_keys) == (split == "validation")
            ]
            sizes = [len(source[k]) for k in selected]
            amounts = allocations(sizes, total)
            group = out.create_group(split)
            records = group.create_dataset(
                "records",
                (total,),
                dtype=dtype,
                chunks=(min(2048, total),),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            group.create_dataset(
                "source_shard",
                (total,),
                dtype="u2",
                chunks=(min(65536, total),),
                compression="gzip",
                compression_opts=1,
            )
            group.create_dataset(
                "source_row",
                (total,),
                dtype="i8",
                chunks=(min(65536, total),),
                compression="gzip",
                compression_opts=1,
            )
            group.attrs["source_shard_names"] = json.dumps(selected)
            offset = 0
            for shard, (key, n) in enumerate(zip(selected, amounts)):
                indices = np.sort(rng.choice(len(source[key]), int(n), replace=False))
                for begin in range(0, len(source[key]), 100000):
                    chosen = indices[(indices >= begin) & (indices < begin + 100000)]
                    if not len(chosen):
                        continue
                    block = source[key][begin : begin + 100000][chosen - begin]
                    end = offset + len(block)
                    records[offset:end] = block
                    group["source_shard"][offset:end] = shard
                    group["source_row"][offset:end] = chosen
                    all_ids.append(block["source_id"].astype(np.int64))
                    all_splits.append(
                        np.full(len(block), split == "validation", dtype=np.uint8)
                    )
                    all_rows.append(np.arange(offset, end, dtype=np.int64))
                    offset = end
                counts[key] = int(n)
                report(stage="sample", shard=key, split=split, sampled_rows=offset)
            assert offset == total
            corr = group.create_dataset(
                "xp_correlations",
                (total, 2, 1485),
                dtype="f4",
                # Random writes must not repeatedly recompress 128-star chunks.
                # Contiguous staging uses the scratch disk/page cache; publish sequentially.
                fillvalue=np.nan,
            )
            corr.attrs["band_order"] = "BP,RP"
            corr.attrs["packing"] = (
                "Strict lower triangle in numpy.tril_indices(55,-1) order; diagonal implicit one"
            )
            meta = group.create_dataset(
                "xp_band_metadata",
                (total, 2, 4),
                dtype="f8",
                chunks=(min(1024, total), 2, 4),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
                fillvalue=np.nan,
            )
            meta.attrs["columns"] = (
                "basis_function_id,n_parameters,n_relevant_bases,standard_deviation"
            )
            group.create_dataset(
                "xp_covariance_available",
                (total,),
                dtype="?",
                chunks=(min(65536, total),),
                compression="gzip",
                compression_opts=1,
            )
        ids = np.concatenate(all_ids)
        splits = np.concatenate(all_splits)
        rows = np.concatenate(all_rows)
        del all_ids, all_splits, all_rows
        order = np.argsort(ids)
        sorted_ids = ids[order]
        if np.any(np.diff(sorted_ids) <= 0):
            raise ValueError("Duplicate or unordered source IDs")
        report(
            stage="correlations",
            sampling_seconds=time.perf_counter() - started,
            counts_by_shard=counts,
            source_id_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        )
        files = sorted(Path(args.csv_dir).glob("XpContinuousMeanSpectrum_*.csv.gz"))
        if not files:
            raise FileNotFoundError(args.csv_dir)
        # Bound queued results: each worker reads one CSV and only returns selected stars.
        found = np.zeros(len(ids), dtype=bool)
        matched = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            pending = {}
            next_file = iter(files)
            for _ in range(args.workers):
                p = next(next_file, None)
                if p:
                    pending[executor.submit(read_correlations, p, sorted_ids)] = p
            while pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    del pending[future]
                    path, pos, corr, meta = future.result()
                    global_rows = order[pos]
                    if np.any(found[global_rows]):
                        raise ValueError("Duplicate CSV source")
                    found[global_rows] = True
                    for split, value in [("train", 0), ("validation", 1)]:
                        mask = splits[global_rows] == value
                        destination = rows[global_rows[mask]]
                        sorting = np.argsort(destination)
                        destination = destination[sorting]
                        if len(destination):
                            # Contiguous runs avoid huge HDF5 fancy-selection lists.
                            c = corr[mask][sorting]
                            m = meta[mask][sorting]
                            for j in range(0, len(destination), 512):
                                ix = destination[j : j + 512]
                                out[split]["xp_correlations"][ix] = c[j : j + 512]
                                out[split]["xp_band_metadata"][ix] = m[j : j + 512]
                                out[split]["xp_covariance_available"][ix] = True
                    matched += len(pos)
                    out.flush()
                    report(
                        stage="correlations",
                        csv=path,
                        matched_rows=matched,
                        missing_rows=len(ids) - matched,
                        output_bytes=temporary.stat().st_size,
                    )
                    p = next(next_file, None)
                    if p:
                        pending[executor.submit(read_correlations, p, sorted_ids)] = p
        if not found.all():
            raise ValueError(
                f"Missing CSV correlations for {(~found).sum()} sampled sources"
            )
        # Round-trip checks retain exact source IDs, errors and packed correlations.
        for split in ("train", "validation"):
            group = out[split]
            if not group["xp_covariance_available"][:].all():
                raise ValueError("Incomplete availability")
            if not np.isfinite(group["xp_correlations"][:128]).all():
                raise ValueError("Invalid covariance probe")
        out.attrs["provenance"] = json.dumps(manifest)
        out.flush()
    report(stage="publishing", staging_bytes=temporary.stat().st_size)
    publish_hdf5(temporary, output)
    report(status="complete", stage="verified", output_bytes=output.stat().st_size)
    temporary.unlink()


if __name__ == "__main__":
    main()
