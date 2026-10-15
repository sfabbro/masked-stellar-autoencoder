"""Read-only benchmark indexes and explicitly derived age/distance posteriors."""

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

BENCHMARK = (
    "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5"
)
EXPECTED_IDS = "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"


def publish(staged, output):
    copying = output.with_suffix(output.suffix + ".copying")
    if output.exists() or copying.exists():
        raise FileExistsError(output)
    shutil.copyfile(staged, copying)
    with h5py.File(staged) as original, h5py.File(copying) as copied:
        assert original.attrs["complete"] and copied.attrs["complete"]
        assert original.attrs["content_id"] == copied.attrs["content_id"]

        def verify(name, node):
            if isinstance(node, h5py.Dataset):
                other = copied[name]
                assert node.shape == other.shape and node.dtype == other.dtype
                for start in range(0, len(node), 131072):
                    left = node[start : start + 131072]
                    right = other[start : start + 131072]
                    if node.dtype.names:
                        assert left.tobytes() == right.tobytes(), name
                    else:
                        np.testing.assert_array_equal(left, right)

        original.visititems(verify)
    copying.replace(output)


def posterior_tables(output, scratch, paths):
    """Preserve all percentiles and flags; validity is not survey quality certification."""
    staged = scratch / "posterior-labels.h5.building"
    summaries = []
    with h5py.File(staged, "x") as out:
        out.attrs["dependency_caveat"] = (
            "StarHorse posteriors use stellar/photometric/astrometric models; "
            "not independent geometric distances or asteroseismic ages. "
            "Survey flags are retained verbatim; quality interpretation pending."
        )
        for index, source in enumerate(paths):
            local = scratch / f"posterior-input-{index}.fits"
            shutil.copyfile(source, local)
            group = out.create_group(Path(source).stem)
            with fits.open(local, memmap=True) as hdul:
                hdu = next(h for h in hdul if isinstance(h, fits.BinTableHDU))
                names = {c.lower(): c for c in hdu.columns.names}
                id_name = next(
                    (names[n] for n in ["gaiaedr3", "source_id"] if n in names), None
                )
                if id_name is None:
                    raise ValueError(f"No Gaia source identifier in {source}")
                ids = np.asarray(hdu.data[id_name], dtype="i8")
                group.create_dataset("source_id", data=ids, compression="gzip")
                group.attrs["source_file"] = str(source)
                group.attrs["gaia_release"] = (
                    "EDR3; exact identity join to DR3 required"
                )
                for prefix, unit in [("Age", "Gyr"), ("Dist", "kpc")]:
                    columns = [
                        prefix + suffix for suffix in ["05", "16", "50", "84", "95"]
                    ]
                    if not set(columns) <= set(hdu.columns.names):
                        raise ValueError(
                            f"Missing posterior percentiles: {source} {prefix}"
                        )
                    for name in columns:
                        if hdu.columns[name].unit != unit:
                            raise ValueError(
                                f"Unexpected unit for {name}: {hdu.columns[name].unit}"
                            )
                    values = np.column_stack([hdu.data[n] for n in columns]).astype(
                        "f8"
                    )
                    valid = (ids > 0) & np.isfinite(values).all(1) & (values > 0).all(1)
                    valid &= (np.diff(values, axis=1) >= 0).all(1)
                    ds = group.create_dataset(
                        prefix + "_percentiles", data=values, compression="gzip"
                    )
                    ds.attrs["unit"] = unit
                    ds.attrs["percentiles"] = [5, 16, 50, 84, 95]
                    group.create_dataset(
                        prefix + "_posterior_valid", data=valid, compression="gzip"
                    )
                for name in hdu.columns.names:
                    if "flag" in name.lower():
                        raw = np.asarray(hdu.data[name])
                        if raw.dtype.kind == "U":
                            raw = raw.astype("S")
                        group.create_dataset(
                            "flags/" + name, data=raw, compression="gzip"
                        )
                group.create_dataset(
                    "source_row", data=np.arange(len(ids)), compression="gzip"
                )
                summaries.append(
                    {"path": str(source), "rows": len(ids), "flags_audited": False}
                )
            local.unlink()
        out.attrs.update(
            complete=True, content_id=json.dumps(summaries, sort_keys=True)
        )
    destination = output / "posterior-labels.h5"
    publish(staged, destination)
    return {
        "file": str(destination),
        "sources": summaries,
        "bytes": destination.stat().st_size,
        "independent_age_distance_truth": False,
        "quality_flag_audit_pending": True,
    }


def dataset_view(
    output,
    scratch,
    registry,
    benchmark=BENCHMARK,
    expected_hash=EXPECTED_IDS,
    expected_counts=(9000000, 1000000),
):
    """Index all benchmark stars; labels carry a separate source-ID holdout split."""
    with h5py.File(registry) as labels:
        selected = labels["selected/labels"][:]
    order = np.argsort(selected["source_id"])
    label_ids = selected["source_id"][order]
    if len(np.unique(label_ids)) != len(label_ids):
        raise ValueError("Registry selected labels must be unique")
    staged = scratch / "dataset-view.h5.building"
    summaries, all_ids = {}, []
    with h5py.File(benchmark, "r") as source, h5py.File(staged, "x") as out:
        provenance = json.loads(source.attrs["provenance"])
        if provenance["source_id_sha256"] != expected_hash:
            raise ValueError("Benchmark provenance hash mismatch")
        for split, count in zip(["train", "validation"], expected_counts):
            group = source[split]
            records = group["records"]
            if len(records) != count:
                raise ValueError(f"Unexpected benchmark {split} count")
            ids = np.asarray(records.fields("source_id")[:], dtype="<i8")
            all_ids.append(ids)
            positions = np.searchsorted(label_ids, ids)
            match = positions < len(label_ids)
            if len(label_ids):
                match &= label_ids[np.minimum(positions, len(label_ids) - 1)] == ids
            else:
                match[:] = False
            label_row = np.full(count, -1, dtype="i8")
            label_row[match] = order[positions[match]]
            partition = np.full(count, 255, dtype="u1")
            partition[match] = selected["split"][label_row[match]]
            g = out.create_group(split)
            for name, values in [
                ("source_id", ids),
                ("selected_label_row", label_row),
                ("label_holdout_split", partition),
            ]:
                g.create_dataset(name, data=values, compression="gzip")
            # Access to every original record, scaler, correlation and row/shard provenance.
            g["benchmark"] = h5py.ExternalLink(str(benchmark), "/" + split)
            summaries[split] = {
                "stars": count,
                "matched_FeH": int(match.sum()),
                "sft_fit_eligible": int((match & (partition == 0)).sum())
                if split == "train"
                else 0,
            }
        joined_ids = np.concatenate(all_ids)
        sorted_ids = np.sort(joined_ids)
        if np.any(sorted_ids[1:] == sorted_ids[:-1]):
            raise ValueError("Duplicate benchmark IDs or train/validation overlap")
        digest = hashlib.sha256(joined_ids.tobytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError("Actual benchmark source-ID hash mismatch")
        out["reference_scaler"] = h5py.ExternalLink(str(benchmark), "/reference_scaler")
        out["label_registry"] = h5py.ExternalLink(str(registry), "/")
        out.attrs.update(
            complete=True,
            content_id=digest,
            benchmark=str(benchmark),
            registry=str(registry),
            split_policy="SFT fits train benchmark stars with label_holdout_split=0 only; "
            "benchmark validation never used for SFT fitting",
            storage="Index with absolute read-only external links; not a standalone data copy",
        )
    destination = output / "dataset-view.h5"
    publish(staged, destination)
    with h5py.File(destination) as check:
        assert check["train/benchmark/records"].dtype == source_dtype(benchmark)
        assert len(check["label_registry/selected/labels"]) == len(selected)
    return {
        "file": str(destination),
        "bytes": destination.stat().st_size,
        "source_id_sha256": digest,
        "counts": summaries,
        "unique_ids_and_no_overlap": True,
        "publication_roundtrip": True,
        "storage": "index plus original immutable benchmark and registry; all three required",
    }


def source_dtype(benchmark):
    with h5py.File(benchmark) as f:
        return f["train/records"].dtype
