"""Source-aware native/PIGS/Li indexes with immutable benchmark holdouts."""

import gzip
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from pristine_quality_views import exact_rows
from stellar_catalog_jobs import atomic_json, identity_split
from stellar_dataset_views import EXPECTED_IDS, publish

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def fit_masks(ids, benchmark_train, benchmark_validation):
    """Source-ID labels alone never authorize fitting a benchmark validation star."""
    ids = np.asarray(ids)
    train = exact_rows(benchmark_train, ids) >= 0
    validation = exact_rows(benchmark_validation, ids) >= 0
    if (train & validation).any():
        raise ValueError("Benchmark train/validation overlap")
    split = identity_split(ids)
    membership = np.where(train, 0, np.where(validation, 1, 255)).astype("u1")
    broad_fit = (split == 0) & ~validation
    strict_fit = broad_fit & train
    return split, membership, broad_fit, strict_fit


def build_view(output, scratch, root=ROOT):
    from li2025_labels import build as build_li

    source_paths = {
        "native": root / "label_registry_v2/labels.h5",
        "PIGS": root / "pigs_labels_v1/pigs-labels.h5",
        "li_gz": root / "li2025_collection_v1/li2025-table2.dat.gz",
        "li_readme": root / "li2025_collection_v1/li2025-ReadMe",
        "base_index": root / "dataset_view_v2/dataset-view.h5",
    }
    provenance = {}
    for key, path in source_paths.items():
        provenance[key] = dict(
            path=str(path),
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    for key, expected in {
        "li_gz": "8eaf96716c4c118db4a1701b5f5a779b077f06de04c2c281c23c0d709ea9b421",
        "li_readme": "c140ce88e8ebbc900397d21038e46fb644e3529dfd5bfb231ab5a33d187f041c",
    }.items():
        if provenance[key]["sha256"] != expected:
            raise ValueError("Published Li source checksum changed")
    with gzip.open(source_paths["li_gz"], "rb") as f:
        table = scratch / "table2.dat"
        table.write_bytes(f.read())
    li_report = build_li(table, source_paths["li_readme"], output / "li2025-silver")
    with h5py.File(source_paths["base_index"]) as f:
        train = f["train/source_id"][:]
        validation = f["validation/source_id"][:]
    if len(train) != 9000000 or len(validation) != 1000000:
        raise ValueError("Expected exact 9M/1M benchmark")
    digest = hashlib.sha256(
        train.astype("<i8").tobytes() + validation.astype("<i8").tobytes()
    ).hexdigest()
    if digest != EXPECTED_IDS or len(np.unique(np.r_[train, validation])) != 10000000:
        raise ValueError("Benchmark ordered ID checksum/uniqueness changed")
    datasets = {}
    with h5py.File(source_paths["native"]) as f:
        datasets["native"] = f["selected/labels"][:]
    with h5py.File(source_paths["PIGS"]) as f:
        datasets["PIGS"] = f["selected/labels"][:]
    with h5py.File(output / "li2025-silver/li2025-silver-labels.h5") as f:
        datasets["Li2025_CaT_silver"] = f["selected/measurements"][:]
    staged = scratch / "enriched-label-view.h5.building"
    summary = {}
    combined_ids = []
    with h5py.File(staged, "x") as out:
        out.attrs["policy"] = (
            "Independent source layers; no mean/offset/error-floor merge. DESI excluded pending uncertainty calibration. Li is CaT silver, not HRS gold. Broader fits exclude benchmark validation; strict fits also require benchmark train. Unsupervised pretraining overlap is not excluded."
        )
        out.attrs["benchmark_source_id_sha256"] = digest
        out.attrs["input_provenance_json"] = json.dumps(provenance)
        content = hashlib.sha256()
        for name, rows in datasets.items():
            ids = rows["source_id"]
            split, membership, broad, strict = fit_masks(ids, train, validation)
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Source selected layer has duplicate IDs")
            if "split" in rows.dtype.names:
                np.testing.assert_array_equal(split, rows["split"])
            group = out.create_group(name)
            group.create_dataset("measurements", data=rows, compression="gzip")
            for key, value in dict(
                label_split=split,
                benchmark_partition=membership,
                broader_fit_eligible=broad,
                strict_fit_eligible=strict,
            ).items():
                group.create_dataset(key, data=value, compression="gzip")
                content.update(value.tobytes())
            content.update(rows.tobytes())
            group.attrs["uncertainty_policy"] = (
                "Reported errors and separate 0.2dex allowance; inclusion unresolved, no gold NLL promotion"
                if name.startswith("Li")
                else "Native published uncertainties retained; PIGS published 0.13dex floor already included, no additional floor applied"
            )
            summary[name] = dict(
                unique_ids=len(ids),
                strict_fit=int(strict.sum()),
                broader_fit=int(broad.sum()),
                benchmark_validation=int((membership == 1).sum()),
            )
            combined_ids.append(ids)
        out.attrs["cross_source_policy"] = (
            "The same ID can appear in multiple layers; consumers must choose source precedence or model correlated observations explicitly before training; never concatenate as independent stars."
        )
        out.attrs["content_id"] = content.hexdigest()
        out.attrs["complete"] = True
    target = output / "enriched-label-view.h5"
    publish(staged, target)
    result = dict(
        source_layers=summary,
        unique_ids_union=len(np.unique(np.concatenate(combined_ids))),
        benchmark_source_id_sha256=digest,
        inputs=provenance,
        li_ingestion=li_report,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        bytes=target.stat().st_size,
        feature_data="Indexed association to existing 10M benchmark; no new 20M-star claim",
        training_performed=False,
    )
    atomic_json(output / "enriched-view-report.json", result)
    return result
