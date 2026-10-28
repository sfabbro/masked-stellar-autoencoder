"""Materialize source-aware FeH features from the actual benchmark and legacy FITS.

Benchmark members always use their immutable original records, even when incomplete.
Only outside-benchmark stars may use the first original legacy FITS row.
"""

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy.io import fits
from pristine_quality_views import exact_rows
from stellar_age_cohorts import file_digest
from stellar_catalog_jobs import atomic_json
from stellar_dataset_views import BENCHMARK, EXPECTED_IDS, publish
from stellar_enriched_views import ROOT, fit_masks

LAYERS = ("native", "PIGS", "Li2025_CaT_silver")
LABEL_SHA = "dea153be909762099a095b6eb40628de8d932f22d6a8af1903d8e4a09e640d50"
LEGACY_SHA = "b8b502c3c55672557a3e0f32ad488bec8af248062ca74b904ac1bb6c82ee6621"
SCALER_SHA = "5fd976729dfe5ccd7b4ec2ce080e679c49f1ca5244f779cad45e148fa4836037"


def clean(values):
    result = np.asarray(values, dtype="f4").copy()
    result[~np.isfinite(result) | (result == -9999)] = np.nan
    return result


def canonical_rows(measurements, ids, available):
    """Fixed native > PIGS > Li precedence, independent of values/errors/outcomes."""
    layer = np.full(len(ids), -1, dtype="i1")
    row = np.full(len(ids), -1, dtype="i8")
    for number, name in enumerate(LAYERS):
        match = exact_rows(measurements[name]["source_id"], ids)
        take = available & (layer < 0) & (match >= 0)
        layer[take], row[take] = number, match[take]
    return layer, row


def record_projection(records, positions, expected_ids, features, errors, block=8192):
    """One bounded read per occupied contiguous block, preserving original row order.

    Never address train/source_id: the original schema is train/records (compound).
    Field projection avoids materializing covariance arrays in Python memory.
    """
    positions, expected_ids = np.asarray(positions), np.asarray(expected_ids)
    if records.dtype.names is None or "source_id" not in records.dtype.names:
        raise ValueError("Require original compound records with source_id")
    if (
        len(np.unique(positions)) != len(positions)
        or (positions < 0).any()
        or (positions >= len(records)).any()
    ):
        raise ValueError("Ambiguous or out-of-range original benchmark row")
    fields = ["source_id"] + list(
        dict.fromkeys(
            n
            for n in [*features, *errors]
            if n is not None and n in records.dtype.names
        )
    )
    values = np.full((len(positions), len(features)), np.nan, dtype="f4")
    uncertainty = np.full_like(values, np.nan)
    for cell in np.unique(positions // block):
        where = np.flatnonzero(positions // block == cell)
        start = int(cell) * block
        raw = records.fields(fields)[start : min(start + block, len(records))]
        selected = raw[positions[where] - start]
        np.testing.assert_array_equal(selected["source_id"], expected_ids[where])
        for j, name in enumerate(features):
            if name in selected.dtype.names:
                values[where, j] = clean(selected[name])
        for j, name in enumerate(errors):
            if name is not None and name in selected.dtype.names:
                uncertainty[where, j] = clean(selected[name])
    return values, uncertainty


def materialize(
    output,
    scratch,
    *,
    root=ROOT,
    benchmark=BENCHMARK,
    legacy=None,
    expected_ids=EXPECTED_IDS,
    expected_counts=(9000000, 1000000),
    expected_label_sha=LABEL_SHA,
    expected_legacy_sha=LEGACY_SHA,
    expected_scaler_sha=SCALER_SHA,
    config_path="configs/pretrain.canfar.yaml",
):
    started = time.monotonic()
    output, scratch, root = Path(output), Path(scratch), Path(root)
    legacy = Path(
        legacy
        or "/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits"
    )
    label_path = root / "enriched_label_view_v1/enriched-label-view.h5"
    if (
        file_digest(label_path) != expected_label_sha
        or file_digest(legacy) != expected_legacy_sha
    ):
        raise ValueError("Verified input source checksum changed")
    config = yaml.safe_load(Path(config_path).read_text())["data"]
    features, errors = config["feature_cols"], config["error_cols"]
    if len(features) != len(errors) or len(set(features)) != len(features):
        raise ValueError("Feature/error schema mismatch")
    xp = [
        j
        for j, n in enumerate(features)
        if n.startswith(("bp_", "rp_")) and n.split("_")[-1].isdigit()
    ]
    if len(xp) != 110:
        raise ValueError("Require exactly110 XP coefficients")
    with h5py.File(root / "dataset_view_v2/dataset-view.h5") as f:
        train, val = f["train/source_id"][:], f["validation/source_id"][:]
    digest = hashlib.sha256(
        train.astype("<i8").tobytes() + val.astype("<i8").tobytes()
    ).hexdigest()
    if (
        digest != expected_ids
        or (len(train), len(val)) != tuple(expected_counts)
        or len(np.unique(np.r_[train, val])) != len(train) + len(val)
    ):
        raise ValueError("Immutable ordered benchmark index changed")
    with h5py.File(label_path) as f:
        measurements = {n: f[n + "/measurements"][:] for n in LAYERS}
        for n, data in measurements.items():
            if (data["source_id"] <= 0).any() or len(
                np.unique(data["source_id"])
            ) != len(data):
                raise ValueError("Invalid layer IDs")
            if (
                "quantity" in data.dtype.names
                and not (data["quantity"] == b"FeH").all()
            ):
                raise ValueError("MH is not FeH")
            split, membership, broad, strict = fit_masks(data["source_id"], train, val)
            for key, value in dict(
                label_split=split,
                benchmark_partition=membership,
                broader_fit_eligible=broad,
                strict_fit_eligible=strict,
            ).items():
                np.testing.assert_array_equal(f[n + "/" + key][:], value)
    ids = np.unique(np.concatenate([v["source_id"] for v in measurements.values()]))
    values = np.full((len(ids), len(features)), np.nan, dtype="f4")
    uncertainty = np.full_like(values, np.nan)
    input_source = np.full(len(ids), 3, dtype="u1")  # 0train/1val/2legacy/3no input
    input_row = np.full(len(ids), -1, dtype="i8")
    benchmark_summary = {}
    with h5py.File(benchmark) as f:
        if json.loads(f.attrs["provenance"])["source_id_sha256"] != expected_ids:
            raise ValueError("Benchmark provenance changed")
        center, scale = f["reference_scaler/center"][:], f["reference_scaler/scale"][:]
        if (
            hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest()
            != expected_scaler_sha
        ):
            raise ValueError("Frozen scaler changed")
        if (
            center.shape != (len(features),)
            or scale.shape != center.shape
            or not (
                np.isfinite(center).all()
                and np.isfinite(scale).all()
                and (scale > 0).all()
            )
        ):
            raise ValueError("Invalid original feature scaler")
        for code, (split, index) in enumerate([("train", train), ("validation", val)]):
            records = f[split + "/records"]
            if len(records) != expected_counts[code]:
                raise ValueError("Original record count changed")
            positions = exact_rows(index, ids)
            take = np.flatnonzero(positions >= 0)
            v, e = record_projection(
                records, positions[take], ids[take], features, errors
            )
            values[take], uncertainty[take] = v, e
            input_source[take], input_row[take] = code, positions[take]
            benchmark_summary[split] = dict(
                matched_labels=len(take),
                present_feature_columns=[
                    n for n in features if n in records.dtype.names
                ],
                missing_feature_columns=[
                    n for n in features if n not in records.dtype.names
                ],
                record_chunk_shape=records.chunks,
            )
    with fits.open(legacy, memmap=True) as f:
        hdu = next(h for h in f if isinstance(h, fits.BinTableHDU))
        legacy_ids = np.asarray(hdu.data["source_id"], dtype="i8")
        unique, first = np.unique(legacy_ids, return_index=True)
        keep = unique > 0
        unique, first = unique[keep], first[keep]
        match = exact_rows(unique, ids)
        lr = np.where(match >= 0, first[np.maximum(match, 0)], -1)
        outside = (input_source == 3) & (lr >= 0)
        take = np.flatnonzero(outside)
        rows = lr[take]
        np.testing.assert_array_equal(hdu.data["source_id"][rows], ids[take])
        for j, name in enumerate(features):
            if name in hdu.columns.names:
                v = clean(hdu.data[name][rows])
                null = hdu.columns[name].null
                if null is not None:
                    v[v == null] = np.nan
                values[take, j] = v
        for j, name in enumerate(errors):
            if name is not None and name in hdu.columns.names:
                uncertainty[take, j] = clean(hdu.data[name][rows])
        input_source[take], input_row[take] = 2, rows
        # Independent matched-star unit check BEFORE any legacy fallback is published.
        candidates = np.flatnonzero(
            (input_source < 2)
            & (lr >= 0)
            & np.isfinite(values[:, xp]).all(1)
            & np.isfinite(values[:, features.index("G")])
        )
        common = []
        for start in range(0, len(candidates), 2048):
            candidate = candidates[start : start + 2048]
            raw = hdu.data[lr[candidate]]
            finite = np.isfinite(clean(raw["G"]))
            for j in xp:
                finite &= np.isfinite(clean(raw[features[j]]))
            common.extend(candidate[finite].tolist())
            if len(common) >= 128:
                break
        common = np.asarray(common[:128], dtype="i8")
        if len(common) < 128:
            raise ValueError("Insufficient common128 legacy/benchmark unit stars")
        raw = hdu.data[lr[common]]
        g = clean(raw["G"])
        coeff = np.column_stack([clean(raw[features[j]]) for j in xp])
        target = values[common][:, xp]
        np.testing.assert_allclose(
            g, values[common, features.index("G")], rtol=0, atol=0.002
        )
        np.testing.assert_allclose(coeff, target, rtol=0.001, atol=1e-6)
        relative = np.linalg.norm(coeff.astype("f8") - target, axis=1) / np.maximum(
            np.linalg.norm(target, axis=1), 1e-12
        )
        if relative.max() > 0.001:
            raise ValueError("XP normalization mismatch")
        unit_audit = dict(
            matched_stars=len(common),
            all110_coefficients=True,
            chosen_units="g_normalized",
            maximum_relative_l2=float(relative.max()),
            source_id_sha256=hashlib.sha256(ids[common].tobytes()).hexdigest(),
        )
    available = np.isfinite(values[:, xp]).all(1)
    split, membership, broad, strict = fit_masks(ids, train, val)
    broad &= available
    strict &= available
    chosen_layer, chosen_row = canonical_rows(measurements, ids, available)
    selected = np.flatnonzero(chosen_layer >= 0)
    staged = scratch / "enriched-feature-cohort.h5.building"
    summary = {}
    with h5py.File(label_path) as src, h5py.File(staged, "x") as f:
        raw_layers = f.create_group("raw_layers")
        for key, value in src.attrs.items():
            raw_layers.attrs[key] = value
        for n in LAYERS:
            src.copy(n, raw_layers, name=n)
        for key, value in dict(
            source_id=ids,
            values=values,
            reported_feature_errors=uncertainty,
            feature_observed=np.isfinite(values),
            error_observed=np.isfinite(uncertainty) & (uncertainty > 0),
            input_source=input_source,
            input_original_row=input_row,
            feature_available=available,
            label_split=split,
            benchmark_partition=membership,
        ).items():
            f.create_dataset(
                "features/" + key, data=value, compression="gzip", compression_opts=1
            )
        f.create_dataset("reference_scaler/center", data=center)
        f.create_dataset("reference_scaler/scale", data=scale)
        for n, data in measurements.items():
            rows = exact_rows(ids, data["source_id"])
            sp, part, br, st = fit_masks(data["source_id"], train, val)
            br &= available[rows]
            st &= available[rows]
            for key, value in dict(
                feature_row=rows,
                feature_available=available[rows],
                broader_fit=br,
                strict_fit=st,
                identity_test=available[rows] & (sp == 2),
            ).items():
                f.create_dataset(
                    "layer_index/" + n + "/" + key, data=value, compression="gzip"
                )
            summary[n] = dict(
                feature_available=int(available[rows].sum()),
                strict_fit=int(st.sum()),
                broader_fit=int(br.sum()),
                identity_test=int((available[rows] & (sp == 2)).sum()),
                strict_metalpoor=int((st & (data["value"] < -2)).sum()),
            )
        for key, value in dict(
            feature_row=selected,
            source_id=ids[selected],
            layer=chosen_layer[selected],
            layer_row=chosen_row[selected],
            strict_fit=strict[selected],
            broader_fit=broad[selected],
            identity_test=(split[selected] == 2),
        ).items():
            f.create_dataset("canonical/" + key, data=value, compression="gzip")
        f.attrs.update(
            complete=True,
            benchmark_source_id_sha256=digest,
            label_source_sha256=expected_label_sha,
            scaler_sha256=expected_scaler_sha,
            schema_version="enriched-feature-cohort-v1",
            legacy_source_sha256=expected_legacy_sha,
            feature_columns_json=json.dumps(features),
            error_columns_json=json.dumps(errors),
            input_source_codes="0 benchmark train,1 benchmark validation,2 outside-benchmark first-original legacy FITS,3 unavailable",
            layer_codes=json.dumps(LAYERS),
            policy="Raw source layers unchanged; fixed native>PIGS>Li precedence without averaging/shifts/error minimization; correlated repeatedIDs not independent stars. FeH not MH. No DESI gold merge or calibrated NLL sigma claim. Benchmark members always original benchmark inputs, no legacy fallback for incomplete benchmark rows. Broader and strict fits exclude EVERY benchmark validation ID. Source-ID label holdouts do not exclude unsupervised pretraining.",
            unit_audit_json=json.dumps(unit_audit),
            benchmark_input=str(benchmark),
            legacy_input=str(legacy),
        )
        content = hashlib.sha256()

        def update(name, node):
            if isinstance(node, h5py.Dataset):
                content.update(name.encode())
                for start in range(0, len(node), 8192):
                    content.update(node[start : start + 8192].tobytes())

        f.visititems(update)
        f.attrs["content_id"] = content.hexdigest()
    target = output / "enriched-feature-cohort.h5"
    publish(staged, target)
    report = dict(
        source_layers=summary,
        unique_feature_ids=len(ids),
        feature_available_unique=int(available.sum()),
        canonical_selected=len(selected),
        feature_source_counts={
            str(k): int((input_source == k).sum()) for k in range(4)
        },
        benchmark_projection=benchmark_summary,
        unit_audit=unit_audit,
        content_id=content.hexdigest(),
        sha256=file_digest(target),
        bytes=target.stat().st_size,
        runtime_seconds=time.monotonic() - started,
        training_performed=False,
        no_calibrated_NLL_merge=True,
        raw_benchmark_and_legacy_immutable=True,
    )
    atomic_json(output / "enriched-feature-report.json", report)
    return report
