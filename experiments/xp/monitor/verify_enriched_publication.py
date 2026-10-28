"""Audit actual enriched publication bytes without using its builder.

A missing original benchmark leaves its raw-record verification explicitly pending.
Run with --benchmark on a host where the unchanged original is readable for the
complete gate. No training, output publication or remote sessions are performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

BASE = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
LAYERS = ("native", "PIGS", "Li2025_CaT_silver")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8388608), b""):
            h.update(b)
    return h.hexdigest()


def cleaned(a):
    a = np.array(a, dtype="f4")
    a[(a == -9999) | ~np.isfinite(a)] = np.nan
    return a


def split(ids):
    x = ids.astype("u8") ^ np.uint64(42)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return np.select([x % 1000 < 700, x % 1000 < 850], [0, 1], default=2).astype("u1")


def verify(args):
    assert (
        sha(args.labels)
        == "dea153be909762099a095b6eb40628de8d932f22d6a8af1903d8e4a09e640d50"
    )
    assert (
        sha(args.legacy)
        == "b8b502c3c55672557a3e0f32ad488bec8af248062ca74b904ac1bb6c82ee6621"
    )
    with h5py.File(args.index) as idx:
        train, val = idx["train/source_id"][:], idx["validation/source_id"][:]
    allids = np.r_[train, val]
    assert len(train) == 9000000 and len(val) == 1000000
    assert len(np.unique(allids)) == 10000000
    ordered_sha = hashlib.sha256(allids.astype("<i8").tobytes()).hexdigest()
    assert (
        ordered_sha
        == "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
    )
    report = {}
    with h5py.File(args.cohort) as f, h5py.File(args.labels) as labels:
        assert (
            f.attrs["complete"] and f.attrs["benchmark_source_id_sha256"] == ordered_sha
        )
        h = hashlib.sha256()

        def check_content(name, node):
            if isinstance(node, h5py.Dataset):
                h.update(name.encode())
                for start in range(0, len(node), 8192):
                    h.update(node[start : start + 8192].tobytes())

        f.visititems(check_content)
        assert h.hexdigest() == f.attrs["content_id"]
        for key, value in labels.attrs.items():
            np.testing.assert_array_equal(f["raw_layers"].attrs[key], value)

        def check_raw(name, node):
            actual = f["raw_layers/" + name]
            for key, value in node.attrs.items():
                np.testing.assert_array_equal(actual.attrs[key], value)
            if isinstance(node, h5py.Dataset):
                assert node.shape == actual.shape and node.dtype == actual.dtype
                assert node[:].tobytes() == actual[:].tobytes(), name

        labels.visititems(check_raw)
        ids = f["features/source_id"][:]
        np.testing.assert_array_equal(
            ids,
            np.unique(
                np.concatenate(
                    [labels[n + "/measurements"]["source_id"] for n in LAYERS]
                )
            ),
        )
        values, errors = (
            f["features/values"][:],
            f["features/reported_feature_errors"][:],
        )
        names, error_names = (
            json.loads(f.attrs["feature_columns_json"]),
            json.loads(f.attrs["error_columns_json"]),
        )
        xp = [
            j
            for j, n in enumerate(names)
            if n.startswith(("bp_", "rp_")) and n.split("_")[-1].isdigit()
        ]
        assert len(xp) == 110 and values.shape == errors.shape == (len(ids), len(names))
        np.testing.assert_array_equal(
            f["features/feature_observed"][:], np.isfinite(values)
        )
        np.testing.assert_array_equal(
            f["features/error_observed"][:], np.isfinite(errors) & (errors > 0)
        )
        available = np.isfinite(values[:, xp]).all(1)
        np.testing.assert_array_equal(f["features/feature_available"][:], available)
        sp = split(ids)
        partition = np.select(
            [np.isin(ids, train), np.isin(ids, val)], [0, 1], default=255
        ).astype("u1")
        np.testing.assert_array_equal(f["features/label_split"][:], sp)
        np.testing.assert_array_equal(f["features/benchmark_partition"][:], partition)
        source, rows = (
            f["features/input_source"][:],
            f["features/input_original_row"][:],
        )
        for code, index in enumerate([train, val]):
            take = partition == code
            assert (source[take] == code).all()
            np.testing.assert_array_equal(index[rows[take]], ids[take])
        assert ((source == 2) | (source == 3))[partition == 255].all()
        canonical_layer = np.full(len(ids), -1, dtype="i1")
        canonical_row = np.full(len(ids), -1, dtype="i8")
        summary = {}
        for code, n in enumerate(LAYERS):
            measurement = labels[n + "/measurements"][:]
            positions = np.searchsorted(ids, measurement["source_id"])
            np.testing.assert_array_equal(ids[positions], measurement["source_id"])
            np.testing.assert_array_equal(
                f["layer_index/" + n + "/feature_row"][:], positions
            )
            a, s, p = available[positions], sp[positions], partition[positions]
            for key, v in dict(
                feature_available=a,
                strict_fit=a & (s == 0) & (p == 0),
                broader_fit=a & (s == 0) & (p != 1),
                identity_test=a & (s == 2),
            ).items():
                np.testing.assert_array_equal(f["layer_index/" + n + "/" + key][:], v)
            take = a & (canonical_layer[positions] < 0)
            canonical_layer[positions[take]] = code
            canonical_row[positions[take]] = np.flatnonzero(take)
            summary[n] = dict(
                available=int(a.sum()),
                strict=int((a & (s == 0) & (p == 0)).sum()),
                broader=int((a & (s == 0) & (p != 1)).sum()),
                test=int((a & (s == 2)).sum()),
            )
        chosen = np.flatnonzero(canonical_layer >= 0)
        for key, v in dict(
            feature_row=chosen,
            source_id=ids[chosen],
            layer=canonical_layer[chosen],
            layer_row=canonical_row[chosen],
            strict_fit=(sp[chosen] == 0) & (partition[chosen] == 0),
            broader_fit=(sp[chosen] == 0) & (partition[chosen] != 1),
            identity_test=sp[chosen] == 2,
        ).items():
            np.testing.assert_array_equal(f["canonical/" + key][:], v)
        with fits.open(args.legacy, memmap=True) as fitsfile:
            data = fitsfile[1].data
            unique, first = np.unique(
                np.asarray(data["source_id"], dtype="i8"), return_index=True
            )
            outside = np.flatnonzero(source == 2)
            pos = np.searchsorted(unique, ids[outside])
            np.testing.assert_array_equal(unique[pos], ids[outside])
            np.testing.assert_array_equal(first[pos], rows[outside])
            for j, n in enumerate(names):
                raw = (
                    cleaned(data[n][rows[outside]])
                    if n in data.names
                    else np.full(len(outside), np.nan, dtype="f4")
                )
                np.testing.assert_array_equal(raw, values[outside, j])
                en = error_names[j]
                eraw = (
                    cleaned(data[en][rows[outside]])
                    if en is not None and en in data.names
                    else np.full(len(outside), np.nan, dtype="f4")
                )
                np.testing.assert_array_equal(eraw, errors[outside, j])
            absent = np.flatnonzero(source == 3)
            assert (
                not np.isin(ids[absent], unique).any()
                and np.isnan(values[absent]).all()
                and (rows[absent] == -1).all()
            )
            # All legacy-overlapping benchmark inputs, not a chosen favorable subset.
            common = np.flatnonzero((source < 2) & np.isin(ids, unique))
            original = first[np.searchsorted(unique, ids[common])]
            matched_xp = 0
            for j in [names.index("G"), *xp]:
                raw = cleaned(data[names[j]][original])
                both = np.isfinite(raw) & np.isfinite(values[common, j])
                np.testing.assert_allclose(
                    values[common[both], j],
                    raw[both],
                    rtol=0.001,
                    atol=0.002 if names[j] == "G" else 1e-6,
                )
                if j in xp:
                    matched_xp += int(both.sum())
        center, scale = f["reference_scaler/center"][:], f["reference_scaler/scale"][:]
        assert (
            hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest()
            == f.attrs["scaler_sha256"]
            == "5fd976729dfe5ccd7b4ec2ce080e679c49f1ca5244f779cad45e148fa4836037"
        )
        benchmark_checked = args.benchmark is not None
        if args.benchmark:
            with h5py.File(args.benchmark) as bench:
                assert (
                    json.loads(bench.attrs["provenance"])["source_id_sha256"]
                    == ordered_sha
                )
                for code, part in enumerate(["train", "validation"]):
                    ds = bench[part + "/records"]
                    assert len(ds) == [9000000, 1000000][code]
                    wanted = np.flatnonzero(source == code)
                    order = wanted[np.argsort(rows[wanted])]
                    # Independent h5py sorted fancy reads, without builder block projection.
                    for start in range(0, len(order), 256):
                        take = order[start : start + 256]
                        raw = ds[rows[take]]
                        np.testing.assert_array_equal(raw["source_id"], ids[take])
                        for j, n in enumerate(names):
                            expected = (
                                cleaned(raw[n])
                                if n in raw.dtype.names
                                else np.full(len(take), np.nan, dtype="f4")
                            )
                            np.testing.assert_array_equal(values[take, j], expected)
                            n = error_names[j]
                            expected = (
                                cleaned(raw[n])
                                if n is not None and n in raw.dtype.names
                                else np.full(len(take), np.nan, dtype="f4")
                            )
                            np.testing.assert_array_equal(errors[take, j], expected)
        report.update(
            cohort_sha256=sha(args.cohort),
            bytes=Path(args.cohort).stat().st_size,
            content_id=h.hexdigest(),
            source_layers=summary,
            all_raw_layers_byte_exact=True,
            all_masks_and_precedence_independently_verified=True,
            all_legacy_fallback_fields_verified=True,
            benchmark_legacy_overlap_ids=len(common),
            benchmark_legacy_overlap_finite_xp_values=matched_xp,
            all_original_benchmark_raw_records_verified=benchmark_checked,
            publication_gate_complete=benchmark_checked,
            pending=[]
            if benchmark_checked
            else [
                "Independent full original benchmark raw-record comparison; no refined GPU launch until resolved"
            ],
        )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True)
    p.add_argument(
        "--labels", default=str(BASE / "enriched_label_view_v1/enriched-label-view.h5")
    )
    p.add_argument("--index", default=str(BASE / "dataset_view_v2/dataset-view.h5"))
    p.add_argument("--legacy", default="/tmp/msa-labels-complete-20261015.fits")
    p.add_argument("--benchmark")
    p.add_argument("--report", required=True)
    verify(p.parse_args())
