import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, "scripts")
from stellar_catalog_jobs import identity_split

root = Path("experiments/xp/results-stellar-improvement")


def unique_best(records, indices):
    order = np.lexsort((records["uncertainty"], records["source_id"]))
    records = records[order]
    indices = indices[order]
    keep = (
        np.r_[True, np.diff(records["source_id"]) != 0]
        if len(records)
        else np.array([], bool)
    )
    return records[keep], indices[keep]


def summaries(delta, sigma):
    if not len(delta):
        return {"n": 0}
    med = float(np.median(delta))
    z = delta / sigma
    return dict(
        n=len(delta),
        median_delta_dex=med,
        mae_delta_dex=float(abs(delta).mean()),
        robust_scatter_dex=float(1.4826 * np.median(abs(delta - med))),
        median_reported_pair_sigma_dex=float(np.median(sigma)),
        fraction_abs_delta_gt_3sigma=float((abs(z) > 3).mean()),
        robust_scatter_normalized=float(1.4826 * np.median(abs(z - np.median(z)))),
    )


# Nonmonotonic fixture: lowest quoted uncertainty retained per duplicate ID.
a = np.array(
    [(9, 0.4), (2, 0.1), (9, 0.2)], dtype=[("source_id", "i8"), ("uncertainty", "f8")]
)
a, ii = unique_best(a, np.arange(3))
assert a["source_id"].tolist() == [2, 9] and ii.tolist() == [1, 2]
with h5py.File(root / "label_registry_v2/labels.h5") as f:
    ds = f["measurements"]
    data = ds.fields(
        [
            "source_id",
            "value",
            "uncertainty",
            "survey",
            "quality_pass",
            "quantity",
            "split",
        ]
    )[:]
original = np.arange(len(data))
by = {}
for tag in [b"galah", b"apogee"]:
    sel = (data["survey"] == tag) & data["quality_pass"] & (data["quantity"] == b"FeH")
    by[tag] = unique_best(data[sel], original[sel])
g, gi = by[b"galah"]
a, ai = by[b"apogee"]
ids, ig, ia = np.intersect1d(
    g["source_id"], a["source_id"], assume_unique=True, return_indices=True
)
g, a = g[ig], a[ia]
gi, ai = gi[ig], ai[ia]
assert np.array_equal(g["source_id"], a["source_id"])
split = identity_split(ids)
np.testing.assert_array_equal(split, g["split"])
np.testing.assert_array_equal(split, a["split"])
delta = g["value"] - a["value"]
sigma = np.hypot(g["uncertainty"], a["uncertainty"])
offset = float(np.median(delta[split == 0])) if (split == 0).any() else None
report = {
    "comparison": "native quality-pass GALAH minus APOGEE; exact GaiaDR3 IDs; smallest reported error per survey if duplicate",
    "pairs": len(ids),
    "training_only_median_offset_dex": offset,
    "bins": {},
    "caveats": [
        "Neither survey is treated as absolute truth.",
        "Quadrature errors assume independent random errors; shared systematics/covariances may matter.",
        "An additive training-only median is an exploratory diagnostic, not an approved abundance correction or calibrated total uncertainty.",
        "APOGEE quality cuts and survey overlap select this sample; metal-poor overlap may be sparse.",
    ],
}
for splitid, label in [(0, "train"), (1, "tune"), (2, "test")]:
    for lo, hi, b in [
        (-np.inf, np.inf, "all"),
        (-np.inf, -2, "APOGEE_FeH_lt_minus2"),
        (-2, -1, "APOGEE_FeH_minus2_to_minus1"),
        (-1, np.inf, "APOGEE_FeH_ge_minus1"),
    ]:
        mask = (split == splitid) & (a["value"] >= lo) & (a["value"] < hi)
        r = summaries(delta[mask], sigma[mask])
        r["after_train_offset"] = (
            summaries(delta[mask] - offset, sigma[mask]) if offset is not None else None
        )
        report["bins"][label + "_" + b] = r
np.savez_compressed(
    root / "native-overlap-pairs.npz",
    source_id=ids,
    split=split,
    galah_feh=g["value"],
    apogee_feh=a["value"],
    galah_uncertainty=g["uncertainty"],
    apogee_uncertainty=a["uncertainty"],
    galah_registry_measurement_row=gi,
    apogee_registry_measurement_row=ai,
)
(root / "native-overlap-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
