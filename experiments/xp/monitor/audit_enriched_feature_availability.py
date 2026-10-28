"""Local actual-XP coverage audit; source layers stay separate and unshifted."""

import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pristine_quality_views import exact_rows
from stellar_age_cohorts import feature_index, file_digest
from stellar_catalog_jobs import identity_split
from stellar_dataset_views import EXPECTED_IDS

root = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
out = root / "enriched-feature-availability-local-v1"
out.mkdir(exist_ok=False)
source = root / "enriched_label_view_v1/enriched-label-view.h5"
assert (
    file_digest(source)
    == "dea153be909762099a095b6eb40628de8d932f22d6a8af1903d8e4a09e640d50"
)
config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
xp = [
    n
    for n in config["data"]["feature_cols"]
    if n.startswith(("bp_", "rp_")) and n.split("_")[-1].isdigit()
]
assert len(xp) == 110
started = time.monotonic()
featureids, featurerows, valid, _ = feature_index(
    "/tmp/msa-labels-complete-20261015.fits", xp
)
with h5py.File(root / "dataset_view_v2/dataset-view.h5") as f:
    train, val = f["train/source_id"][:], f["validation/source_id"][:]
assert (
    hashlib.sha256(
        train.astype("<i8").tobytes() + val.astype("<i8").tobytes()
    ).hexdigest()
    == EXPECTED_IDS
)
summary = {}
allids = []
canonical = {}
with h5py.File(source) as src, h5py.File(out / "feature-availability.h5", "x") as dest:
    dest.attrs["LOCAL_only_not_ARC_published"] = True
    dest.attrs["training_performed"] = False
    dest.attrs["source_sha256"] = file_digest(source)
    dest.attrs["policy"] = (
        "Independent original measurement layers; no shifts/averaging/error-floor changes. Complete finite110XP and first original feature-row availability. Fit masks exclude every benchmark validation ID. Repeated IDs remain correlated layers; no training concatenation."
    )
    for layer in ["native", "PIGS", "Li2025_CaT_silver"]:
        g = src[layer]
        measurement = g["measurements"][:]
        ids = measurement["source_id"]
        assert (ids > 0).all() and len(np.unique(ids)) == len(ids)
        split = identity_split(ids)
        np.testing.assert_array_equal(g["label_split"][:], split)
        membership = np.where(
            np.isin(ids, train), 0, np.where(np.isin(ids, val), 1, 255)
        ).astype("u1")
        np.testing.assert_array_equal(g["benchmark_partition"][:], membership)
        np.testing.assert_array_equal(
            g["strict_fit_eligible"][:], (split == 0) & (membership == 0)
        )
        np.testing.assert_array_equal(
            g["broader_fit_eligible"][:], (split == 0) & (membership != 1)
        )
        index = exact_rows(featureids, ids)
        available = (index >= 0) & valid[np.maximum(index, 0)]
        row = np.where(available, featurerows[np.maximum(index, 0)], -1)
        strict = available & g["strict_fit_eligible"][:]
        broad = available & g["broader_fit_eligible"][:]
        test = available & (split == 2)
        assert not np.isin(ids[strict | broad], val).any()
        group = dest.create_group(layer)
        src.copy(g["measurements"], group, name="original_measurements")
        for key, value in dict(
            source_id=ids,
            feature_row=row,
            feature_available=available,
            label_split=split,
            benchmark_partition=membership,
            strict_fit=strict,
            broader_fit=broad,
            identity_test=test,
        ).items():
            group.create_dataset(key, data=value, compression="gzip")
        np.testing.assert_array_equal(group["original_measurements"][:], measurement)
        value = measurement["value"]
        bins = {
            "below_minus3": value < -3,
            "minus3_to_minus2": (value >= -3) & (value < -2),
            "minus2_to_minus1": (value >= -2) & (value < -1),
            "at_least_minus1": value >= -1,
        }
        summary[layer] = dict(
            raw_labels=len(ids),
            feature_available=int(available.sum()),
            strict_fits=int(strict.sum()),
            broader_fits=int(broad.sum()),
            identity_test=int(test.sum()),
            strict_bins={k: int((strict & v).sum()) for k, v in bins.items()},
            test_bins={k: int((test & v).sum()) for k, v in bins.items()},
        )
        allids.append(ids[available])
        for i in np.flatnonzero(available):
            canonical.setdefault(int(ids[i]), layer)
    dest.attrs["complete"] = True
report = dict(
    LOCAL_only_not_ARC_published=True,
    source_layers=summary,
    feature_available_unique_union=len(canonical),
    duplicate_feature_available_measurement_layers=sum(map(len, allids))
    - len(canonical),
    all_original_measurement_bytes_unchanged=True,
    all_fit_masks_independently_rederived=True,
    no_training_or_NLL_merge=True,
    pretraining_independence_claim=False,
    runtime_seconds=time.monotonic() - started,
)
(out / "coverage-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
