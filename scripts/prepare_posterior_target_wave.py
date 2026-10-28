"""Prepare real StarHorse age/distance posterior targets and parent86 latents.

No acquisition. Targets are catalogue posterior medians, not raw observations or
independent physical truth. All APOKASC and legacy reference families stay out
of FIT/TUNE. Original benchmark validation never enters FIT/TUNE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import run_xp_representation_probes as probe
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

PARENT_SHA = "d1ed550e3650817df1e1337b4e405fd50fefa43010866b828faa1aedf5040fb3"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def role_indices(ids, masks, *, excluded_ids=(), caps=(4096, 1024, 1024, 2048)):
    """Deterministic identity split: split old TEST into separate CAL/TEST."""
    ids = np.asarray(ids)
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Exact unique integer identities required")
    excluded = np.isin(ids, np.asarray(list(excluded_ids), dtype="i8"))
    buckets = np.array(
        [
            int.from_bytes(hashlib.sha256(str(int(s)).encode()).digest()[:4], "little")
            % 2
            for s in ids
        ]
    )
    candidates = [
        masks["strict_fit"] & ~excluded,
        masks["tune"] & ~excluded,
        masks["identity_test"] & (buckets == 0),
        masks["identity_test"] & (buckets == 1),
    ]
    roles = np.full(len(ids), "", dtype="U5")
    for role, candidate, cap in zip(
        ("fit", "tune", "cal", "test"), candidates, caps, strict=True
    ):
        selected = sorted(
            np.flatnonzero(candidate),
            key=lambda i: hashlib.sha256(str(int(ids[i])).encode()).digest(),
        )[:cap]
        if len(selected) < 16:
            raise ValueError(f"Insufficient real {role} candidates")
        if np.any(roles[selected] != ""):
            raise ValueError("Role identities overlap")
        roles[selected] = role
    return roles


def prepare(base, output):
    if output.exists():
        raise FileExistsError(output)
    parent = base / "main-checkpoints/checkpoint-86.pth"
    if sha(parent) != PARENT_SHA:
        raise ValueError("Parent86 source pin differs")
    age = base / "age_feature_cohort_v1/age-feature-cohort.h5"
    distance = base / "distance-feature-production-local-v2/distance-feature-cohort.h5"
    feature = base / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    anchors = base / "age_teacher_anchors_v2/apokasc3.parquet"
    with h5py.File(age) as a, h5py.File(distance) as d:
        ag, dg = a["selected"], d["selected"]
        ids = ag["source_id"][:]
        dids = dg["source_id"][:]
        index = {int(s): i for i, s in enumerate(dids)}
        mapped = np.array([index.get(int(s), -1) for s in ids])
        age_bounds = ag["age_percentiles"][:, [1, 2, 3]]
        all_distance = dg["distance_percentiles"][:]
        distance_bounds = all_distance[np.maximum(mapped, 0)][:, [1, 2, 3]]
        with h5py.File(feature) as f:
            available_feature_ids = f["features/source_id"][:]
        valid = (mapped >= 0) & np.isin(ids, available_feature_ids)
        for bounds in (age_bounds, distance_bounds):
            valid &= (
                np.isfinite(bounds).all(1)
                & (bounds[:, 0] > 0)
                & (bounds[:, 0] < bounds[:, 1])
                & (bounds[:, 1] < bounds[:, 2])
            )
        masks = {
            k: ag[k][:].astype(bool) & valid
            for k in ("strict_fit", "tune", "identity_test")
        }
        for key in masks:
            masks[key] &= dg[key][:][np.maximum(mapped, 0)].astype(bool)
        # Stronger union holdout includes all Hipparcos and seismic IDs.
        excluded = set(d["ALL_reference_holdout_source_ids"][:].tolist())
        excluded |= set(pd.read_parquet(anchors).source_id.astype(int))
        roles = role_indices(ids, masks, excluded_ids=excluded)
        selected = roles != ""
        ids, roles = ids[selected], roles[selected]
        target = np.column_stack(
            (np.log(age_bounds[selected, 1]), np.log(distance_bounds[selected, 1]))
        )
        quantiles = np.stack((age_bounds[selected], distance_bounds[selected]), axis=1)
        if np.any(np.isin(ids[np.isin(roles, ["fit", "tune"])], list(excluded))):
            raise ValueError("Reference/anchor family leakage into FIT/TUNE")
        partitions = ag["benchmark_partition"][:][selected]
        if np.any(partitions[np.isin(roles, ["fit", "tune"])] == 1):
            raise ValueError("Benchmark validation leakage into FIT/TUNE")
    with h5py.File(feature) as f:
        fids = f["features/source_id"][:]
        index = {int(s): i for i, s in enumerate(fids)}
        idx = np.array([index[int(s)] for s in ids])
        order = np.argsort(idx)
        raw = np.empty((len(ids), f["features/values"].shape[1]))
        raw[order] = f["features/values"][idx[order]]
        names = json.loads(f.attrs["feature_columns_json"])
    checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    signature = checkpoint["run_signature"]
    if names != signature["feature_cols"]:
        raise ValueError("Feature names differ from parent86")
    values = (raw - np.asarray(signature["scaler_center"])) / np.asarray(
        signature["scaler_scale"]
    )
    allowed = np.array(
        [
            n in ("W1", "W2", "G", "BP", "RP") or n.startswith(("bp_", "rp_"))
            for n in names
        ]
    )
    values[:, ~allowed] = -9999
    values[(~np.isfinite(values)) | (raw == -9999)] = -9999
    model = probe.load_model(
        yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
        checkpoint,
        torch.device("cpu"),
    )
    model.eval()
    torch.set_num_threads(4)
    with torch.no_grad():
        embeddings = torch.cat(
            [
                model.encoder(torch.tensor(values[i : i + 128], dtype=torch.float32))
                for i in range(0, len(values), 128)
            ]
        ).numpy()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "prepared.npz",
        source_id=ids,
        role=roles,
        embeddings=embeddings,
        target=target,
        posterior_quantiles=quantiles,
        benchmark_partition=partitions,
    )
    report = dict(
        status="prepared_source_bound_teacher_posterior_targets",
        input_sha256=sha(output / "prepared.npz"),
        encoder_checkpoint_sha256=PARENT_SHA,
        source_provenance={str(p): sha(p) for p in (age, distance, feature, anchors)},
        target_names=["log_age_Gyr", "log_distance_kpc"],
        target_units=["ln(Gyr)", "ln(kpc)"],
        posterior_quantiles=[16, 50, 84],
        role_counts={
            r: int((roles == r).sum()) for r in ("fit", "tune", "cal", "test")
        },
        feature_policy="Parent86 photometry+XP only; astrometry/EBV withheld",
        family_holdout="All APOKASC and Hipparcos/seismic reference union excluded FIT/TUNE",
        target_estimand="StarHorse model/parallax-dependent posterior medians; quantile widths are posterior uncertainties, not independent measurement errors",
        raw_signed_parallax_training=False,
        independent_age_truth_claim=False,
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=Path("experiments/xp/results-stellar-improvement")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base, args.output)


if __name__ == "__main__":
    main()
