"""Prepare source-pinned APOKASC3 ages with object/family-disjoint role IDs.

Uses corrected signed-error anchor v2, not raw released Willett Gaia IDs.
No StarHorse/latent-teacher identification claim: this is a separate native
APOKASC3 age-scale prediction experiment. No acquisitions or encoder updates.
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
from prepare_posterior_target_wave import PARENT_SHA, role_indices, sha

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted


def prepare(base, output):
    if output.exists():
        raise FileExistsError(output)
    parent = base / "main-checkpoints/checkpoint-86.pth"
    anchors = base / "age_teacher_anchors_v2/apokasc3.parquet"
    feature = base / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    age = base / "age_feature_cohort_v1/age-feature-cohort.h5"
    if (
        sha(parent) != PARENT_SHA
        or sha(anchors)
        != "ab1cdcc3bce73a36c2dde85508fbafb0a7234d18540599e9674adeace9fff2b3"
    ):
        raise ValueError("Corrected APOKASC v2 or parent86 pin changed")
    apo = pd.read_parquet(anchors)
    if apo.source_id.duplicated().any():
        raise ValueError("APOKASC identities duplicated")
    apo = apo.loc[apo.age_bounds_valid].sort_values("source_id")
    with h5py.File(age) as h5:
        reference_ids = h5["external_reference/source_id"][:]
    with h5py.File(feature) as f:
        fids = f["features/source_id"][:]
        index = {int(s): i for i, s in enumerate(fids)}
        apo = apo.loc[
            np.isin(apo.source_id, fids) & ~np.isin(apo.source_id, reference_ids)
        ]
        ids = apo.source_id.to_numpy(dtype="i8")
        rows = np.array([index[int(s)] for s in ids])
        partitions = np.asarray(f["features/benchmark_partition"][:])[rows]
        bucket = np.array(
            [
                int.from_bytes(
                    hashlib.sha256(str(int(s)).encode()).digest()[:4], "little"
                )
                % 10
                for s in ids
            ]
        )
        masks = dict(
            strict_fit=(bucket < 6) & (partitions != 1),
            tune=(bucket >= 6) & (bucket < 8) & (partitions != 1),
            identity_test=bucket >= 8,
        )
        roles = role_indices(ids, masks, caps=(4096, 1024, 1024, 2048))
        chosen = roles != ""
        ids, roles, rows, partitions = (
            ids[chosen],
            roles[chosen],
            rows[chosen],
            partitions[chosen],
        )
        apo = apo.iloc[np.flatnonzero(chosen)]
        bounds = apo[["age16", "age50", "age84"]].to_numpy()
        target = np.log(bounds[:, 1:2])
        order = np.argsort(rows)
        raw = np.empty((len(rows), f["features/values"].shape[1]))
        raw[order] = f["features/values"][rows[order]]
        names = json.loads(f.attrs["feature_columns_json"])
    checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    signature = checkpoint["run_signature"]
    if names != signature["feature_cols"]:
        raise ValueError("Parent86 feature names changed")
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
        posterior_quantiles=bounds[:, None, :],
        benchmark_partition=partitions,
    )
    report = dict(
        status="prepared_source_bound_apokasc3_age_scale",
        input_sha256=sha(output / "prepared.npz"),
        encoder_checkpoint_sha256=PARENT_SHA,
        source_provenance={str(p): sha(p) for p in (anchors, feature, age)},
        target_names=["log_age_Gyr"],
        target_units=["ln(Gyr)"],
        posterior_quantiles=[16, 50, 84],
        role_counts={
            r: int((roles == r).sum()) for r in ("fit", "tune", "cal", "test")
        },
        family_holdout="Object IDs are shared across APOKASC/Willett Kepler family; SHA256 exactGaiaID roles prevent same-star family leakage; all legacy66reference IDs excluded all roles; benchmarkvalidation excludedFIT/TUNE",
        feature_policy="Parent86 photometry+XP only, no astrometry/EBV",
        target_estimand="APOKASC3 model-dependent seismic age posterior median in Gyr; corrected age16=age50+signedlower, age84=age50+upper; not independent absolute age truth",
        cross_teacher_scale_identified=False,
        raw_signed_parallax_training=False,
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base", type=Path, default=Path("experiments/xp/results-stellar-improvement")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.base, a.output)


if __name__ == "__main__":
    main()
