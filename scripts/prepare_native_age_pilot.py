"""Prepare bounded, source-bound parent86 embeddings for the joint-age pilot.

Uses existing local catalogues only. No acquisition, label imputation or encoder
updates. Independent object hash roles are for this pilot only, not a new
independent science test after repeated programme exploration.
"""

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


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base", type=Path, default=Path("experiments/xp/results-stellar-improvement")
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    base = args.base
    args.output.mkdir(parents=True, exist_ok=False)
    parent = base / "main-checkpoints/checkpoint-86.pth"
    if (
        sha(parent)
        != "d1ed550e3650817df1e1337b4e405fd50fefa43010866b828faa1aedf5040fb3"
    ):
        raise ValueError("Parent86 SHA differs")
    cohort = base / "age_feature_cohort_v1/age-feature-cohort.h5"
    anchors = base / "age_teacher_anchors_v2/apokasc3.parquet"
    feature = base / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    labels = {}
    with h5py.File(cohort) as f:
        reference_ids = set(f["external_reference/source_id"][:].tolist())
        selected = f["selected"]
        ages = selected["age_percentiles"][:]
        ids = selected["source_id"][:]
        eligible = selected["strict_fit"][:].astype(bool)
        for sid, bounds in zip(ids[eligible], ages[eligible][:, [1, 2, 3]]):
            if np.isfinite(bounds).all() and 0 < bounds[0] < bounds[1] < bounds[2]:
                labels.setdefault(int(sid), {})[0] = (
                    np.log(bounds[1]),
                    (np.log(bounds[2]) - np.log(bounds[0])) / 2,
                )
    apo = pd.read_parquet(anchors)
    if apo.source_id.duplicated().any():
        raise ValueError("Duplicate anchor identities")
    for row in apo.loc[apo.age_bounds_valid].itertuples():
        labels.setdefault(int(row.source_id), {})[1] = (
            np.log(row.age50),
            (np.log(row.age84) - np.log(row.age16)) / 2,
        )
    with h5py.File(feature) as f:
        feature_ids = f["features/source_id"][:]
        if len(np.unique(feature_ids)) != len(feature_ids):
            raise ValueError("Duplicate feature identities")
        index = {int(s): i for i, s in enumerate(feature_ids)}
        chosen = []
        roles = []
        counts = {"fit": 0, "tune": 0, "test": 0}
        for sid in sorted(
            labels, key=lambda s: hashlib.sha256(str(s).encode()).digest()
        ):
            bucket = (
                int.from_bytes(hashlib.sha256(str(sid).encode()).digest()[:4], "little")
                % 10
            )
            role = "fit" if bucket < 6 else "tune" if bucket < 8 else "test"
            cap = 256 if role == "fit" else 64
            if sid in index and sid not in reference_ids and counts[role] < cap:
                chosen.append(sid)
                roles.append(role)
                counts[role] += 1
        if min(counts.values()) < 16:
            raise ValueError("Insufficient native pilot roles")
        idx = np.array([index[s] for s in chosen])
        order = np.argsort(idx)
        raw = np.empty((len(idx), f["features/values"].shape[1]), dtype="f8")
        raw[order] = f["features/values"][idx[order]]
        names = json.loads(f.attrs["feature_columns_json"])
    checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    signature = checkpoint["run_signature"]
    if names != signature["feature_cols"]:
        raise ValueError("Parent/feature column order differs")
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
                model.encoder(torch.tensor(x, dtype=torch.float32))
                for x in np.array_split(values, 6)
            ]
        ).numpy()
    tau = np.full((2, len(chosen)), np.nan)
    sigma = np.full_like(tau, np.nan)
    quality = np.zeros_like(tau)
    for j, sid in enumerate(chosen):
        for teacher, (t, s) in labels[sid].items():
            tau[teacher, j] = t
            sigma[teacher, j] = s
            quality[teacher, j] = 1.0
    path = args.output / "pilot.npz"
    np.savez_compressed(
        path,
        embeddings=embeddings,
        source_id=np.array(chosen, dtype="i8"),
        role=np.array(roles),
        teacher_tau=tau,
        teacher_sigma=sigma,
        teacher_quality=quality,
        teacher_features=np.zeros((len(chosen), 5)),
    )
    document = dict(
        schema_version=1,
        target="age",
        input_sha256=sha(path),
        encoder_checkpoint_sha256=sha(parent),
        frozen_embeddings=True,
        split_provenance="SHA256(source_id) modulo10: fit0-5/tune6-7/test8-9; all legacy seismic reference IDs excluded; bounded pilot roles only",
        teacher_names=["starhorse", "apokasc3"],
        age_scales=["starhorse", "apokasc3"],
        reference_teacher="starhorse",
        teacher_feature_policy="unused_affine",
        units={"age": "ln(age/Gyr)"},
        source_provenance={str(p): sha(p) for p in (cohort, anchors, feature)},
        role_counts=counts,
        teacher_counts=(quality > 0).sum(1).tolist(),
        overlap_rows=int(((quality > 0).sum(0) > 1).sum()),
        scope="Native bounded code-path pilot; no giant-scale transfer or independent science claim",
    )
    (args.output / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document))


if __name__ == "__main__":
    main()
