"""Bounded age/distance head integration on prepared, frozen encoder embeddings.

This entrypoint does not acquire data, run the encoder, or certify native source
associations. The preparation manifest binds its NPZ and records those external
provenance obligations. Only FIT labels enter gradients; TUNE selects one state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from masked_stellar_autoencoder.training.distance_heads import LogDistanceGaussianHead
from masked_stellar_autoencoder.training.multi_teacher_age import (
    DEFAULT_TEACHER_FEATURE_NAMES,
    MultiTeacherAgeHead,
    TeacherLabelBatch,
)

ROLES = ("fit", "tune", "test")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prepared(path: Path, manifest_path: Path, target: str):
    """Fail closed on missing provenance, malformed arrays, units and split IDs."""
    manifest = json.loads(manifest_path.read_text())
    required = {
        "schema_version",
        "target",
        "input_sha256",
        "encoder_checkpoint_sha256",
        "source_provenance",
        "frozen_embeddings",
        "units",
        "split_provenance",
    }
    if required - manifest.keys():
        raise ValueError(
            f"missing preparation metadata: {sorted(required - manifest.keys())}"
        )
    if manifest["schema_version"] != 1 or manifest["target"] != target:
        raise ValueError("manifest target/schema mismatch")
    if manifest["input_sha256"] != digest(path):
        raise ValueError("NPZ SHA mismatch")
    encoder_sha = manifest["encoder_checkpoint_sha256"]
    if (
        not isinstance(encoder_sha, str)
        or len(encoder_sha) != 64
        or any(c not in "0123456789abcdef" for c in encoder_sha)
    ):
        raise ValueError("encoder checkpoint SHA must be hexadecimal SHA256")
    if (
        manifest["frozen_embeddings"] is not True
        or not manifest["source_provenance"]
        or not manifest["split_provenance"]
    ):
        raise ValueError("frozen embedding/source/split provenance required")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    ids, roles, embeddings = data["source_id"], data["role"], data["embeddings"]
    n = len(ids)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != n:
        raise ValueError("source IDs must be unique integers across all roles")
    if roles.shape != (n,) or set(roles.tolist()) != set(ROLES):
        raise ValueError("exactly nonempty fit/tune/test roles required")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != n
        or embeddings.shape[1] < 1
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("invalid frozen embeddings")
    if target == "age":
        if manifest.get("teacher_feature_policy") != "unused_affine":
            raise ValueError("pilot requires teacher_feature_policy=unused_affine")
        if manifest["units"].get("age") != "ln(age/Gyr)":
            raise ValueError("age units must be ln(age/Gyr)")
        names, scales = manifest["teacher_names"], manifest["age_scales"]
        if (
            len(names) != len(set(names))
            or not names
            or len(scales) != len(names)
            or not all(scales)
        ):
            raise ValueError("unique named teachers and age scales required")
        if manifest["reference_teacher"] not in names:
            raise ValueError("reference teacher must define a named scale")
        t = len(names)
        for key in ("teacher_tau", "teacher_sigma", "teacher_quality"):
            if data[key].shape != (t, n):
                raise ValueError(f"{key} must be [teacher,row]")
        if (
            data["teacher_features"].shape != (n, len(DEFAULT_TEACHER_FEATURE_NAMES))
            or not np.isfinite(data["teacher_features"]).all()
        ):
            raise ValueError(
                "teacher features must match declared physical feature order"
            )
        batch = TeacherLabelBatch(
            tuple(names),
            tuple(scales),
            torch.as_tensor(data["teacher_tau"], dtype=torch.float64),
            torch.as_tensor(data["teacher_sigma"], dtype=torch.float64),
            torch.as_tensor(np.isfinite(data["teacher_tau"])),
            torch.as_tensor(data["teacher_quality"], dtype=torch.float64),
        )
        if bool(((batch.quality != 0) & (batch.quality != 1)).any()):
            raise ValueError("joint age pilot requires binary quality")
        for role in ROLES:
            if not bool(
                (
                    batch.observed[:, roles == role]
                    & (batch.quality[:, roles == role] > 0)
                ).any()
            ):
                raise ValueError("each role requires usable teacher evidence")
    else:
        if (
            manifest["units"] != {"parallax": "mas", "latent": "ln(distance/kpc)"}
            or not manifest.get("zero_point_metadata")
            or not manifest.get("effective_sigma_metadata")
        ):
            raise ValueError(
                "distance units, zero-point and effective-sigma provenance required"
            )
        for key in ("parallax_mas", "sigma_eff_mas", "zero_point_mas"):
            if data[key].shape != (n,) or not np.isfinite(data[key]).all():
                raise ValueError(f"invalid {key}")
        if not (data["sigma_eff_mas"] > 0).all():
            raise ValueError("effective sigma must be positive")
    return data, manifest


def run_pilot(
    path: Path,
    manifest_path: Path,
    output: Path,
    *,
    target: str,
    steps: int = 20,
    width: int = 32,
    seed: int = 42,
    lr: float = 1e-3,
    objective: str = "joint",
):
    """Train only a fresh head on frozen FIT embeddings and select on TUNE NLL."""
    if objective not in ("joint", "composite"):
        raise ValueError("Unknown age objective")
    if (
        target not in ("age", "distance")
        or not 1 <= steps <= 1000
        or not 1 <= width <= 256
        or not np.isfinite(lr)
        or not 0 < lr <= 0.1
    ):
        raise ValueError("invalid bounded pilot parameters")
    data, manifest = load_prepared(path, manifest_path, target)
    output.mkdir(parents=True, exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if target == "age":
            head = MultiTeacherAgeHead(
                manifest["teacher_names"],
                input_dim=data["embeddings"].shape[1],
                width=width,
                reference_teacher=manifest["reference_teacher"],
                calibration_mode="affine",
            ).double()
        else:
            head = LogDistanceGaussianHead(
                input_dim=data["embeddings"].shape[1], width=width
            ).double()
    embeddings = torch.tensor(data["embeddings"], dtype=torch.float64)
    masks = {role: data["role"] == role for role in ROLES}

    def loss(role):
        mask = masks[role]
        if target == "age":
            tau = torch.tensor(data["teacher_tau"][:, mask], dtype=torch.float64)
            labels = TeacherLabelBatch(
                tuple(manifest["teacher_names"]),
                tuple(manifest["age_scales"]),
                tau,
                torch.tensor(data["teacher_sigma"][:, mask], dtype=torch.float64),
                torch.isfinite(tau),
                torch.tensor(data["teacher_quality"][:, mask], dtype=torch.float64),
            )
            return head.loss(
                embeddings[mask],
                torch.tensor(data["teacher_features"][mask], dtype=torch.float64),
                labels,
                objective=objective,
            )
        return head.loss(
            embeddings[mask],
            torch.tensor(data["parallax_mas"][mask], dtype=torch.float64),
            torch.tensor(data["sigma_eff_mas"][mask], dtype=torch.float64),
            zero_point_mas=torch.tensor(
                data["zero_point_mas"][mask], dtype=torch.float64
            ),
            order=64,
        )

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    best, selected, history = float("inf"), None, []
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        fit = loss("fit").mean
        if not bool(torch.isfinite(fit)):
            raise ValueError("nonfinite FIT loss")
        fit.backward()
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in head.parameters()
        ):
            raise ValueError("nonfinite head gradients")
        optimizer.step()
        with torch.no_grad():
            tune = float(loss("tune").mean)
        if not np.isfinite(tune):
            raise ValueError("nonfinite TUNE loss")
        history.append({"step": step, "fit_nll": float(fit.detach()), "tune_nll": tune})
        if tune < best:
            best, selected = (
                tune,
                {
                    "head": copy.deepcopy(head.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "step": step,
                },
            )
    head.load_state_dict(selected["head"], strict=True)
    torch.save(selected, output / "checkpoint.pth")
    arrays = {}
    with torch.no_grad():
        for role in ROLES:
            prediction = head(embeddings[masks[role]])
            arrays[f"{role}_source_id"] = data["source_id"][masks[role]]
            if not bool(
                torch.isfinite(prediction["mean"]).all()
                & torch.isfinite(prediction["log_scale"]).all()
            ):
                raise ValueError("nonfinite selected predictions")
            arrays[f"{role}_mean"] = prediction["mean"].numpy()
            arrays[f"{role}_log_scale"] = prediction["log_scale"].numpy()
    np.savez(output / "predictions.npz", **arrays)
    result = {
        "status": "bounded_frozen_embedding_pilot_complete",
        "target": target,
        "scope": "head integration only; no native-source, encoder, quadrature-convergence or production certificate",
        "input_sha256": digest(path),
        "preparation_manifest_sha256": digest(manifest_path),
        "runner_sha256": digest(Path(__file__)),
        "encoder_checkpoint_sha256": manifest["encoder_checkpoint_sha256"],
        "checkpoint_sha256": digest(output / "checkpoint.pth"),
        "predictions_sha256": digest(output / "predictions.npz"),
        "head_configuration": {
            "class": type(head).__name__,
            "input_dim": embeddings.shape[1],
            "width": width,
            "calibration_mode": "affine" if target == "age" else None,
        },
        "selected_step": selected["step"],
        "completed_steps": steps,
        "seed": seed,
        "selection": "first minimum tune NLL; test targets never used in optimization or selection",
        "role_counts": {role: int(mask.sum()) for role, mask in masks.items()},
        "history": history,
        "objective": (
            "shared_latent_joint_gaussian"
            if objective == "joint"
            else "weighted_composite_marginal"
        )
        if target == "age"
        else "signed_parallax_convolution_GH64",
        "coordinate": manifest["units"],
        "preparation": manifest,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("age", "distance"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    run_pilot(
        args.input, args.manifest, args.output, target=args.target, steps=args.steps
    )


if __name__ == "__main__":
    main()
