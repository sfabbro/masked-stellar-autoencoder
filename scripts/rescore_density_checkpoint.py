"""Versioned full-role FP64 fixed-checkpoint scoring; no fitting or old-output edits.

The required policy JSON pins every input and computational implementation. Output
is separate from the original run and does not clear any previous failed audit.
A subsequent independent comparison is required before corrected metrics ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import run_multilabel_screen as scoring
import run_xp_representation_probes as architecture
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training import multilabel_experiments as heads

SCORING_VERSION = "fp64-full-role-v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_policy(policy, directory, parent, model_config, *, installed_revision):
    """Validate immutable artifact and implementation pins before loading weights."""
    required = (
        "run_id",
        "source_sha",
        "library_revision",
        "checkpoint_sha256",
        "paired_sha256",
        "manifest_sha256",
        "parent_sha256",
        "model_config_sha256",
        "scoring_code_sha256",
        "head_code_sha256",
        "architecture_code_sha256",
        "head_kwargs",
    )
    if any(key not in policy for key in required):
        raise ValueError("policy missing required artifact/implementation pin")
    for key in required:
        if key.endswith("_sha256") and not re.fullmatch("[0-9a-f]{64}", policy[key]):
            raise ValueError(f"invalid SHA256 policy: {key}")
    if installed_revision != policy["library_revision"]:
        raise ValueError("installed torchregress revision differs from policy")
    paths = {
        "checkpoint_sha256": directory / "checkpoint.pth",
        "paired_sha256": directory / "paired-cohort.npz",
        "manifest_sha256": directory / "manifest.json",
        "parent_sha256": parent,
        "model_config_sha256": model_config,
        "scoring_code_sha256": Path(scoring.__file__),
        "head_code_sha256": Path(heads.__file__),
        "architecture_code_sha256": Path(architecture.__file__),
    }
    for key, path in paths.items():
        if digest(path) != policy[key]:
            raise ValueError(f"artifact/implementation mismatch: {key}")
    old = json.loads((directory / "manifest.json").read_text())
    if old["status"] != "complete" or old["source_sha"] != policy["source_sha"]:
        raise ValueError("original source/status mismatch")
    if (
        directory.name != policy["run_id"]
        or old.get("run_id", directory.name) != policy["run_id"]
    ):
        raise ValueError("run identity mismatch")
    if old["policy"]["library_revision"] != policy["library_revision"]:
        raise ValueError("original library revision mismatch")
    if old["checkpoint_sha256"] != policy["checkpoint_sha256"]:
        raise ValueError("original checkpoint receipt mismatch")
    if (
        old["policy"].get("checkpoint_sha256", policy["parent_sha256"])
        != policy["parent_sha256"]
    ):
        raise ValueError("original parent checkpoint mismatch")
    kwargs = policy["head_kwargs"]
    if kwargs.get("mc_samples", 32) != old["policy"].get("noise_mc_samples", 32):
        raise ValueError("original measurement integration sample count mismatch")
    if kwargs.get("kind") not in heads.DENSITY_KINDS:
        raise ValueError("explicit distributional head kind required")
    for key, default in (
        ("kind", None),
        ("flow_type", "nsf"),
        ("noise_aware", False),
        ("flow_objective", "nll"),
        ("contrastive_weight", 0.1),
    ):
        if kwargs.get(key, default) != old["arm"].get(key, default):
            raise ValueError(f"original head policy mismatch: {key}")
    return old


def validate_roles(data):
    ids = np.asarray(data["source_id"])
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("unique integer source IDs required")
    roles = {}
    for role in ("test", "calibration"):
        rows = np.asarray(data[f"{role}_rows"])
        if (
            rows.ndim != 1
            or not len(rows)
            or rows.dtype.kind not in "iu"
            or np.any(rows < 0)
            or np.any(rows >= len(ids))
            or len(np.unique(rows)) != len(rows)
        ):
            raise ValueError(f"invalid full {role} role")
        roles[role] = rows
    if np.intersect1d(*roles.values()).size:
        raise ValueError("test/calibration overlap")
    return roles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/pretrain.canfar.yaml")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args(argv)
    if (
        args.output.resolve() == args.directory.resolve()
        or args.directory.resolve() in args.output.resolve().parents
    ):
        raise ValueError("output must be outside the original run directory")
    policy = json.loads(args.policy.read_text())
    distribution = importlib.metadata.distribution("torchregress")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    revision = direct.get("vcs_info", {}).get("commit_id")
    validate_policy(
        policy,
        args.directory,
        args.parent,
        args.model_config,
        installed_revision=revision,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit CPU fallback")
    with np.load(args.directory / "paired-cohort.npz", allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    roles = validate_roles(data)
    # Reserve a distinct output exclusively, after cheap input checks and before scoring.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        scoring_version=SCORING_VERSION,
        policy=policy,
        policy_sha256=digest(args.policy),
        training_updates=0,
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        original_failures_cleared=False,
        independent_verification="pending",
        device=args.device,
        role_counts={k: len(v) for k, v in roles.items()},
    )

    def save():
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    save()
    try:
        state = torch_load_trusted(
            args.directory / "checkpoint.pth", map_location="cpu", weights_only=False
        )
        parent = torch_load_trusted(args.parent, map_location="cpu")
        model = architecture.load_model(
            yaml.safe_load(args.model_config.read_text()), parent, torch.device("cpu")
        )
        model.load_state_dict(state["model_state_dict"], strict=True)
        head = heads.MultiLabelHead(**policy["head_kwargs"])
        head.load_state_dict(state["head_state_dict"], strict=True)
        out = {}
        for role, rows in roles.items():
            result = scoring.predictions(
                model, head, data, rows, torch.device(args.device)
            )
            out.update({f"{role}_{key}": value for key, value in result.items()})
            out[f"{role}_source_id"] = data["source_id"][rows]
            out[f"{role}_rows"] = rows
        path = args.output / "predictions-fp64.npz"
        np.savez_compressed(path, **out)
        manifest.update(status="complete_scoring_only", predictions_sha256=digest(path))
        save()
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
