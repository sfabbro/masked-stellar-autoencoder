"""Matched GPU heads for real source-bound posterior age/distance labels.

Frozen parent86 features, fresh heads, no encoder/reconstruction SGD. Catalogue
posterior medians are supervised labels; this does not fit raw parallaxes or
certify physical ages/distances. TUNE alone selects a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from masked_stellar_autoencoder.training.multilabel_experiments import MultiLabelHead


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_normalization(target, roles):
    fit = np.asarray(target)[np.asarray(roles) == "fit"]
    if len(fit) < 16 or not np.isfinite(fit).all():
        raise ValueError("Insufficient finite FIT labels")
    center, scale = fit.mean(0), fit.std(0)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Degenerate FIT target normalization")
    return center, scale


def run(
    prepared, preparation_manifest, output, *, kind, seed, steps=1000, require_cuda=True
):
    metadata = json.loads(preparation_manifest.read_text())
    if metadata["input_sha256"] != digest(prepared):
        raise ValueError("Prepared source SHA mismatch")
    if metadata["target_names"] not in (
        ["log_age_Gyr", "log_distance_kpc"],
        ["log_age_Gyr"],
    ):
        raise ValueError("Unexpected posterior target contract")
    if not metadata["source_provenance"] or not metadata["family_holdout"]:
        raise ValueError("Source/family holdout evidence required")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Production target-head screen requires actual CUDA")
    device = torch.device("cuda" if require_cuda else "cpu")
    archive = np.load(prepared, allow_pickle=False)
    arrays = {k: archive[k] for k in archive.files}
    ids, roles, target = arrays["source_id"], arrays["role"], arrays["target"]
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError("Repeated or noninteger identities")
    if set(roles.tolist()) != {"fit", "tune", "cal", "test"}:
        raise ValueError("Four disjoint roles required")
    if np.any(arrays["benchmark_partition"][np.isin(roles, ["fit", "tune"])] == 1):
        raise ValueError("Benchmark validation leakage")
    center, scale = fit_normalization(target, roles)
    x = torch.tensor(arrays["embeddings"], dtype=torch.float64, device=device)
    y = torch.tensor((target - center) / scale, dtype=torch.float64, device=device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = (
        MultiLabelHead(
            kind="flow" if kind == "nsf" else kind,
            target_dim=target.shape[1],
            latent_dim=x.shape[1],
            width=64,
            flow_type="nsf",
        )
        .double()
        .to(device)
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    fit = np.flatnonzero(roles == "fit")
    tune = np.flatnonzero(roles == "tune")
    rng = np.random.default_rng(seed)
    best, selected, history = float("inf"), None, []
    stream_digest = hashlib.sha256()
    for step in range(steps):
        rows = rng.choice(fit, size=min(256, len(fit)), replace=False)
        stream_digest.update(np.asarray(rows, dtype="<i8").tobytes())
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -head.log_prob(x[rows], y[rows]).mean() / target.shape[1]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite actual FIT density")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite actual FIT gradient")
        optimizer.step()
        if (step + 1) % 25 == 0 or step + 1 == steps:
            head.eval()
            with torch.no_grad():
                score = float(-head.log_prob(x[tune], y[tune]).mean() / target.shape[1])
            history.append(
                dict(
                    step=step + 1,
                    fit_loss=float(loss.detach()),
                    tune_nll_per_target=score,
                )
            )
            if score < best:
                best, selected = (
                    score,
                    dict(
                        step=step + 1,
                        head=copy.deepcopy(head.state_dict()),
                        optimizer=copy.deepcopy(optimizer.state_dict()),
                    ),
                )
    if selected is None:
        raise ValueError("No finite tuning selection")
    head.load_state_dict(selected["head"], strict=True)
    head.eval()
    predictions = {}
    for role in ("cal", "test"):
        index = np.flatnonzero(roles == role)
        means, variances, lp = [], [], []
        with torch.no_grad():
            for chunk in np.array_split(index, max(1, (len(index) + 31) // 32)):
                mean, var = head.moments(x[chunk])
                means.append(mean.cpu().numpy())
                variances.append(var.cpu().numpy())
                lp.append(head.log_prob(x[chunk], y[chunk]).cpu().numpy())
        standardized_mean, standardized_variance = (
            np.concatenate(means),
            np.concatenate(variances),
        )
        if not (
            np.isfinite(standardized_mean).all()
            and np.isfinite(standardized_variance).all()
            and (standardized_variance >= 0).all()
            and np.isfinite(np.concatenate(lp)).all()
        ):
            raise FloatingPointError(
                "Nonfinite/invalid actual selected-model predictions"
            )
        log_mean = center + scale * standardized_mean
        # This point is exp(E[log label]), not the arithmetic predictive mean.
        predictions.update(
            {
                f"{role}_source_id": ids[index],
                f"{role}_target_log": target[index],
                f"{role}_mean_log": log_mean,
                f"{role}_variance_log": scale**2 * standardized_variance,
                f"{role}_geometric_point": np.exp(log_mean),
                f"{role}_joint_log_prob_log_units": np.concatenate(lp)
                - np.log(scale).sum(),
                f"{role}_joint_log_prob_native": np.concatenate(lp)
                - np.log(scale).sum()
                - target[index].sum(1),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint.pth"
    torch.save(
        dict(
            head_state_dict=head.cpu().state_dict(),
            selected_optimizer_state_dict=selected["optimizer"],
            selected_step=selected["step"],
            training_steps=steps,
            center=center,
            scale=scale,
            source_input_sha256=metadata["input_sha256"],
            kind=kind,
            seed=seed,
        ),
        checkpoint,
    )
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if any(
        not torch.equal(v, reloaded["head_state_dict"][k])
        for k, v in head.state_dict().items()
    ):
        raise ValueError("Checkpoint reload differs")
    np.savez_compressed(output / "predictions.npz", **predictions)
    manifest = dict(
        status="complete",
        kind=kind,
        seed=seed,
        device=str(device),
        actual_gpu=torch.cuda.get_device_name() if require_cuda else None,
        input_sha256=metadata["input_sha256"],
        preparation_manifest_sha256=digest(preparation_manifest),
        checkpoint_sha256=digest(checkpoint),
        selected_step=selected["step"],
        training_steps=steps,
        head_only_label_presentations=steps * min(256, len(fit)),
        encoder_updates=0,
        reconstruction_presentations=0,
        target_estimand=metadata["target_estimand"],
        raw_signed_parallax_training=False,
        target_names=metadata["target_names"],
        target_units=metadata["target_units"],
        role_counts=metadata["role_counts"],
        source_provenance=metadata["source_provenance"],
        family_holdout=metadata["family_holdout"],
        point_statistic="exp(E[log catalogue-posterior median | photometry+XP])",
        scoring_precision="float64",
        checkpoint_reload_exact=True,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        row_stream_sha256=stream_digest.hexdigest(),
        runner_sha256=digest(Path(__file__)),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--preparation-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kind", choices=("gaussian", "mdn", "nsf"), required=True)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--allow-cpu-smoke", action="store_true")
    args = p.parse_args()
    print(
        json.dumps(
            run(
                args.prepared,
                args.preparation_manifest,
                args.output,
                kind=args.kind,
                seed=args.seed,
                steps=args.steps,
                require_cuda=not args.allow_cpu_smoke,
            )
        )
    )


if __name__ == "__main__":
    main()
