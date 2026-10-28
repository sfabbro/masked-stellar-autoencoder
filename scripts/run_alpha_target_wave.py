"""GPU prediction heads for independently source-audited native DESI alpha.

Alpha is a signed, linear bracket-abundance coordinate in dex. No exp/log label
transform, acquisitions, cross-survey correction or physical-truth claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
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


def alpha_native_coordinates(mean, variance, log_prob, center, scale):
    """Ordinary affine dex coordinates; signed alpha never receives exp/log."""
    center, scale = np.asarray(center), np.asarray(scale)
    if (
        center.shape != (1,)
        or scale.shape != (1,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Alpha affine normalizer must be finite positive scalar")
    return center + scale * mean, scale**2 * variance, log_prob - np.log(scale).sum()


def run(
    prepared,
    preparation_manifest,
    output,
    *,
    kind,
    seed,
    steps=1000,
    require_cuda=True,
    expected_library_revision=None,
):
    library_info = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    actual_library_revision = library_info.get("vcs_info", {}).get("commit_id")
    if (
        expected_library_revision is not None
        and actual_library_revision != expected_library_revision
    ):
        raise ValueError(
            "Installed alpha library revision differs from published policy"
        )
    metadata = json.loads(preparation_manifest.read_text())
    if metadata["input_sha256"] != digest(prepared):
        raise ValueError("Prepared alpha source SHA mismatch")
    if metadata.get("status") != "source_audited_alpha_features_complete":
        raise ValueError("Actual alpha source preparation has not passed")
    if (
        metadata.get("target_names") != ["desi_rvs_alpha_fe"]
        or metadata.get("target_units") != ["dex"]
        or metadata.get("coordinate") != "linear_native_bracket_abundance"
    ):
        raise ValueError("Alpha must use the native signed linear dex coordinate")
    source_audit = preparation_manifest.parent / "independent-source-audit.json"
    if digest(source_audit) != metadata.get("source_audit_sha256"):
        raise ValueError("Alpha independent original-source audit SHA changed")
    audit = json.loads(source_audit.read_text())
    if audit.get("status") != "all_original_projection_rows_independently_verified":
        raise ValueError(
            "Alpha original source associations are not independently verified"
        )
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
    if target.shape != (len(ids), 1) or not np.isfinite(target).all():
        raise ValueError("Native alpha targets must be finite [row,1]")
    if np.any(arrays["raw_feh"][np.isin(roles, ["fit", "tune"])] <= -2):
        raise ValueError("Unrecommended metal-poor DESI alpha labels entered FIT/TUNE")
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
        native_mean, native_variance, native_lp = alpha_native_coordinates(
            standardized_mean, standardized_variance, np.concatenate(lp), center, scale
        )
        predictions.update(
            {
                f"{role}_source_id": ids[index],
                f"{role}_target_native": target[index],
                f"{role}_mean_native": native_mean,
                f"{role}_variance_native": native_variance,
                f"{role}_raw_feh": arrays["raw_feh"][index],
                f"{role}_joint_log_prob_standardized": np.concatenate(lp),
                f"{role}_joint_log_prob_native": native_lp,
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
        library_revision=actual_library_revision,
        library_source=library_info,
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
        point_statistic="E[DESI native alpha/Fe dex | photometry+XP]",
        density_estimand="Native DESI alpha measurement-label density; not deconvolved physical alpha",
        quoted_errors_used_in_training=False,
        metal_poor_labels_recommended=False,
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
    p.add_argument("--library-revision", required=True)
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
                expected_library_revision=args.library_revision,
            )
        )
    )


if __name__ == "__main__":
    main()
