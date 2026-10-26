"""CANFAR CUDA-only synthetic coupled-six-head admission; never loads MSA weights."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from masked_stellar_autoencoder.training.remediation_joint6 import (
    JointSixHead,
    masked_gaussian_log_prob,
)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc():
    return dt.datetime.now(dt.UTC).isoformat()


def truth_parameters(*, device="cpu", dtype=torch.float64):
    lower = torch.eye(6, device=device, dtype=dtype)
    lower[1, 0] = 0.65
    lower[2, 1] = -0.45
    lower[4, 3] = 0.55
    lower[5, 0] = 0.30
    slope = torch.tensor(
        [
            [0.2, 0, -0.1, 0],
            [0, 0.2, 0, 0.1],
            [0.1, 0, 0.1, 0],
            [0, -0.15, 0, 0.1],
            [0.1, 0, 0, -0.1],
            [0, 0.1, 0.1, 0],
        ],
        device=device,
        dtype=dtype,
    )
    return lower, slope


def synthetic_batch(count, generator, *, device="cpu", dtype=torch.float64):
    lower, slope = truth_parameters(device=device, dtype=dtype)
    context = torch.randn((count, 4), device=device, dtype=dtype, generator=generator)
    mean = context @ slope.T
    eps = torch.randn((count, 6), device=device, dtype=dtype, generator=generator)
    return context, mean + eps @ lower.T, mean


def all_masks(count=128, *, device="cpu"):
    code = torch.arange(count, device=device) % 64
    return (code[:, None] & (1 << torch.arange(6, device=device))) != 0


def covariance_checks(samples, truth_covariance, gates):
    """Compare whole-vector sample covariance with known conditional truth."""
    covariance = torch.cov(samples.T)
    relative = torch.linalg.matrix_norm(
        covariance - truth_covariance
    ) / torch.linalg.matrix_norm(truth_covariance)
    pairs = [(0, 1), (1, 2), (3, 4), (0, 5)]
    errors = [
        (covariance[i, j] - truth_covariance[i, j]).abs().item() for i, j in pairs
    ]
    signs = [bool(covariance[i, j] * truth_covariance[i, j] > 0) for i, j in pairs]
    mean_error = samples.mean(0).abs().max().item()
    passed = bool(
        relative <= gates["covariance_relative_frobenius_max"]
        and max(errors) <= gates["cross_covariance_absolute_error_max"]
        and all(signs)
        and mean_error <= gates["mean_absolute_error_max"]
    )
    return dict(
        passed=passed,
        covariance=covariance.tolist(),
        truth_covariance=truth_covariance.tolist(),
        covariance_relative_frobenius=float(relative),
        cross_pairs=pairs,
        cross_absolute_errors=errors,
        cross_sign_matches=signs,
        mean_absolute_error_max=mean_error,
    )


def sparse_checks(diagnostic, mask, gates):
    active = mask.any(-1)
    if diagnostic.get("exact"):
        return dict(
            passed=True,
            method="exact Gaussian/MDN observed-subvector law",
            active_rows=int(active.sum()),
        )
    relative = diagnostic["relative_ess"][active]
    replica = diagnostic["independent_replica_log_prob_gap"][active]
    refinement = diagnostic["refinement_log_prob_gap"][active]
    finite = bool(
        torch.isfinite(relative).all()
        and torch.isfinite(replica).all()
        and torch.isfinite(refinement).all()
    )
    result = dict(
        finite=finite,
        active_rows=int(active.sum()),
        relative_ess_median=float(relative.median()),
        relative_ess_min=float(relative.min()),
        independent_replica_gap_p95=float(torch.quantile(replica, 0.95)),
        refinement_gap_p95=float(torch.quantile(refinement, 0.95)),
    )
    result["passed"] = bool(
        finite
        and result["relative_ess_median"] >= gates["relative_ess_median_min"]
        and result["relative_ess_min"] >= gates["relative_ess_min"]
        and result["independent_replica_gap_p95"] <= gates["replica_gap_p95_max"]
        and result["refinement_gap_p95"] <= gates["refinement_gap_p95_max"]
    )
    return result


def complete_joint_log_prob(head, context, target):
    """The fully observed noiseless joint needs no missing/noise integration."""
    if head.kind == "nsf":
        return head.flow.distribution(context).log_prob(target)
    return head.log_prob(
        context,
        target,
        torch.zeros_like(target),
        torch.ones_like(target, dtype=torch.bool),
    )


def complete_objective(head, context, target):
    count = target.numel()
    objective = -complete_joint_log_prob(head, context, target).sum() / count
    if head.kind == "nsf":
        mean, lower = head.proposal_parameters(context)
        proposal = (
            -masked_gaussian_log_prob(
                mean,
                lower,
                target,
                torch.zeros_like(target),
                torch.ones_like(target, dtype=torch.bool),
            ).sum()
            / count
        )
        objective = objective + head.proposal_weight * proposal
    return objective


def validate_config(cfg):
    if cfg["run_id"] not in {
        "remediation_joint6_preflight_v1",
        "remediation_joint6_preflight_v2",
    } or cfg["kinds"] != [
        "gaussian",
        "mdn",
        "nsf",
    ]:
        raise ValueError("Exact coupled-six preflight identity/families required")
    if (
        not 500 <= cfg["updates"] <= 1000
        or cfg["diagnostic_rows"] != 128
        or cfg["diagnostic_particles"] != 64
    ):
        raise ValueError(
            "Bounded training and fixed128-row/64+128-particle diagnostic required"
        )
    if (
        cfg["precision"] != "float32_train_float64_diagnostics"
        or cfg["library_revision"] != "6bd8b93b35fe4fe216319870e72e94eb78429106"
    ):
        raise ValueError("Immutable library/declared precision contract differs")
    return cfg


def run(args):
    cfg = validate_config(yaml.safe_load(args.config.read_text()))
    if args.run_id != cfg["run_id"]:
        raise ValueError("Wrong preflight run ID")
    if args.output.exists():
        raise FileExistsError("Never overwrite an attempted preflight output")
    if (
        not torch.cuda.is_available()
        or os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
        or not Path("/arc/projects").is_dir()
    ):
        raise RuntimeError(
            "Actual fit requires CANFAR headless CUDA, never local execution"
        )
    args.output.mkdir(parents=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(
        run_id=args.run_id,
        source_commit=source,
        library_revision=cfg["library_revision"],
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        head_module_sha256=hashlib.sha256(
            Path(inspect.getfile(JointSixHead)).read_bytes()
        ).hexdigest(),
        status="running",
        started=utc(),
        scope="Synthetic learned correlated six-dimensional heads only; no MSA/checkpoint/physical-truth certification",
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        torchregress_version=importlib.metadata.version("torchregress"),
        checks={},
    )
    atomic(args.output / "manifest.json", manifest)
    try:
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Actual torchregress immutable library pin differs")
        if args.expected_source and source != args.expected_source:
            raise ValueError("Actual source commit differs")
        if cfg.get("reuse_preflight"):
            prior_spec = cfg["reuse_preflight"]
            prior_dir = Path(prior_spec["path"])
            prior = json.loads((prior_dir / "manifest.json").read_text())
            if (
                prior.get("source_commit") != prior_spec["source_commit"]
                or prior.get("head_module_sha256") != manifest["head_module_sha256"]
            ):
                raise ValueError(
                    "Completed synthetic check source/head bindings differ"
                )
            for family in ("gaussian", "mdn"):
                check = prior.get("checks", {}).get(family, {})
                artifact = prior_dir / f"{family}-samples-and-diagnostics.npz"
                if not check.get("passed") or hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest() != check.get("sample_sha256"):
                    raise ValueError(
                        "Completed synthetic check/content certificate differs"
                    )
                manifest["checks"][family] = check
            manifest["reused_completed_checks"] = dict(
                path=str(prior_dir),
                source_commit=prior["source_commit"],
                families=["gaussian", "mdn"],
                no_retraining=True,
            )
            atomic(args.output / "manifest.json", manifest)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        lower, _ = truth_parameters(device="cuda")
        truth_covariance = lower @ lower.T
        for index, kind in enumerate(cfg["kinds"]):
            if kind in manifest["checks"]:
                continue
            manifest.update(current_family=kind, phase="initializing")
            atomic(args.output / "manifest.json", manifest)
            torch.manual_seed(cfg["seed"] + index)
            torch.cuda.manual_seed_all(cfg["seed"] + index)
            generator = torch.Generator(device="cuda").manual_seed(cfg["seed"] + 100)
            head = JointSixHead(
                kind,
                latent_dim=4,
                width=cfg["width"],
                components=cfg["components"],
                particles=cfg["training_particles"],
                seed=cfg["seed"],
            ).cuda()
            optimizer = torch.optim.Adam(head.parameters(), lr=cfg["learning_rate"])
            hold_gen = torch.Generator(device="cuda").manual_seed(cfg["seed"] + 200)
            hx, hy, _ = synthetic_batch(
                cfg["holdout_rows"], hold_gen, device="cuda", dtype=torch.float32
            )
            with torch.no_grad():
                initial_nll = float(-complete_joint_log_prob(head, hx, hy).mean())
            for update in range(cfg["updates"]):
                x, y, _ = synthetic_batch(
                    cfg["batch_size"], generator, device="cuda", dtype=torch.float32
                )
                optimizer.zero_grad(set_to_none=True)
                loss = complete_objective(head, x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{kind}: nonfinite synthetic objective")
                loss.backward()
                grads = [p.grad for p in head.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError(f"{kind}: nonfinite or absent gradients")
                optimizer.step()
                if update % 50 == 0 or update + 1 == cfg["updates"]:
                    atomic(
                        args.output / "progress.json",
                        dict(
                            kind=kind,
                            update=update + 1,
                            updates=cfg["updates"],
                            loss=float(loss.detach()),
                            time=utc(),
                        ),
                    )
            head = head.double().eval()
            with torch.no_grad():
                final_nll = float(
                    -complete_joint_log_prob(head, hx.double(), hy.double()).mean()
                )
                draws = head.sample(
                    torch.zeros((1, 4), device="cuda", dtype=torch.float64),
                    cfg["draws"],
                    torch.Generator(device="cuda").manual_seed(cfg["seed"] + 300),
                )[:, 0]
                covariance = covariance_checks(draws, truth_covariance, cfg["gates"])
                dx, dy, dm = synthetic_batch(
                    128,
                    torch.Generator(device="cuda").manual_seed(cfg["seed"] + 400),
                    device="cuda",
                )
                mask = all_masks(128, device="cuda")
                sigma = torch.full_like(dy, cfg["diagnostic_sigma"])
                noisy = dy + sigma * torch.randn(
                    dy.shape,
                    device="cuda",
                    dtype=dy.dtype,
                    generator=torch.Generator(device="cuda").manual_seed(
                        cfg["seed"] + 401
                    ),
                )
                diagnostics = head.diagnostics(dx, noisy, sigma, mask, particles=64)
                sparse = sparse_checks(diagnostics, mask, cfg["gates"])
                improvement = initial_nll - final_nll
                law = torch.distributions.MultivariateNormal(
                    dm, scale_tril=lower.expand(128, 6, 6)
                )
                truth_nll = float(-law.log_prob(dy).mean())
                passed = bool(
                    covariance["passed"]
                    and sparse["passed"]
                    and improvement >= cfg["gates"]["nll_improvement_min"]
                )
                result = dict(
                    passed=passed,
                    updates=cfg["updates"],
                    initial_holdout_joint_nll=initial_nll,
                    final_holdout_joint_nll=final_nll,
                    nll_improvement=improvement,
                    independent_truth_sample_nll=truth_nll,
                    covariance=covariance,
                    sparse_noisy=sparse,
                    target_dim=6,
                    training="complete/noiseless synthetic vectors; partial/noisy likelihood admitted separately",
                    precision=cfg["precision"],
                    numerical_parity_tolerance_unchanged=5e-5,
                )
                arrays = dict(
                    samples=draws.cpu().numpy(),
                    diagnostic_context=dx.cpu().numpy(),
                    diagnostic_target=noisy.cpu().numpy(),
                    diagnostic_mask=mask.cpu().numpy(),
                    diagnostic_sigma=sigma.cpu().numpy(),
                    log_prob=diagnostics["log_prob"].cpu().numpy(),
                )
                for key in (
                    "relative_ess",
                    "independent_replica_log_prob_gap",
                    "refinement_log_prob_gap",
                ):
                    if key in diagnostics:
                        arrays[key] = diagnostics[key].cpu().numpy()
                np.savez_compressed(
                    args.output / f"{kind}-samples-and-diagnostics.npz", **arrays
                )
                result["sample_sha256"] = hashlib.sha256(
                    (args.output / f"{kind}-samples-and-diagnostics.npz").read_bytes()
                ).hexdigest()
                atomic(args.output / f"{kind}-checks.json", result)
                manifest["checks"][kind] = result
                atomic(args.output / "manifest.json", manifest)
            del optimizer, head
            torch.cuda.empty_cache()
        manifest["status"] = (
            "remediation_joint6_preflight_pass"
            if all(x["passed"] for x in manifest["checks"].values())
            else "remediation_joint6_preflight_failed"
        )
        manifest["finished"] = utc()
        manifest["gates"] = cfg["gates"]
        atomic(args.output / "manifest.json", manifest)
        if manifest["status"] != "remediation_joint6_preflight_pass":
            raise RuntimeError(
                "Declared coupled-six synthetic admission gates failed; no automatic retry"
            )
    except Exception as error:
        manifest.update(
            status="remediation_joint6_preflight_failed",
            error_type=type(error).__name__,
            error=str(error),
            finished=utc(),
        )
        atomic(args.output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/remediation-joint-preflight-20261026.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
