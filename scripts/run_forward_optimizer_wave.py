"""Versioned FP64 existing-decoder optimizer / flow-prior MAP comparison.

ZERO decoder/encoder/reconstruction training. A new unconditional joint label
flow is fitted on FIT only for the EB arm. This is local MAP/Laplace, not a
normalized posterior and not the old failed scalar quadrature experiment.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchregress.losses.nflows import create_flow_model

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.forward_optimizer_wave import (
    adaptive_prior_convergence,
    defensive_flow_posterior,
    measurement_nodes,
    observed_prior_log_prob,
    solve_library_gauss_newton,
    solve_one,
)


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["mle", "mle_gauss_newton", "flow_prior_map", "flow_prior_posterior"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-steps", type=int, default=1000)
    parser.add_argument("--reuse-prior", type=Path)
    parser.add_argument("--expected-prior-sha")
    parser.add_argument("--adaptive-prior", action="store_true")
    parser.add_argument("--prior-node-cap", type=int, default=16384)
    parser.add_argument("--resume-prior-convergence", type=Path)
    parser.add_argument("--expected-convergence-sha")
    parser.add_argument("--test-count", type=int, default=2048)
    parser.add_argument("--cal-count", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("Production experiment requires actual CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    checkpoint = args.original / "checkpoint.pth"
    source = {
        p.name: sha(p)
        for p in [
            checkpoint,
            args.original / "paired-cohort.npz",
            args.original / "forward-xp-inputs.npz",
        ]
    }
    original_manifest = json.loads((args.original / "manifest.json").read_text())
    if original_manifest.get("status") != "complete":
        raise ValueError("Original source experiment must be complete")
    if original_manifest.get("checkpoint_sha256") != source["checkpoint.pth"]:
        raise ValueError("Original checkpoint SHA differs from producer manifest")
    pair = dict(np.load(args.original / "paired-cohort.npz", allow_pickle=False))
    xp = dict(np.load(args.original / "forward-xp-inputs.npz", allow_pickle=False))
    saved = torch_load_trusted(checkpoint, map_location=device, weights_only=False)
    reg = saved["forward_regressor"]
    decoder = reg.decoder_.to(device).double().eval().requires_grad_(False)
    center = np.asarray(saved["label_center"], dtype="f8")
    scale = np.asarray(saved["label_scale"], dtype="f8")
    labels = torch.tensor(
        (pair["value"] - center) / scale, dtype=torch.double, device=device
    )
    errors = torch.tensor(pair["error"] / scale, dtype=torch.double, device=device)
    rows_xp = xp["feature_rows"]
    fit = pair["fit_rows"]
    assert not set(fit).intersection(pair["test_rows"])
    assert not set(fit).intersection(pair["calibration_rows"])
    prior = None
    metadata = dict(
        status="running",
        mode=args.mode,
        source=source,
        original_source_revision=original_manifest.get("source_revision"),
        seed=args.seed,
        torchregress_version=__import__("importlib.metadata").metadata.version(
            "torchregress"
        ),
        cuda=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        reconstruction_presentations=0,
        decoder_updates=0,
        normalized_posterior=False,
        posterior_estimator="defensive_self_normalized_importance_sampling"
        if args.mode == "flow_prior_posterior"
        else None,
        prior_semantics="joint intrinsic prior under diagonal quoted-label-error convolution; finite QMC approximation"
        if args.mode not in {"mle", "mle_gauss_newton"}
        else None,
    )
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    if args.mode not in {"mle", "mle_gauss_newton"}:
        prior = (
            create_flow_model(
                n_features=3,
                context_dim=0,
                flow_type="nsf",
                n_transforms=3,
                hidden_features=[32, 32],
                bins=8,
            )
            .to(device)
            .double()
        )
        if args.reuse_prior:
            if (
                not args.adaptive_prior
                or not args.expected_prior_sha
                or sha(args.reuse_prior) != args.expected_prior_sha
            ):
                raise ValueError(
                    "Reused prior needs explicit SHA and adaptive validation"
                )
            failed = json.loads((args.reuse_prior.parent / "manifest.json").read_text())
            if (
                failed.get("status") != "failed_prior_quadrature_convergence"
                or failed.get("source") != source
            ):
                raise ValueError(
                    "Reused prior is not the exact source-bound failed predecessor"
                )
            restored = torch_load_trusted(
                args.reuse_prior, map_location=device, weights_only=False
            )
            if restored["seed"] != args.seed or not np.array_equal(
                restored["fit_rows"], fit
            ):
                raise ValueError("Prior seed/FIT role differs from failed predecessor")
            prior.load_state_dict(restored["prior"], strict=True)
            metadata.update(
                reused_prior=str(args.reuse_prior),
                reused_prior_sha256=args.expected_prior_sha,
                prior_training_updates=0,
                prior_training_approximation="unchanged predecessor64-node objective; no refit",
            )
        else:
            optimizer = torch.optim.Adam(prior.parameters(), lr=1e-3)
            generator = torch.Generator().manual_seed(args.seed)
            nodes = measurement_nodes(64, device=device)
            for step in range(2 if args.smoke else args.prior_steps):
                selected = fit[
                    torch.randperm(len(fit), generator=generator)[:256].numpy()
                ]
                optimizer.zero_grad(set_to_none=True)
                loss = -observed_prior_log_prob(
                    prior(), labels[selected], errors[selected], nodes
                ).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite prior objective")
                loss.backward()
                optimizer.step()
            torch.save(
                {
                    "prior": prior.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "fit_rows": fit,
                    "seed": args.seed,
                },
                args.output / "prior.pth",
            )
        prior.eval().requires_grad_(False)
        if args.adaptive_prior:
            resume = None
            if args.resume_prior_convergence:
                if (
                    not args.expected_convergence_sha
                    or sha(args.resume_prior_convergence)
                    != args.expected_convergence_sha
                ):
                    raise ValueError("Prior convergence report SHA changed")
                before = json.loads(
                    (args.resume_prior_convergence.parent / "manifest.json").read_text()
                )
                resume = json.loads(args.resume_prior_convergence.read_text())
                if (
                    before.get("status")
                    != "failed_adaptive_prior_quadrature_convergence"
                    or before.get("source") != source
                    or before.get("reused_prior_sha256") != args.expected_prior_sha
                    or before.get("seed") != args.seed
                    or before.get("adaptive_prior_convergence") != resume
                ):
                    raise ValueError(
                        "Convergence predecessor source/prior binding differs"
                    )
                metadata.update(
                    resumed_convergence_sha256=args.expected_convergence_sha,
                    recomputed_final_baseline_nodes=resume["nodes"],
                    earlier_resolution_results_reused=True,
                )
            convergence = adaptive_prior_convergence(
                prior(),
                labels[pair["tune_rows"][:128]],
                errors[pair["tune_rows"][:128]],
                max_nodes=args.prior_node_cap,
                resume_report=resume,
            )
            (args.output / "prior-quadrature-convergence.json").write_text(
                json.dumps(convergence, indent=2)
            )
            metadata["adaptive_prior_convergence"] = convergence
            if convergence["status"] != "converged_finite_qmc_only":
                metadata["status"] = "failed_adaptive_prior_quadrature_convergence"
                (args.output / "manifest.json").write_text(
                    json.dumps(metadata, indent=2)
                )
                raise RuntimeError(
                    "Prior convergence failed at unchanged tolerance and bounded cap"
                )
        else:
            # Independent nested-node convergence check uses TUNE only, never test.
            tune = pair["tune_rows"][:128]
            with torch.no_grad():
                convergences = {}
                previous = None
                for count in [64, 128, 256, 512]:
                    value = observed_prior_log_prob(
                        prior(),
                        labels[tune],
                        errors[tune],
                        measurement_nodes(count, device=device),
                    )
                    if previous is not None:
                        convergences[str(count)] = float((value - previous).abs().max())
                    previous = value
            (args.output / "prior-quadrature-convergence.json").write_text(
                json.dumps(convergences, indent=2)
            )
            # Preserve approximation evidence; a failed gate stops before predictions.
            if not args.smoke and convergences["512"] > 0.01:
                metadata.update(
                    status="failed_prior_quadrature_convergence",
                    convergence=convergences,
                )
                (args.output / "manifest.json").write_text(
                    json.dumps(metadata, indent=2)
                )
                raise RuntimeError(
                    "Prior convolution QMC not converged at fixed absolute .01 log-density tolerance"
                )
    if args.mode == "flow_prior_posterior":
        # Freeze sample count on TUNE before looking at any TEST likelihood.
        checks = []
        tune_rows = pair["fit_rows"][:1] if args.smoke else pair["tune_rows"][:16]
        tune_inputs = {}
        if not args.smoke:
            import h5py

            from masked_stellar_autoencoder.training.input_measurement import (
                standardized_xp_errors,
                xp_indices,
            )

            cfg = saved["policy"]
            if sha(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
                raise ValueError("Original parent hash changed")
            parent = torch_load_trusted(
                cfg["checkpoint"], map_location="cpu", weights_only=False
            )
            feature_scale = np.asarray(parent["run_signature"]["scaler_scale"])
            columns = xp_indices(cfg["feature_columns"])
            with h5py.File(cfg["cohort"]) as source_hdf:
                for row in tune_rows:
                    if source_hdf["features/source_id"][row] != pair["source_id"][row]:
                        raise ValueError("TUNE raw-source identity differs")
                    raw = source_hdf["features/values"][row : row + 1]
                    error = source_hdf["features/reported_feature_errors"][
                        row : row + 1
                    ]
                    sigma = standardized_xp_errors(
                        error,
                        feature_scale,
                        columns,
                        g_magnitude=raw[:, cfg["feature_columns"].index("G")],
                        error_convention="raw_xp",
                    )[0]
                    tune_inputs[int(row)] = (pair["input"][row, columns], sigma)
        for row in tune_rows:
            position = np.searchsorted(rows_xp, row)
            values, sigma = (
                (xp["standardized_input"][position], xp["standardized_sigma"][position])
                if args.smoke
                else tune_inputs[int(row)]
            )
            x = torch.tensor(values, dtype=torch.double, device=device)
            variance = (
                torch.tensor(sigma, dtype=torch.double, device=device).square()
                + reg.noise_variance_.to(device).double()
            )
            solution = solve_one(
                decoder,
                x,
                variance,
                torch.zeros((1, 3), dtype=torch.double, device=device),
                log_prior=prior().log_prob,
                max_iter=5 if args.smoke else args.max_iter,
            )
            estimates = [
                defensive_flow_posterior(
                    decoder,
                    x,
                    variance,
                    solution["prediction"],
                    solution["hessian"],
                    prior(),
                    count=count,
                    seed=args.seed,
                )
                for count in [4096, 16384]
            ]
            checks.append(
                dict(
                    row=int(row),
                    log_normalizer_delta=abs(
                        estimates[1]["log_normalizer"] - estimates[0]["log_normalizer"]
                    ),
                    mean_delta=float(
                        (estimates[1]["mean"] - estimates[0]["mean"]).abs().max()
                    ),
                    effective_samples=estimates[1]["effective_samples"],
                )
            )
        (args.output / "posterior-tune-convergence.json").write_text(
            json.dumps(checks, indent=2)
        )
        if not args.smoke and any(
            c["log_normalizer_delta"] > 0.05
            or c["mean_delta"] > 0.05
            or c["effective_samples"] < 100
            for c in checks
        ):
            metadata.update(status="failed_posterior_importance_convergence")
            (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
            raise RuntimeError(
                "Defensive importance convergence/ESS gate failed on TUNE"
            )
    for role, limit in [("test", args.test_count), ("calibration", args.cal_count)]:
        rows = pair[role + "_rows"]
        ordered = sorted(
            rows,
            key=lambda r: hashlib.sha256(
                str(int(pair["source_id"][r])).encode()
            ).digest(),
        )
        chosen = np.asarray(ordered[: 2 if args.smoke else limit])
        positions = np.searchsorted(rows_xp, chosen)
        np.testing.assert_array_equal(rows_xp[positions], chosen)
        outputs = []
        for index, position in enumerate(positions):
            x = torch.tensor(
                xp["standardized_input"][position], dtype=torch.double, device=device
            )
            sx = torch.tensor(
                xp["standardized_sigma"][position], dtype=torch.double, device=device
            )
            variance = sx.square() + reg.noise_variance_.to(device).double()
            starts = torch.stack(
                [
                    torch.zeros(3, dtype=torch.double, device=device),
                    labels[fit[index % len(fit)]].detach(),
                ]
            )
            # Common two FIT-defined starts; labels of evaluated objects never enter inversion.
            if args.mode == "mle_gauss_newton":
                result = solve_library_gauss_newton(
                    reg, x, sx, starts, max_iter=5 if args.smoke else args.max_iter
                )
            else:
                result = solve_one(
                    decoder,
                    x,
                    variance,
                    starts,
                    log_prior=None if prior is None else prior().log_prob,
                    max_iter=5 if args.smoke else args.max_iter,
                )
            outputs.append(result)
            if args.mode == "flow_prior_posterior":
                posterior = defensive_flow_posterior(
                    decoder,
                    x,
                    variance,
                    result["prediction"],
                    result["hessian"],
                    prior(),
                    count=16384,
                    seed=args.seed + int(chosen[index]),
                )
                np.savez_compressed(
                    args.output / f"{role}-posterior-{index:06d}.npz",
                    row=chosen[index],
                    mean=posterior["mean"].cpu().numpy(),
                    covariance=posterior["covariance"].cpu().numpy(),
                    quantiles=posterior["quantiles"].cpu().numpy(),
                    log_normalizer=posterior["log_normalizer"],
                    effective_samples=posterior["effective_samples"],
                    samples=posterior["samples"],
                    sample_values=posterior["sample_values"].cpu().numpy(),
                    normalized_weights=posterior["normalized_weights"].cpu().numpy(),
                )
            np.savez_compressed(
                args.output / f"{role}-object-{index:06d}.npz",
                row=chosen[index],
                source_id=pair["source_id"][chosen[index]],
                prediction=result["prediction"].cpu().numpy(),
                objective=result["objective"],
                gradient_norm=result["gradient_norm"],
                hessian=result["hessian"].cpu().numpy(),
                hessian_min_eigenvalue=result["hessian_min_eigenvalue"],
            )
        np.savez_compressed(
            args.output / f"{role}-predictions.npz",
            rows=chosen,
            source_id=pair["source_id"][chosen],
            standardized_prediction=np.stack(
                [r["prediction"].cpu().numpy() for r in outputs]
            ),
            native_prediction=np.stack([r["prediction"].cpu().numpy() for r in outputs])
            * scale
            + center,
            gradient_norm=[r["gradient_norm"] for r in outputs],
            local_stationary=[r["local_stationary"] for r in outputs],
        )
    metadata.update(
        status="complete",
        source=source,
        scope="hash-selected bounded optimizer screen; no full-cohort claim",
        physical_truth_calibration=False,
    )
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            if output.is_dir():
                (output / "failure.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "exception": type(exc).__name__,
                            "message": str(exc),
                        },
                        indent=2,
                    )
                )
                manifest = output / "manifest.json"
                if manifest.exists():
                    state = json.loads(manifest.read_text())
                    if state.get("status") == "running":
                        state["status"] = "failed"
                        manifest.write_text(json.dumps(state, indent=2))
        raise
