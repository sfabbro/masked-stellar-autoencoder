"""Audited XP-only forward/FlowEB diagnostic; no encoder or replay updates."""

import argparse
import copy
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from run_enriched_value import COHORT_SHA
from run_frozen_head_screen import save_host_npz
from run_prediction_comparison import digest, native_inputs
from run_stellar_label_experiments import BENCH
from stellar_catalog_jobs import atomic_json
from torch import nn
from torchregress.algorithms import ForwardInversionRegressor

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.forward_eiv import (
    IntegratedForward,
    decoder,
    grid_predictions,
)
from masked_stellar_autoencoder.training.input_measurement import (
    audit_xp_error_sources,
    standardized_xp_errors,
    xp_indices,
)


def fit_model(method, x, sx, y, sy, policy, smoke):
    torch.manual_seed(42)
    epochs = 1 if smoke else policy["epochs"]
    if method == "forward_plugin":
        # Production library implementation, not a similarly named reimplementation.
        model = ForwardInversionRegressor(
            decoder=decoder(x.shape[1]).to(x.device),
            sigma_x=sx,
            learn_noise=True,
            epochs=epochs,
            batch_size=policy["batch_size"],
            lr=policy["lr"],
            seed=42,
            device=x.device,
            weight_decay=0,
        ).fit(x, y)
        state = dict(
            decoder=model.decoder_.state_dict(),
            intrinsic_variance=model.noise_variance_.cpu(),
        )
        return model.decoder_.eval(), state, None
    direct = method == "direct_mlp"
    model = (
        nn.Sequential(
            nn.Linear(x.shape[1], 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        if direct
        else IntegratedForward(x.shape[1], flow=method == "flow_eb")
    ).to(x.device)
    if not direct:
        # Match the library plug-in arm's fit-only initial nuisance variance.
        with torch.no_grad():
            initial = (0.01 * x.var(dim=0, unbiased=False).mean()).clamp_min(1e-8)
            model.log_intrinsic_variance.fill_(initial.log())
    optimizer = torch.optim.AdamW(model.parameters(), lr=policy["lr"], weight_decay=0)
    generator = torch.Generator(device=x.device).manual_seed(42)
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(
            len(x), generator=generator, device=x.device
        ).split(policy["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = (
                (model(x[indices]).flatten() - y[indices]).square().mean()
                if direct
                else model.objective(
                    x[indices],
                    sx[indices],
                    y[indices],
                    sy[indices],
                    order=policy["quadrature_nodes"],
                )
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(method + " nonfinite objective")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError(method + " nonfinite gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    state = dict(
        state=model.state_dict(), optimizer=optimizer.state_dict(), losses=losses
    )
    return model.eval(), state, None


@torch.no_grad()
def evaluate(method, model, state, x, sx, grid):
    if method == "direct_mlp":
        return torch.cat([model(block).cpu() for block in x.split(512)]).numpy()
    if method == "forward_plugin":
        return grid_predictions(
            model, state["intrinsic_variance"].to(x.device), x, sx, grid
        )
    return grid_predictions(
        model.decoder, model.intrinsic_variance, x, sx, grid, model.prior
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="forward_eiv_xp_screen_v1")
    parser.add_argument("--config", default="configs/forward-eiv-screen.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cohort",
        default="/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/enriched_feature_cohort_v1/enriched-feature-cohort.h5",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--benchmark", default=BENCH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    policy = yaml.safe_load(Path(args.config).read_text())
    if args.run_id != policy["run_id"]:
        raise ValueError("Run differs from declared screen")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        policy=policy,
        smoke=args.smoke,
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        scope="XP-only data diagnostic; zero encoder/reconstruction updates; not a 10M replay run",
    )
    atomic_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        )
        if direct["vcs_info"]["commit_id"] != policy["torchregress_revision"]:
            raise ValueError("Installed torchregress pin differs")
        parent = Path(args.checkpoint or policy["checkpoint"])
        if (
            digest(parent) != policy["checkpoint_sha256"]
            or digest(args.cohort) != COHORT_SHA
        ):
            raise ValueError("Parent/cohort digest differs")
        if not args.smoke and not torch.cuda.is_available():
            raise ValueError("Production requires actual GPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data, fit, test, safe, _, center, scale = native_inputs(args.cohort)
        parent_state = torch_load_trusted(
            parent, map_location="cpu", weights_only=False
        )
        np.testing.assert_array_equal(
            parent_state["run_signature"]["scaler_center"], data["center"]
        )
        np.testing.assert_array_equal(
            parent_state["run_signature"]["scaler_scale"], data["scale"]
        )
        with h5py.File(args.cohort) as source:
            errors = source["features/reported_feature_errors"][:][data["feature_row"]]
            audit = (
                {"smoke_fixture_only": True}
                if args.smoke
                else audit_xp_error_sources(
                    args.cohort,
                    args.benchmark,
                    str(source.attrs["legacy_input"]),
                    data["feature_row"],
                    data["feature_columns"],
                )
            )
            units = (
                {"smoke_fixture_only": True}
                if args.smoke
                else probe.audit_fits_xp_units(
                    args.benchmark,
                    data["feature_columns"],
                    source["features/values"][:],
                    source["features/source_id"][:],
                    "g_normalized",
                )
            )
        atomic_json(args.output / "xp-error-source-audit.json", audit)
        atomic_json(args.output / "xp-unit-audit.json", units)
        columns = xp_indices(data["feature_columns"])
        sigma = standardized_xp_errors(
            errors,
            data["scale"],
            columns,
            g_magnitude=data["raw_values"][:, data["feature_columns"].index("G")],
            error_convention="raw_xp",
        )
        values = safe[:, columns]
        if not np.isfinite(data["raw_values"][:, columns]).all():
            raise ValueError("This pilot requires complete XP observations")
        measurement = data["fit_native_measurements"]
        y = (measurement["value"] - center) / scale
        sy = np.asarray(measurement["uncertainty"], dtype="f8") / scale
        if not np.isfinite(sy).all() or np.any(sy <= 0):
            raise ValueError("Quoted FeH errors must be positive, no fabricated floors")
        grid = np.linspace(y.min() - 1, y.max() + 1, policy["grid_points"], dtype="f4")
        if args.smoke:
            fit, measurement, y, sy = fit[:128], measurement[:128], y[:128], sy[:128]
            test = {
                "used_row": test["used_row"][:128],
                "measurements": test["measurements"][:128],
            }
        rows = test["used_row"]
        save_host_npz(
            args.output / "paired-cohort.npz",
            fit_source_id=data["source_id"][fit],
            fit_measurements=measurement,
            test_measurements=test["measurements"],
            fit_x=values[fit],
            test_x=values[rows],
            fit_sigma=sigma[fit],
            test_sigma=sigma[rows],
            fit_label_standardized=y,
            fit_label_sigma_standardized=sy,
            grid_standardized=grid,
            label_center=center,
            label_scale=scale,
            xp_columns=columns,
            scaler_center=data["center"],
            scaler_scale=data["scale"],
        )
        x, sx, target, label_sigma, tx, tsx, tg = [
            torch.as_tensor(a, dtype=torch.float32, device=device)
            for a in [values[fit], sigma[fit], y, sy, values[rows], sigma[rows], grid]
        ]
        states, predictions, verification, comparisons = {}, {}, {}, {}
        for method in policy["methods"]:
            model, state, _ = fit_model(
                method, x, sx, target, label_sigma, policy, args.smoke
            )
            prediction = evaluate(method, model, state, tx, tsx, tg)
            # Actual disk serialization and strict state reload, including CUDA->CPU.
            path = args.output / (method + ".pth")
            torch.save(state, path)
            saved = torch_load_trusted(path, map_location=device, weights_only=False)
            restored = copy.deepcopy(model)
            restored.load_state_dict(
                saved["decoder"] if method == "forward_plugin" else saved["state"],
                strict=True,
            )
            repeat = evaluate(method, restored, saved, tx, tsx, tg)
            np.testing.assert_array_equal(prediction, repeat)
            verification[method] = dict(
                strict_disk_reload_exact=True,
                rows=len(tx),
                checkpoint_sha256=digest(path),
            )
            states[method] = state
            if method in {"integrated_flat", "flow_eb"}:
                q = {
                    order: float(
                        model.objective(
                            x[:128],
                            sx[:128],
                            target[:128],
                            label_sigma[:128],
                            order=order,
                        ).detach()
                    )
                    for order in [32, 64, 128]
                }
                verification[method]["fit_only_quadrature_nll_per_xp"] = q
                if abs(q[64] - q[128]) > 0.01 * max(1, abs(q[128])):
                    raise ValueError(method + " GH64/128 convergence gate failed")
            chosen = prediction[:, 1] if method == "flow_eb" else prediction[:, 0]
            dex = chosen * scale + center
            predictions[method] = dex
            predictions[method + "_raw"] = prediction
            comparisons[method] = {
                name: probe.score(dex[keep], test["measurements"]["value"][keep])
                for name, keep in {
                    "all": np.ones(len(rows), bool),
                    "below_minus2": test["measurements"]["value"] < -2,
                }.items()
            }
            if method != "direct_mlp":
                comparisons[method]["grid_mle_all"] = probe.score(
                    prediction[:, 0] * scale + center, test["measurements"]["value"]
                )
                comparisons[method]["posterior_mean_all"] = probe.score(
                    prediction[:, 1] * scale + center, test["measurements"]["value"]
                )
            atomic_json(args.output / "checkpoint_verification.json", verification)
            print(method + " complete", flush=True)
        save_host_npz(
            args.output / "predictions.npz",
            source_id=test["measurements"]["source_id"],
            **predictions,
        )
        atomic_json(args.output / "comparison.json", comparisons)
        atomic_json(
            args.output / "artifact-sha256.json",
            {
                p.name: dict(sha256=digest(p), bytes=p.stat().st_size)
                for p in args.output.iterdir()
                if p.is_file()
                and p.name not in {"manifest.json", "artifact-sha256.json"}
            },
        )
        manifest.update(
            status="complete",
            elapsed_s=time.monotonic() - started,
            fit_count=len(fit),
            test_count=len(rows),
            label_center=center,
            label_scale=scale,
            grid_bounds_dex=[
                float(grid[0] * scale + center),
                float(grid[-1] * scale + center),
            ],
            result=comparisons,
        )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
