"""Label-free, reset-per-object TTA screen on the audited fixed10M cohort."""

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from run_enriched_value import COHORT_SHA
from run_native_age import state_digest
from run_prediction_comparison import digest, native_inputs
from run_stellar_label_experiments import BENCH
from stellar_catalog_jobs import atomic_json, identity_split

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.input_measurement import (
    audit_xp_error_sources,
    standardized_xp_errors,
    xp_indices,
)
from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_prediction_head,
)
from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)
from masked_stellar_autoencoder.training.test_time_experiments import (
    adapt_one,
    average_predictions,
)


def hashed_selection(ids, count, seed):
    """Select using IDs only; target values never enter subset eligibility."""
    keys = np.array(
        [
            int.from_bytes(
                hashlib.sha256(f"{int(s)}:{seed}".encode()).digest()[:8], "little"
            )
            for s in ids
        ],
        dtype="u8",
    )
    return np.argsort(keys, kind="stable")[: min(count, len(ids))]


@torch.no_grad()
def baseline_predictions(model, head, values, device):
    return np.concatenate(
        [
            head(
                model.encoder(torch.tensor(block, dtype=torch.float32, device=device))
            )[:, :, 0]
            .mean(1)
            .cpu()
            .numpy()
            for block in np.array_split(values, max(1, int(np.ceil(len(values) / 512))))
        ]
    )


def tune_inputs(cohort, data, count, seed):
    with h5py.File(cohort) as f:
        m = f["raw_layers/native/measurements"][:]
        rows = f["layer_index/native/feature_row"][:]
        ids = f["features/source_id"][:]
        available = f["features/feature_available"][:]
        partition = f["features/benchmark_partition"][:]
        good = (
            available[rows] & (partition[rows] == 0) & (identity_split(ids[rows]) == 1)
        )
        candidates = np.flatnonzero(good)
        choice = candidates[hashed_selection(m["source_id"][candidates], count, seed)]
        rows = rows[choice]
        # h5py slicing uses an ordered source-index read; restore hash-selection order.
        order = np.argsort(rows)
        inverse = np.argsort(order)
        raw = f["features/values"][rows[order]][inverse]
        errors = f["features/reported_feature_errors"][rows[order]][inverse]
    scaled = (raw - data["center"]) / data["scale"]
    safe, visible = spectrophotometric_inputs(
        torch.tensor(scaled, dtype=torch.float32),
        torch.tensor(np.isfinite(raw)),
        data["feature_columns"],
    )
    assert (identity_split(m["source_id"][choice]) == 1).all()
    return rows, m[choice], raw, errors, safe.numpy(), visible.numpy()


def evaluate_views(model, head, x, visible, sigma, ids, xp, policy, device):
    points = {}
    for label, noise, mask in [
        ("mask_average", 0.0, 0.1),
        ("noise_average", 1.0, 0.0),
        ("noise_mask_average", 1.0, 0.1),
    ]:
        chunks = []
        for start in range(0, len(x), 512):
            stop = start + 512
            values = torch.tensor(x[start:stop], dtype=torch.float32, device=device)
            shown = torch.tensor(visible[start:stop], dtype=torch.bool, device=device)
            sx = torch.tensor(sigma[start:stop], dtype=torch.float32, device=device)
            chunks.append(
                average_predictions(
                    model,
                    head,
                    values,
                    shown,
                    sx,
                    ids[start:stop],
                    xp,
                    views=policy["views"],
                    noise_scale=noise,
                    mask_probability=mask,
                    seed=policy["seed"],
                )
                .cpu()
                .numpy()
            )
        points[label] = np.concatenate(chunks)
    return points


def evaluate_adaptation(model, head, x, visible, ids, xp, policy, device, output, role):
    outputs = {}
    for method, mode, consistency, lr in [
        ("masked_view", "masked_view", 0.0, 0.01),
        ("latent_recon", "latent_recon", 0.0, 0.01),
        ("layernorm_recon", "layernorm_recon", 0.0, policy["layernorm_lr"]),
        (
            "layernorm_recon_consistency",
            "layernorm_recon",
            policy["consistency"],
            policy["layernorm_lr"],
        ),
    ]:
        values = []
        diagnostics = []
        for row, source in enumerate(ids):
            v = torch.tensor(x[row : row + 1], dtype=torch.float32, device=device)
            shown = torch.tensor(
                visible[row : row + 1], dtype=torch.bool, device=device
            )
            point, diagnostic = adapt_one(
                model,
                head,
                v,
                shown,
                xp,
                int(source),
                mode=mode,
                steps=policy["steps"],
                lr=lr,
                mask_probability=policy["adapt_mask_probability"],
                anchor=policy["anchor"],
                consistency=consistency,
                seed=policy["seed"],
            )
            values.append(float(point.cpu()))
            diagnostics.append(diagnostic)
        outputs[method] = np.asarray(values, dtype="f4")
        np.savez_compressed(
            output / (role + "-" + method + ".npz"),
            source_id=ids,
            standardized_prediction=outputs[method],
        )
        if len(ids):
            check, _ = adapt_one(
                model,
                head,
                torch.tensor(x[:1], dtype=torch.float32, device=device),
                torch.tensor(visible[:1], dtype=torch.bool, device=device),
                xp,
                int(ids[0]),
                mode=mode,
                steps=policy["steps"],
                lr=lr,
                mask_probability=policy["adapt_mask_probability"],
                anchor=policy["anchor"],
                consistency=consistency,
                seed=policy["seed"],
            )
            np.testing.assert_array_equal(
                np.asarray(float(check.cpu()), dtype="f4"), outputs[method][0]
            )
        atomic_json(output / (role + "-" + method + "-diagnostics.json"), diagnostics)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="test_time_reconstruction_screen_v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", default="configs/test-time-screen.yaml")
    p.add_argument(
        "--cohort",
        default="/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/enriched_feature_cohort_v1/enriched-feature-cohort.h5",
    )
    p.add_argument("--checkpoint")
    p.add_argument("--benchmark", default=BENCH)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    policy = yaml.safe_load(Path(args.config).read_text())
    if args.run_id != policy["run_id"]:
        raise ValueError("run absent from TTA policy")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        policy=policy,
        smoke=args.smoke,
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        training_presentations=0,
        reconstruction_training_updates=0,
        scope="Fixed10M source cohort, inference-only; not10M reconstruction presentations",
    )
    atomic_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        checkpoint = Path(args.checkpoint or policy["checkpoint"])
        if (
            digest(checkpoint) != policy["checkpoint_sha256"]
            or digest(args.cohort) != COHORT_SHA
        ):
            raise ValueError("actual winner/cohort differs")
        if not args.smoke and not torch.cuda.is_available():
            raise ValueError("production requires GPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data, fit, test, safe, visible, center, scale = native_inputs(args.cohort)
        disk = torch_load_trusted(checkpoint, map_location="cpu", weights_only=False)
        np.testing.assert_array_equal(
            [center, scale], [disk["label_center"], disk["label_scale"]]
        )
        assert data["feature_columns"] == disk["feature_columns"]
        cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        model = probe.load_model(cfg, disk, device).eval()
        head = build_prediction_head("residual").to(device)
        head.load_state_dict(disk["head_state_dict"], strict=True)
        head.eval()
        before = dict(model=state_digest(model), head=state_digest(head))
        xp = xp_indices(data["feature_columns"]).tolist()
        tune_rows, tune_m, tune_raw, tune_errors, tune_x, tune_visible = tune_inputs(
            args.cohort,
            data,
            4 if args.smoke else policy["tune_objects"],
            policy["seed"],
        )
        with h5py.File(args.cohort) as f:
            errors = f["features/reported_feature_errors"][:][data["feature_row"]]
            audit = (
                {"smoke_only": True}
                if args.smoke
                else audit_xp_error_sources(
                    args.cohort,
                    args.benchmark,
                    str(f.attrs["legacy_input"]),
                    np.unique(np.r_[data["feature_row"], tune_rows]),
                    data["feature_columns"],
                )
            )
            units = (
                {"smoke_only": True}
                if args.smoke
                else probe.audit_fits_xp_units(
                    args.benchmark,
                    data["feature_columns"],
                    f["features/values"][:],
                    f["features/source_id"][:],
                    "g_normalized",
                )
            )
        atomic_json(args.output / "xp-error-source-audit.json", audit)
        atomic_json(args.output / "xp-unit-audit.json", units)
        sigma = standardized_xp_errors(
            errors,
            data["scale"],
            np.asarray(xp),
            g_magnitude=data["raw_values"][:, data["feature_columns"].index("G")],
            error_convention="raw_xp",
        )
        tune_sigma = standardized_xp_errors(
            tune_errors,
            data["scale"],
            np.asarray(xp),
            g_magnitude=tune_raw[:, data["feature_columns"].index("G")],
            error_convention="raw_xp",
        )
        rows = test["used_row"][:128] if args.smoke else test["used_row"]
        measurement = test["measurements"][: len(rows)]
        ids = measurement["source_id"]
        full = baseline_predictions(model, head, safe[rows], device)
        reference = np.load(checkpoint.parent / "predictions.npz", allow_pickle=False)[
            "member_outputs"
        ][: len(rows), :, 0].mean(1)
        np.testing.assert_allclose(full, reference, rtol=5e-5, atol=5e-5)
        selection = hashed_selection(
            ids, 4 if args.smoke else policy["adapt_objects"], policy["seed"]
        )
        np.savez_compressed(
            args.output / "paired-inputs.npz",
            test_measurements=measurement,
            test_input=safe[rows],
            test_visible=visible[rows],
            test_sigma_xp=sigma[rows],
            xp_columns=xp,
            adapt_selection=selection,
            tune_measurements=tune_m,
            tune_input=tune_x,
            tune_visible=tune_visible,
            tune_sigma_xp=tune_sigma,
            tune_feature_row=tune_rows,
            label_center=center,
            label_scale=scale,
        )
        test_views = evaluate_views(
            model, head, safe[rows], visible[rows], sigma[rows], ids, xp, policy, device
        )
        test_adapt = evaluate_adaptation(
            model,
            head,
            safe[rows][selection],
            visible[rows][selection],
            ids[selection],
            xp,
            policy,
            device,
            args.output,
            "test",
        )
        tune_base = baseline_predictions(model, head, tune_x, device)
        tune_views = evaluate_views(
            model,
            head,
            tune_x,
            tune_visible,
            tune_sigma,
            tune_m["source_id"],
            xp,
            policy,
            device,
        )
        tune_adapt = evaluate_adaptation(
            model,
            head,
            tune_x,
            tune_visible,
            tune_m["source_id"],
            xp,
            policy,
            device,
            args.output,
            "tune",
        )
        # No parameters/buffers, original gradients or training modes may leak across stars.
        assert before == dict(model=state_digest(model), head=state_digest(head))
        np.testing.assert_array_equal(
            baseline_predictions(model, head, safe[rows], device), full
        )
        np.savez_compressed(
            args.output / "predictions.npz",
            source_id=ids,
            baseline=full * scale + center,
            **{k: v * scale + center for k, v in test_views.items()},
            **{k + "_subset": v * scale + center for k, v in test_adapt.items()},
        )
        np.savez_compressed(
            args.output / "tune-predictions.npz",
            source_id=tune_m["source_id"],
            baseline=tune_base * scale + center,
            **{k: v * scale + center for k, v in tune_views.items()},
            **{k + "_subset": v * scale + center for k, v in tune_adapt.items()},
        )
        summary = {}
        for role, truth, base, views, adapt, selected in [
            ("test", measurement["value"], full, test_views, test_adapt, selection),
            (
                "tune",
                tune_m["value"],
                tune_base,
                tune_views,
                tune_adapt,
                np.arange(len(tune_m)),
            ),
        ]:
            summary[role] = {
                "baseline_full": probe.score(base * scale + center, truth),
                "baseline_adapt_subset": probe.score(
                    base[selected] * scale + center, truth[selected]
                ),
            }
            summary[role].update(
                {k: probe.score(v * scale + center, truth) for k, v in views.items()}
            )
            summary[role].update(
                {
                    k: probe.score(v * scale + center, truth[selected])
                    for k, v in adapt.items()
                }
            )
        atomic_json(args.output / "comparison.json", summary)
        manifest.update(
            status="complete",
            elapsed_s=time.monotonic() - started,
            full_test_count=len(ids),
            adapt_test_count=len(selection),
            tune_count=len(tune_m),
            original_model_head_unchanged=True,
            baseline_winner_max_abs=float(np.max(abs(full - reference))),
        )
        atomic_json(
            args.output / "artifact-sha256.json",
            {
                p.name: dict(sha256=digest(p), bytes=p.stat().st_size)
                for p in args.output.iterdir()
                if p.is_file()
                and p.name not in {"artifact-sha256.json", "manifest.json"}
            },
        )
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
