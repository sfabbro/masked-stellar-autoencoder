"""Native CANFAR inference/storage benchmark; no optimization or model selection."""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


def storage_bytes(stars, latent_dim, quantiles=33):
    if min(stars, latent_dim, quantiles) <= 0:
        raise ValueError("Positive storage dimensions required")
    return dict(
        source_id_int64=stars * 8,
        latent_float32=stars * latent_dim * 4,
        six_marginal_quantile_grids_float32=stars * 6 * quantiles * 4,
        joint_1024_draws_float32=stars * 1024 * 6 * 4,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-failed-output", action="store_true")
    a = p.parse_args()
    if a.output.exists() and a.reuse_failed_output:
        previous = json.loads((a.output / "manifest.json").read_text())
        if previous.get("status") != "failed" or previous.get("training_performed"):
            raise ValueError("Only failed read-only benchmark output may be reused")
        attempt = 1
        while (a.output / f"failed-attempt-{attempt}.json").exists():
            attempt += 1
        with (a.output / f"failed-attempt-{attempt}.json").open("xb") as stream:
            stream.write((a.output / "manifest.json").read_bytes())
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.remediation_energy6 import (
        EnergyJointSixHead,
    )

    cfg = yaml.safe_load(a.config.read_text())
    m = dict(
        status="loading", training_performed=False, models_changed=False, benchmarks=[]
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Native CANFAR CUDA required")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Library revision differs")
        source = json.loads(Path(cfg["source_manifest"]).read_text())
        for path, expected in [
            (cfg["encoder_parent"], cfg["encoder_parent_sha256"]),
            (cfg["reference_checkpoint"], cfg["reference_checkpoint_sha256"]),
            (source["prepared_path"], source["prepared_sha256"]),
        ]:
            if prior.sha(path) != expected:
                raise ValueError("Model/input digest changed")
        device = torch.device("cuda")
        torch.set_num_threads(4)
        reference = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        state = torch_load_trusted(
            cfg["encoder_parent"], map_location="cpu", weights_only=False
        )
        model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
        model = native.probe.load_model(model_cfg, reference, device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval().requires_grad_(False)
        latent_dim = model_cfg["model"]["layer_dims"][-1]
        head = EnergyJointSixHead(
            latent_dim=latent_dim,
            width=cfg["head_width"],
            particles=cfg["particles"],
            seed=cfg["seed"],
            components=cfg["mixture_components"],
            energy_training_draws=32,
            unknown_error_floor_normalized=cfg["unknown_error_floor_normalized"],
        ).to(device)
        head.load_state_dict(state["head_state_dict"], strict=True)
        head.eval().requires_grad_(False)
        with h5py.File(source["prepared_path"]) as f:
            rows = np.arange(min(65536, len(f["source_id"])), dtype="i8")
            eligible = f["primary_input_eligible"][: len(rows)].astype(bool)
            rows = rows[eligible][:4096]
            if len(rows) < 256:
                raise ValueError("Too few natural-input stars for bounded benchmark")
            values = prior.profile_batch(f, rows, "phot_xp_measured_parallax", source)
        x = torch.as_tensor(values, device=device, dtype=torch.float32)

        def measure(name, batch, draws, operation):
            with torch.inference_mode():
                operation()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                elapsed = []
                for _ in range(3):
                    begin = time.perf_counter()
                    result = operation()
                    torch.cuda.synchronize()
                    elapsed.append(time.perf_counter() - begin)
                    if not torch.isfinite(result).all():
                        raise FloatingPointError("Nonfinite benchmark output")
            item = dict(
                operation=name,
                batch=batch,
                draws=draws,
                median_seconds=float(np.median(elapsed)),
                stars_per_second=float(batch / np.median(elapsed)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            )
            m["benchmarks"].append(item)
            save()
            gc.collect()
            torch.cuda.empty_cache()

        with torch.inference_mode():
            z = model.encoder(x)
        for batch in [64, 256, 1024, 4096]:
            if batch <= len(x):
                measure("encoder", batch, 0, lambda b=batch: model.encoder(x[:b]))
        for batch in [16, 64, 256]:
            for draws in [32, 128, 512]:
                gen = torch.Generator(device=device).manual_seed(2026110790)
                try:
                    measure(
                        "flow_samples_given_latent",
                        batch,
                        draws,
                        lambda b=batch, d=draws, g=gen: head.sample(z[:b], d, g),
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    m["benchmarks"].append(
                        dict(
                            operation="flow_samples_given_latent",
                            batch=batch,
                            draws=draws,
                            status="out_of_memory",
                        )
                    )
                    save()
        gc.collect()
        torch.cuda.empty_cache()
        m["allocated_before_quantile_bytes"] = torch.cuda.memory_allocated()
        with torch.inference_mode():
            gen = torch.Generator(device=device).manual_seed(2026110791)
            pieces = []
            for start in range(0, 128, 8):
                pieces.append(
                    prior.bounded_whole_draws(
                        head, z[start : start + 8], 1024, gen, 128
                    )
                )
                gc.collect()
            draw = np.concatenate(pieces, axis=1)
        convergence = {}
        full = np.quantile(draw, [0.05, 0.5, 0.95], axis=0)
        for count in [32, 128, 256, 512]:
            delta = np.quantile(draw[:count], [0.05, 0.5, 0.95], axis=0) - full
            convergence[str(count)] = dict(
                median_absolute_standardized_quantile_difference_per_target=np.median(
                    abs(delta), axis=(0, 1)
                ).tolist(),
                p95_absolute_standardized_quantile_difference_per_target=np.quantile(
                    abs(delta), 0.95, axis=(0, 1)
                ).tolist(),
            )
        m.update(
            status="complete",
            checkpoint_sha256=cfg["encoder_parent_sha256"],
            source_sha256=source["prepared_sha256"],
            library_revision=cfg["library_revision"],
            gpu=torch.cuda.get_device_name(),
            latent_dim=latent_dim,
            input_rows=len(x),
            storage_218131884_stars=storage_bytes(218131884, latent_dim),
            storage_174922161_stars=storage_bytes(174922161, latent_dim),
            sample_quantile_convergence=convergence,
            scope="Warm GPU architecture benchmark on an existing frozen checkpoint; no physical accuracy, PDF normalization, marginal quadrature or training choice inferred. Timing excludes catalogue I/O, CPU conversion, quantile reduction, compression and job scheduling. Flow+exact float32 context defines on-demand whole-vector draws; stored marginal grids are approximations, not exact analytic marginal densities. Finite-draw quantile convergence uses 1,024 nested draws as a noisy reference, not ground truth. No per-star 1,024-draw all-sky catalogue is launched by this benchmark.",
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
