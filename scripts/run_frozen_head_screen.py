"""Fit-only VIR/CNF screens on an immutable encoder; no reconstruction SGD."""

import argparse
import hashlib
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
from run_prediction_comparison import digest, native_inputs
from run_stellar_label_experiments import BENCH
from stellar_catalog_jobs import atomic_json
from torch import nn

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.cnf_regression_features import (
    CNFRegressionFeatures,
    density_shaping_loss,
)
from masked_stellar_autoencoder.training.prediction_vir import VIRPredictionHead


def save_host_npz(path, **arrays):
    """Explicit host transfer: NumPy must never implicitly coerce CUDA tensors."""
    host = {
        key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in arrays.items()
    }
    atomic_json(
        Path(path).with_suffix(".storage.json"),
        {
            key: {
                "type": type(value).__name__,
                "device": str(value.device)
                if isinstance(value, torch.Tensor)
                else "host",
            }
            for key, value in arrays.items()
        },
    )
    np.savez_compressed(path, **host)


def stream(count, updates, batch, seed=42):
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        yield rng.integers(count, size=batch)


def point_head(dim):
    head = nn.Sequential(
        nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


@torch.no_grad()
def evaluate(head, values, *, vir=False):
    head.eval()
    outputs, posteriors = [], []
    for block in values.split(512):
        if vir:
            outputs.append(head(block)[:, 0, 0].cpu().numpy())
            posterior = head.posterior(block)
            posteriors.append(
                torch.stack(
                    (posterior.gamma, posterior.nu, posterior.alpha, posterior.beta), -1
                )
                .cpu()
                .numpy()
            )
        else:
            outputs.append(head(block).flatten().cpu().numpy())
    return np.concatenate(outputs), np.concatenate(posteriors) if vir else None


def train_head(fit, labels, test, mode, policy, out, smoke):
    torch.manual_seed(42)
    vir = mode in VIRPredictionHead.MODES
    head = (
        VIRPredictionHead(labels, mode=mode) if vir else point_head(fit.shape[1])
    ).to(fit.device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=policy["head_lr"], weight_decay=1e-4
    )
    updates = 4 if smoke else policy["head_updates"]
    batch = min(policy["batch_size"], len(fit))
    epoch_updates = int(np.ceil(len(fit) / batch))
    initial, _ = evaluate(head, test, vir=vir)
    label_hash = hashlib.sha256()
    for step, indices in enumerate(stream(len(fit), updates, batch)):
        if vir and mode == "vir" and step > 0 and step % epoch_updates == 0:
            head.refresh_statistics(fit, labels)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = fit[indices], labels[indices]
        loss = head.objective(x, y) if vir else (head(x).flatten() - y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite head loss")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in head.parameters()
        ):
            raise FloatingPointError("Nonfinite head gradient")
        optimizer.step()
        label_hash.update(indices.tobytes())
        if (step + 1) % 100 == 0 or step + 1 == updates:
            with (out / "metrics.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        dict(method=mode, step=step + 1, loss=float(loss.detach()))
                    )
                    + "\n"
                )
    if vir and mode == "vir":
        head.refresh_statistics(fit, labels)
    final, posterior = evaluate(head, test, vir=vir)
    state = dict(
        head=head.state_dict(),
        optimizer=optimizer.state_dict(),
        mode=mode,
        updates=updates,
        stream_sha256=label_hash.hexdigest(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )
    torch.save(state, out / (mode + "-head.pth"))
    # Actual disk reload, not an in-memory copy.
    restored = (
        VIRPredictionHead(labels, mode=mode) if vir else point_head(fit.shape[1])
    ).to(fit.device)
    restored.load_state_dict(
        torch_load_trusted(
            out / (mode + "-head.pth"), map_location=fit.device, weights_only=False
        )["head"],
        strict=True,
    )
    replay, replay_posterior = evaluate(restored, test, vir=vir)
    np.testing.assert_array_equal(final, replay)
    if vir:
        np.testing.assert_array_equal(posterior, replay_posterior)
    save_host_npz(
        out / (mode + "-predictions.npz"),
        initial_standardized=initial,
        final_standardized=final,
        **({"nig_posterior": posterior} if vir else {}),
    )
    return dict(
        method=mode,
        updates=updates,
        label_presentations=updates * batch,
        stream_sha256=label_hash.hexdigest(),
        strict_disk_reload_all_test_exact=True,
    )


def flow_features(fit, test, raw_labels, mode, policy, out, smoke):
    from zuko.flows import NSF

    # Fit-only PCA; no target/test-driven dimension selection.
    _, singular, vectors = torch.linalg.svd(fit.double(), full_matrices=False)
    components = vectors[: policy["pca_dimensions"]].T.float()
    projection = fit @ components
    projection_scale = projection.std(0, unbiased=False).clamp_min(1e-6)
    fit_projection = projection / projection_scale
    test_projection = test @ components / projection_scale
    torch.manual_seed(42)
    pair = CNFRegressionFeatures(
        NSF(
            features=components.shape[1], transforms=3, hidden_features=(64, 64), bins=8
        ),
        NSF(
            features=components.shape[1], transforms=3, hidden_features=(64, 64), bins=8
        ),
        cut_dex=policy["cnf_cut_dex"],
    ).to(fit.device)
    signal = pair.partition(raw_labels)
    positive, negative = torch.where(signal)[0], torch.where(~signal)[0]
    optimizer = torch.optim.AdamW(
        pair.parameters(), lr=policy["flow_lr"], weight_decay=1e-4
    )
    rng = np.random.default_rng(43)
    updates = 2 if smoke else policy["flow_updates"]
    sample_hash = hashlib.sha256()
    clamped = [0, 0]
    for step in range(updates):
        a = positive[rng.integers(len(positive), size=256)]
        b = negative[rng.integers(len(negative), size=256)]
        sample_hash.update(a.cpu().numpy().tobytes())
        sample_hash.update(b.cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        signal_loss = density_shaping_loss(
            pair.signal_flow,
            fit_projection[a],
            fit_projection[b],
            contrastive_c=2.0 if mode == "cnf" else None,
        )
        background_loss = density_shaping_loss(
            pair.background_flow,
            fit_projection[b],
            fit_projection[a],
            contrastive_c=0.5 if mode == "cnf" else None,
        )
        clamped[0] += int(float(signal_loss.detach()) == -50.0)
        clamped[1] += int(float(background_loss.detach()) == -50.0)
        loss = signal_loss + background_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite flow loss")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in pair.parameters()
        ):
            raise FloatingPointError("Nonfinite flow gradient")
        optimizer.step()
    pair.freeze_flows()
    with torch.no_grad():
        fit_aug = torch.cat(
            [
                pair(z, p)
                for z, p in zip(fit.split(512), fit_projection.split(512), strict=True)
            ]
        )
        test_aug = torch.cat(
            [
                pair(z, p)
                for z, p in zip(
                    test.split(512), test_projection.split(512), strict=True
                )
            ]
        )
        phi_center = fit_aug[:, -2:].mean(0)
        phi_scale = fit_aug[:, -2:].std(0, unbiased=False)
        clip = {}
        for label, flow in [
            ("signal", pair.signal_flow),
            ("background", pair.background_flow),
        ]:
            logs = torch.cat([flow().log_prob(p) for p in fit_projection.split(512)])
            clip[label] = float(((logs < -30) | (logs > 30)).float().mean())
        if not torch.isfinite(phi_scale).all() or (phi_scale < 1e-12).any():
            raise ValueError("Constant/saturated Phi features: reject before head SGD")
        fit_aug[:, -2:] = (fit_aug[:, -2:] - phi_center) / phi_scale
        test_aug[:, -2:] = (test_aug[:, -2:] - phi_center) / phi_scale
    flow_state = dict(
        state=pair.state_dict(),
        optimizer=optimizer.state_dict(),
        sampling_rng=rng.bit_generator.state,
        sampling_sha256=sample_hash.hexdigest(),
        class_presentations_per_flow=updates * 256,
        clamped_steps=clamped,
        pca_components=components.cpu(),
        pca_scale=projection_scale.cpu(),
        phi_center=phi_center.cpu(),
        phi_scale=phi_scale.cpu(),
        flow_updates=updates,
        signal_fit_count=int(signal.sum()),
        background_fit_count=int((~signal).sum()),
        fit_clipped_fraction=clip,
    )
    torch.save(flow_state, out / (mode + "-features.pth"))
    restored = CNFRegressionFeatures(
        NSF(
            features=components.shape[1], transforms=3, hidden_features=(64, 64), bins=8
        ),
        NSF(
            features=components.shape[1], transforms=3, hidden_features=(64, 64), bins=8
        ),
        cut_dex=policy["cnf_cut_dex"],
    ).to(fit.device)
    disk = torch_load_trusted(
        out / (mode + "-features.pth"), map_location=fit.device, weights_only=False
    )
    restored.load_state_dict(disk["state"], strict=True)
    disk_projection = test @ disk["pca_components"] / disk["pca_scale"]
    restored.freeze_flows()
    with torch.no_grad():
        check = torch.cat(
            [
                restored(z, p)
                for z, p in zip(
                    test.split(512), disk_projection.split(512), strict=True
                )
            ]
        )
        check[:, -2:] = (check[:, -2:] - disk["phi_center"]) / disk["phi_scale"]
        torch.testing.assert_close(check, test_aug, rtol=0, atol=0)
    atomic_json(
        out / (mode + "-feature-report.json"),
        dict(
            signal_fit_count=int(signal.sum()),
            background_fit_count=int((~signal).sum()),
            phi_scale=phi_scale.tolist(),
            fit_clipped_fraction=clip,
            flow_updates=updates,
            class_presentations_per_flow=updates * 256,
            sampling_sha256=sample_hash.hexdigest(),
            clamped_steps=clamped,
            strict_disk_reload_features_all_test_exact=True,
        ),
    )
    return fit_aug, test_aug


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="configs/frozen-head-screen.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--cohort",
        default="/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/enriched_feature_cohort_v1/enriched-feature-cohort.h5",
    )
    parser.add_argument("--benchmark", default=BENCH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    policy = yaml.safe_load(Path(args.config).read_text())
    suite = next(a for a in policy["jobs"] if a["run_id"] == args.run_id)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        policy=policy,
        suite=suite,
        smoke=args.smoke,
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        reconstruction_presentations=0,
        encoder_updates=0,
        comparison_scope="Frozen encoder/head-only screen; not matched to prior encoder-adaptation10M experiments",
    )
    atomic_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        )
        if direct["vcs_info"]["commit_id"] != policy["torchregress_revision"]:
            raise ValueError("Installed library differs from immutable screen policy")
        parent = Path(args.checkpoint or policy["checkpoint"])
        if (
            digest(parent) != policy["checkpoint_sha256"]
            or digest(args.cohort) != COHORT_SHA
        ):
            raise ValueError("Actual parent/cohort digest differs")
        if not args.smoke and not torch.cuda.is_available():
            raise ValueError("Production screen requires actual GPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save_host_npz(
            args.output / "device-serialization-preflight.npz",
            probe=torch.arange(6.0, device=device).reshape(2, 3),
        )
        with np.load(args.output / "device-serialization-preflight.npz") as serialized:
            np.testing.assert_array_equal(serialized["probe"], [[0, 1, 2], [3, 4, 5]])
        data, fit, test, safe, _, yc, ys = native_inputs(args.cohort)
        if args.smoke:
            # Include an actual fit-tail star so binary density smoke has support.
            all_targets = data["fit_truth"][fit]
            selection = np.r_[
                np.flatnonzero(all_targets < policy["cnf_cut_dex"])[:16],
                np.flatnonzero(all_targets >= policy["cnf_cut_dex"])[:112],
            ]
            fit = fit[selection]
            data["fit_native_measurements"] = data["fit_native_measurements"][selection]
            test = {
                "used_row": test["used_row"][:128],
                "measurements": test["measurements"][:128],
            }
            yc, ys = (
                float(data["fit_truth"][fit].mean()),
                float(data["fit_truth"][fit].std()),
            )
        with h5py.File(args.cohort) as source:
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
        checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
        np.testing.assert_array_equal(
            checkpoint["run_signature"]["scaler_center"], data["center"]
        )
        np.testing.assert_array_equal(
            checkpoint["run_signature"]["scaler_scale"], data["scale"]
        )
        config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        model = (
            probe.load_model(config, checkpoint, device).eval().requires_grad_(False)
        )
        all_rows = np.concatenate((fit, test["used_row"]))
        latent = probe.encode(model, safe[all_rows], device)
        train_z = latent[: len(fit)]
        raw_test_latent = latent[len(fit) :]
        latent_center = train_z.mean(0)
        latent_scale = train_z.std(0)
        latent_scale = np.maximum(latent_scale, 1e-6)
        fit_z = torch.as_tensor((train_z - latent_center) / latent_scale, device=device)
        test_z = torch.as_tensor(
            (raw_test_latent - latent_center) / latent_scale, device=device
        )
        labels = torch.as_tensor(
            (data["fit_truth"][fit] - yc) / ys, dtype=torch.float32, device=device
        )
        raw_labels = torch.as_tensor(
            data["fit_truth"][fit], dtype=torch.float32, device=device
        )
        save_host_npz(
            args.output / "paired-cohort.npz",
            fit_source_id=data["source_id"][fit],
            fit_measurements=data["fit_native_measurements"],
            test_measurements=test["measurements"],
            fit_input=safe[fit],
            test_input=safe[test["used_row"]],
            fit_latent=train_z,
            test_latent=raw_test_latent,
            latent_center=latent_center,
            latent_scale=latent_scale,
            label_center=yc,
            label_scale=ys,
            fit_target=data["fit_truth"][fit],
        )
        with np.load(
            args.output / "paired-cohort.npz", allow_pickle=False
        ) as preserved:
            np.testing.assert_array_equal(preserved["fit_latent"], latent[: len(fit)])
            np.testing.assert_array_equal(preserved["test_latent"], latent[len(fit) :])
        before = hashlib.sha256(
            b"".join(
                v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
            )
        ).hexdigest()
        results = []
        for mode in suite["methods"]:
            fit_x, test_x = fit_z, test_z
            if suite["kind"] == "cnf":
                if mode == "raw_mse":
                    fit_x = torch.cat((fit_z, fit_z.new_zeros(len(fit_z), 2)), 1)
                    test_x = torch.cat((test_z, test_z.new_zeros(len(test_z), 2)), 1)
                else:
                    fit_x, test_x = flow_features(
                        fit_z, test_z, raw_labels, mode, policy, args.output, args.smoke
                    )
            result = train_head(
                fit_x, labels, test_x, mode, policy, args.output, args.smoke
            )
            results.append(result)
            atomic_json(
                args.output / "progress.json",
                dict(completed_methods=results, total_methods=len(suite["methods"])),
            )
        after = hashlib.sha256(
            b"".join(
                v.detach().cpu().numpy().tobytes() for v in model.state_dict().values()
            )
        ).hexdigest()
        assert before == after
        manifest.update(
            status="complete",
            parent_sha256=digest(parent),
            cohort_sha256=COHORT_SHA,
            unit_audit=units,
            native_fit=len(fit),
            native_test=len(test["used_row"]),
            label_center=yc,
            label_scale=ys,
            methods=results,
            encoder_state_sha256=before,
            encoder_unchanged=True,
            library=direct,
            elapsed_seconds=time.monotonic() - started,
            artifact_sha256={
                p.name: digest(p)
                for p in args.output.iterdir()
                if p.suffix in {".npz", ".pth"}
            },
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch_version=torch.__version__,
        )
        atomic_json(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        atomic_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
