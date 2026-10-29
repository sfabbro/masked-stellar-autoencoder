"""CANFAR-only evaluation of an existing six-target NSF checkpoint, without training."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import run_remediation_energy_pilot as prior
import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"


def whole_draws(head, z, count, chunk, generator):
    import torch

    if (
        not isinstance(count, int)
        or not isinstance(chunk, int)
        or count < 2
        or chunk < 1
    ):
        raise ValueError("Invalid sample count/chunk")
    blocks = []
    with torch.no_grad():
        for start in range(0, count, chunk):
            blocks.append(
                head.sample(z, min(chunk, count - start), generator)
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(blocks, axis=0)


def admission(
    evidence,
    original_run,
    checkpoint,
    checkpoint_sha,
    expected_update,
    manifest_sha=None,
):
    if manifest_sha is not None and prior.sha(evidence) != manifest_sha:
        raise ValueError("Original manifest SHA differs")
    m = prior.load_json(evidence)
    if (
        m.get("run_id") != original_run
        or m.get("source_commit") != "f70aedca512925481f29a5022bf0184084282374"
        or m.get("status") != "failed"
        or m.get("error_type") != "OutOfMemoryError"
        or m.get("completed_updates") != 1000
        or m.get("selected_update") != expected_update
    ):
        raise ValueError(
            "Original finished training and out-of-memory evidence does not match"
        )
    if prior.sha(checkpoint) != checkpoint_sha:
        raise ValueError("Selected checkpoint SHA differs")
    return m


def run(cfg, arm, recovery, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": recovery["run_id"],
        "status": "admitting",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "training_performed": False,
        "optimizer_created": False,
        "original_run": arm["run_id"],
        "original_selected_checkpoint_sha256": recovery["checkpoint_sha256"],
    }
    prior.atomic(output / "manifest.json", manifest)
    try:
        original = admission(
            ROOT + arm["run_id"] + "/manifest.json",
            arm["run_id"],
            ROOT + arm["run_id"] + "/selected-checkpoint.pth",
            recovery["checkpoint_sha256"],
            recovery["selected_update"],
            recovery["manifest_sha256"],
        )
        source, audit, preflight = prior.admission(cfg, arm)
        prepared = Path(source["prepared_path"])
        if (
            prior.sha(prepared) != source["prepared_sha256"]
            or not Path("/arc/projects/k-pop").is_dir()
        ):
            raise ValueError("Prepared source SHA or CANFAR mount differs")
        import run_multilabel_screen as native
        import torch

        from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
        from masked_stellar_autoencoder.training import remediation_joint6

        if not torch.cuda.is_available():
            raise RuntimeError("CANFAR CUDA required")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("torchregress revision differs")
        if prior.sha(remediation_joint6.__file__) != preflight.get(
            "head_module_sha256"
        ):
            raise ValueError("Preflight head source differs")
        for key in ("checkpoint", "reference_checkpoint"):
            if prior.sha(cfg[key]) != cfg[key + "_sha256"]:
                raise ValueError("Parent checkpoint SHA differs")
        reference = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        signature = reference["run_signature"]
        model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
        device = torch.device("cuda")
        torch.set_num_threads(4)
        with h5py.File(prepared, "r") as data:
            if (
                data.attrs.get("schema") != "remediation-sparse-six-target-source-v1"
                or data.attrs.get("source_commit") != source["source_commit"]
            ):
                raise ValueError("Prepared source binding differs")
            names = data["feature_names"][:].astype("U").tolist()
            if names != signature["feature_cols"] or data["target_names"][:].astype(
                "U"
            ).tolist() != list(prior.NAMES):
                raise ValueError("Feature/target order differs")
            for key in ("scaler_center", "scaler_scale"):
                np.testing.assert_array_equal(data[key][:], signature[key])
            ids = data["source_id"][:]
            roles = data["role"][:].astype("U")
            if (
                ids.dtype.kind not in "iu"
                or np.any(ids <= 0)
                or np.any(np.diff(ids) <= 0)
            ):
                raise ValueError("Source IDs invalid")
            values, errors = data["value"][:], data["error"][:]
            quality = data["observed"][:].astype(bool) & data["quality"][:].astype(bool)
            y, sigma, mask, center, scale, error_known = prior.normalize_targets(
                values, quality, errors, data["error_observed"][:], roles == "fit"
            )
            native_scale = np.array(
                [values[(roles == "fit") & mask[:, j], j].std() for j in range(6)]
            )
            if not np.isfinite(native_scale).all() or not (native_scale > 0).all():
                raise ValueError("FIT scale invalid")
            tune = prior.fixed_role_rows(
                ids, roles, mask, "tune", cfg["tune_per_label"]
            )
            dev = prior.fixed_role_rows(
                ids, roles, mask, "devtest", cfg["evaluation_per_label"]
            )
            corner = prior.heldout_corner_rows(ids, roles, cfg["corner_source_ids"])
            model = native.probe.load_model(model_cfg, reference, device)
            del reference
            model.eval()
            head = prior.build_head(
                "nsf_energy", model_cfg["model"]["layer_dims"][-1], cfg
            ).to(device)
            head.eval()
            state = torch_load_trusted(
                ROOT + arm["run_id"] + "/selected-checkpoint.pth",
                map_location="cpu",
                weights_only=False,
            )
            if (
                state.get("run_id") != arm["run_id"]
                or state.get("update") != recovery["selected_update"]
                or state.get("prepared_sha256") != source["prepared_sha256"]
                or state.get("parent_sha256") != cfg["checkpoint_sha256"]
                or state.get("arm") != arm
            ):
                raise ValueError("Checkpoint lineage differs")
            np.testing.assert_array_equal(state["center"], center)
            np.testing.assert_array_equal(state["scale"], scale)
            model.load_state_dict(state["model_state_dict"], strict=True)
            head.load_state_dict(state["head_state_dict"], strict=True)
            del state
            gc.collect()
            for p in model.parameters():
                p.requires_grad_(False)
            for p in head.parameters():
                p.requires_grad_(False)
            profile = arm.get("input_profile", "phot_xp")

            def latent(rows):
                x = torch.as_tensor(
                    prior.profile_batch(data, rows, profile, source),
                    dtype=torch.float32,
                    device=device,
                )
                return model.encoder(x)

            def evaluate(rows, path):
                points = []
                pits = []
                observed_pits = []
                for start in range(0, len(rows), recovery["batch_size"]):
                    rr = rows[start : start + recovery["batch_size"]]
                    with torch.no_grad():
                        z = latent(rr)
                        gen = torch.Generator(device=device).manual_seed(
                            cfg["seed"] + int(start)
                        )
                        draws = whole_draws(
                            head,
                            z,
                            cfg["evaluation_draws"],
                            recovery["draw_chunk"],
                            gen,
                        )
                    points.append(np.median(draws, axis=0))
                    pits.append(np.mean(draws <= y[rr][None], axis=0))
                    er = np.random.default_rng(cfg["seed"] + 701 + int(start))
                    noisy = draws + er.standard_normal(draws.shape) * sigma[rr][None]
                    observed_pits.append(
                        np.where(
                            error_known[rr],
                            np.mean(noisy <= y[rr][None], axis=0),
                            np.nan,
                        )
                    )
                pred = np.concatenate(points)
                native_pred = pred * scale + center
                native_pred[:, 4:] = np.exp(native_pred[:, 4:])
                if not np.isfinite(native_pred).all():
                    raise FloatingPointError("Nonfinite predictions")
                n = mask[rows].sum(axis=0)
                if (n == 0).any():
                    raise ValueError("Missing panel label")
                per = (
                    np.where(
                        mask[rows], np.abs(native_pred - values[rows]) / native_scale, 0
                    ).sum(axis=0)
                    / n
                )
                np.savez_compressed(
                    path,
                    source_id=ids[rows],
                    row=rows,
                    value=values[rows],
                    observed=mask[rows],
                    error=errors[rows],
                    error_observed=error_known[rows],
                    label_source=data["label_source"][:].astype("U64")[rows],
                    standardized_median=pred,
                    native_median=native_pred,
                    empirical_intrinsic_pit=np.where(
                        mask[rows], np.concatenate(pits), np.nan
                    ),
                    empirical_observed_pit=np.where(
                        mask[rows], np.concatenate(observed_pits), np.nan
                    ),
                    role=roles[rows],
                    real_xp_present=data["real_xp_present"][:][rows],
                    real_no_xp=data["real_no_xp"][:][rows],
                    measured_parallax_observed=data["measured_parallax/observed"][:][
                        rows
                    ],
                )
                return float(per.mean()), per.tolist()

            manifest.update(
                status="evaluating",
                prepared_sha256=source["prepared_sha256"],
                selected_update=recovery["selected_update"],
                original_failure=original["error_type"],
                batch_size=recovery["batch_size"],
                draw_chunk=recovery["draw_chunk"],
            )
            prior.atomic(output / "manifest.json", manifest)
            tune_score, tune_per = evaluate(tune, output / "tune-predictions.npz")
            dev_score, dev_per = evaluate(dev, output / "development-predictions.npz")
            with torch.no_grad():
                z = latent(corner)
                gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 9999)
                draws = whole_draws(
                    head, z, cfg["corner_draws"], recovery["draw_chunk"], gen
                )
            native_draws = draws * scale + center
            native_draws[:, :, 4:] = np.exp(native_draws[:, :, 4:])
            if not np.isfinite(native_draws).all():
                raise FloatingPointError("Nonfinite joint samples")
            np.savez_compressed(
                output / "whole-vector-corners.npz",
                source_id=ids[corner],
                row=corner,
                standardized_draws=draws,
                native_draws=native_draws,
                standardized_covariance=np.asarray(
                    [np.cov(draws[:, j, :], rowvar=False) for j in range(len(corner))]
                ),
                native_covariance=np.asarray(
                    [
                        np.cov(native_draws[:, j, :], rowvar=False)
                        for j in range(len(corner))
                    ]
                ),
                reference=values[corner],
                observed=mask[corner],
            )
            np.savez_compressed(
                output / "normalization.npz",
                center=center,
                scale=scale,
                native_fit_scale=native_scale,
                feature_names=np.asarray(names),
                target_names=np.asarray(prior.NAMES),
            )
            manifest.update(
                status="nsf_energy_evaluation_recovery_complete",
                tune_score=tune_score,
                tune_per_label=tune_per,
                development_score=dev_score,
                development_per_label=dev_per,
                scope="Inference-only; chunked actual joint samples, previously used development panel, no exact NLL or physical truth claim",
                artifacts={
                    p.name: {"bytes": p.stat().st_size, "sha256": prior.sha(p)}
                    for p in output.iterdir()
                    if p.is_file() and p.name != "manifest.json"
                },
            )
    except Exception as e:
        manifest.update(status="failed", error_type=type(e).__name__, error=str(e))
        prior.atomic(output / "manifest.json", manifest)
        raise
    prior.atomic(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.config).read_text())
    cfg = yaml.safe_load(Path(plan["training_config"]).read_text())
    recovery = [r for r in plan["recoveries"] if r["run_id"] == args.run_id]
    if len(recovery) != 1:
        raise ValueError("Unknown recovery run")
    arm = [a for a in cfg["arms"] if a["run_id"] == recovery[0]["original_run"]]
    if len(arm) != 1:
        raise ValueError("Original arm differs")
    run(cfg, arm[0], recovery[0], args.output)


if __name__ == "__main__":
    main()
