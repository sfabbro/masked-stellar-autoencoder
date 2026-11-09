"""Native CUDA PDF QA queue: score selected immutable checkpoints after training."""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if not ROOT.is_dir():
        raise RuntimeError("CANFAR only")
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.decision_pdf_metrics import sample_scores

    cfg = yaml.safe_load(a.config.read_text())
    arms = {v["run_id"]: v for v in cfg["arms"]}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    direct = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
        raise ValueError("Library differs")
    source, _, _ = prior.admission(cfg, next(iter(arms.values())))
    if prior.sha(source["prepared_path"]) != source["prepared_sha256"]:
        raise ValueError("Prepared source differs")
    reference = torch_load_trusted(
        cfg["reference_checkpoint"], map_location="cpu", weights_only=False
    )
    if prior.sha(cfg["reference_checkpoint"]) != cfg["reference_checkpoint_sha256"]:
        raise ValueError("Reference checkpoint differs")
    model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
    device = torch.device("cuda")
    torch.set_num_threads(4)
    results = {}
    pending = set(arms)
    deadline = time.monotonic() + 24 * 3600

    def save():
        prior.atomic(
            a.output / "manifest.json",
            dict(
                status="waiting_or_scoring"
                if pending
                else "six_target_pdf_qa_complete",
                pending=sorted(pending),
                results=results,
                training_performed=False,
                optimizer_created=False,
                scope="Sample CRPS, interval widths and energy scores against sparse reference values; age/distance teachers and survey-scale dependence retained; no certified physical six-D coverage claim",
            ),
        )

    save()
    with h5py.File(source["prepared_path"]) as data:
        ids = data["source_id"][:]
        roles = data["role"][:].astype("U")
        values = data["value"][:]
        quality = data["observed"][:].astype(bool) & data["quality"][:].astype(bool)
        y, sigma, mask, center, scale, known = prior.normalize_targets(
            values, quality, data["error"][:], data["error_observed"][:], roles == "fit"
        )
        for key in ["scaler_center", "scaler_scale"]:
            np.testing.assert_array_equal(data[key][:], reference["run_signature"][key])
        panel = prior.fixed_role_rows(ids, roles, mask, "devtest", 2048)
        while pending and time.monotonic() < deadline:
            ready = []
            for rid in sorted(pending):
                path = ROOT / rid / "manifest.json"
                if not path.exists():
                    continue
                try:
                    m = json.loads(path.read_text())
                except (ValueError, OSError):
                    continue
                if m.get("status") == "failed":
                    results[rid] = dict(status="upstream_failed", error=m.get("error"))
                    pending.remove(rid)
                elif m.get("status") == "remediation_pilot_complete":
                    ready.append((rid, m))
            for rid, manifest in ready:
                model = head = None
                try:
                    arm = arms[rid]
                    state = torch_load_trusted(
                        ROOT / rid / "selected-checkpoint.pth",
                        map_location="cpu",
                        weights_only=False,
                    )
                    if (
                        state["run_id"] != rid
                        or state["prepared_sha256"] != source["prepared_sha256"]
                        or state["parent_sha256"] != cfg["checkpoint_sha256"]
                        or state["arm"] != arm
                        or state["update"] != manifest["selected_update"]
                    ):
                        raise ValueError("Selected state binding differs")
                    np.testing.assert_array_equal(state["center"], center)
                    np.testing.assert_array_equal(state["scale"], scale)
                    model = native.probe.load_model(model_cfg, reference, device)
                    model.load_state_dict(state["model_state_dict"], strict=True)
                    model.eval().requires_grad_(False)
                    head = prior.build_head(
                        arm["kind"], model_cfg["model"]["layer_dims"][-1], cfg
                    ).to(device)
                    head.load_state_dict(state["head_state_dict"], strict=True)
                    head.eval().requires_grad_(False)
                    del state
                    blocks = []
                    for start in range(0, len(panel), 4):
                        rows = panel[start : start + 4]
                        with torch.no_grad():
                            z = model.encoder(
                                torch.as_tensor(
                                    prior.profile_batch(
                                        data,
                                        rows,
                                        arm.get("input_profile", "phot_xp"),
                                        source,
                                    ),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                            gen = torch.Generator(device=device).manual_seed(
                                2026110600 + start
                            )
                            draws = prior.bounded_whole_draws(head, z, 1024, gen, 128)
                        standardized = sample_scores(draws, y[rows], mask[rows])
                        native_draws = draws * scale + center
                        native_draws[:, :, 4:] = np.exp(native_draws[:, :, 4:])
                        if not np.isfinite(native_draws).all():
                            raise ValueError("Native draws overflowed; no clipping")
                        scores = sample_scores(native_draws, values[rows], mask[rows])
                        noise = (
                            np.random.default_rng(2026110610 + start).standard_normal(
                                draws.shape
                            )
                            * sigma[rows][None]
                        )
                        noisy = sample_scores(
                            draws + noise, y[rows], mask[rows] & known[rows]
                        )
                        blocks.append(
                            dict(
                                crps=scores["crps"],
                                pit=scores["pit"],
                                quantiles=scores["quantiles"],
                                standardized_crps=standardized["crps"],
                                energy=standardized["energy"],
                                known_noise_crps=noisy["crps"],
                            )
                        )
                        if start % 1024 == 0:
                            prior.atomic(
                                a.output / (rid + "-progress.json"),
                                dict(rows=start + len(rows), total=len(panel)),
                            )
                    arrays = {
                        key: np.concatenate(
                            [b[key] for b in blocks],
                            axis=1 if key == "quantiles" else 0,
                        )
                        for key in blocks[0]
                    }
                    path = a.output / (rid + "-pdf-scores.npz")
                    np.savez_compressed(
                        path,
                        source_id=ids[panel],
                        reference=values[panel],
                        observed=mask[panel],
                        **arrays,
                    )
                    results[rid] = dict(
                        status="pdf_qa_complete",
                        rows=len(panel),
                        per_target=[
                            dict(
                                target=t,
                                n=int(mask[panel, j].sum()),
                                mean_crps=float(np.nanmean(arrays["crps"][:, j])),
                                mean_standardized_crps=float(
                                    np.nanmean(arrays["standardized_crps"][:, j])
                                ),
                                coverage90=float(
                                    np.nanmean(
                                        (arrays["pit"][:, j][mask[panel, j]] >= 0.05)
                                        & (arrays["pit"][:, j][mask[panel, j]] <= 0.95)
                                    )
                                ),
                                median_width90=float(
                                    np.median(
                                        (
                                            arrays["quantiles"][7, :, j]
                                            - arrays["quantiles"][1, :, j]
                                        )[mask[panel, j]]
                                    )
                                ),
                            )
                            for j, t in enumerate(prior.NAMES)
                        ],
                        mean_masked_standardized_energy=float(
                            np.nanmean(arrays["energy"])
                        ),
                        selected_checkpoint_sha256=prior.sha(
                            ROOT / rid / "selected-checkpoint.pth"
                        ),
                        scores_sha256=prior.sha(path),
                        complete_six_references=int(mask[panel].all(1).sum()),
                    )
                except Exception as e:
                    results[rid] = dict(status="pdf_qa_failed", error=repr(e))
                finally:
                    del model, head
                    gc.collect()
                    torch.cuda.empty_cache()
                    pending.remove(rid)
                    save()
            if pending:
                time.sleep(45)
    if pending:
        for rid in pending:
            results[rid] = dict(status="upstream_incomplete_at_24h")
        pending.clear()
        save()


if __name__ == "__main__":
    main()
