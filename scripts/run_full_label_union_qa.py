"""CANFAR CUDA inference over the complete usable labelled-source registry.

An immutable first large-run snapshot; never optimizes or selects a model.
Whole-vector flow draws produce approximate marginal quantiles. All eligible
roles are scored, including reserved cluster members, with exact row/ID binding.
"""

import argparse
import gc
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
LEVELS = [0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=128)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    if a.draws < 128 or not 1 <= a.batch <= 64:
        raise ValueError("At least 128 draws and a bounded star batch required")
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.cached_inputs import cache_natural_inputs
    from masked_stellar_autoencoder.training.remediation_energy6 import (
        EnergyJointSixHead,
    )
    from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap

    cfg = yaml.safe_load(a.config.read_text())
    m = dict(
        status="waiting_for_first_checkpoint",
        training_performed=False,
        scope="Full usable source registry, not a tiny QA sample. First snapshot, not final selected model. Marginal quantiles are finite-draw estimates; no CAL interval correction applied. Mapped observations retain noisy survey labels; no physical ground truth claim.",
    )

    def save():
        prior.atomic(a.output / "manifest.json", m)

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Native CANFAR CUDA required")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Library differs")
        # Share the exact snapshot held by the existing high-draw diagnostic job.
        qa = ROOT / "expanded_first_checkpoint_qa_20261107_v1"
        deadline = time.monotonic() + 72 * 3600
        while time.monotonic() < deadline:
            try:
                q = json.loads((qa / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                q = {}
            if q.get("status") in ["failed", "upstream_failed"]:
                raise RuntimeError(
                    "First-checkpoint producer failed: " + str(q.get("error"))
                )
            snapshot = qa / "large-snapshot.pth"
            if q.get("large_snapshot_sha256") and snapshot.exists():
                shutil.copyfile(snapshot, a.output / "snapshot.pth")
                if prior.sha(a.output / "snapshot.pth") != q["large_snapshot_sha256"]:
                    raise ValueError("Snapshot digest changed")
                break
            time.sleep(30)
        else:
            raise TimeoutError("No immutable snapshot in 72h")
        smpath = ROOT / cfg["expanded_source_run"] / "manifest.json"
        source = json.loads(smpath.read_text())
        if (
            source.get("status") != "expanded_six_source_complete"
            or prior.sha(source["prepared_path"]) != source["prepared_sha256"]
        ):
            raise ValueError("Source digest/status differs")
        if prior.sha(cfg["reference_checkpoint"]) != cfg["reference_checkpoint_sha256"]:
            raise ValueError("Reference differs")
        state = torch_load_trusted(
            a.output / "snapshot.pth", map_location="cpu", weights_only=False
        )
        if state["source_sha256"] != source["prepared_sha256"]:
            raise ValueError("Checkpoint/source binding differs")
        reference = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
        device = torch.device("cuda")
        torch.set_num_threads(4)
        model = native.probe.load_model(model_cfg, reference, device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval().requires_grad_(False)
        head = EnergyJointSixHead(
            latent_dim=model_cfg["model"]["layer_dims"][-1],
            width=cfg["head_width"],
            particles=cfg["particles"],
            seed=cfg["seed"],
            components=cfg["mixture_components"],
            energy_training_draws=32,
            unknown_error_floor_normalized=cfg["unknown_error_floor_normalized"],
        ).to(device)
        head.load_state_dict(state["head_state_dict"], strict=True)
        head.eval().requires_grad_(False)
        center, scale = np.asarray(state["center"]), np.asarray(state["scale"])
        ms = state["map_state_dict"]
        maps = SixSurveyMap(
            ms["support"].cpu(),
            ms["reference"].cpu(),
            zero_sum=torch.nonzero(ms["zero_sum"].cpu(), as_tuple=False)
            .flatten()
            .tolist(),
            kind=state["decision"]["mapping_kind"],
        )
        maps.load_state_dict(ms, strict=True)
        bias, slopes = maps.coefficients()
        np.savez_compressed(
            a.output / "map-coefficients.npz",
            bias=bias.detach().numpy(),
            slopes=slopes.detach().numpy(),
            support=ms["support"].cpu().numpy(),
            kind=np.array(state["decision"]["mapping_kind"]),
            source_registry=np.asarray(source["source_registry"]),
            center=center,
            scale=scale,
        )
        m.update(
            status="preparing",
            checkpoint_sha256=prior.sha(a.output / "snapshot.pth"),
            source_manifest_path=str(smpath),
            source_sha256=source["prepared_sha256"],
            checkpoint_update=state.get("update"),
            checkpoint_epoch=state.get("epoch"),
            decision=state["decision"],
            draws=a.draws,
            quantile_levels=LEVELS,
            gpu=torch.cuda.get_device_name(),
            completed_rows=0,
        )
        save()
        del state, reference, maps
        with h5py.File(source["prepared_path"]) as f:
            ids, roles = f["source_id"][:], f["role"][:].astype("S8")
            eligible = f["primary_input_eligible"][:].astype(bool) & np.isin(
                roles, [b"fit", b"tune", b"cal", b"devtest", b"external"]
            )
            rows = np.flatnonzero(eligible)
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Duplicate registry IDs")
            x = cache_natural_inputs(
                f, source["feature_profiles"]["phot_xp_measured_parallax"]
            )
            check = rows[np.linspace(0, len(rows) - 1, 128, dtype=int)]
            np.testing.assert_array_equal(
                x[check],
                prior.profile_batch(f, check, "phot_xp_measured_parallax", source),
            )
            m.update(
                status="inference",
                total_rows=len(rows),
                role_counts={
                    r.decode(): int(((roles == r) & eligible).sum())
                    for r in np.unique(roles[eligible])
                },
                source_registry=source["source_registry"],
            )
            save()
            begin = time.monotonic()
            with h5py.File(a.output / "predictions.h5", "w") as out:
                out.attrs["source_sha256"] = source["prepared_sha256"]
                out.attrs["checkpoint_sha256"] = m["checkpoint_sha256"]
                for name, v in [
                    ("source_id", ids),
                    ("role", roles),
                    ("eligible", eligible),
                    ("quantile_levels", np.asarray(LEVELS)),
                ]:
                    out.create_dataset(name, data=v)
                med = out.create_dataset(
                    "median",
                    shape=(len(ids), 6),
                    dtype="f4",
                    chunks=(8192, 6),
                    compression="gzip",
                    compression_opts=1,
                    fillvalue=np.nan,
                )
                quant = out.create_dataset(
                    "quantiles",
                    shape=(len(ids), 9, 6),
                    dtype="f4",
                    chunks=(2048, 9, 6),
                    compression="gzip",
                    compression_opts=1,
                    fillvalue=np.nan,
                )
                # Accumulate a bounded I/O block; small GPU batches avoid flow OOM.
                for start in range(0, len(rows), 4096):
                    rr = rows[start : start + 4096]
                    blocks = []
                    for k in range(0, len(rr), a.batch):
                        r = rr[k : k + a.batch]
                        with torch.inference_mode():
                            z = model.encoder(
                                torch.as_tensor(
                                    x[r], device=device, dtype=torch.float32
                                )
                            )
                            gen = torch.Generator(device=device).manual_seed(
                                2026110770 + start + k
                            )
                            draws = prior.bounded_whole_draws(
                                head, z, a.draws, gen, 128
                            )
                        physical = draws * scale + center
                        physical[:, :, 4:] = np.exp(physical[:, :, 4:])
                        if not np.isfinite(physical).all():
                            raise FloatingPointError("Nonfinite draw, no clipping")
                        blocks.append(
                            np.quantile(physical, LEVELS, axis=0)
                            .transpose(1, 0, 2)
                            .astype("f4")
                        )
                        del draws, physical, z
                        if k % 512 == 0:
                            gc.collect()
                    values = np.concatenate(blocks)
                    med[rr] = values[:, 4]
                    quant[rr] = values
                    out.flush()
                    m.update(
                        completed_rows=start + len(rr),
                        elapsed_seconds=time.monotonic() - begin,
                    )
                    save()
                cluster_rows = np.flatnonzero(
                    f["reserved_globular_member"][:].astype(bool) & eligible
                )
                if len(cluster_rows):
                    cd = out.create_dataset(
                        "cluster_draws",
                        shape=(len(cluster_rows), 1024, 6),
                        dtype="f4",
                        chunks=(1, 1024, 6),
                        compression="gzip",
                        compression_opts=1,
                    )
                    out.create_dataset("cluster_row", data=cluster_rows)
                    for k in range(0, len(cluster_rows), 8):
                        r = cluster_rows[k : k + 8]
                        with torch.inference_mode():
                            z = model.encoder(
                                torch.as_tensor(
                                    x[r], device=device, dtype=torch.float32
                                )
                            )
                            gen = torch.Generator(device=device).manual_seed(
                                2026110780 + k
                            )
                            v = (
                                prior.bounded_whole_draws(head, z, 1024, gen, 128)
                                * scale
                                + center
                            )
                        v[:, :, 4:] = np.exp(v[:, :, 4:])
                        if not np.isfinite(v).all():
                            raise FloatingPointError("Nonfinite cluster draws")
                        cd[k : k + len(r)] = v.transpose(1, 0, 2)
                        gc.collect()
                m["cluster_draw_stars"] = len(cluster_rows)
        m.update(
            status="full_label_union_inference_complete",
            predictions_sha256=prior.sha(a.output / "predictions.h5"),
            map_sha256=prior.sha(a.output / "map-coefficients.npz"),
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
