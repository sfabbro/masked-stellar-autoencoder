"""Native CUDA common-space QA of the first immutable large-run snapshot."""

import argparse
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-waiting-output", action="store_true")
    p.add_argument(
        "--checkpoint-name",
        choices=["latest-checkpoint.pth", "selected-checkpoint.pth"],
        default="latest-checkpoint.pth",
    )
    a = p.parse_args()
    if a.output.exists() and a.reuse_waiting_output:
        previous = json.loads((a.output / "manifest.json").read_text())
        if (
            previous.get("status")
            not in ["waiting_for_first_checkpoint", "upstream_failed"]
            or list(a.output.glob("*.pth*"))
            or list(a.output.glob("*.npz"))
        ):
            raise ValueError("Cannot replace QA after snapshot/evaluation admission")
        attempt = 1
        while (a.output / f"pre-snapshot-attempt-{attempt}.json").exists():
            attempt += 1
        with (a.output / f"pre-snapshot-attempt-{attempt}.json").open("xb") as stream:
            stream.write((a.output / "manifest.json").read_bytes())
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch
    from masked_stellar_autoencoder.training.remediation_energy6 import (
        EnergyJointSixHead,
    )

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.cached_inputs import cache_natural_inputs
    from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap

    cfg = yaml.safe_load(a.config.read_text())
    run = ROOT / cfg["run_id"]
    deadline = time.monotonic() + 48 * 3600
    m = dict(
        status="waiting_for_first_checkpoint",
        model_training_performed=False,
        models={},
        scope="Immutable large-run snapshot versus frozen earlier six-target NSF. Common reference predictions; source-native labels retain their scales. Preliminary checkpoint may precede a full FIT epoch. Benchmark/cluster parent exposure and teacher dependencies remain; raw density QA is not a release certificate.",
    )

    def save():
        prior.atomic(a.output / "manifest.json", m)

    save()
    while time.monotonic() < deadline:
        try:
            progress = json.loads((run / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            progress = {}
        if progress.get("status") == "failed":
            m.update(status="upstream_failed", error=progress.get("error"))
            save()
            return
        path = run / a.checkpoint_name
        if path.exists():
            # Hold one inode while the producer atomically replaces later snapshots.
            with (
                path.open("rb") as src,
                (a.output / "large-snapshot.pth").open("xb") as dst,
            ):
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            break
        time.sleep(30)
    else:
        raise TimeoutError("Large run produced no checkpoint in 48h")
    if not torch.cuda.is_available():
        raise RuntimeError("CANFAR CUDA required")
    direct = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
        raise ValueError("Library differs")
    source = json.loads(
        (ROOT / cfg["expanded_source_run"] / "manifest.json").read_text()
    )
    if source.get("status") != "expanded_six_source_complete":
        raise ValueError("Expanded source not complete")
    if prior.sha(source["prepared_path"]) != source["prepared_sha256"]:
        raise ValueError("Source SHA differs")
    reference = torch_load_trusted(
        cfg["reference_checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        prior.sha(cfg["reference_checkpoint"]) != cfg["reference_checkpoint_sha256"]
        or prior.sha(cfg["encoder_parent"]) != cfg["encoder_parent_sha256"]
    ):
        raise ValueError("Reference/baseline SHA differs")
    model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
    device = torch.device("cuda")
    torch.set_num_threads(4)
    refs = ROOT / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
    refmanifest = json.loads((refs.parent / "manifest.json").read_text())
    if prior.sha(refs) != refmanifest["sources"]["apokasc3"]["reference_sha256"]:
        raise ValueError("Seismic source differs")
    seismo = pd.read_parquet(refs)
    tess_dir = ROOT / "tess_k2_local_gaia_candidates_20261107_v1"
    tess_manifest = json.loads((tess_dir / "manifest.json").read_text())
    tess_path = tess_dir / "seismic-gaia-qa-candidates.parquet"
    if (
        tess_manifest.get("status") != "conservative_seismic_gaia_candidates_complete"
        or prior.sha(tess_path) != tess_manifest["candidate_sha256"]
    ):
        raise ValueError("TESS QA candidate identity binding changed")
    tess = pd.read_parquet(tess_path)
    tess = tess.loc[
        (tess.catalogue == "tess_hon_2021")
        & np.isfinite(tess.logg_nu_scaling)
        & np.isfinite(tess.e_logg_stat)
        & (tess.e_logg_stat > 0)
    ].copy()
    m["tess_candidate_sha256"] = tess_manifest["candidate_sha256"]
    with h5py.File(source["prepared_path"]) as data:
        ids = data["source_id"][:]
        roles = data["role"][:].astype("S8")
        eligible = data["primary_input_eligible"][:].astype(bool)
        panels = {
            role: np.sort(
                np.random.default_rng(2026110742).choice(
                    np.flatnonzero((roles == role.encode()) & eligible),
                    2048,
                    replace=False,
                )
            )
            for role in ["fit", "devtest"]
        }
        panels["external"] = np.flatnonzero((roles == b"external") & eligible)
        panels["seismic_devtest"] = np.flatnonzero(
            (roles == b"devtest") & eligible & np.isin(ids, seismo.source_id.to_numpy())
        )
        panels["tess_devtest"] = np.flatnonzero(
            (roles == b"devtest")
            & eligible
            & np.isin(ids, tess.source_id.to_numpy(dtype="i8"))
        )
        panels["tess_external"] = np.flatnonzero(
            (roles == b"external")
            & eligible
            & np.isin(ids, tess.source_id.to_numpy(dtype="i8"))
        )
        panel = np.unique(np.concatenate(list(panels.values())))
        corner = prior.heldout_corner_rows(
            ids, roles.astype("U"), cfg["corner_source_ids"]
        )
        panel = np.union1d(panel, corner)
        np.savez_compressed(
            a.output / "panel.npz",
            source_id=ids[panel],
            row=panel,
            role=roles[panel],
            input_eligible=eligible[panel],
            reserved_benchmark=data["reserved_benchmark"][:][panel],
            reserved_globular_member=data["reserved_globular_member"][:][panel],
            seismic_devtest=np.isin(panel, panels["seismic_devtest"]),
            corner=np.isin(panel, corner),
            tess_devtest=np.isin(panel, panels["tess_devtest"]),
            tess_external=np.isin(panel, panels["tess_external"]),
            parallax_mas=data["measured_parallax/parallax_mas"][:][panel],
            parallax_error_mas=data["measured_parallax/parallax_error_mas"][:][panel],
            parallax_observed=data["measured_parallax/observed"][:][panel],
        )
        m.update(
            status="scoring",
            stars=len(panel),
            source_sha256=source["prepared_sha256"],
            source_manifest_path=str(
                ROOT / cfg["expanded_source_run"] / "manifest.json"
            ),
            large_snapshot_sha256=prior.sha(a.output / "large-snapshot.pth"),
        )
        save()
        cached_inputs = cache_natural_inputs(
            data, source["feature_profiles"]["phot_xp_measured_parallax"]
        )
        check_rows = panel[
            np.linspace(0, len(panel) - 1, min(128, len(panel)), dtype=int)
        ]
        np.testing.assert_array_equal(
            cached_inputs[check_rows],
            prior.profile_batch(data, check_rows, "phot_xp_measured_parallax", source),
        )
        m["cached_inputs_verified_rows"] = len(check_rows)
        m["tess_devtest_stars"] = len(panels["tess_devtest"])
        save()
        for name, path in [
            ("large_first_checkpoint", a.output / "large-snapshot.pth"),
            ("earlier_pooled_nsf", Path(cfg["encoder_parent"])),
        ]:
            state = torch_load_trusted(path, map_location="cpu", weights_only=False)
            if (
                name == "large_first_checkpoint"
                and state.get("source_sha256", state.get("prepared_sha256"))
                != source["prepared_sha256"]
            ):
                raise ValueError("Snapshot source binding differs")
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
            pieces = []
            corners = []
            gravity = []
            for start in range(0, len(panel), 4):
                rr = panel[start : start + 4]
                with torch.no_grad():
                    z = model.encoder(
                        torch.as_tensor(
                            cached_inputs[rr],
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    gen = torch.Generator(device=device).manual_seed(2026110750 + start)
                    draw = prior.bounded_whole_draws(head, z, 1024, gen, 128)
                physical = draw * scale + center
                physical[:, :, 4:] = np.exp(physical[:, :, 4:])
                if not np.isfinite(physical).all():
                    raise FloatingPointError("Nonfinite native draws, no clipping")
                pieces.append(
                    dict(
                        median=np.median(physical, axis=0),
                        quantiles=np.quantile(
                            physical,
                            [0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975],
                            axis=0,
                        ),
                        covariance=np.asarray(
                            [np.cov(draw[:, j], rowvar=False) for j in range(len(rr))]
                        ),
                        age_tail=(physical[:, :, 4] > 20).mean(0),
                    )
                )
                gravity.append(physical[:, :, 1].astype("f4"))
                stress = np.isin(rr, corner)
                if stress.any():
                    corners.append(
                        (ids[rr[stress]], draw[:, stress], physical[:, stress])
                    )
                if start % 512 == 0:
                    prior.atomic(
                        a.output / (name + "-progress.json"),
                        dict(rows=start + len(rr), total=len(panel)),
                    )
            arrays = {
                k: np.concatenate(
                    [v[k] for v in pieces], axis=1 if k == "quantiles" else 0
                )
                for k in pieces[0]
            }
            np.savez_compressed(
                a.output / (name + "-common.npz"),
                source_id=ids[panel],
                center=center,
                scale=scale,
                logg_draws=np.concatenate(gravity, axis=1),
                **arrays,
            )
            if corners:
                np.savez_compressed(
                    a.output / (name + "-single-star.npz"),
                    source_id=np.concatenate([c[0] for c in corners]),
                    standardized_draws=np.concatenate([c[1] for c in corners], axis=1),
                    native_draws=np.concatenate([c[2] for c in corners], axis=1),
                )
            if name == "large_first_checkpoint":
                map_state = state["map_state_dict"]
                maps = SixSurveyMap(
                    map_state["support"].cpu(),
                    map_state["reference"].cpu(),
                    zero_sum=torch.nonzero(map_state["zero_sum"].cpu(), as_tuple=False)
                    .flatten()
                    .tolist(),
                    kind=state["decision"]["mapping_kind"],
                ).to(device)
                maps.load_state_dict(map_state, strict=True)
                bias, slopes = maps.coefficients()
                np.savez_compressed(
                    a.output / "snapshot-map-coefficients.npz",
                    bias=bias.detach().cpu().numpy(),
                    slopes=slopes.detach().cpu().numpy(),
                    kind=np.array(state["decision"]["mapping_kind"]),
                    source_registry=np.asarray(source["source_registry"]),
                    center=center,
                    scale=scale,
                )
                del maps
                m["large_decision"] = state["decision"]
            m["models"][name] = dict(
                status="complete",
                checkpoint_sha256=prior.sha(path),
                training_update=state.get("update"),
                epoch=state.get("epoch"),
                artifacts={
                    p.name: dict(sha256=prior.sha(p), bytes=p.stat().st_size)
                    for p in a.output.glob(name + "-*.npz")
                },
            )
            save()
            del state, model, head
            torch.cuda.empty_cache()
        m.update(
            status="first_checkpoint_common_qa_complete",
            panel_sha256=prior.sha(a.output / "panel.npz"),
            seismic_reference_sha256=refmanifest["sources"]["apokasc3"][
                "reference_sha256"
            ],
        )
        save()


if __name__ == "__main__":
    main()
