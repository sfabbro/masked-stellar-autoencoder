"""CUDA whole-vector draws for a model-blind random development panel and a separate stress panel."""

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


def select_panel(ids, roles, eligible, stress_ids, size=1024, seed=2026110642):
    """Selection uses IDs/role/input eligibility only, never predictions or labels."""
    ids, roles, eligible = (
        np.asarray(ids),
        np.asarray(roles),
        np.asarray(eligible, bool),
    )
    candidates = np.flatnonzero((roles == "devtest") & eligible)
    if len(candidates) < size:
        raise ValueError("Insufficient eligible devtest stars")
    random_rows = np.sort(
        np.random.default_rng(seed).choice(candidates, size=size, replace=False)
    )
    stress_rows = candidates[np.isin(ids[candidates], stress_ids)]
    panel = np.union1d(random_rows, stress_rows)
    return panel, np.isin(panel, random_rows), np.isin(panel, stress_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

    if not torch.cuda.is_available():
        raise RuntimeError("CANFAR CUDA required")
    cfg = yaml.safe_load(a.config.read_text())
    direct = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
        raise ValueError("Library differs")
    source, _, _ = prior.admission(cfg, cfg["arms"][0])
    if prior.sha(source["prepared_path"]) != source["prepared_sha256"]:
        raise ValueError("Prepared source differs")
    if prior.sha(cfg["reference_checkpoint"]) != cfg["reference_checkpoint_sha256"]:
        raise ValueError("Reference checkpoint differs")
    reference = torch_load_trusted(
        cfg["reference_checkpoint"], map_location="cpu", weights_only=False
    )
    model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
    arms = {v["run_id"]: v for v in cfg["arms"]}
    pending = set(arms)
    results = {}
    device = torch.device("cuda")
    torch.set_num_threads(4)
    deadline = time.monotonic() + 24 * 3600
    with h5py.File(source["prepared_path"]) as data:
        ids = data["source_id"][:]
        roles = data["role"][:].astype("U")
        eligible = data["primary_input_eligible"][:].astype(bool)
        panel, random_panel, stress_panel = select_panel(
            ids, roles, eligible, cfg["corner_source_ids"]
        )
        if not len(panel):
            raise ValueError("No eligible development stars")
        panel_hash = (
            __import__("hashlib").sha256(ids[panel].astype("<i8").tobytes()).hexdigest()
        )
        prior.atomic(
            a.output / "panel-selection.json",
            dict(
                source_id=ids[panel],
                random_panel=random_panel,
                stress_panel=stress_panel,
                random_seed=2026110642,
                selected_ids_sha256=panel_hash,
                population="Uniform without replacement among primary-input-eligible prepared devtest stars; not all-Gaia representative",
                stress_selection="Predeclared corner IDs, reported separately; not included as extra random observations",
            ),
        )

        def save():
            prior.atomic(
                a.output / "manifest.json",
                dict(
                    status="waiting_or_scoring" if pending else "complete",
                    pending=sorted(pending),
                    results=results,
                    stars=len(panel),
                    selected_ids_sha256=__import__("hashlib")
                    .sha256(ids[panel].astype("<i8").tobytes())
                    .hexdigest(),
                    random_stars=int(random_panel.sum()),
                    stress_stars=int(stress_panel.sum()),
                    training_performed=False,
                    optimizer_created=False,
                    scope="Model-blind random development panel, plus separately flagged predeclared stress stars. Genuine coupled six-vector draws; not all-Gaia frequency estimation or certified physical dependence.",
                ),
            )

        save()
        while pending and time.monotonic() < deadline:
            for rid in sorted(pending):
                path = ROOT / rid / "manifest.json"
                if not path.exists():
                    continue
                m = json.loads(path.read_text())
                if m.get("status") == "failed":
                    results[rid] = dict(status="upstream_failed")
                    pending.remove(rid)
                    save()
                    continue
                if m.get("status") != "remediation_pilot_complete":
                    continue
                model = head = None
                try:
                    checkpoint = ROOT / rid / "selected-checkpoint.pth"
                    if prior.sha(checkpoint) != m["selected_checkpoint_sha256"]:
                        raise ValueError("Selected checkpoint SHA differs")
                    state = torch_load_trusted(
                        checkpoint, map_location="cpu", weights_only=False
                    )
                    if (
                        state["run_id"] != rid
                        or state["prepared_sha256"] != source["prepared_sha256"]
                        or state["parent_sha256"] != cfg["checkpoint_sha256"]
                        or state["arm"] != arms[rid]
                    ):
                        raise ValueError("State binding differs")
                    model = native.probe.load_model(model_cfg, reference, device)
                    model.load_state_dict(state["model_state_dict"], strict=True)
                    model.eval().requires_grad_(False)
                    head = prior.build_head(
                        arms[rid]["kind"], model_cfg["model"]["layer_dims"][-1], cfg
                    ).to(device)
                    head.load_state_dict(state["head_state_dict"], strict=True)
                    head.eval().requires_grad_(False)
                    center, scale = (
                        np.asarray(state["center"]),
                        np.asarray(state["scale"]),
                    )
                    del state
                    pieces = []
                    for start in range(0, len(panel), 4):
                        rows = panel[start : start + 4]
                        with torch.no_grad():
                            z = model.encoder(
                                torch.as_tensor(
                                    prior.profile_batch(
                                        data,
                                        rows,
                                        arms[rid].get("input_profile", "phot_xp"),
                                        source,
                                    ),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                            gen = torch.Generator(device=device).manual_seed(
                                2026110619 + start
                            )
                            draws = prior.bounded_whole_draws(head, z, 1024, gen, 128)
                        physical = draws * scale + center
                        physical[:, :, 4:] = np.exp(physical[:, :, 4:])
                        pieces.append(physical)
                    draws = np.concatenate(pieces, axis=1)
                    if not np.isfinite(draws).all():
                        raise ValueError(
                            "Nonfinite physical six-target draws; no clipping"
                        )
                    result = dict(
                        status="complete",
                        stars=len(panel),
                        random_stars=int(random_panel.sum()),
                        stress_stars=int(stress_panel.sum()),
                        selected_update=m["selected_update"],
                        selected_checkpoint_sha256=m["selected_checkpoint_sha256"],
                        panel_ids_sha256=panel_hash,
                    )
                    artifact = a.output / (rid + "-representative-draws.npz")
                    np.savez_compressed(
                        artifact,
                        source_id=ids[panel],
                        native_draws=draws,
                        random_panel=random_panel,
                        stress_panel=stress_panel,
                        reference=data["value"][:][panel],
                        observed=data["observed"][:][panel].astype(bool)
                        & data["quality"][:][panel].astype(bool),
                    )
                    result["artifact_sha256"] = prior.sha(artifact)
                    results[rid] = result
                    pending.remove(rid)
                    save()
                except Exception as exc:
                    results[rid] = dict(status="evaluation_failed", error=repr(exc))
                    pending.remove(rid)
                    save()
                finally:
                    del model, head
                    gc.collect()
                    torch.cuda.empty_cache()
            if pending:
                time.sleep(45)
        if pending:
            raise TimeoutError("Upstream wait expired; pending arms preserved")


if __name__ == "__main__":
    main()
