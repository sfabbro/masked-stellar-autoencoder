"""CUDA seismic QA on all available decision-source devtest stars, without panel caps."""

import argparse
import gc
import importlib.metadata
import json
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
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.decision_pdf_metrics import sample_scores

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
    refs = ROOT / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
    refmanifest = json.loads((refs.parent / "manifest.json").read_text())
    refsha = refmanifest["sources"]["apokasc3"]["reference_sha256"]
    if prior.sha(refs) != refsha:
        raise ValueError("Seismic reference differs")
    ref = pd.read_parquet(refs)
    ridx = {int(s): i for i, s in enumerate(ref.source_id)}
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
        panel = np.array(
            [
                i
                for i, sid in enumerate(ids)
                if roles[i] == "devtest" and eligible[i] and int(sid) in ridx
            ],
            dtype="i8",
        )
        if not len(panel):
            raise ValueError("No seismic devtest input matches")
        rr = np.array([ridx[int(ids[i])] for i in panel])
        truth = ref.loggSeis.to_numpy()[rr, None]
        errors = ref.e_loggSeis.to_numpy()[rr, None]

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
                    reference_sha256=refsha,
                    training_performed=False,
                    optimizer_created=False,
                    scope="All eligible exact-ID APOKASC Gold/Silver devtest stars in current prepared source, no hashed panel cap. Fine-tuning holdout; historical teacher/parent and seismic calibration dependencies remain. PDF QA uses whole six-vector draws, scoring the logg marginal only.",
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
                        pieces.append(draws[:, :, 1:2] * scale[1] + center[1])
                    draws = np.concatenate(pieces, axis=1)
                    if not np.isfinite(draws).all():
                        raise ValueError("Nonfinite physical gravity draws")
                    raw = sample_scores(draws, truth, np.ones_like(truth, bool))
                    noisy = sample_scores(
                        draws
                        + np.random.default_rng(2026110620).standard_normal(draws.shape)
                        * errors[None],
                        truth,
                        np.ones_like(truth, bool),
                    )
                    prediction = np.median(draws[:, :, 0], axis=0)
                    delta = prediction - truth[:, 0]
                    result = dict(
                        status="complete",
                        stars=len(panel),
                        selected_update=m["selected_update"],
                        selected_checkpoint_sha256=m["selected_checkpoint_sha256"],
                        median_bias=float(np.median(delta)),
                        mae=float(abs(delta).mean()),
                        intrinsic_crps=float(raw["crps"].mean()),
                        convolved_crps=float(noisy["crps"].mean()),
                        intrinsic_coverage90=float(
                            ((raw["pit"] >= 0.05) & (raw["pit"] <= 0.95)).mean()
                        ),
                        convolved_coverage90=float(
                            ((noisy["pit"] >= 0.05) & (noisy["pit"] <= 0.95)).mean()
                        ),
                    )
                    artifact = a.output / (rid + "-seismic-pdf.npz")
                    np.savez_compressed(
                        artifact,
                        source_id=ids[panel],
                        logg_seismic=truth[:, 0],
                        reference_error=errors[:, 0],
                        logg_draws=draws[:, :, 0],
                        intrinsic_pit=raw["pit"],
                        convolved_pit=noisy["pit"],
                        intrinsic_crps=raw["crps"],
                        convolved_crps=noisy["crps"],
                    )
                    result["artifact_sha256"] = prior.sha(artifact)
                    results[rid] = result
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
