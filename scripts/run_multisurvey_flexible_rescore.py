"""Evaluate saved TUNE snapshots with a defensive integration proposal; no SGD."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml
from run_multisurvey_consensus_preflight import atomic, sha
from run_multisurvey_flexible import (
    attach_defensive_proposal,
    evaluate,
    integration_gate,
    tensor_data,
)


def load_head(path, kind, config, *, expected_sha=None):
    import torch

    from masked_stellar_autoencoder.training.multisurvey_flow import (
        CopulaResidualHead,
        GaussianResidualHead,
        MixtureResidualHead,
        NSFResidualHead,
    )

    before = sha(path)
    if expected_sha is not None and before != expected_sha:
        raise ValueError("Selected checkpoint SHA differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        sha(path) != before
        or saved["kind"] != kind
        or saved["source_commit"] != config["training_source_commit"]
        or saved["cache_sha256"] != config["fit_cache_sha256"]
        or saved["reference_sha256"] != config["suite"]["density_reference_sha256"]
    ):
        raise ValueError("Snapshot source/cache/reference contract differs")
    cls = dict(
        gaussian=GaussianResidualHead,
        mixture=MixtureResidualHead,
        copula=CopulaResidualHead,
        nsf=NSFResidualHead,
    )[kind]
    torch.manual_seed(config["suite"]["seed"])
    head = cls(saved["context_dim"], **saved["head_config"]).double().cuda()
    head.load_state_dict(saved["head"], strict=True)
    head.eval().requires_grad_(False)
    return head, saved, before


def run(config, output, commit):
    import numpy as np
    import torch

    if (
        not Path("/arc/projects/k-pop").is_dir()
        or os.environ.get("PYTHONNOUSERSITE") != "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Defensive rescoring requires headless CANFAR CUDA")
    installed = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if (
        installed.get("vcs_info", {}).get("commit_id")
        != config["suite"]["parent"]["library_revision"]
    ):
        raise ValueError("Installed density library differs")
    if sha(config["producer_manifest"]) != config["producer_manifest_sha256"]:
        raise ValueError("Completed training producer changed")
    original = json.loads(Path(config["producer_manifest"]).read_text())
    if (
        original["status"] != "multisurvey_flexible_suite_complete"
        or original["source_commit"] != config["training_source_commit"]
        or original["config"] != config["suite"]
        or original["cache"]["cache_sha256"] != config["fit_cache_sha256"]
        or sha(original["cache"]["cache_path"]) != config["fit_cache_sha256"]
    ):
        raise ValueError("Completed training/cache contract differs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        run_id=config["run_id"],
        source_commit=commit,
        config=config,
        cache=original["cache"],
        status="rescoring_saved_snapshots",
        training_performed=False,
        selection_on_tune=True,
        scope="Same trained densities; defensive 50:50 original/fitted-Gaussian reference mixture improves integration only; TUNE selection, no CAL access",
    )
    atomic(output / "manifest.json", manifest)
    try:
        data, metadata = tensor_data(original["cache"]["cache_path"], roles=("tune",))
        data = data["tune"]
        root = Path(config["producer_manifest"]).parent
        gaussian, saved, _ = load_head(
            root / "gaussian-selected.pth",
            "gaussian",
            config,
            expected_sha=original["heads"]["gaussian"]["checkpoint_sha256"],
        )
        np.testing.assert_array_equal(
            metadata["context_center"], saved["context_center"]
        )
        np.testing.assert_array_equal(metadata["context_scale"], saved["context_scale"])
        attach_defensive_proposal(data, gaussian, torch.device("cuda"))
        states = {}
        for kind in ("gaussian", "mixture"):
            state = json.loads(json.dumps(original["heads"][kind]))
            if state["status"] != "complete":
                raise ValueError("Analytic control did not complete")
            path = root / (kind + "-tune.npz")
            if sha(path) != state["evaluation"]["prediction_sha256"]:
                raise ValueError("Control prediction changed")
            shutil.copyfile(path, output / path.name)
            state["checkpoint_path"] = str(root / (kind + "-selected.pth"))
            states[kind] = state
        for kind in ("copula", "nsf"):
            state = dict(
                status="no_eligible_fitted_checkpoint",
                best_step=None,
                best_nll=None,
                snapshots=[],
            )
            for step in config["suite"]["snapshots"]:
                path = root / f"{kind}-step-{step:04d}.pth"
                if not path.exists():
                    continue
                head, saved, digest = load_head(path, kind, config)
                if saved["step"] != step:
                    raise ValueError("Snapshot update differs")
                np.testing.assert_array_equal(
                    metadata["context_center"], saved["context_center"]
                )
                np.testing.assert_array_equal(
                    metadata["context_scale"], saved["context_scale"]
                )
                state.update(
                    parameters=sum(p.numel() for p in head.parameters()),
                    head_config=saved["head_config"],
                )
                gate = integration_gate(
                    head,
                    kind,
                    data,
                    torch.device("cuda"),
                    particles=config["particles"],
                    seed=20261105 + 1100000,
                )
                score = (
                    evaluate(
                        head,
                        kind,
                        data,
                        torch.device("cuda"),
                        particles=config["particles"],
                        seed=20261105 + 1200000,
                    )
                    if gate["pass_gate"]
                    else None
                )
                state["snapshots"].append(
                    dict(
                        step=step,
                        gate=gate,
                        score=score,
                        checkpoint_sha256=digest,
                        checkpoint_path=str(path),
                    )
                )
                if score is not None and (
                    state["best_nll"] is None or score["nll"] < state["best_nll"]
                ):
                    state.update(
                        best_step=step,
                        best_nll=score["nll"],
                        checkpoint_sha256=digest,
                        checkpoint_path=str(output / (kind + "-selected.pth")),
                    )
                    shutil.copyfile(path, output / (kind + "-selected.pth"))
                print(
                    json.dumps(
                        dict(
                            head=kind,
                            step=step,
                            score=score,
                            gate={k: v for k, v in gate.items() if k != "raw"},
                        )
                    ),
                    flush=True,
                )
                del head
                states[kind] = state
                atomic(output / "head-status.json", states)
            if state["best_step"] is not None:
                head, _, _ = load_head(
                    output / (kind + "-selected.pth"),
                    kind,
                    config,
                    expected_sha=state["checkpoint_sha256"],
                )
                gate = integration_gate(
                    head,
                    kind,
                    data,
                    torch.device("cuda"),
                    particles=config["particles"],
                    seed=20261105 + 1300000,
                )
                state["selected_gate"] = gate
                state["status"] = "selected_gate_failed"
                if gate["pass_gate"]:
                    state["evaluation"] = evaluate(
                        head,
                        kind,
                        data,
                        torch.device("cuda"),
                        particles=config["particles"],
                        seed=20261105 + 1400000,
                        output=output / (kind + "-tune.npz"),
                        draws=2048,
                    )
                    state["status"] = "complete"
            states[kind] = state
            atomic(output / "head-status.json", states)
        manifest.update(status="multisurvey_flexible_suite_complete", heads=states)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic(output / "manifest.json", manifest)
        raise
    atomic(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, commit)


if __name__ == "__main__":
    main()
