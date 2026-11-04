"""Evaluation-only CAL-role check of already selected flexible density heads."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml
from run_multisurvey_consensus_preflight import atomic, sha
from run_multisurvey_flexible import (
    attach_defensive_proposal,
    build_cache,
    evaluate,
    integration_gate,
    tensor_data,
)


def run(config, output, commit):
    import numpy as np
    import torch

    from masked_stellar_autoencoder.training.multisurvey_flow import (
        CopulaResidualHead,
        GaussianResidualHead,
        MixtureResidualHead,
        NSFResidualHead,
    )

    if (
        not Path("/arc/projects/k-pop").is_dir()
        or os.environ.get("PYTHONNOUSERSITE") != "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("CAL density evaluation requires headless CANFAR CUDA")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        run_id=config["run_id"],
        status="admitting",
        source_commit=commit,
        evaluation_role="cal",
        training_performed=False,
        config=config,
        scope="Fixed TUNE-selected heads; CAL opened for evaluation only, not fitting or hyperparameter selection; parent exposure and physical-scale limitations remain",
    )
    atomic(output / "manifest.json", manifest)
    try:
        if sha(config["selection_manifest"]) != config["selection_manifest_sha256"]:
            raise ValueError("Fixed TUNE selection manifest changed")
        selection = json.loads(Path(config["selection_manifest"]).read_text())
        if selection["status"] != "multisurvey_flexible_suite_complete":
            raise ValueError("TUNE selection is not complete")
        expected_primary = min(
            (k for k, v in selection["heads"].items() if v["status"] == "complete"),
            key=lambda k: selection["heads"][k]["evaluation"]["nll"],
        )
        if config["primary_head"] != expected_primary:
            raise ValueError("Primary candidate must be fixed from TUNE before CAL")
        models = {}
        normalization = None
        constructors = dict(
            gaussian=GaussianResidualHead,
            mixture=MixtureResidualHead,
            copula=CopulaResidualHead,
            nsf=NSFResidualHead,
        )
        for kind, item in config["heads"].items():
            selected = selection["heads"][kind]
            if (
                selected["status"] != "complete"
                or selected["checkpoint_path"] != item["path"]
                or selected["checkpoint_sha256"] != item["sha256"]
            ):
                raise ValueError("CAL heads differ from immutable TUNE selection")
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Selected density-head SHA differs")
            saved = torch.load(item["path"], map_location="cpu", weights_only=True)
            if (
                saved["kind"] != kind
                or saved["reference_sha256"]
                != config["suite"]["density_reference_sha256"]
                or saved["cache_sha256"] != config["fit_cache_sha256"]
                or saved["source_commit"] != config["training_source_commit"]
            ):
                raise ValueError(
                    "Selected head and original training/reference contracts differ"
                )
            current = dict(center=saved["context_center"], scale=saved["context_scale"])
            if normalization is not None:
                np.testing.assert_array_equal(
                    current["center"], normalization["center"]
                )
                np.testing.assert_array_equal(current["scale"], normalization["scale"])
            normalization = current
            torch.manual_seed(config["suite"]["seed"])
            model = (
                constructors[kind](saved["context_dim"], **saved["head_config"])
                .double()
                .cuda()
            )
            model.load_state_dict(saved["head"], strict=True)
            model.eval().requires_grad_(False)
            models[kind] = (model, saved)
        manifest["cache"] = build_cache(
            config["suite"],
            output,
            commit,
            role_plan=(("cal", config["cal_stars"]),),
            context_normalization=normalization,
        )
        data, _ = tensor_data(manifest["cache"]["cache_path"], roles=("cal",))
        if config["integration_proposal"] != "defensive_mixture":
            raise ValueError("Unexpected CAL integration proposal")
        attach_defensive_proposal(
            data["cal"], models["gaussian"][0], torch.device("cuda")
        )
        manifest["status"] = "evaluating_cal"
        atomic(output / "manifest.json", manifest)
        states = {}
        for kind, (model, saved) in models.items():
            gate = integration_gate(
                model,
                kind,
                data["cal"],
                torch.device("cuda"),
                particles=config["evaluation_particles"],
                seed=20261105 + 700000,
            )
            state = dict(
                status="cal_numerical_gate_failed",
                initial_gate=gate,
                selected_gate=gate,
                best_step=saved["step"],
                parameters=sum(p.numel() for p in model.parameters()),
                snapshots=[],
                checkpoint_sha256=config["heads"][kind]["sha256"],
                checkpoint_path=config["heads"][kind]["path"],
                head_config=saved["head_config"],
            )
            if gate["pass_gate"]:
                state["evaluation"] = evaluate(
                    model,
                    kind,
                    data["cal"],
                    torch.device("cuda"),
                    particles=config["evaluation_particles"],
                    seed=20261105 + 800000,
                    output=output / (kind + "-cal.npz"),
                    draws=2048,
                )
                if kind in {"copula", "nsf"}:
                    # Independent, doubled-particle scores on every CAL star;
                    # no checkpoint changes or selection follow this check.
                    refined = evaluate(
                        model,
                        kind,
                        data["cal"],
                        torch.device("cuda"),
                        particles=2 * config["evaluation_particles"],
                        seed=20261105 + 900000,
                        include_scores=True,
                    )
                    with np.load(
                        output / (kind + "-cal.npz"), allow_pickle=False
                    ) as predictions:
                        active = predictions["mask"].any(1)
                        first = predictions["per_star_nll"][active]
                        ids = predictions["source_id"][active]
                    second = np.asarray(refined["per_star_nll"])
                    state["score_refinement"] = dict(
                        particles=2 * config["evaluation_particles"],
                        mean_signed_difference=float((second - first).mean()),
                        absolute_difference_p95=float(
                            np.quantile(abs(second - first), 0.95)
                        ),
                        absolute_difference_max=float(abs(second - first).max()),
                        source_id=ids.tolist(),
                        first_nll=first.tolist(),
                        refined_nll=second.tolist(),
                        scope="Independent particle doubling on all scored CAL stars; no reselection",
                    )
                state["status"] = "complete"
            states[kind] = state
            atomic(output / "head-status.json", states)
            print(
                json.dumps(
                    dict(
                        head=kind,
                        status=state["status"],
                        evaluation=state.get("evaluation"),
                    )
                ),
                flush=True,
            )
        manifest["heads"] = states
        manifest["status"] = "multisurvey_flexible_suite_complete"
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
