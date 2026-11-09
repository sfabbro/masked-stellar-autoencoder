"""Frozen CANFAR CUDA replay of exact trained affine survey observation models."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from run_multisurvey_consensus_preflight import native_preflight, sha

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    affine_survey_components,
    marginal_scores,
    summarize_residuals,
    summarize_scores,
)
from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
from masked_stellar_autoencoder.training.multisurvey_flow import MixtureResidualHead
from masked_stellar_autoencoder.training.multisurvey_joint import (
    RoleMemoryDataset,
    mixture_terms,
    physical_components,
)
from masked_stellar_autoencoder.training.multisurvey_overlay import (
    sparse_dataset_from_manifest,
)

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = ROOT + "multisurvey_joint_fullfit_20261105_v1/"
SELECTED = RUN + "checkpoint-00005088.pth"
SELECTED_SHA = "9328ab102a54fae7fb541f08f4b478128f327834c4398bebccfff07112205fd7"
PREDICTIONS = {
    "devtest": "b73bd23c939ec8e36a51abcec29a9a306033e07025940c5cfd5702c644c2ccd9",
    "external": "aca9420f31ec08fc1a3d8a10cece8989e532f371d877f2c0150e16df4ff89335",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path("configs/multisurvey-joint-20261105.yaml").read_text())
    manifest = dict(
        status="admitting",
        training_performed=False,
        optimizer_created=False,
        updates=0,
        checkpoint_sha256=SELECTED_SHA,
        roles={},
    )

    def save():
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )

    save()
    if not torch.cuda.is_available() or sha(SELECTED) != SELECTED_SHA:
        raise RuntimeError("Frozen selected native checkpoint and CUDA required")

    def execute(model, relative, dataset, floors, **bindings):
        device = next(model.parameters()).device
        saved = torch.load(SELECTED, map_location="cpu", weights_only=False)
        if (
            saved["source_binding"] != dataset.source_binding
            or saved["normalization_sha256"] != bindings["normalization_sha256"]
        ):
            raise ValueError("Selected checkpoint source/normalization binding differs")
        model.load_state_dict(saved["model"], strict=True)
        relative.load_state_dict(saved["relative"], strict=True)
        shape = torch.load(
            config["mixture_checkpoint"], map_location="cpu", weights_only=True
        )
        mixture = (
            MixtureResidualHead(shape["context_dim"], **shape["head_config"])
            .double()
            .to(device)
        )
        mixture.load_state_dict(saved["mixture"], strict=True)
        center = saved["context_center"].to(device=device, dtype=torch.float64)
        scale = saved["context_scale"].to(device=device, dtype=torch.float64)
        for module in [model, relative, mixture]:
            module.eval().requires_grad_(False)
        del saved, shape
        floor_array = np.asarray(
            floors.detach().cpu() if torch.is_tensor(floors) else floors
        )
        manifest["floors"] = floor_array.tolist()
        for role, expected in PREDICTIONS.items():
            path = RUN + "predictions-" + role + ".h5"
            if sha(path) != expected:
                raise ValueError("Saved prediction hash differs")
            source = sparse_dataset_from_manifest(
                config["parent"]["overlay_manifest"],
                expected_manifest_sha256=config["parent"]["overlay_manifest_sha256"],
                role=role,
            )
            memory = RoleMemoryDataset(source)
            total = 0.0
            active = 0
            max_drift = np.zeros(3)
            target = output / ("mapped-" + role + ".h5")
            manifest["status"] = "replaying_" + role
            save()
            with h5py.File(path) as prior, h5py.File(target, "x") as f, torch.no_grad():
                f.attrs["checkpoint_sha256"] = SELECTED_SHA
                f.attrs["role"] = role
                f.attrs["observation_model"] = (
                    "exact affine training map at per-star reference Gaussian mean"
                )
                for start in range(0, len(memory), 512):
                    batch = collate_multisurvey(
                        [
                            memory[i]
                            for i in range(start, min(start + 512, len(memory)))
                        ],
                        repeat_policy="lowest_source_row",
                    )
                    terms = mixture_terms(
                        model, mixture, relative, batch, floors, center, scale
                    )
                    total += float(terms["nll_sum"])
                    active += terms["active"]
                    weights, mu, lower, mean, cov = physical_components(
                        mixture,
                        terms["context"],
                        terms["reference_mean"],
                        terms["reference_lower"],
                    )
                    mean = mean.cpu().numpy()
                    n = len(mean)
                    np.testing.assert_array_equal(
                        batch["source_id"].numpy(),
                        prior["source_id"][start : start + n],
                    )
                    previous = prior["mean"][start : start + n]
                    for target_index, tolerance in enumerate((0.02, 2e-5, 2e-5)):
                        np.testing.assert_allclose(
                            mean[:, target_index],
                            previous[:, target_index],
                            rtol=1e-6,
                            atol=tolerance,
                        )
                    max_drift = np.maximum(
                        max_drift, np.max(abs(mean - previous), axis=0)
                    )
                    offsets, derivatives = relative(terms["reference_mean"])
                    mapped, factors = affine_survey_components(
                        mu.cpu().numpy(),
                        lower.cpu().numpy(),
                        terms["reference_mean"].cpu().numpy(),
                        offsets.cpu().numpy(),
                        derivatives.cpu().numpy(),
                    )
                    arrays = dict(
                        source_id=batch["source_id"].numpy(),
                        weights=weights.cpu().numpy(),
                        reference_mean=terms["reference_mean"].cpu().numpy(),
                        component_mean=mapped,
                        component_variance=(factors**2).sum(-1),
                    )
                    for key, value in arrays.items():
                        if key not in f:
                            f.create_dataset(
                                key,
                                shape=(len(memory), *value.shape[1:]),
                                dtype=value.dtype,
                                chunks=True,
                            )
                        f[key][start : start + n] = value
                    if start % 32768 == 0:
                        print(
                            json.dumps(
                                dict(phase=role, rows=start + n, total=len(memory))
                            ),
                            flush=True,
                        )
                result = dict(
                    stars=len(memory),
                    observed_stars=active,
                    joint_observed_nll=total / active,
                    replayed_common_mean_max_abs_difference=max_drift.tolist(),
                    surveys={},
                )
                w = f["weights"][:]
                for s, survey in enumerate(["APOGEE", "DESI", "GALAH"]):
                    result["surveys"][survey] = {}
                    for t, name in enumerate(["Teff", "logg", "FeH"]):
                        keep = prior["native_observed"][:, s, t].astype(bool)
                        if not keep.any():
                            continue
                        mapped_mean = f["component_mean"][:, s, :, t][keep]
                        variance = f["component_variance"][:, s, :, t][keep]
                        observed = prior["native_value"][:, s, t][keep]
                        quoted = prior["native_error"][:, s, t][keep]
                        error = np.sqrt(
                            np.where(np.isfinite(quoted) & (quoted > 0), quoted**2, 0)
                            + floor_array[s, t] ** 2
                        )
                        score = marginal_scores(
                            w[keep], mapped_mean, variance, observed, error
                        )
                        summary = summarize_residuals(
                            (w[keep] * mapped_mean).sum(1), observed
                        )
                        summary["distribution"] = summarize_scores(score)
                        result["surveys"][survey][name] = summary
                manifest["roles"][role] = result
            source.close()
            manifest["roles"][role]["artifact_sha256"] = sha(target)
            save()
        return dict(updates=0, roles=list(PREDICTIONS))

    native_preflight(
        config["parent"],
        config["parent"]["overlay_manifest"],
        "407c6af02dea19f5bc848d8336c049f097229ab5",
        continuation=execute,
    )
    manifest["status"] = "frozen_survey_qa_complete"
    save()
    print(json.dumps(dict(status=manifest["status"], updates=0)), flush=True)


if __name__ == "__main__":
    main()
