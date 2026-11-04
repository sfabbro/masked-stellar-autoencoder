"""Bounded three-survey joint Gaussian fits and saved distribution diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml
from run_multisurvey_consensus_preflight import atomic, native_preflight, sha


def labeled_positions(dataset):
    import numpy as np

    rows = np.unique(np.concatenate([x[0] for x in dataset._indices.values()]))
    rows = np.intersect1d(rows, dataset.star_rows)
    positions = np.searchsorted(dataset.star_rows, rows)
    if not len(positions):
        raise ValueError("No source-associated stars in the requested role")
    return positions


def evaluate(model, relative, dataset, floors, positions, output, *, draws=8192):
    import numpy as np
    import torch
    from torchregress.losses import MarginalizedMultiSourceGaussianNLLLoss

    from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
    from masked_stellar_autoencoder.training.multisurvey_distribution import (
        joint_gaussian_draws,
        observation_marginals,
    )
    from masked_stellar_autoencoder.training.multisurvey_model import (
        pack_multisurvey_observations,
    )

    model.eval()
    relative.eval()
    device = next(model.parameters()).device
    canonical, records, corners, nlls = [], [], [], []
    generator = torch.Generator(device=device).manual_seed(20261104)
    with torch.no_grad():
        for start in range(0, len(positions), 64):
            stars = [dataset[int(i)] for i in positions[start : start + 64]]
            batch = collate_multisurvey(stars, repeat_policy="lowest_source_row")
            mean, lower = model(batch["input"].to(device))
            packed = pack_multisurvey_observations(
                mean,
                batch,
                relative,
                floors,
                expected_registry_sha256=dataset.registry_sha256,
                expected_source_registry=dataset.names,
                assume_diagonal_catalog_covariance=True,
            )
            mapped, variance, pit = observation_marginals(mean, lower, packed)
            nll = MarginalizedMultiSourceGaussianNLLLoss(reduction="none")(
                mean, lower, packed
            )
            active = packed.mask.any(-1)
            nlls.extend(nll[active].cpu().tolist())
            ids = batch["source_id"].numpy()
            canonical.append((ids, mean.cpu().numpy(), lower.cpu().numpy()))
            m = batch["observed"].shape[1]
            for b in range(len(ids)):
                for r in range(m):
                    for d in range(3):
                        k = r * 3 + d
                        if not packed.mask[b, k]:
                            continue
                        records.append(
                            [
                                int(ids[b]),
                                int(batch["source_index"][b, r]),
                                d,
                                float(packed.values[b, k]),
                                float(mapped[b, k]),
                                float(variance[b, k]),
                                float(pit[b, k]),
                            ]
                        )
            if start == 0:
                samples = joint_gaussian_draws(
                    mean[:10], lower[:10], draws, generator=generator
                )
                corners = [ids[:10], samples.cpu().numpy()]
    # Keep Gaia identities as int64: never put them in a floating-point table.
    array = np.asarray([r[1:] for r in records], dtype="f8")
    np.savez_compressed(
        output,
        source_id=np.concatenate([c[0] for c in canonical]),
        mean=np.concatenate([c[1] for c in canonical]),
        scale_tril=np.concatenate([c[2] for c in canonical]),
        record_source_id=np.asarray([r[0] for r in records], dtype="i8"),
        record_source_index=array[:, 0].astype("i8"),
        record_target_index=array[:, 1].astype("i8"),
        reference=array[:, 2],
        observed_mean=array[:, 3],
        observed_variance=array[:, 4],
        observed_pit=array[:, 5],
        corner_source_id=corners[0],
        joint_draws=corners[1],
        per_star_nll=np.asarray(nlls),
    )
    metrics = []
    for s, source in enumerate(dataset.names):
        for d, target in enumerate(("teff", "logg", "fe_h")):
            selected = (array[:, 0] == s) & (array[:, 1] == d)
            values = array[selected]
            if not len(values):
                continue
            residual = values[:, 3] - values[:, 2]
            metrics.append(
                dict(
                    source=source,
                    target=target,
                    count=len(values),
                    mae=float(np.abs(residual).mean()),
                    bias=float(residual.mean()),
                    observed_coverage68=float(
                        ((values[:, 5] >= 0.16) & (values[:, 5] <= 0.84)).mean()
                    ),
                    observed_coverage90=float(
                        ((values[:, 5] >= 0.05) & (values[:, 5] <= 0.95)).mean()
                    ),
                    pit_histogram=np.histogram(
                        values[:, 5], bins=np.linspace(0, 1, 11)
                    )[0].tolist(),
                )
            )
    return dict(
        mean_joint_observation_nll=float(np.mean(nlls)),
        metrics=metrics,
        scored_stars=len(nlls),
        array_sha256=sha(output),
        scope="Previously assigned TUNE only; first-order survey observation model with declared floors, not physical truth calibration",
    )


def fit(config, output, source_commit):
    import numpy as np
    import torch

    from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
    from masked_stellar_autoencoder.training.multisurvey_overlay import (
        sparse_dataset_from_manifest,
    )
    from masked_stellar_autoencoder.training.multisurvey_training import (
        ConsensusTrainer,
        FitIdentityGate,
    )

    def continuation(model, relative, dataset, floors, **bindings):
        # No encoder dropout or batch-normalization updates in the frozen control.
        frozen = config["encoder"] == "frozen"
        for p in model.mae.parameters():
            p.requires_grad_(not frozen)
        # Decoder is unused in this supervised-only comparison.
        encoder_ids = {id(p) for p in model.mae.encoder.parameters()}
        for p in model.mae.parameters():
            if id(p) not in encoder_ids:
                p.requires_grad_(False)
        groups = [
            {
                "params": list(model.head.parameters()) + list(relative.parameters()),
                "lr": 0.001,
            }
        ]
        if not frozen:
            groups.append(
                {"params": list(model.mae.encoder.parameters()), "lr": 1.25e-6}
            )
        optimizer = torch.optim.Adam(groups)
        ids_sha = hashlib.sha256(
            np.asarray(dataset.ids[dataset.star_rows], dtype="<i8").tobytes()
        ).hexdigest()
        gate = FitIdentityGate(
            dataset,
            source_binding=dataset.source_binding,
            fit_ids_sha256=ids_sha,
            repeat_policy="lowest_source_row",
        )
        trainer = ConsensusTrainer(
            model,
            relative,
            optimizer,
            gate,
            floors,
            parent_sha256=bindings["parent_sha256"],
            normalization_sha256=bindings["normalization_sha256"],
            library_revision=bindings["library_revision"],
            msa_revision=source_commit,
            assume_diagonal_catalog_covariance=True,
            calibration_regularization_weight=0.0001,
        )
        tune = sparse_dataset_from_manifest(
            config["parent"]["overlay_manifest"],
            expected_manifest_sha256=config["parent"]["overlay_manifest_sha256"],
            role="tune",
        )
        try:
            if np.intersect1d(
                dataset.ids[dataset.star_rows], tune.ids[tune.star_rows]
            ).size:
                raise ValueError("FIT/TUNE source identity overlap")
            rng = np.random.default_rng(config["seed"])
            fit_positions = labeled_positions(dataset)
            tune_positions = labeled_positions(tune)
            tune_positions = np.sort(
                rng.choice(
                    tune_positions, min(1024, len(tune_positions)), replace=False
                )
            )
            before = evaluate(
                model,
                relative,
                tune,
                floors,
                tune_positions,
                output / "initial-tune.npz",
            )
            seen = set()
            with (output / "training.jsonl").open("x") as log:
                for step in range(config["updates"]):
                    positions = rng.choice(
                        fit_positions, config["batch_size"], replace=False
                    )
                    stars = [dataset[int(i)] for i in positions]
                    seen.update(s["source_id"] for s in stars)
                    batch = collate_multisurvey(
                        stars, repeat_policy="lowest_source_row"
                    )
                    metrics = trainer.step(batch)
                    log.write(
                        json.dumps(dict(step=step + 1, **metrics), allow_nan=False)
                        + "\n"
                    )
                    log.flush()
                    if (step + 1) % 100 == 0:
                        print(json.dumps(dict(step=step + 1, **metrics)), flush=True)
            checkpoint_sha = trainer.save(
                output / "final.pth", sampler_state=rng.bit_generator.state
            )
            # Reload the actual full saved state before producing released arrays.
            trainer.load(output / "final.pth", expected_sha256=checkpoint_sha)
            after = evaluate(
                model, relative, tune, floors, tune_positions, output / "final-tune.npz"
            )
            from plot_multisurvey_distribution import render

            render(output / "final-tune.npz", output / "plots")
            np.save(
                output / "fit-seen-source-ids.npy", np.asarray(sorted(seen), dtype="i8")
            )
            return dict(
                updates=trainer.steps,
                supervised_xp_masking=False,
                initial=before,
                final=after,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_reload="pass",
                unique_fit_seen=len(seen),
                fit_eligible_source_associated=len(fit_positions),
                encoder=config["encoder"],
            )
        finally:
            tune.close()

    return native_preflight(
        config["parent"],
        config["parent"]["overlay_manifest"],
        source_commit,
        continuation=continuation,
    )


def run(config, output, *, source_commit, execute=fit):
    if (
        set(config) != {"run_id", "encoder", "updates", "batch_size", "seed", "parent"}
        or config["encoder"] not in {"frozen", "finetune"}
        or config["updates"] != 1000
        or config["batch_size"] != 64
        or config["seed"] != config["parent"]["seed"]
    ):
        raise ValueError("Pilot must retain the bounded matched experiment contract")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        run_id=config["run_id"],
        source_commit=source_commit,
        config=config,
        status="admitting",
        scope="1000-update three-survey Gaussian distribution pilot; TUNE diagnostics only; no final-test promotion",
    )
    atomic(output / "manifest.json", manifest)
    try:
        producer = json.loads(
            Path(config["parent"]["overlay_producer_manifest"]).read_text()
        )
        result = producer.get("result", {})
        if (
            producer.get("status") != "multisurvey_desi_overlay_complete"
            or producer.get("source_commit")
            != "cf40e37c87d4e724b84fd1dffcac6d9e1363560d"
            or result.get("overlay_manifest_path")
            != config["parent"]["overlay_manifest"]
            or result.get("overlay_manifest_sha256")
            != config["parent"]["overlay_manifest_sha256"]
        ):
            raise ValueError("Completed overlay producer binding differs")
        manifest["status"] = "running"
        atomic(output / "manifest.json", manifest)
        manifest["result"] = execute(config, output, source_commit)
        if manifest["result"].get("updates") != config["updates"]:
            raise ValueError("Incomplete optimizer budget")
        manifest["status"] = "multisurvey_distribution_pilot_complete"
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configs = yaml.safe_load(args.config.read_text())["experiments"]
    config = next(c for c in configs if c["run_id"] == args.run_id)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(config, args.output, source_commit=commit)


if __name__ == "__main__":
    main()
