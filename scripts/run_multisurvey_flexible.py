"""CANFAR-only matched joint NF/mixture/Gaussian density experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml
from run_multisurvey_consensus_preflight import atomic, native_preflight, sha


def build_cache(config, output, commit, *, role_plan=None, context_normalization=None):
    import h5py
    import numpy as np
    import torch

    from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
    from masked_stellar_autoencoder.training.multisurvey_flow import (
        reference_posterior,
        whiten_observations,
    )
    from masked_stellar_autoencoder.training.multisurvey_model import (
        pack_multisurvey_observations,
    )
    from masked_stellar_autoencoder.training.multisurvey_overlay import (
        sparse_dataset_from_manifest,
    )

    def collect(model, relative, fit_dataset, floors, **bindings):
        if sha(config["density_reference"]) != config["density_reference_sha256"]:
            raise ValueError("Fine-tuned reference checkpoint digest differs")
        saved = torch.load(
            config["density_reference"], map_location="cpu", weights_only=True
        )
        contract = saved["contract"]
        digest = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            digest != saved["contract_sha256"]
            or contract["parent_sha256"] != config["parent"]["checkpoint_sha256"]
            or contract["normalization_sha256"] != bindings["normalization_sha256"]
            or contract["source_binding"] != fit_dataset.source_binding
            or saved["steps"] != 1000
        ):
            raise ValueError(
                "Reference Gaussian source/normalization/ancestry contract differs"
            )
        model.load_state_dict(saved["model"], strict=True)
        relative.load_state_dict(saved["relative_head"], strict=True)
        model.eval().requires_grad_(False)
        relative.eval().requires_grad_(False)
        del saved
        device = next(model.parameters()).device
        rng = np.random.default_rng(config["seed"])
        path = output / "frozen-contexts.h5"
        metadata = dict(
            source_binding=fit_dataset.source_binding,
            registry_sha256=fit_dataset.registry_sha256,
            source_registry=list(fit_dataset.names),
            reference_sha256=config["density_reference_sha256"],
            msa_revision=commit,
            library_revision=config["parent"]["library_revision"],
            scope="Frozen feature-only encoder and fixed first-order survey observation maps; finite-QMC is an integration approximation, not a posterior prediction",
        )
        with h5py.File(path, "x") as cache:
            cache.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
            plan = (
                role_plan
                if role_plan is not None
                else (("fit", config["fit_stars"]), ("tune", config["tune_stars"]))
            )
            for role, maximum in plan:
                dataset = (
                    fit_dataset
                    if role == "fit"
                    else sparse_dataset_from_manifest(
                        config["parent"]["overlay_manifest"],
                        expected_manifest_sha256=config["parent"][
                            "overlay_manifest_sha256"
                        ],
                        role=role,
                    )
                )
                try:
                    associated = np.unique(
                        np.concatenate([v[0] for v in dataset._indices.values()])
                    )
                    rows = np.intersect1d(dataset.star_rows, associated)
                    positions = np.searchsorted(dataset.star_rows, rows)
                    positions = np.sort(
                        rng.choice(
                            positions, min(maximum, len(positions)), replace=False
                        )
                    )
                    if len(positions) < 128:
                        raise ValueError("Insufficient fixed-role source identities")
                    group = cache.create_group(role)
                    ids = dataset.ids[dataset.star_rows[positions]]
                    group.create_dataset("source_id", data=ids)
                    group.attrs["identity_sha256"] = hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()
                    ).hexdigest()
                    for start in range(0, len(positions), 128):
                        chosen = positions[start : start + 128]
                        # The cache writer owns construction, so no caller can
                        # substitute a forged batch. The verified reader checks
                        # attachment immutability on every access.
                        stars = [dataset[int(i)] for i in chosen]
                        batch = collate_multisurvey(
                            stars, repeat_policy="lowest_source_row"
                        )
                        np.testing.assert_array_equal(
                            batch["source_id"].numpy(), ids[start : start + len(chosen)]
                        )
                        if (
                            any(x != role for x in batch["role"])
                            or batch["source_binding"] != dataset.source_binding
                        ):
                            raise ValueError(
                                "Role or source binding changed during cache construction"
                            )
                        with torch.no_grad():
                            latent = model.mae.encoder(batch["input"].to(device))
                            mean, lower = model.head(latent.double())
                            packed = pack_multisurvey_observations(
                                mean,
                                batch,
                                relative,
                                floors,
                                expected_registry_sha256=dataset.registry_sha256,
                                expected_source_registry=dataset.names,
                                assume_diagonal_catalog_covariance=True,
                            )
                            observation = whiten_observations(lower, packed)
                            proposal_mean, proposal_lower, reference_lp = (
                                reference_posterior(observation)
                            )
                            physical_design = (
                                torch.where(
                                    packed.mask[..., None], packed.jacobian, 0.0
                                )
                                @ lower
                            )
                        b, k = packed.mask.shape
                        if k > 9:
                            raise ValueError(
                                "Unexpected extra source records after repeat resolution"
                            )
                        values = dict(
                            context=latent,
                            reference_mean=mean,
                            reference_lower=lower,
                            posterior_mean=proposal_mean,
                            posterior_lower=proposal_lower,
                            reference_lp=reference_lp,
                            count=observation["count"],
                            xp_available_count=batch["input_available"][:, 5:115].sum(
                                -1
                            ),
                        )
                        slot_values = dict(
                            design=observation["design"],
                            residual=observation["residual"],
                            physical_design=physical_design,
                            reference_mapping=packed.mapped_mean,
                            native_value=packed.values,
                            native_mask=packed.mask,
                            native_noise_variance=packed.measurement_covariance.diagonal(
                                dim1=-2, dim2=-1
                            ),
                        )
                        for name, tensor in slot_values.items():
                            array = tensor.detach().cpu().numpy()
                            shape = (b, 9, *array.shape[2:])
                            padded = np.zeros(shape, dtype=array.dtype)
                            padded[:, :k] = array
                            values[name] = padded
                        sources = (
                            batch["source_index"].repeat_interleave(3, dim=1).numpy()
                        )
                        source_index = np.full((b, 9), -2, dtype="i8")
                        source_index[:, :k] = sources
                        values["source_index"] = source_index
                        values["noise_logdet"] = observation["noise_logdet"]
                        for name, tensor in values.items():
                            array = (
                                tensor.detach().cpu().numpy()
                                if isinstance(tensor, torch.Tensor)
                                else tensor
                            )
                            if name not in group:
                                group.create_dataset(
                                    name,
                                    shape=(len(ids), *array.shape[1:]),
                                    dtype=array.dtype,
                                    chunks=(min(256, len(ids)), *array.shape[1:]),
                                    compression="lzf",
                                )
                            group[name][start : start + b] = array
                        if start % 4096 == 0:
                            print(
                                json.dumps(
                                    dict(
                                        phase="cache",
                                        role=role,
                                        rows=start + b,
                                        total=len(ids),
                                    )
                                ),
                                flush=True,
                            )
                    dataset._check_source_binding()
                finally:
                    dataset.close()
            if (
                "fit" in cache
                and "tune" in cache
                and np.intersect1d(
                    cache["fit/source_id"][:], cache["tune/source_id"][:]
                ).size
            ):
                raise ValueError("FIT/TUNE identity overlap")
            if context_normalization is None:
                context = cache["fit/context"][:].astype("f8")
                center, scale = context.mean(0), context.std(0)
                scale = np.where(scale > 1e-8, scale, 1.0)
            else:
                center = np.asarray(context_normalization["center"], dtype="f8")
                scale = np.asarray(context_normalization["scale"], dtype="f8")
                if (
                    not np.isfinite(center).all()
                    or not np.isfinite(scale).all()
                    or (scale <= 0).any()
                ):
                    raise ValueError("Invalid copied FIT context normalization")
            cache.create_dataset("context_center", data=center)
            cache.create_dataset("context_scale", data=scale)
        return dict(
            cache_path=str(path),
            cache_sha256=sha(path),
            metadata=metadata,
            updates=0,
            supervised_xp_masking=False,
            model_weights_saved=False,
        )

    receipt = native_preflight(
        config["parent"],
        config["parent"]["overlay_manifest"],
        commit,
        continuation=collect,
    )
    receipt["model_weights_saved"] = False
    return receipt


def tensor_data(path, *, roles=("fit", "tune")):
    import h5py
    import numpy as np
    import torch

    with h5py.File(path, "r") as cache:
        metadata = json.loads(cache.attrs["metadata"])
        metadata["context_center"] = cache["context_center"][:].tolist()
        metadata["context_scale"] = cache["context_scale"][:].tolist()
        data = {}
        for role in roles:
            data[role] = {
                name: torch.from_numpy(cache[role][name][:]) for name in cache[role]
            }
            raw = data[role]["context"].double()
            data[role]["context"] = (
                raw - torch.from_numpy(cache["context_center"][:])
            ) / torch.from_numpy(cache["context_scale"][:])
            ids = data[role]["source_id"].numpy()
            if (
                hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
                != cache[role].attrs["identity_sha256"]
            ):
                raise ValueError("Cached source-ID digest differs")
    return data, metadata


def subset(data, indices, device):
    return {k: v[indices].to(device) for k, v in data.items()}


def attach_defensive_proposal(data, gaussian, device):
    """Freeze feature-only Gaussian-control parameters in residual coordinates."""
    import torch

    means, lowers = [], []
    with torch.no_grad():
        for start in range(0, len(data["context"]), 128):
            mean, lower = gaussian(data["context"][start : start + 128].to(device))
            means.append(mean.cpu())
            lowers.append(lower.cpu())
    data["proposal_mean"] = torch.cat(means)
    data["proposal_lower"] = torch.cat(lowers)


def log_prob(head, kind, batch, *, particles, seed):
    from masked_stellar_autoencoder.training.multisurvey_flow import (
        defensive_importance_log_prob,
        gaussian_observation_log_prob,
        importance_log_prob,
        normal_nodes,
    )

    if kind == "gaussian":
        mean, lower = head(batch["context"])
        return gaussian_observation_log_prob(mean, lower, batch), None
    if kind == "mixture":
        return head.observation_log_prob(batch["context"], batch), None
    if "proposal_mean" in batch:
        nodes = normal_nodes(
            particles, len(batch["context"]), 4, seed=seed, like=batch["context"]
        )
        return defensive_importance_log_prob(
            head,
            batch["context"],
            batch["proposal_mean"],
            batch["proposal_lower"],
            batch,
            nodes,
        )
    nodes = normal_nodes(
        particles, len(batch["context"]), 3, seed=seed, like=batch["context"]
    )
    return importance_log_prob(
        head,
        batch["context"],
        batch["posterior_mean"],
        batch["posterior_lower"],
        batch["reference_lp"],
        batch["count"],
        nodes,
    )


def integration_gate(head, kind, data, device, *, particles, seed):
    import numpy as np
    import torch

    if kind in ("gaussian", "mixture"):
        return dict(pass_gate=True, exact=True)
    mask, sources = data["native_mask"].numpy(), data["source_index"].numpy()
    values = data["native_value"].numpy()
    chosen = set()
    signatures = np.concatenate([mask.astype("i8"), sources], axis=1)
    _, groups = np.unique(signatures, axis=0, return_inverse=True)
    for group in np.unique(groups):
        chosen.update(np.flatnonzero(groups == group)[:8].tolist())
    for target, direction in ((0, -1), (1, -1), (1, 1), (2, 1)):
        active = mask & ((np.arange(9) % 3) == target)[None]
        score = np.where(active, values, np.inf if direction == 1 else -np.inf)
        score = score.min(1) if direction == 1 else score.max(1)
        order = np.argsort(score) if direction == 1 else np.argsort(-score)
        chosen.update(order[:12].tolist())
    chosen.update(np.flatnonzero(data["xp_available_count"].numpy() == 0)[:12].tolist())
    rng = np.random.default_rng(20261105)
    priority = np.asarray(sorted(chosen), dtype="i8")[:128]
    remaining = np.setdiff1d(np.arange(len(mask)), priority)
    indices = np.sort(
        np.concatenate(
            [
                priority,
                rng.choice(
                    remaining, min(128 - len(priority), len(remaining)), replace=False
                ),
            ]
        )
    )
    (
        ess_values,
        replica,
        refinement,
        first_values,
        second_values,
        refined_values,
        ids,
    ) = [], [], [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), 8):
            batch = subset(data, indices[start : start + 8], device)
            first, ess = log_prob(
                head, kind, batch, particles=particles, seed=seed + start
            )
            second, _ = log_prob(
                head, kind, batch, particles=particles, seed=seed + start + 1
            )
            refined, _ = log_prob(
                head, kind, batch, particles=particles * 2, seed=seed + start + 2
            )
            active = batch["count"] > 0
            ess_values.extend(ess[active].cpu().tolist())
            replica.extend((first - second)[active].abs().cpu().tolist())
            refinement.extend((first - refined)[active].abs().cpu().tolist())
            first_values.extend(first[active].cpu().tolist())
            second_values.extend(second[active].cpu().tolist())
            refined_values.extend(refined[active].cpu().tolist())
            ids.extend(batch["source_id"][active].cpu().tolist())
    result = dict(
        exact=False,
        particles=particles,
        rows=len(ess_values),
        relative_ess_median=float(np.median(ess_values)),
        relative_ess_min=float(np.min(ess_values)),
        replica_gap_p95=float(np.quantile(replica, 0.95)),
        replica_gap_max=float(np.max(replica)),
        refinement_gap_p95=float(np.quantile(refinement, 0.95)),
        refinement_gap_max=float(np.max(refinement)),
    )
    result["panel_sha256"] = hashlib.sha256(
        np.asarray(ids, dtype="<i8").tobytes()
    ).hexdigest()
    result["raw"] = dict(
        source_id=ids,
        relative_ess=ess_values,
        log_prob=first_values,
        replica_log_prob=second_values,
        refined_log_prob=refined_values,
    )
    result["pass_gate"] = (
        result["relative_ess_median"] >= 0.1
        and result["relative_ess_min"] >= 0.01
        and result["replica_gap_p95"] <= 0.1
        and result["refinement_gap_p95"] <= 0.1
    )
    return result


def evaluate(
    head,
    kind,
    data,
    device,
    *,
    particles,
    seed,
    output=None,
    draws=1024,
    corner_draws=8192,
    include_scores=False,
):
    import numpy as np
    import torch

    from masked_stellar_autoencoder.training.multisurvey_flow import ensemble_crps

    head.eval()
    scores, saved = [], {}
    with torch.no_grad():
        for start in range(0, len(data["source_id"]), 16):
            batch = subset(data, slice(start, start + 16), device)
            lp, _ = log_prob(head, kind, batch, particles=particles, seed=seed + start)
            scores.extend((-lp[batch["count"] > 0]).cpu().tolist())
            if output is None:
                continue
            torch.manual_seed(seed + start)
            samples = head.sample(batch["context"], draws)
            if not torch.isfinite(samples).all():
                raise FloatingPointError("Nonfinite intrinsic prediction samples")
            physical = batch["reference_mean"][None] + torch.einsum(
                "bij,kbj->kbi", batch["reference_lower"], samples
            )
            mapped = batch["reference_mapping"][None] + torch.einsum(
                "bjd,kbd->kbj", batch["physical_design"], samples
            )
            noise = batch["native_noise_variance"].clamp_min(1e-30).sqrt()
            pit = torch.special.ndtr(
                (batch["native_value"][None] - mapped) / noise[None]
            ).mean(0)
            noisy = mapped + torch.randn_like(mapped) * noise[None]
            values = dict(
                source_id=batch["source_id"],
                reference=batch["native_value"],
                mask=batch["native_mask"],
                source_index=batch["source_index"],
                xp_available_count=batch["xp_available_count"],
                observed_pit=pit,
                observed_crps=ensemble_crps(noisy, batch["native_value"]),
                observed_median=noisy.median(0).values,
                observed_low=torch.quantile(noisy, 0.05, dim=0),
                observed_high=torch.quantile(noisy, 0.95, dim=0),
                canonical_median=physical.median(0).values,
                canonical_low=torch.quantile(physical, 0.05, dim=0),
                canonical_high=torch.quantile(physical, 0.95, dim=0),
                per_star_nll=-lp,
            )
            for name, value in values.items():
                saved.setdefault(name, []).append(value.cpu().numpy())
            if start == 0:
                joint = head.sample(batch["context"][:10], corner_draws)
                joint = batch["reference_mean"][:10][None] + torch.einsum(
                    "bij,kbj->kbi", batch["reference_lower"][:10], joint
                )
                corner_ids = batch["source_id"][:10].cpu().numpy()
                joint_draws = joint.permute(1, 0, 2).cpu().numpy()
    if output is not None:
        np.savez_compressed(
            output,
            **{k: np.concatenate(v) for k, v in saved.items()},
            corner_source_id=corner_ids,
            joint_draws=joint_draws,
        )
    result = dict(
        nll=float(np.mean(scores)),
        scored_stars=len(scores),
        prediction_sha256=sha(output) if output is not None else None,
        observation_cdf_draws=draws if output is not None else None,
    )
    if include_scores:
        result["per_star_nll"] = scores
    return result


def train(config, output, cache_receipt, commit, *, device=None):
    import numpy as np
    import torch

    from masked_stellar_autoencoder.training.multisurvey_flow import (
        CopulaResidualHead,
        GaussianResidualHead,
        MixtureResidualHead,
        NSFResidualHead,
    )

    if sha(cache_receipt["cache_path"]) != cache_receipt["cache_sha256"]:
        raise ValueError("Frozen-context cache digest differs")
    data, metadata = tensor_data(cache_receipt["cache_path"])
    if (
        metadata["reference_sha256"] != config["density_reference_sha256"]
        or metadata["library_revision"] != config["parent"]["library_revision"]
    ):
        raise ValueError("Frozen cache reference or library contract differs")
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Flexible stellar head fitting requires CANFAR CUDA")
        device = torch.device("cuda")
    else:
        device = torch.device(device)
        if device.type != "cuda" and metadata.get("synthetic") is not True:
            raise RuntimeError(
                "CPU density execution is restricted to synthetic checks"
            )
    constructors = dict(
        gaussian=GaussianResidualHead,
        mixture=MixtureResidualHead,
        copula=CopulaResidualHead,
        nsf=NSFResidualHead,
    )
    from torchregress.losses.nflows import recommended_tail_bound

    extents = (
        data["fit"]["posterior_mean"].abs()
        + 4 * data["fit"]["posterior_lower"].square().sum(-1).sqrt()
    )
    flow_bound = recommended_tail_bound(extents)
    heads, optimizers, states = {}, {}, {}
    for kind, constructor in constructors.items():
        torch.manual_seed(config["seed"])
        head_config = (
            dict(bound=flow_bound, bins=16) if kind in ("copula", "nsf") else {}
        )
        head = (
            constructor(data["fit"]["context"].shape[1], **head_config)
            .double()
            .to(device)
        )
        heads[kind] = head
        optimizers[kind] = torch.optim.Adam(head.parameters(), lr=config["head_lr"])
        initial_gate = integration_gate(
            head,
            kind,
            data["tune"],
            device,
            particles=config.get("gate_particles", config["particles"]),
            seed=config["seed"],
        )
        states[kind] = dict(
            status="running" if initial_gate["pass_gate"] else "initial_gate_failed",
            initial_gate=initial_gate,
            parameters=sum(p.numel() for p in head.parameters()),
            best_nll=None,
            best_step=None,
            snapshots=[],
            head_config=head_config,
        )
    atomic(output / "head-status.json", states)
    rng = np.random.default_rng(config["seed"])
    order = rng.permutation(len(data["fit"]["source_id"]))
    cursor = 0
    presented = []
    with (output / "training.jsonl").open("x") as log:
        for step in range(1, config["updates"] + 1):
            if cursor + config["batch_size"] > len(order):
                order = rng.permutation(len(order))
                cursor = 0
            indices = order[cursor : cursor + config["batch_size"]]
            cursor += config["batch_size"]
            presented.append(data["fit"]["source_id"][indices].numpy())
            batch = subset(data["fit"], indices, device)
            active_count = int((batch["count"] > 0).sum())
            if active_count == 0:
                raise ValueError("No observed labels in optimizer batch")
            for kind, head in heads.items():
                if states[kind]["status"] != "running":
                    continue
                try:
                    head.train()
                    optimizer = optimizers[kind]
                    optimizer.zero_grad(set_to_none=True)
                    total = 0.0
                    for start in range(0, len(indices), config["micro_batch"]):
                        micro = {
                            k: v[start : start + config["micro_batch"]]
                            for k, v in batch.items()
                        }
                        lp, _ = log_prob(
                            head,
                            kind,
                            micro,
                            particles=config["particles"],
                            seed=config["seed"] + step * 1024 + start,
                        )
                        loss = -lp.sum() / active_count
                        if not torch.isfinite(loss):
                            raise FloatingPointError(
                                "Nonfinite flexible observation objective"
                            )
                        loss.backward()
                        total += float(loss.detach())
                    grads = [p.grad for p in head.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError("Nonfinite or missing head gradients")
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            head.parameters(), 10.0, error_if_nonfinite=True
                        )
                    )
                    optimizer.step()
                    log.write(
                        json.dumps(
                            dict(
                                step=step,
                                head=kind,
                                nll=total,
                                gradient_norm=gradient_norm,
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    log.flush()
                except Exception as error:
                    states[kind].update(
                        status="training_failed", failed_step=step, error=str(error)
                    )
                    atomic(output / "head-status.json", states)
            if step in config["snapshots"]:
                for kind, head in heads.items():
                    if states[kind]["status"] != "running":
                        continue
                    head.eval()
                    payload = dict(
                        head=head.state_dict(),
                        optimizer=optimizers[kind].state_dict(),
                        kind=kind,
                        step=step,
                        cache_sha256=cache_receipt["cache_sha256"],
                        reference_sha256=config["density_reference_sha256"],
                        source_commit=commit,
                        context_dim=data["fit"]["context"].shape[1],
                        context_center=metadata["context_center"],
                        context_scale=metadata["context_scale"],
                        head_config=states[kind]["head_config"],
                    )
                    if config.get("save_all_snapshots", False):
                        torch.save(payload, output / f"{kind}-step-{step:04d}.pth")
                    gate = integration_gate(
                        head,
                        kind,
                        data["tune"],
                        device,
                        particles=config.get("gate_particles", config["particles"]),
                        seed=config["seed"] + 100000,
                    )
                    score = (
                        evaluate(
                            head,
                            kind,
                            data["tune"],
                            device,
                            particles=config["evaluation_particles"],
                            seed=config["seed"] + 200000,
                        )
                        if gate["pass_gate"]
                        else None
                    )
                    snapshot = dict(step=step, gate=gate, score=score)
                    states[kind]["snapshots"].append(snapshot)
                    if score is not None and (
                        states[kind]["best_nll"] is None
                        or score["nll"] < states[kind]["best_nll"]
                    ):
                        states[kind].update(best_nll=score["nll"], best_step=step)
                        temporary = output / (kind + "-selected.partial")
                        torch.save(payload, temporary)
                        temporary.replace(output / (kind + "-selected.pth"))
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
                atomic(output / "head-status.json", states)
    stream = np.concatenate(presented).astype("<i8")
    np.save(output / "fit-presented-source-ids.npy", stream)
    atomic(
        output / "sampling.json",
        dict(
            presentations=len(stream),
            unique_stars=len(np.unique(stream)),
            stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest(),
            seed=config["seed"],
            policy="uniform shuffled FIT cache, shared across all still-active heads",
        ),
    )
    for kind, head in heads.items():
        if states[kind]["best_step"] is None:
            if states[kind]["status"] == "running":
                states[kind]["status"] = "no_eligible_fitted_checkpoint"
            continue
        selected = torch.load(
            output / (kind + "-selected.pth"), map_location=device, weights_only=True
        )
        head.load_state_dict(selected["head"], strict=True)
        head.eval()
        gate = integration_gate(
            head,
            kind,
            data["tune"],
            device,
            particles=config["evaluation_particles"],
            seed=config["seed"] + 300000,
        )
        states[kind]["selected_gate"] = gate
        if not gate["pass_gate"]:
            states[kind]["status"] = "selected_gate_failed"
            continue
        states[kind]["evaluation"] = evaluate(
            head,
            kind,
            data["tune"],
            device,
            particles=config["evaluation_particles"],
            seed=config["seed"] + 400000,
            output=output / (kind + "-tune.npz"),
            draws=config["draws"],
            corner_draws=config.get("corner_draws", 8192),
        )
        states[kind]["checkpoint_sha256"] = sha(output / (kind + "-selected.pth"))
        states[kind]["status"] = "complete"
    atomic(output / "head-status.json", states)
    return states


def run(config, output, commit):
    if (
        not Path("/arc/projects/k-pop").is_dir()
        or os.environ.get("PYTHONNOUSERSITE") != "1"
    ):
        raise RuntimeError("Native flexible suite requires headless CANFAR hygiene")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        run_id=config["run_id"],
        status="admitting",
        config=config,
        source_commit=commit,
        scope="Matched 3D joint flexible densities, fixed natural-input encoder and affine survey maps; existing TUNE only, not physical-truth calibration",
    )
    atomic(output / "manifest.json", manifest)
    try:
        manifest["cache"] = build_cache(config, output, commit)
        manifest["status"] = "frozen_cache_complete"
        atomic(output / "manifest.json", manifest)
        manifest["heads"] = train(config, output, manifest["cache"], commit)
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
    config = yaml.safe_load(args.config.read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(config, args.output, commit)


if __name__ == "__main__":
    main()
