"""CANFAR full-FIT mixture consensus, retained reconstruction, final catalogue export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import yaml
from run_multisurvey_consensus_preflight import atomic, native_preflight, sha


def run(config, output, commit):
    import h5py
    import numpy as np
    import torch

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
    from masked_stellar_autoencoder.training.multisurvey_training import FitIdentityGate

    if (
        not Path("/arc/projects/k-pop").is_dir()
        or not torch.cuda.is_available()
        or os.environ.get("PYTHONNOUSERSITE") != "1"
    ):
        raise RuntimeError("Native joint training requires headless CANFAR CUDA")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        run_id=config["run_id"],
        status="admitting",
        source_commit=commit,
        config=config,
        scope="Full primary-eligible FIT passes, shared non-Gaussian stellar mixture and zero-sum relative survey maps; natural supervised and separately masked reconstruction forwards; devtest opened only after TUNE selection",
    )
    atomic(output / "manifest.json", manifest)

    def execute(model, relative, dataset, floors, **bindings):
        device = next(model.parameters()).device
        if (
            sha(config["reference_checkpoint"]) != config["reference_sha256"]
            or sha(config["mixture_checkpoint"]) != config["mixture_sha256"]
        ):
            raise ValueError("Warm-start checkpoint SHA differs")
        saved = torch.load(
            config["reference_checkpoint"], map_location="cpu", weights_only=True
        )
        contract = saved["contract"]
        if (
            hashlib.sha256(
                json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            != saved["contract_sha256"]
            or contract["source_binding"] != dataset.source_binding
            or contract["normalization_sha256"] != bindings["normalization_sha256"]
            or contract["parent_sha256"] != config["parent"]["checkpoint_sha256"]
        ):
            raise ValueError("Reference source/scaler/parent contract differs")
        model.load_state_dict(saved["model"], strict=True)
        relative.load_state_dict(saved["relative_head"], strict=True)
        del saved
        saved = torch.load(
            config["mixture_checkpoint"], map_location="cpu", weights_only=True
        )
        if (
            saved["kind"] != "mixture"
            or saved["reference_sha256"] != config["reference_sha256"]
        ):
            raise ValueError("Mixture reference differs")
        mixture = (
            MixtureResidualHead(saved["context_dim"], **saved["head_config"])
            .double()
            .to(device)
        )
        mixture.load_state_dict(saved["head"], strict=True)
        center = torch.tensor(
            saved["context_center"], dtype=torch.float64, device=device
        )
        scale = torch.tensor(saved["context_scale"], dtype=torch.float64, device=device)
        for p in model.mae.parameters():
            p.requires_grad_(True)
        model.head.requires_grad_(False)
        relative.requires_grad_(True)
        mixture.requires_grad_(True)
        encoder = list(model.mae.encoder.parameters())
        enc_ids = {id(p) for p in encoder}
        decoder = [p for p in model.mae.parameters() if id(p) not in enc_ids]
        optimizer = torch.optim.Adam(
            [
                dict(params=encoder, lr=config["encoder_lr"]),
                dict(params=decoder, lr=config["decoder_lr"]),
                dict(params=mixture.parameters(), lr=config["head_lr"]),
                dict(params=relative.parameters(), lr=config["survey_lr"]),
            ]
        )

        def reader(role):
            return sparse_dataset_from_manifest(
                config["parent"]["overlay_manifest"],
                expected_manifest_sha256=config["parent"]["overlay_manifest_sha256"],
                role=role,
            )

        print(json.dumps(dict(phase="caching_fit", stars=len(dataset))), flush=True)
        fit = RoleMemoryDataset(dataset)
        ids = np.asarray(fit.ids[fit.star_rows], dtype="<i8")
        ids_sha = hashlib.sha256(ids.tobytes()).hexdigest()
        gate = FitIdentityGate(
            fit,
            source_binding=fit.source_binding,
            fit_ids_sha256=ids_sha,
            repeat_policy="lowest_source_row",
        )
        tune_source = reader("tune")
        tune = RoleMemoryDataset(tune_source)
        tune_ids = tune.ids[tune.star_rows]
        if np.intersect1d(ids, tune_ids).size:
            raise ValueError("FIT/TUNE overlap")
        rng = np.random.default_rng(config["seed"])
        tune_positions = np.sort(
            rng.choice(len(tune), min(config["tune_stars"], len(tune)), replace=False)
        )
        np.save(output / "tune-source-ids.npy", tune_ids[tune_positions])
        generator = torch.Generator().manual_seed(config["seed"])
        width = bindings["reconstruction_width"]
        manifest.update(
            status="training",
            fit_stars=len(fit),
            fit_ids_sha256=ids_sha,
            tune_stars=len(tune_positions),
            reference_contract_sha256=hashlib.sha256(
                json.dumps(contract, sort_keys=True).encode()
            ).hexdigest(),
            epochs=config["epochs"],
        )
        atomic(output / "manifest.json", manifest)
        best = None
        best_score = float("inf")
        step = 0
        started = time.monotonic()

        def snapshot(epoch, cursor, score=None):
            path = output / f"checkpoint-{step:08d}.pth"
            tmp = path.with_suffix(".partial")
            torch.save(
                dict(
                    model=model.state_dict(),
                    relative=relative.state_dict(),
                    mixture=mixture.state_dict(),
                    optimizer=optimizer.state_dict(),
                    context_center=center.cpu(),
                    context_scale=scale.cpu(),
                    source_commit=commit,
                    library_revision=config["parent"]["library_revision"],
                    config=config,
                    fit_ids_sha256=ids_sha,
                    source_binding=fit.source_binding,
                    normalization_sha256=bindings["normalization_sha256"],
                    epoch=epoch,
                    cursor=cursor,
                    steps=step,
                    mask_rng=generator.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    tune_nll=score,
                ),
                tmp,
            )
            tmp.replace(path)
            receipt = dict(
                path=str(path),
                sha256=sha(path),
                step=step,
                epoch=epoch,
                cursor=cursor,
                tune_nll=score,
            )
            atomic(output / "last-checkpoint.json", receipt)
            return receipt

        def score_tune():
            model.eval()
            mixture.eval()
            relative.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for start in range(0, len(tune_positions), config["micro_batch"]):
                    batch = collate_multisurvey(
                        [
                            tune[int(i)]
                            for i in tune_positions[
                                start : start + config["micro_batch"]
                            ]
                        ],
                        repeat_policy="lowest_source_row",
                    )
                    t = mixture_terms(
                        model, mixture, relative, batch, floors, center, scale
                    )
                    total += float(t["nll_sum"])
                    count += t["active"]
            if not count:
                raise ValueError("No TUNE measurements")
            return total / count

        with (output / "training.jsonl").open("x") as log:
            for epoch in range(config["epochs"]):
                order = np.random.default_rng(config["seed"] + epoch).permutation(
                    len(fit)
                )
                np.save(output / f"fit-epoch-{epoch + 1}-source-ids.npy", ids[order])
                for start in range(0, len(order), config["batch_size"]):
                    dataset._check_source_binding()
                    stars = [
                        fit[int(i)] for i in order[start : start + config["batch_size"]]
                    ]
                    whole = collate_multisurvey(
                        stars, repeat_policy="lowest_source_row"
                    )
                    gate.verify_batch(whole)
                    active = int(whole["observed"].any(-1).any(-1).sum())
                    probabilities = torch.full_like(
                        whole["input"], config["photometry_mask_fraction"]
                    )
                    probabilities[:, 5:115] = config["xp_mask_fraction"]
                    probabilities[:, width:] = 0
                    mask = (
                        torch.rand(whole["input"].shape, generator=generator)
                        < probabilities
                    ) & whole["input_available"]
                    hidden = int(mask.sum())
                    if not hidden:
                        raise ValueError(
                            "No present reconstruction targets in full batch"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    mixture.train()
                    relative.train()
                    nll = 0.0
                    recon = 0.0
                    reg = float(relative.regularization().detach())
                    for lo in range(0, len(stars), config["micro_batch"]):
                        micro = collate_multisurvey(
                            stars[lo : lo + config["micro_batch"]],
                            repeat_policy="lowest_source_row",
                        )
                        t = mixture_terms(
                            model,
                            mixture,
                            relative,
                            micro,
                            floors,
                            center,
                            scale,
                            mask=mask[lo : lo + len(micro["source_id"])],
                            reconstruction_width=width,
                        )
                        loss = (
                            t["nll_sum"] / max(active, 1)
                            + config["reconstruction_weight"]
                            * t["reconstruction_sum"]
                            / hidden
                        )
                        if lo == 0:
                            loss = (
                                loss
                                + config["survey_regularization"]
                                * relative.regularization()
                            )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite joint loss")
                        loss.backward()
                        nll += float(t["nll_sum"].detach())
                        recon += float(t["reconstruction_sum"].detach())
                    groups = dict(
                        encoder=encoder,
                        decoder=decoder,
                        mixture=list(mixture.parameters()),
                        survey=list(relative.parameters()),
                    )
                    norms = {}
                    for name, params in groups.items():
                        gs = [p.grad for p in params if p.grad is not None]
                        if not gs or not all(torch.isfinite(g).all() for g in gs):
                            raise FloatingPointError("Invalid gradients: " + name)
                        norms[name] = (
                            sum(float(g.detach().double().square().sum()) for g in gs)
                            ** 0.5
                        )
                        if step < 2 and norms[name] <= 0:
                            raise FloatingPointError("Inactive training path: " + name)
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]],
                        10.0,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    step += 1
                    metrics = dict(
                        step=step,
                        epoch=epoch + 1,
                        epoch_stars=start + len(stars),
                        batch_stars=len(stars),
                        labeled_stars=active,
                        masked_elements=hidden,
                        nll=nll / max(active, 1),
                        reconstruction=recon / hidden,
                        regularization=reg,
                        gradient_norms=norms,
                        elapsed_seconds=time.monotonic() - started,
                    )
                    log.write(json.dumps(metrics, allow_nan=False) + "\n")
                    log.flush()
                    if step <= 2 or step % 100 == 0:
                        atomic(output / "progress.json", metrics)
                        print(json.dumps(metrics), flush=True)
                    if step % config["checkpoint_interval"] == 0:
                        snapshot(epoch, start + len(stars))
                tune_nll = score_tune()
                if not np.isfinite(tune_nll):
                    raise FloatingPointError("Nonfinite TUNE score")
                ckpt = snapshot(epoch + 1, 0, tune_nll)
                if tune_nll < best_score:
                    best_score = tune_nll
                    best = ckpt
                    atomic(output / "selected-checkpoint.json", best)
                print(
                    json.dumps(
                        dict(
                            phase="epoch_complete",
                            epoch=epoch + 1,
                            tune_nll=tune_nll,
                            selected=best,
                        )
                    ),
                    flush=True,
                )
        tune_source.close()
        if sha(best["path"]) != best["sha256"]:
            raise ValueError("Selected checkpoint changed before final inference")
        selected = torch.load(best["path"], map_location=device, weights_only=True)
        model.load_state_dict(selected["model"], strict=True)
        relative.load_state_dict(selected["relative"], strict=True)
        mixture.load_state_dict(selected["mixture"], strict=True)
        model.eval()
        mixture.eval()
        relative.eval()
        manifest.update(status="exporting", selected=best)
        atomic(output / "manifest.json", manifest)
        exports = {}
        for role in config["export_roles"]:
            source = dataset if role == "fit" else reader(role)
            memory = fit if role == "fit" else RoleMemoryDataset(source)
            role_ids = np.asarray(memory.ids[memory.star_rows], dtype="<i8")
            if role != "fit" and np.intersect1d(ids, role_ids).size:
                raise ValueError("Export role overlaps FIT")
            path = output / f"predictions-{role}.h5"
            temporary = path.with_suffix(".partial")
            with h5py.File(temporary, "x") as h5, torch.no_grad():
                h5.attrs["role"] = role
                h5.attrs["source_binding"] = memory.source_binding
                h5.attrs["selected_checkpoint_sha256"] = best["sha256"]
                h5.attrs["artificial_masking"] = False
                for start in range(0, len(memory), config["inference_batch"]):
                    batch = collate_multisurvey(
                        [
                            memory[i]
                            for i in range(
                                start,
                                min(start + config["inference_batch"], len(memory)),
                            )
                        ],
                        repeat_policy="lowest_source_row",
                    )
                    x = batch["input"].to(device)
                    latent = model.mae.encoder(x)
                    mean, lower = model.head(latent.double())
                    context = (latent.double() - center) / scale
                    w, mu, chol, average, cov = physical_components(
                        mixture, context, mean, lower
                    )
                    fields = dict(
                        source_id=batch["source_id"].numpy(),
                        weights=w.cpu().numpy(),
                        component_mean=mu.cpu().numpy(),
                        component_lower=chol.cpu().numpy(),
                        mean=average.cpu().numpy(),
                        covariance=cov.cpu().numpy(),
                    )
                    for name in ("value", "error", "observed"):
                        array = batch[name].numpy()
                        padded = (
                            np.full((len(x), 3, 3), np.nan)
                            if name != "observed"
                            else np.zeros((len(x), 3, 3), dtype=bool)
                        )
                        for i in range(len(x)):
                            for j in range(array.shape[1]):
                                s = int(batch["source_index"][i, j])
                                if s >= 0:
                                    padded[i, s] = array[i, j]
                        fields["native_" + name] = padded
                    for name, array in fields.items():
                        if name not in h5:
                            h5.create_dataset(
                                name,
                                shape=(len(memory), *array.shape[1:]),
                                dtype=array.dtype,
                                chunks=(
                                    min(1024, max(1, len(memory))),
                                    *array.shape[1:],
                                ),
                                compression="lzf",
                            )
                        h5[name][start : start + len(x)] = array
                    if start % 32768 == 0:
                        print(
                            json.dumps(
                                dict(
                                    phase="export",
                                    role=role,
                                    rows=start + len(x),
                                    total=len(memory),
                                )
                            ),
                            flush=True,
                        )
            temporary.replace(path)
            exports[role] = dict(
                path=str(path),
                sha256=sha(path),
                stars=len(memory),
                source_ids_sha256=hashlib.sha256(role_ids.tobytes()).hexdigest(),
            )
            atomic(output / "exports.json", exports)
            if role != "fit":
                source.close()
        from plot_multisurvey_joint_kiel import plot

        for role in ("fit", "devtest"):
            plot(output / f"predictions-{role}.h5", output)
        return dict(
            updates=step,
            fit_stars=len(fit),
            epochs=config["epochs"],
            selected=best,
            exports=exports,
            supervised_xp_masking=False,
            reconstruction_weight=config["reconstruction_weight"],
        )

    try:
        manifest["result"] = native_preflight(
            config["parent"],
            config["parent"]["overlay_manifest"],
            commit,
            continuation=execute,
        )
        manifest["status"] = "multisurvey_joint_complete"
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic(output / "manifest.json", manifest)
        raise
    atomic(output / "manifest.json", manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(
        yaml.safe_load(a.config.read_text()),
        a.output,
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )


if __name__ == "__main__":
    main()
