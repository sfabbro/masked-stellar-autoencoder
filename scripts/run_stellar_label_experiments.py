"""Bounded source-aware supervised adaptation, sampling and latent refinement.

First-wave labels are explicitly provisional; no main checkpoint is changed.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from astropy.io import fits

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

PARENT = "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_57.pth"
PARENT_SHA = "9b033ae63a3b87c93dcfeffd989172f7888d029f77143e6cf315cf7a032ab2ca"
BENCH = "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5"
LEGACY = "/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits"


def sample_probabilities(y, emphasis=0.0):
    bins = np.digitize(y, [-2.5, -2.0, -1.0, 0.0])
    counts = np.bincount(bins, minlength=5)
    balanced = 1 / np.maximum(counts[bins], 1)
    balanced /= balanced.sum()
    return (1 - emphasis) / len(y) + emphasis * balanced


def supervised_loss(mean, logvar, target, sigma, kind, weights=None):
    if kind == "noisy_nll":
        from torchregress.losses import NoisyTargetGaussianNLL

        per_row = NoisyTargetGaussianNLL(reduction="none")(
            (mean, logvar), target, target_variance=sigma.square()
        )
        return (per_row * (weights if weights is not None else 1.0)).mean()
    errors = (
        (mean - target).square()
        if kind == "mse"
        else torch.nn.functional.huber_loss(mean, target, reduction="none", delta=1.0)
    )
    return (errors * (weights if weights is not None else 1.0)).mean()


def select_rows(ids, y, quality, dec, split, cap=100000, preserve_rare=True):
    partition = probe.identity_split(ids)
    if split == "sky":
        partition = np.where(dec >= 30, 2, np.where(partition == 1, 1, 0))
    rng = np.random.default_rng(42)
    selected, weights = [], []
    for k, limit in [(0, cap), (1, 20000), (2, 20000)]:
        eligible = np.flatnonzero((partition == k) & quality)
        rare = eligible[y[eligible] < -2] if preserve_rare else np.array([], dtype=int)
        common = np.setdiff1d(eligible, rare)
        rng.shuffle(common)
        chosen = common[: max(0, limit - len(rare))]
        selected.append(np.r_[rare, chosen])
        weights.append(
            np.r_[
                np.ones(len(rare)),
                np.full(len(chosen), len(common) / max(len(chosen), 1)),
            ]
        )
    return selected, weights


def refine(model, x, steps, prior=0.1, visible=None):
    # Fit observed feature values only, no supervised target passed to this function.
    if visible is not None:
        if not visible[:, : model.reconstruction_layer.out_features].any(1).all():
            raise ValueError(
                "Each refined star needs an observed reconstruction feature"
            )
        x = x.masked_fill(~visible, 0)
    with torch.no_grad():
        z0 = model.encoder(x)
    z = z0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        pred = probe.decode(model, z)
        observed = (
            torch.ones_like(pred, dtype=torch.bool)
            if visible is None
            else visible[:, : pred.shape[1]]
        )
        loss = (pred - x[:, : pred.shape[1]]).abs().masked_select(
            observed
        ).mean() + prior * (z - z0).square().mean()
        loss.backward()
        opt.step()
    return z.detach()


def prediction_mean(raw, target, center, scale):
    if target == "parallax":
        return (torch.nn.functional.softplus(raw) * scale - center) / scale
    return raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", default=PARENT)
    p.add_argument("--benchmark", default=BENCH)
    p.add_argument("--labels", default=LEGACY)
    args = p.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    settings = next(x for x in spec["experiments"] if x["run_id"] == args.run_id)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "run_id": args.run_id,
        "settings": settings,
        "parent_sha256": PARENT_SHA,
        "label_version": "legacy_source_separated_provisional_e_errors_filtered_not_native_GALAH_flag_audited",
        "scientific_caveat": "FeH excludes APOGEE M_H; heldout labelled IDs are not certified unseen in unsupervised pretraining",
    }
    probe.write_json(args.output / "manifest.json", manifest)
    try:
        scratch = Path("/scratch/msa-supervised-staging") / args.run_id
        scratch.mkdir(parents=True, exist_ok=False)
        local = scratch / "labels.fits"
        shutil.copyfile(args.labels, local)
        config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        features = config["data"]["feature_cols"]
        target_name = settings.get("target", "fe_h")
        x, y, ids, errors, duplicates = probe.read_fits(
            local, features, ["fe_h"], "g_normalized"
        )
        y = y[:, 0]
        with fits.open(local, memmap=True) as f:
            raw = f[1].data
            raw_ids = np.asarray(raw["source_id"], dtype="i8")
            _, first = np.unique(raw_ids, return_index=True)
            first = np.sort(first)
            tags = np.char.strip(np.asarray(raw["spec_source"]).astype(str))[first]
            e = np.asarray(raw["e_fe_h"], dtype="f4")[first]
            if target_name == "fe_h":
                # Prefer the high-resolution Li measurement when another survey has the same ID.
                order = np.argsort(ids)
                ordered = ids[order]
                alltags = np.char.strip(np.asarray(raw["spec_source"]).astype(str))
                for rr in np.flatnonzero(alltags == "li_et_al_vmps"):
                    position = np.searchsorted(ordered, raw_ids[rr])
                    if position < len(ids) and ordered[position] == raw_ids[rr]:
                        chosen = order[position]
                        y[chosen] = raw["fe_h"][rr]
                        e[chosen] = raw["e_fe_h"][rr]
                        tags[chosen] = "li_et_al_vmps"
            if target_name == "age":
                y = np.asarray(raw["age"], dtype="f4")[first]
                e = np.asarray(raw["e_age"], dtype="f4")[first]
            if target_name == "parallax":
                y = x[:, features.index("PARALLAX")].copy()
                e = np.asarray(raw["e_parallax"], dtype="f4")[first]
        registry = os.environ.get("MSA_LABEL_REGISTRY")
        if registry and target_name == "fe_h":
            with h5py.File(registry, "r") as labels:
                selected = labels["selected/labels"][:]
            order = np.argsort(selected["source_id"])
            selected = selected[order]
            positions = np.searchsorted(selected["source_id"], ids)
            matched = positions < len(selected)
            matched[matched] &= (
                selected["source_id"][positions[matched]] == ids[matched]
            )
            y = np.full(len(ids), np.nan, dtype="f4")
            e = np.full(len(ids), np.nan, dtype="f4")
            y[matched] = selected["value"][positions[matched]]
            e[matched] = selected["uncertainty"][positions[matched]]
            tags = tags.astype("U32")
            tags[matched] = selected["survey"][positions[matched]].astype(str)
            manifest.update(
                label_version="native_quality_registry",
                registry_path=registry,
                registry_matches=int(matched.sum()),
                registry_selected_total=len(selected),
            )
        original_availability = np.isfinite(x)
        quality = np.isfinite(y) & np.isfinite(e) & (e > 0)
        if target_name == "fe_h":
            quality &= (tags != "apogee-latentage") & (e < 0.3) & (y > -6) & (y < 1)
        elif target_name == "age":
            quality &= (tags == "apogee-latentage") & (y > 0) & (y < 14) & (e / y < 0.4)
        elif target_name == "parallax":
            quality &= (e < 2) & (np.abs(y) < 100)
        else:
            raise ValueError("Unsupported target")
        dec = x[:, features.index("DEC")].copy()
        checkpoint_hash = probe.digest(args.checkpoint)
        if checkpoint_hash != PARENT_SHA:
            raise ValueError("Parent checksum mismatch")
        checkpoint = torch_load_trusted(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        signature = checkpoint["run_signature"]
        center = np.asarray(signature["scaler_center"])
        scale = np.asarray(signature["scaler_scale"])
        with h5py.File(args.benchmark) as f:
            np.testing.assert_array_equal(center, f["reference_scaler/center"][:])
            np.testing.assert_array_equal(scale, f["reference_scaler/scale"][:])
            provenance = json.loads(f.attrs["provenance"])
            assert (
                provenance["source_id_sha256"]
                == "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
            )
        unit_audit = probe.audit_fits_xp_units(
            args.benchmark, features, x, ids, "g_normalized"
        )
        blocked = ["RA", "DEC", "pmra", "pmdec", *settings.get("blocked", [])]
        visible = np.isfinite(x) & probe.safe_inputs(
            np.ones_like(x[:1]), features, [target_name], blocked
        ).astype(bool)
        x = probe.safe_inputs((x - center) / scale, features, [target_name], blocked)
        rows, selection_weights = select_rows(
            ids,
            y,
            quality,
            dec,
            settings.get("split", "identity"),
            preserve_rare=target_name == "fe_h",
        )
        if min(map(len, rows)) < 20:
            raise ValueError("Insufficient source-separated labels in split")
        label_center = float(y[rows[0]].mean())
        label_scale = max(float(y[rows[0]].std()), 1e-4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        model = probe.load_model(config, checkpoint, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        head = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.GELU(), torch.nn.Linear(128, 2)
        ).to(device)
        xtrain = torch.tensor(x[rows[0]], dtype=torch.float32, device=device)
        vtrain = torch.tensor(visible[rows[0]], device=device)
        ytrain = torch.tensor(
            (y[rows[0]] - label_center) / label_scale,
            dtype=torch.float32,
            device=device,
        )
        etrain = torch.tensor(
            e[rows[0]] / label_scale, dtype=torch.float32, device=device
        )
        head_opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        ttt = int(settings.get("ttt_steps", 0))
        loss_kind = settings.get("loss", "huber")
        # Every variant has identical fixed head warmup, on training labels only.
        warmup = int(settings.get("warmup_steps", 500))
        steps = int(settings.get("steps", 2000))
        batch = 256
        with torch.no_grad():
            initial = probe.encode(model, x[rows[2]][:128], device)
        np.savez_compressed(
            args.output / "initial_predictions.npz",
            source_ids=ids[rows[2]][:128],
            latent=initial,
            initial_head_output=head(
                torch.tensor(initial, dtype=torch.float32, device=device)
            )
            .detach()
            .cpu()
            .numpy(),
        )
        for _ in range(warmup):
            index = rng.integers(len(xtrain), size=min(batch, len(xtrain)))
            head_opt.zero_grad()
            if ttt:
                z = refine(model, xtrain[index], ttt, visible=vtrain[index])
            else:
                with torch.no_grad():
                    z = model.encoder(xtrain[index])
            output = head(z)
            loss = supervised_loss(
                prediction_mean(output[:, 0], target_name, label_center, label_scale),
                output[:, 1],
                ytrain[index],
                etrain[index],
                loss_kind,
            )
            loss.backward()
            head_opt.step()
        adapt = bool(settings.get("adapt_encoder", True))
        if ttt and adapt:
            raise ValueError(
                "TTT experiments require frozen encoder; refined latent is detached"
            )
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(adapt)
        optimizer = torch.optim.AdamW(
            [
                {"params": head.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
                {
                    "params": model.encoder.parameters(),
                    "lr": float(settings.get("encoder_lr", 1e-6)),
                    "weight_decay": 1e-5,
                },
            ]
        )
        base_prob = (
            sample_probabilities(
                y[rows[0]], float(settings.get("sampling_emphasis", 0))
            )
            if target_name == "fe_h"
            else np.full(len(xtrain), 1 / len(xtrain))
        )
        metrics = []
        started = time.monotonic()
        model.eval()
        head.train()
        for step in range(steps):
            probabilities = base_prob
            if settings.get("curriculum"):
                with torch.no_grad():
                    idx = rng.choice(
                        len(xtrain), size=min(4096, len(xtrain)), replace=False
                    )
                    out = head(model.encoder(xtrain[idx]))[:, 0]
                    difficulty = (out - ytrain[idx]).abs().clamp(max=3).cpu().numpy()
                # Capped train-only hardness; retain at least half natural sampling.
                hardness = np.ones(len(xtrain))
                hardness[idx] = 1 + difficulty
                mix = min(0.5, 0.5 * step / max(steps - 1, 1))
                probabilities = (1 - mix) / len(
                    xtrain
                ) + mix * hardness / hardness.sum()
            index = rng.choice(
                len(xtrain), size=min(batch, len(xtrain)), replace=True, p=probabilities
            )
            optimizer.zero_grad(set_to_none=True)
            if ttt:
                z = refine(model, xtrain[index], ttt, visible=vtrain[index])
            else:
                z = model.encoder(xtrain[index])
            output = head(z)
            weights = torch.tensor(
                selection_weights[0][index] / selection_weights[0].mean(),
                dtype=torch.float32,
                device=device,
            )
            if settings.get("importance_corrected"):
                weights = weights * torch.tensor(
                    1 / (len(xtrain) * probabilities[index]),
                    dtype=torch.float32,
                    device=device,
                )
            mean = prediction_mean(output[:, 0], target_name, label_center, label_scale)
            loss = supervised_loss(
                mean, output[:, 1], ytrain[index], etrain[index], loss_kind, weights
            )
            rec_weight = float(settings.get("reconstruction_weight", 0))
            if rec_weight:
                # Frozen decoder receives gradients through adapted encoder; score only artificially hidden finite inputs.
                hidden = xtrain[index].clone()
                mask = (torch.rand(hidden.shape, device=device) < 0.2) & vtrain[index]
                hidden = hidden.masked_fill(mask, 0)
                pred = probe.decode(model, model.encoder(hidden))
                mask = mask[:, : pred.shape[1]]
                rec = (pred - xtrain[index, : pred.shape[1]]).abs().masked_select(
                    mask
                ).sum() / mask.sum().clamp_min(1)
                loss = loss + rec_weight * rec
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite supervised objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*model.encoder.parameters(), *head.parameters()], 1
            )
            optimizer.step()
            if (step + 1) % 100 == 0:
                metric = {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "elapsed_s": time.monotonic() - started,
                    "encoder_lr": optimizer.param_groups[1]["lr"],
                    "head_lr": optimizer.param_groups[0]["lr"],
                }
                metrics.append(metric)
                with (args.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metric) + "\n")
        model.eval()
        head.eval()
        arrays = {}
        results = {}
        for split, index in [("tune", rows[1]), ("test", rows[2])]:
            predicted = []
            for block in np.array_split(index, max(1, int(np.ceil(len(index) / 512)))):
                tensor = torch.tensor(x[block], dtype=torch.float32, device=device)
                z = (
                    refine(
                        model,
                        tensor,
                        ttt,
                        visible=torch.tensor(visible[block], device=device),
                    )
                    if ttt
                    else model.encoder(tensor)
                )
                with torch.no_grad():
                    output = head(z)
                    mean = output[:, 0]
                    mean = (
                        (
                            torch.nn.functional.softplus(mean) * label_scale
                            - label_center
                        )
                        / label_scale
                        if target_name == "parallax"
                        else mean
                    )
                    predicted.append((mean * label_scale + label_center).cpu().numpy())
            prediction = np.concatenate(predicted)
            arrays.update(
                {
                    split + "_source_id": ids[index],
                    split + "_truth": y[index],
                    split + "_prediction": prediction,
                    split + "_label_uncertainty": e[index],
                    split + "_population_weight": selection_weights[
                        1 if split == "tune" else 2
                    ],
                    split + "_survey": tags[index],
                    split + "_input_available": original_availability[index],
                }
            )
            results[split] = probe.score(prediction, y[index])
            results[split]["population_weighted_mae"] = float(
                np.average(
                    np.abs(prediction - y[index]),
                    weights=selection_weights[1 if split == "tune" else 2],
                )
            )
            if target_name == "fe_h":
                results[split + "_below_minus2"] = probe.score(
                    prediction[y[index] < -2], y[index][y[index] < -2]
                )
        np.savez_compressed(args.output / "predictions.npz", **arrays)
        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "head": head.state_dict(),
                "label_center": label_center,
                "label_scale": label_scale,
                "settings": settings,
            },
            args.output / "checkpoint.pth",
        )
        check = torch_load_trusted(
            args.output / "checkpoint.pth", map_location=device, weights_only=False
        )
        other = copy.deepcopy(head)
        other.load_state_dict(check["head"])
        restored = copy.deepcopy(model.encoder)
        restored.load_state_dict(check["encoder"])
        fixture = torch.tensor(x[rows[2]][:128], dtype=torch.float32, device=device)
        with torch.no_grad():
            torch.testing.assert_close(
                head(model.encoder(fixture)), other(restored(fixture)), rtol=0, atol=0
            )
        probe.write_json(
            args.output / "checkpoint_verification.json",
            {
                "encoder_and_head_reload_exact": True,
                "parent_unchanged": probe.digest(args.checkpoint) == PARENT_SHA,
                "checkpoint_sha256": probe.digest(args.output / "checkpoint.pth"),
            },
        )
        manifest.update(
            status="complete",
            split_counts=list(map(len, rows)),
            split_id_sha256=[
                hashlib.sha256(ids[r].tobytes()).hexdigest() for r in rows
            ],
            unit_audit=unit_audit,
            supervised_steps=steps,
            warmup_steps=warmup,
            duplicates_removed=duplicates,
            results=results,
            target=target_name,
            elapsed_training_s=time.monotonic() - started,
            scaler_sha256=hashlib.sha256(
                center.tobytes() + scale.tobytes()
            ).hexdigest(),
            benchmark_source_id_sha256=provenance["source_id_sha256"],
        )
        manifest.update(
            label_file_sha256=probe.digest(local),
            feature_columns=features,
            original_feature_coverage={
                name: float(original_availability[rows[2], i].mean())
                for i, name in enumerate(features)
            },
            encoder_dropout_during_sft=False,
            label_uncertainties_used_in_loss=loss_kind == "noisy_nll",
            train_sampling_population="Source-separated quality-filtered catalogue; cap inclusion weights and sampler policy recorded",
            age_caveat="If target age: APOGEE astroNN-derived proxy ages only, no independent asteroseismic accuracy assertion",
            distance_caveat="If parallax: positive photometric parallax fit to Gaia likelihood, no independent distance accuracy assertion; no zero-point correction asserted",
        )
        probe.write_json(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        probe.write_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
