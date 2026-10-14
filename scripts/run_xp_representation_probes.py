"""Read-only frozen-encoder probes and frozen-decoder latent refinement.

Checkpoint and benchmark are immutable. Supervised heads fit only train labels;
probe hyperparameters use tuning rows, final heldout rows never select models.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from astropy.io import fits
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.models.model import make_model
from masked_stellar_autoencoder.training.hdf5_io import _clean_column


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".building")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def identity_split(ids, seed=42):
    """Stable split by source identity, never by duplicate catalogue row."""
    ids = np.asarray(ids, dtype=np.uint64)
    x = ids ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    buckets = x % np.uint64(1000)
    return np.where(buckets < 700, 0, np.where(buckets < 850, 1, 2))


def safe_inputs(values, features, targets, blocked=()):
    """Remove direct targets/aliases and target uncertainty inputs before encoding."""
    result = np.array(values, dtype=np.float32, copy=True)
    names = {str(name).lower() for name in [*targets, *blocked]}
    aliases = {
        "teff": {"teff", "t_eff", "temperature", "teff_gspphot", "teff_gspspec"},
        "logg": {"logg", "log_g", "logg_gspphot", "logg_gspspec"},
        "fe_h": {"fe_h", "feh", "mh", "metallicity", "mh_gspphot", "mh_gspspec"},
        "parallax": {"parallax", "distance", "r_med_geo", "r_med_photogeo"},
        "ebv": {"ebv", "e_bv", "ag", "a0", "azero"},
    }
    for target in targets:
        names.update(aliases.get(target.lower(), {target.lower()}))
    for index, feature in enumerate(features):
        normalized = feature.lower()
        if normalized in names or normalized.removeprefix("e_") in names:
            result[:, index] = np.nan
    result[~np.isfinite(result)] = -9999
    return result


def score(prediction, target):
    difference = np.asarray(prediction, dtype=np.float64) - target
    finite = np.isfinite(difference)
    if not finite.any():
        return {"count": 0, "mae": None, "rmse": None, "bias": None}
    difference = difference[finite]
    return {
        "count": int(len(difference)),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "bias": float(difference.mean()),
        "median_absolute_error": float(np.median(np.abs(difference))),
    }


def fit_ridge(train_x, train_y, tune_x, tune_y, test_x):
    """All scales fit train only; alpha selected on separate tuning rows."""
    scaler = StandardScaler().fit(train_x)
    center, scale = float(train_y.mean()), float(train_y.std())
    scale = max(scale, 1e-12)
    tx, vx = scaler.transform(train_x), scaler.transform(tune_x)
    candidates = []
    for alpha in (0.01, 1.0, 100.0, 10000.0):
        model = Ridge(alpha=alpha).fit(tx, (train_y - center) / scale)
        error = np.abs(model.predict(vx) * scale + center - tune_y).mean()
        candidates.append((float(error), alpha, model))
    _, alpha, model = min(candidates, key=lambda item: item[0])
    return model.predict(scaler.transform(test_x)) * scale + center, {
        "alpha": alpha,
        "train_target_center": center,
        "train_target_scale": scale,
        "tuning_mae": min(item[0] for item in candidates),
    }


def fit_head(train_x, train_y, test_x, *, seed, steps, device):
    """Frozen-encoder nonlinear head with fixed steps, no test selection."""
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(train_x)
    center, scale = float(train_y.mean()), max(float(train_y.std()), 1e-12)
    train = torch.as_tensor(
        scaler.transform(train_x), dtype=torch.float32, device=device
    )
    target = torch.as_tensor(
        (train_y - center) / scale, dtype=torch.float32, device=device
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(train.shape[1], 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        rows = torch.randint(
            len(train), (min(256, len(train)),), generator=generator, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(train[rows]).squeeze(1), target[rows])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test = torch.as_tensor(
            scaler.transform(test_x), dtype=torch.float32, device=device
        )
        predictions = [
            model(block).squeeze(1).cpu().numpy() for block in test.split(512)
        ]
    return np.concatenate(predictions) * scale + center


def finetune_head(model, train_x, train_y, test_x, *, seed, steps, device):
    """Equal-budget encoder+head adaptation from an independent checkpoint copy."""
    torch.manual_seed(seed)
    encoder = copy.deepcopy(model.encoder).to(device)
    center, scale = float(train_y.mean()), max(float(train_y.std()), 1e-12)
    train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    target = torch.as_tensor(
        (train_y - center) / scale, dtype=torch.float32, device=device
    )
    encoder.train()
    with torch.no_grad():
        latent_size = encoder(train[:1]).shape[1]
    head = torch.nn.Sequential(
        torch.nn.Linear(latent_size, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": 1e-6},
            {"params": head.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        rows = torch.randint(
            len(train), (min(256, len(train)),), generator=generator, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(
            head(encoder(train[rows])).squeeze(1), target[rows]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite fine-tuning objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *head.parameters()], 1.0)
        optimizer.step()
    encoder.eval()
    head.eval()
    with torch.no_grad():
        test = torch.as_tensor(test_x, dtype=torch.float32, device=device)
        predictions = [
            head(encoder(block)).squeeze(1).cpu().numpy() for block in test.split(512)
        ]
    return np.concatenate(predictions) * scale + center


def observed_objective(prediction, observed, visible, z, z0, zscale, prior):
    """Per-star objective ignores heldout entries, including their values/gradients."""
    errors = (prediction - observed).abs().masked_fill(~visible, 0)
    data = errors.sum(dim=1) / visible.sum(dim=1).clamp_min(1)
    regularizer = ((z - z0) / zscale).square().mean(dim=1)
    return (data + prior * regularizer).mean()


def decode(model, z):
    prediction = model.reconstruction_layer(model.decoder(z))
    if getattr(model, "heteroscedastic", False):
        prediction = prediction.chunk(2, dim=-1)[0]
    return prediction


def refine(model, values, hidden, zscale, *, steps=30, lr=0.01, prior=0.01):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    finite = torch.isfinite(values) & (values != -9999)
    inputs = values.masked_fill(hidden | ~finite, -9999)
    with torch.no_grad():
        initial_prediction, z0 = model(inputs)
    z = z0.detach().clone().requires_grad_(True)
    observed = torch.nan_to_num(values[:, : initial_prediction.shape[1]], nan=0.0)
    visible = (
        finite[:, : initial_prediction.shape[1]]
        & ~hidden[:, : initial_prediction.shape[1]]
    )
    if not visible.any(dim=1).all():
        raise ValueError(
            "Latent refinement requires at least one visible reconstruction feature per star"
        )
    optimizer = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = observed_objective(
            decode(model, z), observed, visible, z, z0, zscale, prior
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite refinement objective")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = decode(model, z)
    return initial_prediction, final


def load_model(config, checkpoint, device):
    c, d = config["model"], config["data"]
    model = make_model(
        len(d["feature_cols"]),
        c["layer_dims"],
        len(d["recon_cols"]),
        c["pt_activ_func"],
        c["rtdl_embed"],
        c["norm"],
        decoder_dims=c.get("decoder_dims"),
        encoder_type=c.get("encoder_type", "resnet"),
        cosine_latent=c.get("cosine_latent", False),
        heteroscedastic=config["training"].get("heteroscedastic", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def encode(model, x, device, batch_size=512):
    with torch.no_grad():
        return np.concatenate(
            [
                model(torch.as_tensor(block, device=device))[1].cpu().numpy()
                for block in np.array_split(
                    x, max(1, int(np.ceil(len(x) / batch_size)))
                )
            ]
        )


def read_benchmark(path, split, features, count):
    with h5py.File(path, "r") as source:
        records = source[f"{split}/records"]
        # The original benchmark is already a sampled catalogue. Read bounded
        # prefixes; do not decompress all 121 GB for a small inference pilot.
        rows = min(count, len(records))
        wanted = [*features, "source_id"]
        blocks = []
        for start in range(0, rows, 65536):
            blocks.append(records.fields(wanted)[start : min(start + 65536, rows)])
        raw = np.concatenate(blocks)
    x = np.column_stack([_clean_column(name, raw[name]) for name in features])
    return x, np.asarray(raw["source_id"], dtype=np.int64)


def read_fits(path, features, targets, xp_units):
    with fits.open(path, memmap=True) as hdul:
        table = next(hdu for hdu in hdul if isinstance(hdu, fits.BinTableHDU))
        names = {name.lower(): name for name in table.columns.names}
        masked_optional = {"RA", "DEC", "pmra", "pmdec"}
        missing = [
            name
            for name in [*features, *targets]
            if name.lower() not in names and name not in masked_optional
        ]
        if missing:
            raise ValueError(f"FITS missing required columns: {missing}")
        ids_name = next(
            (
                names[name]
                for name in ("source_id", "gaia_source_id", "sourceid")
                if name in names
            ),
            None,
        )
        if ids_name is None:
            raise ValueError(
                "FITS requires source IDs for independent, duplicate-safe labelled splits"
            )
        ids = np.asarray(table.data[ids_name], dtype=np.int64)
        x = np.column_stack(
            [
                _clean_column(name, table.data[names[name.lower()]])
                if name.lower() in names
                else np.full(len(ids), np.nan)
                for name in features
            ]
        )
        y = np.column_stack(
            [_clean_column(name, table.data[names[name.lower()]]) for name in targets]
        )
        errors = {
            name: np.asarray(table.data[names[f"e_{name}".lower()]], dtype=np.float32)
            for name in targets
            if f"e_{name}".lower() in names
        }
        for index, name in enumerate(features):
            if name.lower() not in names:
                continue
            null = table.columns[names[name.lower()]].null
            if null is not None:
                x[x[:, index] == null, index] = np.nan
        for index, name in enumerate(targets):
            null = table.columns[names[name.lower()]].null
            if null is not None:
                y[y[:, index] == null, index] = np.nan
        y[y == -9999] = np.nan
        if (ids <= 0).any():
            raise ValueError("FITS contains invalid source IDs")
    x[x == -9999] = np.nan
    _, unique = np.unique(ids, return_index=True)
    keep = np.sort(unique)
    return (
        x[keep],
        y[keep],
        ids[keep],
        {name: value[keep] for name, value in errors.items()},
        len(ids) - len(keep),
    )


def audit_fits_xp_units(benchmark, features, x, ids, requested, min_matches=128):
    """Independently determine normalization from matched immutable benchmark IDs."""
    xp = [i for i, name in enumerate(features) if name.startswith(("bp_", "rp_"))]
    order = np.argsort(ids)
    sorted_ids = ids[order]
    found_fits, found_values, matched_ids = [], [], []
    matched = 0
    fields = ["source_id", "G", *[features[i] for i in xp]]
    with h5py.File(benchmark, "r") as source:
        for split in ("train", "validation"):
            records = source[f"{split}/records"]
            for start in range(0, len(records), 65536):
                block = records.fields(fields)[start : start + 65536]
                bid = np.asarray(block["source_id"], dtype=np.int64)
                positions = np.searchsorted(sorted_ids, bid)
                valid = positions < len(ids)
                valid[valid] &= sorted_ids[positions[valid]] == bid[valid]
                if not valid.any():
                    continue
                fits_rows = order[positions[valid]]
                benchmark_values = np.column_stack(
                    [_clean_column(name, block[name][valid]) for name in fields[1:]]
                )
                usable = np.isfinite(x[fits_rows][:, [features.index("G"), *xp]]).all(
                    axis=1
                )
                usable &= np.isfinite(benchmark_values).all(axis=1)
                found_fits.append(fits_rows[usable])
                found_values.append(benchmark_values[usable])
                matched_ids.append(bid[valid][usable])
                matched += int(usable.sum())
                if matched >= min_matches:
                    break
            if matched >= min_matches:
                break
    if matched < min_matches:
        raise ValueError(
            f"Only {matched} finite FITS/benchmark source matches; require {min_matches} for units audit"
        )
    fits_rows = np.concatenate(found_fits)[:min_matches]
    truth = np.concatenate(found_values)[:min_matches]
    match_ids = np.concatenate(matched_ids)[:min_matches]
    g = x[fits_rows, features.index("G")]
    if not np.allclose(g, truth[:, 0], rtol=0, atol=0.002):
        raise ValueError("Matched FITS G differs from benchmark by more than 0.002 mag")
    coefficients = x[fits_rows][:, xp].astype(np.float64)
    target = truth[:, 1:].astype(np.float64)
    factor = 10 ** ((8.5 - g.astype(np.float64)) / 2.5)
    candidates = {"g_normalized": coefficients, "raw": coefficients / factor[:, None]}
    metrics = {}
    acceptable = []
    for name, values in candidates.items():
        relative = np.linalg.norm(values - target, axis=1) / np.maximum(
            np.linalg.norm(target, axis=1), 1e-12
        )
        passed = (
            bool(np.allclose(values, target, rtol=0.001, atol=1e-6))
            and relative.max() <= 0.001
        )
        metrics[name] = {
            "relative_l2_median": float(np.median(relative)),
            "relative_l2_max": float(relative.max()),
            "passed": bool(passed),
        }
        if passed:
            acceptable.append(name)
    if requested == "auto":
        if len(acceptable) != 1:
            raise ValueError(f"FITS units ambiguous or inconsistent: {metrics}")
        chosen = acceptable[0]
    elif requested not in acceptable:
        raise ValueError(
            f"Requested FITS units {requested} fail independent audit: {metrics}"
        )
    else:
        chosen = requested
    if chosen == "raw":
        all_factor = 10 ** ((8.5 - x[:, features.index("G")].astype(np.float64)) / 2.5)
        x[:, xp] /= all_factor[:, None]
    return {
        "chosen_units": chosen,
        "matched_stars": len(fits_rows),
        "candidates": metrics,
        "source_id_sha256": hashlib.sha256(match_ids.tobytes()).hexdigest(),
        "tolerance": "all110coefficients rtol0.001 atol1e-6 and perstarrelativeL2<=0.001; G abs0.002mag",
    }


def probes(args, config, model, center, scale, device):
    features, targets = config["data"]["feature_cols"], args.targets.split(",")
    x, y, ids, errors, duplicates = read_fits(
        args.labels_fits, features, targets, args.fits_xp_units
    )
    unit_audit = audit_fits_xp_units(
        args.benchmark, features, x, ids, args.fits_xp_units
    )
    write_json(args.output / "fits_units_audit.json", unit_audit)
    splits = identity_split(ids, args.seed)
    if args.split == "sky":
        dec = x[:, features.index("DEC")]
        if not np.isfinite(dec).all():
            raise ValueError("Sky holdout requires finite DEC")
        splits = np.where(
            dec >= 30, 2, np.where(identity_split(ids, args.seed) == 1, 1, 0)
        )
    # Astrophysical labels never provided to encoder; also remove sky to avoid
    # catalogue-footprint shortcuts. Removal policy identical across checkpoints.
    xscaled = safe_inputs(
        (x - center) / scale, features, targets, args.blocked.split(",")
    )
    rng = np.random.default_rng(args.seed)
    selected = []
    for split, cap in ((0, args.train_rows), (1, args.tune_rows), (2, args.test_rows)):
        rows = np.flatnonzero(splits == split)
        rng.shuffle(rows)
        selected.append(rows[:cap])
    rows = np.concatenate(selected)
    latent = encode(model, xscaled[rows], device)
    sizes = list(map(len, selected))
    zs = np.split(latent, np.cumsum(sizes)[:-1])
    ys = [y[index] for index in selected]
    result = {
        "split_counts": sizes,
        "duplicate_source_ids_removed": duplicates,
        "fits_units_audit": unit_audit,
        "targets": {},
    }
    predictions = {}
    for col, target in enumerate(targets):
        masks = [np.isfinite(value[:, col]) for value in ys]
        if min(int(mask.sum()) for mask in masks) < 20:
            raise ValueError(f"Insufficient finite {target} labels in a split")
        tx, vy, testx = zs[0][masks[0]], ys[1][masks[1], col], zs[2][masks[2]]
        ty, vx, testy = ys[0][masks[0], col], zs[1][masks[1]], ys[2][masks[2], col]
        prediction, policy = fit_ridge(tx, ty, vx, vy, testx)
        report = {
            "ridge": score(prediction, testy),
            "ridge_policy": policy,
            "budgets": {},
        }
        predictions[f"{target}_truth"] = testy
        predictions[f"{target}_ridge"] = prediction
        test_rows = selected[2][masks[2]]
        predictions[f"{target}_source_ids"] = ids[test_rows]
        report["ridge_sky_bins"] = {}
        declination = x[test_rows, features.index("DEC")]
        for lo, hi in ((-90, -30), (-30, 0), (0, 30), (30, 90.0001)):
            match = (declination >= lo) & (declination < hi)
            report["ridge_sky_bins"][f"{lo}:{hi}"] = score(
                prediction[match], testy[match]
            )
        if target in errors:
            err = errors[target][test_rows]
            for ceiling in (0.1, 0.2, 100):
                good = np.isfinite(err) & (err > 0) & (err <= ceiling)
                report.setdefault("label_error_subsets", {})[str(ceiling)] = score(
                    prediction[good], testy[good]
                )
        for budget in (1000, 10000, args.train_rows):
            budget = min(budget, len(tx))
            key = str(budget)
            if key in report["budgets"]:
                continue
            small, settings = fit_ridge(tx[:budget], ty[:budget], vx, vy, testx)
            head = fit_head(
                tx[:budget],
                ty[:budget],
                testx,
                seed=args.seed,
                steps=args.head_steps,
                device=device,
            )
            finetuned = finetune_head(
                model,
                xscaled[selected[0][masks[0]]][:budget],
                ty[:budget],
                xscaled[test_rows],
                seed=args.seed,
                steps=args.finetune_steps,
                device=device,
            )
            report["budgets"][key] = {
                "ridge": score(small, testy),
                "head": score(head, testy),
                "ridge_policy": settings,
                "head_steps": args.head_steps,
                "head_lr": 0.001,
                "encoder_frozen": True,
                "finetuned_encoder_head": score(finetuned, testy),
                "finetune_policy": {
                    "steps": args.finetune_steps,
                    "encoder_lr": 1e-6,
                    "head_lr": 1e-3,
                    "no_test_selection": True,
                },
            }
            predictions[f"{target}_head_{key}"] = head
            predictions[f"{target}_finetuned_{key}"] = finetuned
        result["targets"][target] = report
    result["split_source_id_sha256"] = [
        hashlib.sha256(ids[index].tobytes()).hexdigest() for index in selected
    ]
    np.savez_compressed(args.output / "probe_predictions.npz", **predictions)
    return result


def latent_refinement(args, config, model, center, scale, device):
    features = config["data"]["feature_cols"]
    train, train_ids = read_benchmark(
        args.benchmark, "train", features, args.prior_rows
    )
    raw, ids = read_benchmark(args.benchmark, "validation", features, args.test_rows)
    if np.intersect1d(train_ids, ids).size:
        raise ValueError("Train/validation source ID overlap")
    train = safe_inputs((train - center) / scale, features, [])
    prior_latent = encode(model, train, device)
    zscale = torch.as_tensor(np.maximum(prior_latent.std(axis=0), 0.05), device=device)
    values = (raw - center) / scale
    finite = np.isfinite(values)
    rng = np.random.default_rng(args.seed)
    hidden = rng.random(values.shape) < 0.2
    xp = [
        index for index, name in enumerate(features) if name.startswith(("bp_", "rp_"))
    ]
    if args.refinement_mask == "full_xp":
        hidden[:, xp] = True
    else:
        hidden[:, xp] = rng.random((len(values), len(xp))) < 0.25
    hidden &= finite
    prediction0, prediction1 = [], []
    for start in range(0, len(values), args.batch_size):
        v = torch.as_tensor(
            values[start : start + args.batch_size], dtype=torch.float32, device=device
        )
        h = torch.as_tensor(hidden[start : start + args.batch_size], device=device)
        initial, final = refine(
            model,
            v,
            h,
            zscale,
            steps=args.refinement_steps,
            lr=args.refinement_lr,
            prior=args.prior_weight,
        )
        prediction0.append(initial.detach().cpu().numpy())
        prediction1.append(final.detach().cpu().numpy())
    prediction0, prediction1 = np.concatenate(prediction0), np.concatenate(prediction1)
    result = {
        "rows": len(ids),
        "policy": "fixed final latent step; visible-feature fit only, train-fitted prior scales",
        "train_prior_source_id_sha256": hashlib.sha256(train_ids.tobytes()).hexdigest(),
        "validation_source_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "scores": {},
    }
    recon = len(config["data"]["recon_cols"])
    for name, columns in (("all_hidden", list(range(recon))), ("xp_hidden", xp)):
        mask = hidden[:, columns]
        truth = values[:, columns][mask]
        result["scores"][name] = {
            "before": score(prediction0[:, columns][mask], truth),
            "after": score(prediction1[:, columns][mask], truth),
        }
    np.savez_compressed(
        args.output / "latent_refinement.npz",
        source_ids=ids,
        hidden=hidden,
        before=prediction0,
        after=prediction1,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("probes", "latent_refinement"), required=True
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config", default="configs/pretrain.canfar.yaml")
    parser.add_argument(
        "--benchmark",
        default="/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5",
    )
    parser.add_argument(
        "--labels-fits",
        default="/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits",
    )
    parser.add_argument(
        "--fits-xp-units", choices=("auto", "g_normalized", "raw"), default="auto"
    )
    parser.add_argument("--targets", default="teff,logg,fe_h")
    parser.add_argument("--blocked", default="RA,DEC,pmra,pmdec")
    parser.add_argument("--split", choices=("identity", "sky"), default="identity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rows", type=int, default=100000)
    parser.add_argument("--tune-rows", type=int, default=20000)
    parser.add_argument("--test-rows", type=int, default=20000)
    parser.add_argument("--prior-rows", type=int, default=4096)
    parser.add_argument("--head-steps", type=int, default=1000)
    parser.add_argument("--finetune-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--refinement-mask", choices=("full_xp", "partial_xp"), default="full_xp"
    )
    parser.add_argument("--refinement-steps", type=int, default=30)
    parser.add_argument("--refinement-lr", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and (args.output / "manifest.json").exists():
        raise FileExistsError("Existing probe manifest: refusing duplicate/overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint_hash = digest(args.checkpoint)
    if args.checkpoint_sha256 and checkpoint_hash != args.checkpoint_sha256:
        raise ValueError(
            "Checkpoint checksum differs from expected saved final checkpoint"
        )
    checkpoint = torch_load_trusted(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    signature = checkpoint["run_signature"]
    if signature["feature_cols"] != config["data"]["feature_cols"]:
        raise ValueError("Feature order mismatch")
    center, scale = (
        np.asarray(signature["scaler_center"]),
        np.asarray(signature["scaler_scale"]),
    )
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid checkpoint feature scale")
    with h5py.File(args.benchmark, "r") as benchmark:
        np.testing.assert_array_equal(benchmark["reference_scaler/center"][:], center)
        np.testing.assert_array_equal(benchmark["reference_scaler/scale"][:], scale)
        provenance = json.loads(benchmark.attrs["provenance"])
        expected = "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
        if provenance.get("source_id_sha256") != expected:
            raise ValueError("Benchmark source ID provenance mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, checkpoint, device)
    manifest = {
        "status": "running",
        "checkpoint_sha256": checkpoint_hash,
        "parent_checkpoint_sha256": checkpoint.get("experiment", {}).get(
            "parent_checkpoint_sha256"
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scaler_sha256": hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest(),
        "benchmark_source_id_sha256": expected,
        "script_sha256": digest(__file__),
        "labels_caveat": "Supervised source IDs held out from head fitting, not certified unseen during encoder pretraining; strict128matched FITS units audit required",
        "encoder_frozen": True,
        "device": str(device),
    }
    write_json(args.output / "manifest.json", manifest)
    try:
        result = (
            probes(args, config, model, center, scale, device)
            if args.mode == "probes"
            else latent_refinement(args, config, model, center, scale, device)
        )
        write_json(args.output / "results.json", result)
        manifest["status"] = "complete"
        write_json(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
