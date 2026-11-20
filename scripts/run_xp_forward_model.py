"""Train the XP forward model on labelled stars, then infer held-out and core stars.

Model, likelihood and contracts: ``masked_stellar_autoencoder.training.xp_forward_model``
and ``docs/design-20261120-likelihood-inference.md`` (section 3). Outputs in
``--output``: ``model.pt``, ``history.json``, ``train_nuisance.npz``,
``xp_forward_predictions.npz`` (held-out test rows and the frozen-core rows,
``subset`` = "heldout" / "core"), ``report.md`` / ``report.json`` /
``training_curves.png`` and ``manifest.json``.

    python scripts/run_xp_forward_model.py --table finetune.fits --labels labels.npz \\
        --covariance-sidecar sidecar_table.h5 --core-rows rows.npy --output out/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from masked_stellar_autoencoder.training import xp_forward_model as xf  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--table", required=True, help="finetune.fits (XP, photometry, PARALLAX, EBV)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", help="labels.npz (contract in xp_forward_model)")
    src.add_argument("--labels-from-table", action="store_true",
                     help="development stand-in: the table's teff/logg/fe_h and the fine-tune split")  # fmt: skip
    p.add_argument(
        "--split-config",
        help="fine-tune config whose metal-poor filters define the split (with --labels-from-table)",
    )
    p.add_argument(
        "--covariance-sidecar",
        help="sidecar_table.h5 aligned to the table rows; else diagonal table errors",
    )
    p.add_argument(
        "--g-flux-term",
        action="store_true",
        help="add the rank-one G-flux term to the sidecar covariance",
    )
    p.add_argument(
        "--xp-error-convention", default="raw_xp", choices=["raw_xp", "g_normalized_xp"]
    )
    p.add_argument("--parallax-zero-point", choices=["sidecar", "constant", "none"],
                   help="default: sidecar with a sidecar, else constant")  # fmt: skip
    p.add_argument("--zero-point-constant", type=float, default=xf.L21_CONSTANT_ZP)
    p.add_argument(
        "--core-rows",
        help="frozen-core table rows (.npy); excluded from training, inferred",
    )
    p.add_argument("--bands", default=",".join(b.name for b in xf.DEFAULT_BANDS))
    p.add_argument("--output", required=True)
    p.add_argument("--max-train-stars", type=int, default=1_000_000)
    p.add_argument("--max-valid-stars", type=int, default=20_000)
    p.add_argument("--max-heldout-stars", type=int, default=50_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--label-noise", choices=["plugin", "marginal"], default="plugin")
    p.add_argument("--label-noise-samples", type=int, default=8)
    p.add_argument(
        "--valid-rows", type=int, default=4096, help="valid stars profiled per epoch"
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--train-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--infer-dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--infer-chunk-rows", type=int, default=1024)
    p.add_argument("--infer-chunk-size", type=int, default=8192)
    p.add_argument("--n-candidates", type=int, default=64)
    p.add_argument("--bank-theta", type=int, default=1024)
    p.add_argument("--checkpoint", help="skip training: load this model.pt")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _pick(rows: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return (
        np.sort(rng.choice(rows, size=n, replace=False))
        if len(rows) > n
        else np.sort(rows)
    )


def main(argv=None) -> dict:
    a = parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    log_file = (out / "run.log").open("a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if a.device == "auto"
        else a.device
    )
    rng = np.random.default_rng(a.seed)
    core = (
        np.load(a.core_rows).astype(np.int64) if a.core_rows else np.zeros(0, np.int64)
    )
    if a.labels:
        labels = xf.load_labels(a.labels)
        labels_src = a.labels
    else:
        cfg = (
            yaml.safe_load(Path(a.split_config).read_text()) if a.split_config else None
        )
        labels = xf.labels_from_table(a.table, cfg, core)
        labels_src = f"table:{a.table}" + (
            f" split:{a.split_config}" if a.split_config else ""
        )
    theta = np.stack(
        [np.asarray(labels[p], dtype=np.float64) for p in xf.PARAMS], axis=1
    )
    labelled = np.isfinite(theta).all(axis=1)
    table_row = np.asarray(labels["table_row"], dtype=np.int64)
    split = labels["split"]
    not_core = ~np.isin(table_row, core)
    train_rows = _pick(
        table_row[labelled & (split == 0) & not_core], a.max_train_stars, rng
    )
    valid_rows = _pick(
        table_row[labelled & (split == 1) & not_core], a.max_valid_stars, rng
    )
    heldout_rows = _pick(
        table_row[labelled & (split == 2) & not_core], a.max_heldout_stars, rng
    )
    log(
        f"rows: train {len(train_rows)} valid {len(valid_rows)} heldout {len(heldout_rows)} core {len(core)}"
    )
    zero_point = a.parallax_zero_point or (
        "sidecar" if a.covariance_sidecar else "constant"
    )
    sidecar = (
        xf.SidecarXPCovariance(a.covariance_sidecar, g_flux_term=a.g_flux_term)
        if a.covariance_sidecar
        else None
    )
    bands = xf.bands_by_name([b for b in a.bands.split(",") if b])
    load_kw = dict(labels=labels, bands=bands, sidecar=sidecar, xp_error_convention=a.xp_error_convention,
                   zero_point=zero_point, zero_point_constant=a.zero_point_constant, log=log)  # fmt: skip
    manifest = {
        "run": str(out),
        "table": a.table,
        "labels": labels_src,
        "sidecar": a.covariance_sidecar,
        "zero_point": zero_point,
        "args": vars(a),
        "device": str(device),
        "rows": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "heldout": len(heldout_rows),
            "core": len(core),
        },
    }
    train_dtype = getattr(torch, a.train_dtype)
    infer_dtype = getattr(torch, a.infer_dtype)
    history: dict = {"epochs": []}
    if a.checkpoint:
        model, extra = xf.load_model(a.checkpoint)
        theta_bounds = np.asarray(extra["theta_bounds"])
        theta_pool = np.asarray(extra["theta_pool"])
        log(f"loaded {a.checkpoint}")
    else:
        train, diag_train = xf.load_star_data(a.table, train_rows, **load_kw)
        valid, _ = (
            xf.load_star_data(a.table, valid_rows, **load_kw)
            if len(valid_rows)
            else (None, None)
        )
        manifest["data_train"] = diag_train
        model = xf.build_model(
            train, xf.ModelConfig(hidden=a.hidden, depth=a.depth, rank=a.rank)
        )
        tcfg = xf.TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, label_noise=a.label_noise,
                              label_noise_samples=a.label_noise_samples, valid_rows=a.valid_rows, seed=a.seed)  # fmt: skip
        history = xf.train_forward_model(
            model, train, valid, tcfg, device=device, dtype=train_dtype, log=log
        )
        theta_bounds = xf.theta_bounds_from_labels(train.theta)
        theta_pool = train.theta[
            rng.choice(len(train), size=min(len(train), 20000), replace=False)
        ]
        nuis = history.pop("train_nuisance")
        np.savez(
            out / "train_nuisance.npz",
            table_row=train.rows,
            source_id=train.source_id,
            **nuis,
        )
        xf.save_model(model, out / "model.pt", extra={"theta_bounds": theta_bounds, "theta_pool": theta_pool,
                                                      "model_config": asdict(model.config)})  # fmt: skip
        (out / "history.json").write_text(
            json.dumps(history, indent=2, default=float) + "\n"
        )
        del train
    summaries: dict = {}
    label_chi2 = None
    if not a.skip_inference:
        icfg = xf.InferenceConfig(chunk_rows=a.infer_chunk_rows, chunk_size=a.infer_chunk_size,
                                  n_candidates=a.n_candidates, bank_theta=a.bank_theta, seed=a.seed)  # fmt: skip
        parts, subsets = [], []
        for name, rows in (("heldout", heldout_rows), ("core", np.unique(core))):
            if not len(rows):
                continue
            data, diag = xf.load_star_data(a.table, rows, **load_kw)
            manifest[f"data_{name}"] = diag
            t0 = time.perf_counter()
            res = xf.infer_stars(
                model,
                data,
                theta_bounds,
                theta_pool,
                icfg,
                device=device,
                dtype=infer_dtype,
                log=log,
            )
            manifest[f"infer_{name}_seconds"] = round(time.perf_counter() - t0, 1)
            pred = xf.prediction_arrays(data, res, np.full(len(data), name))
            parts.append(pred)
            summaries[name] = xf.recovery_summary(pred)
            if name == "heldout":
                lab = np.isfinite(data.theta).all(axis=1)
                sub = data.subset(np.flatnonzero(lab)[: a.valid_rows])
                prof = xf.profile_nuisance_at_labels(
                    model, sub, device=device, dtype=infer_dtype
                )
                label_chi2 = xf.recovery_summary({
                    "label": sub.theta, "mle": sub.theta, "lo68": sub.theta, "hi68": sub.theta,
                    "lo95": sub.theta, "hi95": sub.theta, "g_mag": sub.g_mag, "chi2_xp": prof["chi2_xp"],
                    "dof_xp": prof["dof_xp"], "at_bound": np.zeros((len(sub), 5), bool),
                })  # fmt: skip
            subsets.append(name)
        if parts:
            fixed = ("param_names", "at_bound_names")
            merged = {
                k: parts[0][k] if k in fixed else np.concatenate([p[k] for p in parts])
                for k in parts[0]
            }
            np.savez(out / "xp_forward_predictions.npz", **merged)
            log(
                f"wrote {out / 'xp_forward_predictions.npz'} ({len(merged['table_row'])} stars)"
            )
    manifest["seconds"] = round(time.perf_counter() - t_start, 1)
    manifest["scatter"] = model.describe()
    xf.write_report(out, history, summaries, label_chi2, manifest)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    if sidecar is not None:
        sidecar.close()
    log(f"done in {manifest['seconds']} s")
    log_file.close()
    return manifest


if __name__ == "__main__":
    main()
