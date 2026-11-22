"""Train the XP forward model on labelled stars, then infer held-out and core stars.

Model, likelihood and contracts: ``masked_stellar_autoencoder.training.xp_forward_model``
and ``docs/design-20261120-likelihood-inference.md`` (section 3). Training uses every
labelled train-split star of the frozen high-resolution core plus a random draw of the
other labelled train-split stars, with the empirical reddening labels as the training
anchor of E(B-V) (``--reddening-labels``; never used at inference). The held-out core
(core rows of the valid/test splits) is used neither for training nor for model
selection. Outputs in ``--output``: ``model.pt``, ``history.json``,
``train_nuisance.npz``, ``xp_forward_predictions.npz`` (held-out test rows and the
frozen-core rows, ``subset`` = "heldout" / "core", with the labels' ``split``),
``report.md`` / ``report.json`` / ``training_curves.png`` and ``manifest.json``.

    python scripts/run_xp_forward_model.py --table finetune.fits --labels labels.npz \\
        --reddening-labels v5_reddening.npz --covariance-sidecar sidecar_table.h5 \\
        --core-rows rows.npy --output out/
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

FIXED = ("param_names", "at_bound_names")  # per-file, not per-star, prediction arrays


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
        "--reddening-labels",
        help="npz with source_id, ebv_star, e_ebv_star (build_reddening_labels.py): "
        "E(B-V) anchor of the training stars",
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
        help="frozen-core table rows (.npy, all splits): train-split rows train, "
        "valid/test rows stay out; all are inferred",
    )
    p.add_argument("--bands", default=",".join(xf.LIKELIHOOD_BANDS))
    p.add_argument("--output", required=True)
    p.add_argument("--max-train-stars", type=int, default=1_000_000,
                   help="train-split core rows are always kept; the rest is a random draw")  # fmt: skip
    p.add_argument("--max-valid-stars", type=int, default=20_000)
    p.add_argument("--max-heldout-stars", type=int, default=50_000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nuisance-steps", type=int, default=3,
                   help="Gauss-Newton steps of the per-star (E, mu) per batch")  # fmt: skip
    p.add_argument("--ebv-mode", choices=("label", "anchor"), default="label",
                   help="labelled stars' E in training: held at the label (fixes the "
                   "E gauge exactly) or fitted with the label as a Gaussian term")  # fmt: skip
    p.add_argument("--mean-gradient", choices=("full", "quasi"), default="full",
                   help="emulator mean gradient: full Gaussian likelihood or "
                   "quasi-likelihood (covariance at the detached prediction)")  # fmt: skip
    p.add_argument("--no-anchor", action="store_true",
                   help="ignore the reddening labels in training (E gauge unfixed)")  # fmt: skip
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
    p.add_argument("--infer-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--infer-chunk-rows", type=int, default=2048)
    p.add_argument("--infer-chunk-size", type=int, default=8192)
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--n-starts", type=int, default=1)
    p.add_argument("--newton-steps", type=int, default=30)
    p.add_argument("--profile-scan", type=int, default=4)
    p.add_argument("--profile-steps", type=int, default=10)
    p.add_argument("--bank-theta", type=int, default=4096)
    p.add_argument("--checkpoint", help="skip training: load this model.pt")
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


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
    sel = xf.select_rows(labels, core, max_train=a.max_train_stars, max_valid=a.max_valid_stars,
                         max_heldout=a.max_heldout_stars, seed=a.seed)  # fmt: skip
    log(" ".join(f"{k} {len(v)}" for k, v in sel.items()))
    zero_point = a.parallax_zero_point or (
        "sidecar" if a.covariance_sidecar else "constant"
    )
    sidecar = (
        xf.SidecarXPCovariance(a.covariance_sidecar, g_flux_term=a.g_flux_term)
        if a.covariance_sidecar
        else None
    )
    reddening = xf.ReddeningLabels(a.reddening_labels) if a.reddening_labels else None
    bands = xf.bands_by_name([b for b in a.bands.split(",") if b])
    load_kw = dict(labels=labels, bands=bands, sidecar=sidecar, reddening=reddening,
                   xp_error_convention=a.xp_error_convention, zero_point=zero_point,
                   zero_point_constant=a.zero_point_constant, log=log)  # fmt: skip
    manifest = {
        "run": str(out),
        "table": a.table,
        "labels": labels_src,
        "reddening_labels": reddening.summary() if reddening else None,
        "sidecar": a.covariance_sidecar,
        "zero_point": zero_point,
        "bands": [b.name for b in bands],
        "args": vars(a),
        "device": str(device),
        "rows": {k: len(v) for k, v in sel.items()},
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
        train, diag_train = xf.load_star_data(a.table, sel["train"], **load_kw)
        valid, _ = (
            xf.load_star_data(a.table, sel["valid"], **load_kw)
            if len(sel["valid"])
            else (None, None)
        )
        manifest["data_train"] = diag_train
        model = xf.build_model(
            train, xf.ModelConfig(hidden=a.hidden, depth=a.depth, rank=a.rank)
        )
        tcfg = xf.TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                              nuisance_steps=a.nuisance_steps, anchor=not a.no_anchor,
                              ebv_mode=a.ebv_mode, mean_gradient=a.mean_gradient,
                              label_noise=a.label_noise, label_noise_samples=a.label_noise_samples,
                              valid_rows=a.valid_rows, seed=a.seed)  # fmt: skip
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
            ebv_label=train.ebv_label if train.ebv_label is not None else np.nan,
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
                                  n_candidates=a.n_candidates, n_starts=a.n_starts,
                                  steps=a.newton_steps, profile_scan=a.profile_scan,
                                  profile_steps=a.profile_steps, bank_theta=a.bank_theta,
                                  seed=a.seed)  # fmt: skip
        parts = []
        for name, rows in (("heldout", sel["heldout"]), ("core", sel["core"])):
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
            seconds = time.perf_counter() - t0
            manifest[f"infer_{name}_seconds"] = round(seconds, 1)
            manifest[f"infer_{name}_stars_per_second"] = round(
                len(data) / max(seconds, 1e-9), 2
            )
            pred = xf.prediction_arrays(data, res, np.full(len(data), name))
            parts.append(pred)
            if name == "core":
                split = pred["split"]
                for tag, mask in (
                    ("core_heldout", np.isin(split, (1, 2))),
                    ("core_train", split == 0),
                ):
                    if mask.any():
                        summaries[tag] = xf.recovery_summary(
                            {k: v if k in FIXED else v[mask] for k, v in pred.items()}
                        )
            else:
                summaries[name] = xf.recovery_summary(pred)
                lab = np.isfinite(data.theta).all(axis=1)
                sub = data.subset(np.flatnonzero(lab)[: a.valid_rows])
                prof = xf.profile_nuisance_at_labels(
                    model, sub, device=device, dtype=torch.float64
                )
                label_chi2 = xf.recovery_summary({
                    "label": sub.theta, "mle": sub.theta, "lo68": sub.theta, "hi68": sub.theta,
                    "lo95": sub.theta, "hi95": sub.theta, "g_mag": sub.g_mag, "chi2_xp": prof["chi2_xp"],
                    "dof_xp": prof["dof_xp"], "at_bound": np.zeros((len(sub), 5), bool),
                    "E_mle": prof["ebv"], "ebv_label": sub.ebv_label, "ebv_label_err": sub.ebv_label_err,
                })  # fmt: skip
        if parts:
            merged = {
                k: parts[0][k] if k in FIXED else np.concatenate([p[k] for p in parts])
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
