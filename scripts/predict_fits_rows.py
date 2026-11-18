"""Predict every row of a FITS table with fine-tuned checkpoints.

For stars outside the fine-tune table's split, e.g. the Gaia benchmark stars
in ``validation.fits``. Preprocessing matches finetune_data: non-finite
features become NaN, the parallax column follows the config's astrometry
policy, then the checkpoint's own feature scaler; labels are inverted with
the checkpoint's label scalers.

``--test-split CONFIG`` predicts only the test split that finetune_data
gives the table under CONFIG (with table_row, as eval_ensemble's
predictions.npz), so models trained on different tables are compared on
the same stars with their own scalers. ``--irfm`` adds the photometric
IRFM-scale Teff of each star (training.irfm_teff) for the Teff check.

Anchored (v6) checkpoints carry their label names and Student-t head;
``--extra-labels`` supplies truth columns kept outside the table (E(B-V)).
Missing feature errors take the checkpoint's training fill values.
Anchored checkpoints may hide input features from the encoder
(``drop_input_features``) and carry a head context; both are applied as in
training. ``--condition-parallax SNR`` conditions the joint Student-t on the
measured Gaia parallax of stars with parallax/error >= SNR (log10 space;
``anchored_survey_map.condition_on_observation``); the unconditioned
prediction is kept as ``median_prior``/``quantiles_prior``.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--test-split", type=Path, help="config whose test split to use")
    p.add_argument("--irfm", action="store_true", help="also store IRFM-scale Teff")
    p.add_argument(
        "--condition-parallax",
        type=float,
        metavar="SNR",
        help="anchored: condition on measured parallax where parallax/error >= SNR",
    )
    p.add_argument(
        "--extra-labels",
        type=Path,
        help="npz of label columns per table row (e.g. NAME_reddening.npz)",
    )
    a = p.parse_args()

    from astropy.io import fits

    from masked_stellar_autoencoder.models.checkpoint_load import (
        torch_load_finetune_checkpoint,
    )
    from masked_stellar_autoencoder.models.model import PredictionHead, make_model
    from masked_stellar_autoencoder.training.astrometry_features import (
        apply_parallax_input_policy,
    )
    from masked_stellar_autoencoder.training.eval_ensemble import (
        _ensemble_median_predictions,
        _inverse_labels,
        _inverse_quantile_block,
    )

    state = torch_load_finetune_checkpoint(a.checkpoint, map_location="cpu")
    anchored = state.get("head") == "student_t"
    config = yaml.safe_load(a.config.read_text())
    cols = config["data"]["feature_cols"]
    error_cols = config["data"]["error_cols"]
    pproc = config.get("preprocessing") or {}
    rows = None
    if a.test_split:
        from masked_stellar_autoencoder.training.finetune_data import (
            kept_rows,
            split_row_positions,
        )

        split_cfg = yaml.safe_load(a.test_split.read_text())
        kept = kept_rows(str(a.fits), split_cfg)
        mp = split_cfg["finetuning"].get("metal_poor") or {}
        rows = kept[split_row_positions(len(kept), mp)[2]]
    with fits.open(a.fits, memmap=True) as hdul:
        d = hdul[1].data
        take = slice(None) if rows is None else rows

        def col(name):
            return np.asarray(d[name][take], dtype="f8")

        X = np.column_stack([col(c) for c in cols])
        E = np.column_stack(
            [col(e) if e else np.full(len(X), np.nan) for e in error_cols]
        )
        source_id = np.asarray(d["source_id"][take], dtype="i8")
        if anchored:
            names = [n for n in state["label_names"] if n != "parallax"]
        else:
            names = [n for n in ("teff", "logg", "fe_h", "alpha") if n in d.names]
        side = {}
        if a.extra_labels:
            with np.load(a.extra_labels) as z:
                sid_all = np.asarray(d["source_id"], dtype="i8")
                if not np.array_equal(z["source_id"], sid_all):
                    raise ValueError("--extra-labels does not match the table rows")
                side = {k: np.asarray(z[k][take], "f8") for k in z.files}
        nan = np.full(len(X), np.nan)
        truth = np.column_stack(
            [col(n) if n in d.names else side.get(n, nan) for n in names]
        )
        extra = {}
        if a.irfm:
            from masked_stellar_autoencoder.training.irfm_teff import (
                photometric_teff,
            )

            extra["teff_irfm"], extra["e_teff_irfm"] = photometric_teff(
                *(col(n) for n in ("G", "BP", "RP", "J", "H", "KS", "EBV")),
                col("logg"), col("fe_h"),
            )  # fmt: skip
        extra["G"], extra["EBV"] = col("G"), col("EBV")
        if rows is not None:
            extra["table_row"] = rows
    X[~np.isfinite(X)] = np.nan
    px = cols.index("PARALLAX")
    plx_mas, e_plx_mas = X[:, px].copy(), E[:, px].copy()  # before the policy
    if state.get("error_fill") is not None:  # training's fill, before the policy
        fill = np.broadcast_to(np.asarray(state["error_fill"]), E.shape)
        E = np.where(np.isfinite(E), E, fill)
    empty = np.empty((0, X.shape[1]))
    apply_parallax_input_policy(
        X, empty.copy(), empty.copy(), E, empty.copy(), empty.copy(),
        cols.index("PARALLAX"), pproc.get("astrometry_input_policy", "legacy_raw"),
        snr_cap=float(pproc.get("astrometry_snr_cap", 10.0)),
    )  # fmt: skip

    scaler = state["featurescaler"]
    Xs = scaler.transform(X).astype(np.float32)
    m = config["model"]
    model = make_model(
        len(cols), m["layer_dims"], len(config["data"]["recon_cols"]),
        m["pt_activ_func"], m["rtdl_embed"], m["norm"],
        decoder_dims=m.get("decoder_dims"), encoder_type=m.get("encoder_type", "resnet"),
        growth_rate=m.get("growth_rate", 64), num_dense_layers=m.get("num_dense_layers", 8),
        cosine_latent=m.get("cosine_latent", False),
        heteroscedastic=config["training"].get("heteroscedastic", False),
    ).to(a.device)  # fmt: skip
    act = {"relu": torch.nn.ReLU(), "elu": torch.nn.ELU(), "gelu": torch.nn.GELU()}[
        config["finetuning"].get("active", "relu")
    ]
    if anchored:
        from masked_stellar_autoencoder.training.anchored_survey_map import (
            StudentTHead,
        )

        n_labels = len(state["label_names"])
        head = StudentTHead(
            m["layer_dims"][-1], n_labels, act, state["nu"],
            context_dim=int(state.get("context_dim", 0)),
        )  # fmt: skip
    else:
        head = PredictionHead(m["layer_dims"][-1], 6, act)
    head = head.to(a.device)
    pack = dict(
        parallax_target_space=pproc.get("parallax_target_space", "linear_mas"),
        teff_target_space=pproc.get("teff_target_space", "linear"),
        parallax_floor_mas=float(pproc.get("parallax_floor_mas", 1e-4)),
    )
    scalers = state["label_scalers"]
    if anchored:
        _anchored(a, state, model, head, Xs, E, scaler, cols, scalers, pack, extra,
                  plx_mas, e_plx_mas, source_id, truth, names)  # fmt: skip
        return
    med = _ensemble_median_predictions(
        model, head, [state], Xs, torch.device(a.device), 256, linear_probe=False
    )
    q = _ensemble_median_predictions(
        model, head, [state], Xs, torch.device(a.device), 256,
        linear_probe=False, return_full_quantiles=True,
    )  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        source_id=source_id,
        truth=truth,
        median=_inverse_labels(med, scalers, pack),
        quantiles=_inverse_quantile_block(q, scalers, pack),
        truth_names=np.array(names),
        **extra,
    )
    print("wrote", a.out, len(source_id), "stars")


def _anchored(a, state, model, head, Xs, E, scaler, cols, scalers, pack, extra,
              plx_mas, e_plx_mas, source_id, truth, names):  # fmt: skip
    from scipy.stats import t as student_t

    from masked_stellar_autoencoder.training.anchored_predict import (
        anchored_predict,
    )
    from masked_stellar_autoencoder.training.checkpoint_keys import (
        autoencoder_state_dict,
        prediction_head_state_dict,
    )
    from masked_stellar_autoencoder.training.eval_ensemble import (
        _inverse_labels,
        _inverse_quantile_block,
    )

    device = torch.device(a.device)
    model.load_state_dict(autoencoder_state_dict(state))
    head.load_state_dict(prediction_head_state_dict(state))
    drop = [cols.index(c) for c in state.get("drop_input_features") or []]
    context = None
    Es = None
    if state.get("context_spec"):
        from masked_stellar_autoencoder.training.context_inputs import ContextInputs

        context = ContextInputs.from_spec(state["context_spec"], cols, scaler)
        context.load_state_dict(state["context_state_dict"])
        context = context.to(device)
        scale = np.where(scaler.scale_ <= 0, 1.0, scaler.scale_)
        Es = (E / scale).astype("f4")
    kw = dict(device=device, E=Es, context=context, drop_idx=drop)
    q84 = float(student_t.ppf(0.84, state["nu"]))

    def block(mu, sd, q):
        quant = np.stack([mu - q * sd, mu, mu + q * sd], axis=2)
        return _inverse_labels(mu, scalers, pack), _inverse_quantile_block(
            quant, scalers, pack
        )

    mu, sd = anchored_predict(model, head, Xs, **kw)
    med, quant = block(mu, sd, q84)
    out = dict(median=med, quantiles=quant)
    if a.condition_parallax is not None:
        j = list(state["label_names"]).index("parallax")
        if pack["parallax_target_space"] != "log10_mas":
            raise SystemExit("--condition-parallax needs log10_mas parallax labels")
        ok = (plx_mas > 0) & (plx_mas >= a.condition_parallax * e_plx_mas)
        with np.errstate(divide="ignore", invalid="ignore"):
            zl = np.log10(plx_mas)
            el = e_plx_mas / (np.log(10.0) * plx_mas)
        m, s = float(scalers[j].mean_[0]), float(scalers[j].scale_[0])
        z = np.where(ok, (zl - m) / s, np.nan).astype("f4")
        var = np.where(ok, (el / s) ** 2, np.nan).astype("f4")
        mu_c, sd_c = anchored_predict(model, head, Xs, condition=(j, z, var), **kw)
        q85 = float(student_t.ppf(0.84, state["nu"] + 1))
        qrow = np.where(ok, q85, q84)[:, None]
        out = dict(median_prior=med, quantiles_prior=quant)
        out["median"], out["quantiles"] = block(mu_c, sd_c, qrow)
        out["parallax_conditioned"] = ok
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out, source_id=source_id, truth=truth, truth_names=np.array(names),
        **out, **extra,
    )  # fmt: skip
    print("wrote", a.out, len(source_id), "stars")


if __name__ == "__main__":
    main()
