"""Frozen encoder latents and posterior-head predictions over a FITS table.

The preprocessing is that of ``scripts/predict_fits_rows.py`` for anchored
(Student-t head) checkpoints: non-finite features become NaN, missing errors
take the checkpoint's training fill, the config's parallax input policy, the
checkpoint's feature scaler, and the dropped input features are hidden from
the encoder. The posterior head is conditioned on the measured parallax where
parallax / error >= SNR (``predict_fits_rows --condition-parallax``). The
table is processed in row chunks, so it never sits in memory; latents are
written as float16 to a ``.npy`` memmap.

Missing inputs are the encoder's NaN token (``encoder_input``: NaN -> -9999,
the value the masked autoencoder sees for masked and missing features), so
``hide`` (e.g. ``("PARALLAX",)``) gives latents that carry no information from
those inputs: they are set to NaN after scaling, exactly like the
checkpoint's ``drop_input_features``. The posterior head keeps the full input
(the comparator is unchanged); with hidden inputs each batch runs the
encoder twice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .anchored_multisurvey import MAPPED
from .anchored_predict import encoder_input, mixture_quantiles
from .anchored_survey_map import condition_mixture


def load_anchored_model(config, state, device):
    """(encoder-decoder model, Student-t head, dropped input indices), eval mode."""
    from masked_stellar_autoencoder.models.model import make_model

    from .anchored_survey_map import StudentTHead
    from .checkpoint_keys import autoencoder_state_dict, prediction_head_state_dict

    if state.get("head") != "student_t":
        raise ValueError("Expected an anchored (Student-t head) checkpoint")
    if state.get("context_spec"):
        raise NotImplementedError("Head context checkpoints are not supported here")
    cols = config["data"]["feature_cols"]
    m = config["model"]
    model = make_model(
        len(cols), m["layer_dims"], len(config["data"]["recon_cols"]),
        m["pt_activ_func"], m["rtdl_embed"], m["norm"],
        decoder_dims=m.get("decoder_dims"), encoder_type=m.get("encoder_type", "resnet"),
        growth_rate=m.get("growth_rate", 64), num_dense_layers=m.get("num_dense_layers", 8),
        cosine_latent=m.get("cosine_latent", False),
        heteroscedastic=config["training"].get("heteroscedastic", False),
    )  # fmt: skip
    act = {"relu": torch.nn.ReLU(), "elu": torch.nn.ELU(), "gelu": torch.nn.GELU()}[
        config["finetuning"].get("active", "relu")
    ]
    head = StudentTHead(
        m["layer_dims"][-1], len(state["label_names"]), act, state["nu"],
        context_dim=int(state.get("context_dim", 0)),
        n_components=int(state.get("n_components", 1)),
        min_scale=float(state.get("min_scale", 1e-3)),
        full_covariance=bool(state.get("full_covariance", True)),
    )  # fmt: skip
    model.load_state_dict(autoencoder_state_dict(state))
    head.load_state_dict(prediction_head_state_dict(state))
    drop = [cols.index(c) for c in state.get("drop_input_features") or []]
    return model.to(device).eval(), head.to(device).eval(), drop


def label_pack(config):
    """Label-space conventions of the config (as ``predict_fits_rows``)."""
    pproc = config.get("preprocessing") or {}
    return dict(
        parallax_target_space=pproc.get("parallax_target_space", "linear_mas"),
        teff_target_space=pproc.get("teff_target_space", "linear"),
        parallax_floor_mas=float(pproc.get("parallax_floor_mas", 1e-4)),
    )


def preprocess(config, state, X, E):
    """Scaled float32 features, raw parallax and its error (mas)."""
    from .astrometry_features import apply_parallax_input_policy

    cols = config["data"]["feature_cols"]
    pproc = config.get("preprocessing") or {}
    X = np.array(X, dtype="f8")
    E = np.array(E, dtype="f8")
    X[~np.isfinite(X)] = np.nan
    px = cols.index("PARALLAX")
    plx, e_plx = X[:, px].copy(), E[:, px].copy()
    if state.get("error_fill") is not None:
        fill = np.broadcast_to(np.asarray(state["error_fill"]), E.shape)
        E = np.where(np.isfinite(E), E, fill)
    empty = np.empty((0, X.shape[1]))
    apply_parallax_input_policy(
        X, empty.copy(), empty.copy(), E, empty.copy(), empty.copy(), px,
        pproc.get("astrometry_input_policy", "legacy_raw"),
        snr_cap=float(pproc.get("astrometry_snr_cap", 10.0)),
    )  # fmt: skip
    return state["featurescaler"].transform(X).astype(np.float32), plx, e_plx


def parallax_condition(state, pack, plx_mas, e_plx_mas, snr):
    """(label index, z, var) for conditioning on parallax/error >= snr, or None."""
    if snr is None or "parallax" not in state["label_names"]:
        return None
    if pack["parallax_target_space"] != "log10_mas":
        raise ValueError("Parallax conditioning needs log10_mas parallax labels")
    j = list(state["label_names"]).index("parallax")
    ok = (plx_mas > 0) & (plx_mas >= snr * e_plx_mas)
    with np.errstate(divide="ignore", invalid="ignore"):
        zl = np.log10(plx_mas)
        el = e_plx_mas / (np.log(10.0) * plx_mas)
    sc = state["label_scalers"][j]
    m, s = float(sc.mean_[0]), float(sc.scale_[0])
    z = np.where(ok, (zl - m) / s, np.nan).astype("f4")
    var = np.where(ok, (el / s) ** 2, np.nan).astype("f4")
    return j, z, var


@torch.no_grad()
def encode_and_predict(model, head, Xs, drop, *, device, batch=8192, condition=None,
                       latent_drop=None):  # fmt: skip
    """Latent (float16), scaled marginal 16/50/84 % quantiles (N, P, 3), conditioned.

    ``latent_drop``: input indices hidden for the returned latent (default
    ``drop``); the posterior head always sees the input with ``drop`` hidden.
    """
    lat, lws, locs, sds, oks = [], [], [], [], []
    for i in range(0, len(Xs), batch):
        xb = torch.as_tensor(Xs[i : i + batch], device=device)
        z = model.encoder(encoder_input(xb, drop))
        logw, mu, L = head.mixture(z)
        ok = torch.zeros(len(z), dtype=torch.bool, device=z.device)
        if condition is not None:
            j, zc, vc = condition
            zt = torch.as_tensor(zc[i : i + batch], device=device)
            vt = torch.as_tensor(vc[i : i + batch], device=device)
            logw, mu, sd = condition_mixture(logw, mu, L, j, zt, vt, head.nu)
            ok = torch.isfinite(zt) & torch.isfinite(vt) & (vt >= 0)
        else:
            sd = L.square().sum(-1).sqrt()
        if latent_drop is not None and sorted(latent_drop) != sorted(drop):
            z = model.encoder(encoder_input(xb, latent_drop))
        lat.append(z.half().cpu())
        lws.append(logw.cpu())
        locs.append(mu.cpu())
        sds.append(sd.cpu())
        oks.append(ok.cpu())
    ok = torch.cat(oks).numpy()
    q = mixture_quantiles(
        torch.cat(lws).numpy().astype("f8"), torch.cat(locs).numpy().astype("f8"),
        torch.cat(sds).numpy().astype("f8"), head.nu + ok.astype("f8"),
    )  # fmt: skip
    return torch.cat(lat).numpy(), q, ok


def encode_table(config, state, fits_path, out_dir, *, device="cpu", chunk_rows=262144,
                 batch=8192, condition_parallax=5.0, hide=(), log=print):  # fmt: skip
    """Write ``latent.npy`` (N, D) float16 and ``predictions.npz`` for every row.

    ``hide``: feature names hidden from the encoder for the latent only
    (recorded in ``latent_meta.json``); the posterior predictions use the
    full input.

    ``predictions.npz``: ``source_id``, ``table_row``, ``G``, the posterior
    head's (Teff, log g, [Fe/H]) ``post_median``/``post_q16``/``post_q84``
    (physical) and ``post_scaled`` (median, scaled label units),
    ``parallax_conditioned``, the Student-t degrees of freedom ``post_nu`` and
    ``post_balanced`` (the head was trained with balanced sampling, so its
    posterior is under the sampling prior).
    """
    from astropy.io import fits

    from .eval_ensemble import _inverse_labels, _inverse_quantile_block

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    model, head, drop = load_anchored_model(config, state, device)
    cols = config["data"]["feature_cols"]
    unknown = [c for c in hide if c not in cols]
    if unknown:
        raise ValueError(f"Cannot hide unknown input features {unknown}")
    latent_drop = sorted(set(drop) | {cols.index(c) for c in hide})
    error_cols = config["data"]["error_cols"]
    pack = label_pack(config)
    scalers = state["label_scalers"]
    li = [list(state["label_names"]).index(p) for p in MAPPED]
    dim = int(config["model"]["layer_dims"][-1])
    t0 = time.time()
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul[1].data
        n = len(data)
        latent = np.lib.format.open_memmap(
            out_dir / "latent.npy.tmp", mode="w+", dtype="f2", shape=(n, dim)
        )
        res = {
            k: np.full((n, 3), np.nan, "f4")
            for k in ("post_median", "post_q16", "post_q84", "post_scaled")
        }
        source_id = np.zeros(n, "i8")
        G = np.full(n, np.nan)
        cond = np.zeros(n, bool)
        nu = np.full(n, float(head.nu), "f4")
        for start in range(0, n, chunk_rows):
            sl = slice(start, min(n, start + chunk_rows))
            chunk = data[sl]

            def col(name, chunk=chunk):
                return np.asarray(chunk[name], dtype="f8")

            X = np.column_stack([col(c) for c in cols])
            nan = np.full(len(X), np.nan)
            E = np.column_stack([col(e) if e else nan for e in error_cols])
            source_id[sl] = np.asarray(chunk["source_id"], dtype="i8")
            G[sl] = col("G")
            Xs, plx, e_plx = preprocess(config, state, X, E)
            condition = parallax_condition(state, pack, plx, e_plx, condition_parallax)
            lat, q, ok = encode_and_predict(
                model, head, Xs, drop, device=device, batch=batch, condition=condition,
                latent_drop=latent_drop,
            )  # fmt: skip
            latent[sl] = lat
            cond[sl] = ok
            nu[sl] = head.nu + ok
            quant = _inverse_quantile_block(q, scalers, pack)[:, li]
            res["post_q16"][sl], res["post_q84"][sl] = quant[..., 0], quant[..., 2]
            res["post_median"][sl] = _inverse_labels(q[..., 1], scalers, pack)[:, li]
            res["post_scaled"][sl] = q[:, li, 1]
            log(f"latents: rows {sl.stop}/{n} ({time.time() - t0:.0f} s)")
        latent.flush()
        del latent
    (out_dir / "latent.npy.tmp").replace(out_dir / "latent.npy")
    meta = dict(rows=int(n), dim=dim, hidden_inputs=list(hide),
                dropped_inputs=[cols[i] for i in drop])  # fmt: skip
    (out_dir / "latent_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    np.savez(
        out_dir / "predictions.npz", source_id=source_id, table_row=np.arange(n),
        G=G, parallax_conditioned=cond, post_nu=nu, label_names=np.array(MAPPED),
        post_balanced=state.get("sampling_ratio") is not None, **res,
    )  # fmt: skip
    return out_dir / "latent.npy", out_dir / "predictions.npz"
