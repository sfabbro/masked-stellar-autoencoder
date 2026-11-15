"""Predict every row of a FITS table with fine-tuned checkpoints.

For stars outside the fine-tune table's split, e.g. the Gaia benchmark stars
in ``validation.fits``. Preprocessing matches finetune_data: non-finite
features become NaN, the parallax column follows the config's astrometry
policy, then the checkpoint's own feature scaler; labels are inverted with
the checkpoint's label scalers.
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

    config = yaml.safe_load(a.config.read_text())
    cols = config["data"]["feature_cols"]
    error_cols = config["data"]["error_cols"]
    pproc = config.get("preprocessing") or {}
    with fits.open(a.fits, memmap=True) as hdul:
        d = hdul[1].data
        X = np.column_stack([np.asarray(d[c], dtype="f8") for c in cols])
        E = np.column_stack(
            [
                np.asarray(d[e], dtype="f8") if e else np.full(len(d), np.nan)
                for e in error_cols
            ]
        )
        source_id = np.asarray(d["source_id"], dtype="i8")
        truth = np.column_stack(
            [np.asarray(d[n], dtype="f8") for n in ("teff", "logg", "fe_h")]
        )
    X[~np.isfinite(X)] = np.nan
    empty = np.empty((0, X.shape[1]))
    apply_parallax_input_policy(
        X, empty.copy(), empty.copy(), E, empty.copy(), empty.copy(),
        cols.index("PARALLAX"), pproc.get("astrometry_input_policy", "legacy_raw"),
        snr_cap=float(pproc.get("astrometry_snr_cap", 10.0)),
    )  # fmt: skip

    state = torch_load_finetune_checkpoint(a.checkpoint, map_location="cpu")
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
    head = PredictionHead(m["layer_dims"][-1], 6, torch.nn.ReLU()).to(a.device)
    pack = dict(
        parallax_target_space=pproc.get("parallax_target_space", "linear_mas"),
        teff_target_space=pproc.get("teff_target_space", "linear"),
        parallax_floor_mas=float(pproc.get("parallax_floor_mas", 1e-4)),
    )
    scalers = state["label_scalers"]
    med = _ensemble_median_predictions(
        model, head, [state], Xs, torch.device(a.device), 256, linear_probe=False
    )
    q = _ensemble_median_predictions(
        model, head, [state], Xs, torch.device(a.device), 256,
        linear_probe=False, return_full_quantiles=True,
    )  # fmt: skip
    np.savez_compressed(
        a.out,
        source_id=source_id,
        truth=truth,
        median=_inverse_labels(med, scalers, pack),
        quantiles=_inverse_quantile_block(q, scalers, pack),
    )
    print("wrote", a.out, len(source_id), "stars")


if __name__ == "__main__":
    main()
