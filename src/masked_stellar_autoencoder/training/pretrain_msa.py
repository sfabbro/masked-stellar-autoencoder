import argparse
import os
import sys

import h5py
import numpy as np
import yaml
from sklearn.preprocessing import RobustScaler

# Add the repo root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from config_paths import expand_config_paths
from feature_noise import pert_channel_scale_vector
from models.model import TabResnetWrapper, make_model


def load_config(config_path: str) -> dict:
    """Load and expand YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    expand_config_paths(config)
    return config


def setup_scaler(pretrain_file: h5py.File, train_key: str, cols: list) -> RobustScaler:
    """Initialize and fit the feature scaler on the first training file split."""
    featurescaler = RobustScaler()
    X = pretrain_file[train_key][:]
    X = np.column_stack([TabResnetWrapper._clean_column(col, X[col]) for col in cols])

    # Validate data before fitting scaler
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Invalid values detected in training data before scaling")
        # Remove rows with all NaN values
        valid_rows = ~np.all(np.isnan(X), axis=1)
        X = X[valid_rows]
        if len(X) == 0:
            raise ValueError("No valid data remaining after removing NaN rows")

    featurescaler.fit(X)

    # Validate scaler was fitted properly
    if not hasattr(featurescaler, "scale_") or featurescaler.scale_ is None:
        raise ValueError("Scaler failed to fit properly - scale_ attribute missing")

    return featurescaler


def build_model_from_config(config: dict, num_cols: int, num_recon_cols: int):
    """Construct the MSA model architecture from configuration."""
    blocks_dims = config["model"]["layer_dims"]
    pt_activ = config["model"]["pt_activ_func"]
    d_embed = config["model"]["rtdl_embed"]
    norm = config["model"]["norm"]
    decoder_dims = config["model"].get(
        "decoder_dims", None
    )  # Optional asymmetric decoder

    return make_model(
        num_cols,
        blocks_dims,
        num_recon_cols,
        pt_activ,
        d_embed,
        norm,
        decoder_dims=decoder_dims,
    )


def create_pretrain_wrapper(
    model,
    pretrain_file: h5py.File,
    featurescaler: RobustScaler,
    config: dict,
    cols: list,
) -> TabResnetWrapper:
    """Initialize the wrapper for the pretraining routine."""
    blocks_dims = config["model"]["layer_dims"]
    recon_cols = config["data"]["recon_cols"]
    error_cols = config["data"]["error_cols"]

    xp_ratio = config["training"]["xp_masking_ratio"]
    m_ratio = config["training"]["m_masking_ratio"]
    lr = config["training"]["lr"]
    wd = config["training"]["weight_decay"]
    lasso = config["training"]["lasso"]
    opt = config["training"]["optimizer"]
    lf = config["training"]["loss_fn"]

    pert_features = config["training"].get("pert_features", False)
    pert_scale = config["training"].get("pert_scale", 1.0)
    pert_ch = pert_channel_scale_vector(
        cols, pert_ebv_scale=float(config["training"].get("pert_ebv_scale", 1.0))
    )

    pt_save_file = config["saving"]["model_str"]
    pt_log_file = config["saving"]["log_file"]
    ci = config["saving"]["checkpoint_interval"]

    return TabResnetWrapper(
        model=model,
        datafile=pretrain_file,
        scaler=featurescaler,
        feature_cols=cols,
        error_cols=error_cols,
        recon_cols=recon_cols,
        xp_masking_ratio=xp_ratio,
        m_masking_ratio=m_ratio,
        latent_size=blocks_dims[-1],
        lr=lr,
        optimizer=opt,
        wd=wd,
        lasso=lasso,
        lf=lf,
        pt_save_str=pt_save_file,
        pt_log_file=pt_log_file,
        checkpoint_interval=ci,
        pert_features=pert_features,
        pert_scale=pert_scale,
        pert_channel_scale=pert_ch,
        mask_mixture_xp_full_frac=float(
            config["training"].get("mask_mixture_xp_full_frac", 0.0)
        ),
        scheduler_cosine_t0=int(config["training"].get("scheduler_cosine_t0", 10)),
        scheduler_cosine_t_mult=int(
            config["training"].get("scheduler_cosine_t_mult", 2)
        ),
        scheduler_eta_min_factor=float(
            config["training"].get("scheduler_eta_min_factor", 0.01)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Train MSA")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    with h5py.File(config["data"]["datafile"], "r") as pretrain_file:
        keys_valid = config["data"]["valid_keys"]
        keys_train = [
            item for item in list(pretrain_file.keys()) if item not in keys_valid
        ]
        cols = config["data"]["feature_cols"]
        recon_cols = config["data"]["recon_cols"]

        featurescaler = setup_scaler(pretrain_file, keys_train[0], cols)
        model = build_model_from_config(config, len(cols), len(recon_cols))

        pretrain_wrapper = create_pretrain_wrapper(
            model, pretrain_file, featurescaler, config, cols
        )

        epochs = config["training"]["epochs"]
        batch = config["training"]["mini_batch_size"]
        presaved = config["training"].get("presaved")
        if presaved is None or presaved == "":
            presaved = None

        # pretrain, train, and predict
        pretrain_wrapper.pretrain_hdf(
            keys_train,
            num_epochs=epochs,
            val_keys=keys_valid,
            mini_batch=batch,
            pretrained=presaved,
        )


if __name__ == "__main__":
    main()
