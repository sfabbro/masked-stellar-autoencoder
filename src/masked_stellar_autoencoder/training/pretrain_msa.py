import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from sklearn.preprocessing import RobustScaler

from masked_stellar_autoencoder.models.model import TabResnetWrapper, make_model

from .config_paths import expand_config_paths
from .feature_noise import pert_channel_scale_vector


def fit_pretrain_scaler(datafile, train_keys, cols, training_config):
    """Fit robust scaling from a bounded, seeded sample across training shards."""
    scaler_keys = training_config.get("scaler_keys") or train_keys
    missing = [key for key in scaler_keys if key not in train_keys]
    if missing:
        raise ValueError(f"scaler_keys must be training shards; not found: {missing}")
    if not scaler_keys:
        raise ValueError("No training shards are available for scaler fitting")

    max_rows = int(training_config.get("scaler_max_rows", 1_000_000))
    if max_rows < 1:
        raise ValueError("training.scaler_max_rows must be positive")

    rng = np.random.default_rng(int(training_config.get("scaler_seed", 42)))
    # ponytail: sample evenly by shard to bound RAM; proportional reservoir sampling
    # is the next step if production shards differ substantially in row count.
    keys = list(scaler_keys)
    if max_rows < len(keys):
        keys = rng.choice(keys, size=max_rows, replace=False).tolist()
        rows_per_key = 1
    else:
        rows_per_key = max_rows // len(keys)

    samples = []
    for key in keys:
        dataset = datafile[key]
        n_rows = len(dataset)
        if n_rows == 0:
            continue
        n_sample = min(n_rows, rows_per_key)
        indices = np.sort(rng.choice(n_rows, size=n_sample, replace=False))
        rows = dataset[indices]
        samples.append(
            np.column_stack(
                [TabResnetWrapper._clean_column(col, rows[col]) for col in cols]
            )
        )

    if not samples:
        raise ValueError("Training shards contain no rows for scaler fitting")

    X = np.concatenate(samples, axis=0)
    X[~np.isfinite(X)] = np.nan
    if not np.isfinite(X).any():
        raise ValueError("Training shards have no finite feature values")

    scaler = RobustScaler().fit(X)
    # RobustScaler preserves all-missing columns as NaN; use identity scaling so
    # those features remain masked instead of poisoning every scaled error.
    scaler.center_[~np.isfinite(scaler.center_)] = 0.0
    invalid_scale = ~np.isfinite(scaler.scale_) | (scaler.scale_ <= 0)
    scaler.scale_[invalid_scale] = 1.0
    return scaler


def _validate_error_columns(feature_cols, error_cols):
    if len(feature_cols) != len(error_cols):
        raise ValueError(
            "data.error_cols must contain one uncertainty column per feature"
        )
    if any(error in set(feature_cols) for error in error_cols if error is not None):
        raise ValueError(
            "data.error_cols duplicates data.feature_cols; map each feature to its "
            "measurement uncertainty before pretraining"
        )


def _pilot_path(path):
    if not path:
        return path
    value = Path(path)
    return str(value.with_name(f"{value.stem}_pilot{value.suffix}"))


def _configure_pilot(config):
    training = config["training"]
    training["epochs"] = 1
    training["mini_batch_size"] = min(int(training["mini_batch_size"]), 128)
    training["max_rows_per_shard"] = 512
    training["scaler_max_rows"] = min(
        int(training.get("scaler_max_rows", 1_000_000)), 10_000
    )
    for key in (
        "model_str",
        "log_file",
        "metrics_file",
        "residual_stats_file",
        "arc_checkpoint_dir",
    ):
        if key in config["saving"]:
            config["saving"][key] = _pilot_path(config["saving"][key])


def _limit_pilot_shards(train_keys, valid_keys):
    # ponytail: two train shards and one validation shard keep smoke runs cheap;
    # raise these caps when a representative pilot is needed.
    return train_keys[:2], valid_keys[:1]


def main():
    parser = argparse.ArgumentParser(description="Train MSA")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run one bounded epoch without overwriting full outputs",
    )
    args = parser.parse_args()

    # load YAML
    with open(args.config) as f:
        config = yaml.safe_load(f)
    expand_config_paths(config)
    if args.pilot:
        _configure_pilot(config)

    cols = config["data"]["feature_cols"]
    _validate_error_columns(cols, config["data"]["error_cols"])

    # Load the pretraining file after checking the feature/uncertainty mapping.
    pretrain_file = h5py.File(config["data"]["datafile"])
    keys_valid = config["data"]["valid_keys"]
    available_keys = list(pretrain_file.keys())
    missing_valid = [key for key in keys_valid if key not in available_keys]
    if missing_valid:
        raise ValueError(f"Validation shards missing from HDF5 file: {missing_valid}")
    keys_train = [item for item in available_keys if item not in keys_valid]
    if not keys_train:
        raise ValueError("No training shards remain after excluding valid_keys")
    if args.pilot:
        keys_train, keys_valid = _limit_pilot_shards(keys_train, keys_valid)
    featurescaler = fit_pretrain_scaler(
        pretrain_file, keys_train, cols, config["training"]
    )

    blocks_dims = config["model"]["layer_dims"]
    pt_activ = config["model"]["pt_activ_func"]
    d_embed = config["model"]["rtdl_embed"]
    norm = config["model"]["norm"]
    decoder_dims = config["model"].get(
        "decoder_dims", None
    )  # Optional asymmetric decoder

    recon_cols = config["data"]["recon_cols"]

    model = make_model(
        len(cols),
        blocks_dims,
        len(recon_cols),
        pt_activ,
        d_embed,
        norm,
        decoder_dims=decoder_dims,
        encoder_type=config["model"].get("encoder_type", "resnet"),
        growth_rate=config["model"].get("growth_rate", 64),
        num_dense_layers=config["model"].get("num_dense_layers", 8),
        cosine_latent=config["model"].get("cosine_latent", False),
        heteroscedastic=config["training"].get("heteroscedastic", False),
    )

    xp_ratio = config["training"]["xp_masking_ratio"]
    m_ratio = config["training"]["m_masking_ratio"]
    lr = config["training"]["lr"]
    wd = config["training"]["weight_decay"]
    lasso = config["training"]["lasso"]
    opt = config["training"]["optimizer"]
    lf = config["training"]["loss_fn"]
    pert_features = config["training"].get(
        "pert_features", False
    )  # Optional data augmentation
    pert_scale = config["training"].get("pert_scale", 1.0)  # Noise scale factor
    pert_ch = pert_channel_scale_vector(
        cols, pert_ebv_scale=float(config["training"].get("pert_ebv_scale", 1.0))
    )

    pt_save_file = config["saving"]["model_str"]
    pt_log_file = config["saving"]["log_file"]
    ci = config["saving"]["checkpoint_interval"]

    error_cols = config["data"]["error_cols"]

    # Initialize the pretraining wrapper
    pretrain_wrapper = TabResnetWrapper(
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
        max_rows_per_key=config["training"].get("max_rows_per_shard"),
    )

    pretrain_wrapper._configure_canfar_output(
        metrics_file=config["saving"].get("metrics_file"),
        residual_stats_file=config["saving"].get("residual_stats_file"),
        arc_checkpoint_dir=config["saving"].get("arc_checkpoint_dir"),
        arc_sync_interval=config["saving"].get("arc_sync_interval", 5),
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

    pretrain_file.close()


if __name__ == "__main__":
    main()
