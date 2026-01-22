import yaml
import h5py
import argparse
from sklearn.preprocessing import RobustScaler
import numpy as np
import os
import sys

# Add the repo root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from models.model import make_model, TabResnetWrapper

def main():

    parser = argparse.ArgumentParser(description="Train MSA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    # load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # loading the pretraining file to pass to the wrapper
    pretrain_file = h5py.File(config['data']['datafile'])

    # splitting up the keys
    keys_valid = config['data']['valid_keys']
    keys_train = [item for item in list(pretrain_file.keys()) if item not in keys_valid]

    cols = config['data']['feature_cols']

    scaler_keys = config['training'].get('scaler_keys', keys_train)
    scaler_max_rows = config['training'].get('scaler_max_rows', None)
    scaler_seed = config['training'].get('scaler_seed', 42)
    rng = np.random.default_rng(scaler_seed)

    X_list = []
    total_rows = 0
    for key in scaler_keys:
        if key not in pretrain_file:
            continue
        X_key = pretrain_file[key][:]
        X_key = np.column_stack([TabResnetWrapper._clean_column(col, X_key[col]) for col in cols])

        if np.any(np.isnan(X_key)) or np.any(np.isinf(X_key)):
            print("Warning: Invalid values detected in training data before scaling")
            valid_rows = ~np.all(np.isnan(X_key), axis=1)
            X_key = X_key[valid_rows]
            if len(X_key) == 0:
                continue

        if scaler_max_rows is not None:
            remaining = scaler_max_rows - total_rows
            if remaining <= 0:
                break
            if X_key.shape[0] > remaining:
                idx = rng.choice(X_key.shape[0], size=remaining, replace=False)
                X_key = X_key[idx]

        X_list.append(X_key)
        total_rows += X_key.shape[0]
        if scaler_max_rows is not None and total_rows >= scaler_max_rows:
            break

    if len(X_list) == 0:
        raise ValueError("No valid data available to fit feature scaler")

    X = np.vstack(X_list)

    featurescaler = RobustScaler()
    featurescaler.fit(X)
    
    # Validate scaler was fitted properly
    if not hasattr(featurescaler, 'scale_') or featurescaler.scale_ is None:
        raise ValueError("Scaler failed to fit properly - scale_ attribute missing")
    
    del X

    blocks_dims = config['model']['layer_dims']
    pt_activ = config['model']['pt_activ_func']
    d_embed = config['model']['rtdl_embed']
    norm = config['model']['norm']
    decoder_dims = config['model'].get('decoder_dims', None)  # Optional asymmetric decoder

    recon_cols = config['data']['recon_cols']

    model = make_model(
        len(cols),
        blocks_dims,
        len(recon_cols),
        pt_activ,
        d_embed,
        norm,
        decoder_dims=decoder_dims,
    )

    xp_ratio = config['training']['xp_masking_ratio']
    m_ratio = config['training']['m_masking_ratio']
    lr = config['training']['lr']
    wd = config['training']['weight_decay']
    lasso = config['training']['lasso']
    opt = config['training']['optimizer']
    lf = config['training']['loss_fn']
    pert_features = config['training'].get('pert_features', False)  # Optional data augmentation
    pert_scale = config['training'].get('pert_scale', 1.0)  # Noise scale factor

    pt_save_file = config['saving']['model_str']
    pt_log_file = config['saving']['log_file']
    ci = config['saving']['checkpoint_interval']

    error_cols = config['data']['error_cols']

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
        force_mask_cols=config['training'].get('force_mask_cols', None),
    )

    epochs = config['training']['epochs']
    batch = config['training']['mini_batch_size']
    presaved = config['training'].get('presaved', None)

    # print(model)

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
