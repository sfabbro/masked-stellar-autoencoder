import torch
import h5py
import numpy as np
import time
import os

from models.model import TabResnetWrapper, make_model

h5_path = 'dummy_data2.h5'

class DummyScaler:
    def __init__(self):
        self.scale_ = np.array([1.0, 1.0, 1.0])
    def transform(self, X):
        return X

model = make_model(
    input_dim=3,
    layer_dims=[64, 64],
    output_dim=2,
    activ='relu',
    rtdl_embed_dim=16,
    norm='layer'
)

# Test with cache_data = False
wrapper_no_cache = TabResnetWrapper(
    model=model,
    datafile=h5_path,
    scaler=DummyScaler(),
    feature_cols=['feat1', 'feat2', 'feat3'],
    error_cols=['err1', 'err2', 'err3'],
    recon_cols=['feat1', 'feat2'],
    latent_size=64,
    xp_masking_ratio=0.1,
    m_masking_ratio=0.1,
    lr=1e-3,
    optimizer='adam',
    cache_data=False
)

train_keys = [f'key_{i}' for i in range(10)]

class SlowFile:
    def __init__(self, f):
        self.f = f
    def __contains__(self, key):
        return key in self.f
    def __getitem__(self, key):
        time.sleep(0.1)
        return self.f[key]

with h5py.File(h5_path, 'r') as f:
    wrapper_no_cache.datafile = SlowFile(f)

    start = time.time()
    wrapper_no_cache.pretrain_hdf(train_keys=train_keys, num_epochs=2, mini_batch=100)
    end = time.time()
    print(f"Elapsed time without cache: {end - start:.2f} seconds")
