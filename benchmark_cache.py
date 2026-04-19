import time
import os
import torch
import h5py
import numpy as np
from sklearn.preprocessing import RobustScaler

from models.model import TabResnetWrapper, make_model

# 1. Create a dummy HDF5 file
h5_path = 'dummy_data.h5'
if os.path.exists(h5_path):
    os.remove(h5_path)
with h5py.File(h5_path, 'w') as f:
    for i in range(10):  # 10 keys
        # Don't create group, just create dataset directly
        data = np.zeros(1000, dtype=[('feat1', 'f4'), ('feat2', 'f4'), ('err1', 'f4'), ('err2', 'f4')])
        data['feat1'] = np.random.randn(1000)
        data['feat2'] = np.random.randn(1000)
        data['err1'] = np.abs(np.random.randn(1000))
        data['err2'] = np.abs(np.random.randn(1000))
        f[f'key_{i}'] = data

# 2. Setup model and wrapper
class DummyScaler:
    def __init__(self):
        self.scale_ = np.array([1.0, 1.0])
    def transform(self, X):
        return X

model = make_model(
    input_dim=2,
    layer_dims=[64, 64],
    output_dim=2,
    activ='relu',
    rtdl_embed_dim=16,
    norm='layer'
)

wrapper = TabResnetWrapper(
    model=model,
    datafile=h5_path,
    scaler=DummyScaler(),
    feature_cols=['feat1', 'feat2'],
    error_cols=['err1', 'err2'],
    recon_cols=['feat1', 'feat2'],
    latent_size=64,
    xp_masking_ratio=0.1,
    m_masking_ratio=0.1,
    lr=1e-3,
    optimizer='adam'
)

train_keys = [f'key_{i}' for i in range(10)]

# Monkey-patch _load_data to simulate slow I/O
original_load_data = wrapper._load_data
def slow_load_data(key):
    time.sleep(0.1)  # Simulate 100ms load time per key
    return original_load_data(key)
wrapper._load_data = slow_load_data

# Measure time
start = time.time()
wrapper.pretrain_hdf(train_keys=train_keys, num_epochs=2, mini_batch=100)
end = time.time()

print(f"Elapsed time: {end - start:.2f} seconds")
