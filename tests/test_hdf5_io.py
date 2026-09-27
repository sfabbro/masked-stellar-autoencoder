from unittest.mock import MagicMock

import h5py
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from masked_stellar_autoencoder.training.hdf5_io import ProjectedHDF5Store


def _table():
    dtype = np.dtype(
        [
            ("feature", "f4"),
            ("error", "f4"),
            ("unused", "f8"),
        ]
    )
    return np.array(
        [(float(i), 0.5 + i / 10, i * 1000.0) for i in range(7)], dtype=dtype
    )


def _collect(store, scaler, *, shuffle=False, seed=0):
    return [
        (x.copy(), e.copy())
        for x, e in store.iter_batches(
            "train",
            scaler,
            np.array([2.0], dtype=np.float32),
            batch_rows=3,
            shuffle=shuffle,
            seed=seed,
        )
    ]


def test_projected_store_stream_and_cache_match_with_partial_batch(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("train", data=_table(), chunks=(2,))

        scaler = RobustScaler().fit(np.array([[0.0], [6.0]], dtype=np.float32))
        stream = ProjectedHDF5Store(
            h5,
            ["feature"],
            ["error"],
            scratch_dir=tmp_path,
            cache_fraction=0.0,
            chunk_rows=2,
        )
        stream.prepare(["train"], ["train"], scaler_max_rows=4, scaler_seed=4)
        assert stream.mode == "stream"
        assert stream.fields == ["feature", "error"]
        stream_batches = _collect(stream, scaler)

        cached = ProjectedHDF5Store(
            h5,
            ["feature"],
            ["error"],
            scratch_dir=tmp_path,
            cache_fraction=1.0,
            chunk_rows=2,
        )
        cached.prepare(["train"], ["train"], scaler_max_rows=4, scaler_seed=4)
        assert cached.mode == "cache"
        cached_batches = _collect(cached, scaler)

    assert [len(x) for x, _ in stream_batches] == [3, 3, 1]
    assert len(stream.scaler_sample) == 4
    np.testing.assert_allclose(stream.scaler_sample, cached.scaler_sample)
    for (stream_x, stream_e), (cache_x, cache_e) in zip(
        stream_batches, cached_batches, strict=True
    ):
        np.testing.assert_allclose(stream_x, cache_x)
        np.testing.assert_allclose(stream_e, cache_e)


def test_projected_store_chunk_shuffle_is_seeded_and_preserves_rows(tmp_path):
    with h5py.File(tmp_path / "input.h5", "w") as h5:
        h5.create_dataset("train", data=_table(), chunks=(2,))
        store = ProjectedHDF5Store(
            h5,
            ["feature"],
            ["error"],
            scratch_dir=tmp_path,
            cache_fraction=0.0,
            chunk_rows=2,
            shuffle_buffer_bytes=64,
        )
        store.prepare(["train"], ["train"], scaler_max_rows=4, scaler_seed=4)
        scaler = RobustScaler().fit(np.array([[0.0], [6.0]], dtype=np.float32))

        first = _collect(store, scaler, shuffle=True, seed=17)
        second = _collect(store, scaler, shuffle=True, seed=17)
        other_seed = _collect(store, scaler, shuffle=True, seed=18)

    first_values = np.concatenate([x[:, 0] for x, _ in first])
    second_values = np.concatenate([x[:, 0] for x, _ in second])
    other_values = np.concatenate([x[:, 0] for x, _ in other_seed])
    np.testing.assert_array_equal(first_values, second_values)
    np.testing.assert_array_equal(np.sort(first_values), np.sort(other_values))
    assert not np.array_equal(first_values, other_values)


def test_projected_store_falls_back_when_scratch_budget_is_too_small(tmp_path):
    with h5py.File(tmp_path / "input.h5", "w") as h5:
        h5.create_dataset("train", data=_table())
        store = ProjectedHDF5Store(
            h5,
            ["feature"],
            ["error"],
            scratch_dir=tmp_path,
            cache_fraction=0.8,
            scratch_free_bytes=1,
        )
        store.prepare(["train"], ["train"], scaler_max_rows=4, scaler_seed=4)

    assert store.mode == "stream"
    assert store.cache_bytes > 1


def test_projected_store_repairs_invalid_errors_like_full_shard_loader(tmp_path):
    from masked_stellar_autoencoder.models.model import TabResnetWrapper

    dtype = np.dtype([("feature", "f4"), ("error", "f4")])
    values = np.array(
        [(1.0, 0.5), (2.0, 3.0), (np.inf, np.nan), (4.0, 0.0)], dtype=dtype
    )
    with h5py.File(tmp_path / "input.h5", "w") as h5:
        h5.create_dataset("train", data=values, chunks=(2,))
        store = ProjectedHDF5Store(
            h5,
            ["feature"],
            ["error"],
            scratch_dir=tmp_path,
            cache_fraction=0.0,
            chunk_rows=2,
        )
        store.prepare(["train"], ["train"], scaler_max_rows=2, scaler_seed=4)
        scaler = RobustScaler().fit(np.array([[1.0], [4.0]], dtype=np.float32))
        batches = list(
            store.iter_batches(
                "train",
                scaler,
                scaler.scale_,
                batch_rows=3,
                shuffle=False,
                seed=0,
            )
        )
        streamed_x = np.concatenate([x for x, _ in batches])
        streamed_e = np.concatenate([e for _, e in batches])

        wrapper = TabResnetWrapper(
            model=torch.nn.Identity(),
            datafile=h5,
            scaler=scaler,
            feature_cols=["feature"],
            error_cols=["error"],
            recon_cols=["feature"],
        )
        full_x, full_e = wrapper._load_data("train")

    np.testing.assert_allclose(streamed_x, full_x.numpy(), equal_nan=True)
    np.testing.assert_allclose(streamed_e, full_e.numpy())


def test_pretrain_training_uses_streamed_batches_without_full_shard_load(tmp_path):
    from masked_stellar_autoencoder.models.model import TabResnetWrapper

    class TinyRecon(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, values):
            return values[:, :115] * self.weight, values[:, :1] * self.weight

    feature_cols = [f"feature_{i}" for i in range(120)]
    dtype = np.dtype([(name, "f4") for name in feature_cols])
    values = np.zeros(5, dtype=dtype)
    for i, name in enumerate(feature_cols):
        values[name] = np.arange(5, dtype=np.float32) + i

    with h5py.File(tmp_path / "input.h5", "w") as h5:
        h5.create_dataset("train", data=values, chunks=(2,))
        features = np.column_stack([values[name] for name in feature_cols])
        scaler = RobustScaler().fit(features)
        store = ProjectedHDF5Store(
            h5,
            feature_cols,
            [None] * len(feature_cols),
            scratch_dir=tmp_path,
            cache_fraction=0.0,
            chunk_rows=2,
        )
        store.prepare(["train"], ["train"], scaler_max_rows=5, scaler_seed=1)
        model = TinyRecon()
        wrapper = TabResnetWrapper(
            model=model,
            datafile=h5,
            scaler=scaler,
            feature_cols=feature_cols,
            error_cols=[None] * len(feature_cols),
            recon_cols=feature_cols[:115],
            data_store=store,
            xp_masking_ratio=0.8,
            m_masking_ratio=0.1,
        )
        wrapper._load_data = MagicMock(side_effect=AssertionError("full load used"))
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        total_loss, rows = wrapper._train_pretrain_key(
            "train", optimizer, 2, 0.0, 0.0, 0
        )

    assert rows == 5
    assert np.isfinite(total_loss)
    wrapper.data_store.close()
