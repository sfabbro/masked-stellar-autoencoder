from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
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


def test_projected_store_stream_and_cache_match_with_partial_batch(tmp_path, capsys):
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
    logs = capsys.readouterr().out
    assert "Pretraining scan starting:" in logs
    assert "Pretraining scan progress: key=train" in logs


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


def test_microbatch_accumulation_matches_full_batch_without_full_shard_load(tmp_path):
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

        def train_once(micro_batch_size):
            np.random.seed(31)
            torch.manual_seed(37)
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
                micro_batch_size=micro_batch_size,
            )
            wrapper._load_data = MagicMock(side_effect=AssertionError("full load used"))
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            total_loss, rows = wrapper._train_pretrain_key(
                "train", optimizer, 2, 0.0, 0.0, 0
            )
            return total_loss, rows, model.weight.detach().clone()

        full_loss, full_rows, full_weight = train_once(2)
        micro_loss, micro_rows, micro_weight = train_once(1)

    assert full_rows == micro_rows == 5
    assert np.isfinite(full_loss)
    assert np.isfinite(micro_loss)
    torch.testing.assert_close(micro_weight, full_weight, rtol=1e-5, atol=1e-6)
    store.close()


def test_microbatch_accumulation_rejects_batchnorm(tmp_path):
    from masked_stellar_autoencoder.models.model import TabResnetWrapper

    class TinyBatchNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(120)
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, values):
            return self.norm(values)[:, :115] * self.weight, values[:, :1]

    feature_cols = [f"feature_{i}" for i in range(120)]
    dtype = np.dtype([(name, "f4") for name in feature_cols])
    values = np.zeros(2, dtype=dtype)
    with h5py.File(tmp_path / "input.h5", "w") as h5:
        h5.create_dataset("train", data=values)
        store = ProjectedHDF5Store(
            h5,
            feature_cols,
            [None] * len(feature_cols),
            scratch_dir=tmp_path,
            cache_fraction=0.0,
        )
        store.prepare(["train"], ["train"], scaler_max_rows=2)
        features = np.zeros((2, len(feature_cols)), dtype=np.float32)
        wrapper = TabResnetWrapper(
            model=TinyBatchNorm(),
            datafile=h5,
            scaler=RobustScaler().fit(features),
            feature_cols=feature_cols,
            error_cols=[None] * len(feature_cols),
            recon_cols=feature_cols[:115],
            data_store=store,
            micro_batch_size=1,
        )
        optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="changes BatchNorm statistics"):
            wrapper._train_pretrain_key("train", optimizer, 2, 0.0, 0.0, 0)
        store.close()
