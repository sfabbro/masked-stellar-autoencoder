import numpy as np
import pytest

from masked_stellar_autoencoder.training.fullfit_records import (
    epoch_record_batches,
    fixed_coordinates,
)


def test_one_epoch_presents_every_record_once_with_empty_source_and_short_last_batch():
    batches = list(epoch_record_batches([3, 0, 5], 3, np.random.default_rng(7)))
    actual = [
        (int(s), int(r))
        for sources, rows in batches
        for s, r in zip(sources, rows, strict=True)
    ]
    assert sorted(actual) == [(0, 0), (0, 1), (0, 2)] + [(2, r) for r in range(5)]
    assert len(batches[-1][0]) == 2


def test_fixed_normalization_never_refits_on_new_survey_values_and_rejects_teacher_widths():
    value = np.array([[5000, 2, -1, 0.2, 10, 2], [6000, 4, 0, 0.1, -1, np.nan]])
    observed = np.isfinite(value)
    known = np.zeros_like(observed)
    known[:, :3] = True
    error = np.ones_like(value) * 0.1
    center = np.array([5000, 3, -0.5, 0, 1, 0])
    scale = np.array([1000, 1, 1, 1, 1, 1])
    y, sigma, valid, k = fixed_coordinates(value, error, observed, known, center, scale)
    np.testing.assert_allclose(y[:, 0], [0, 1])
    assert y[0, 4] == pytest.approx(np.log(10) - 1)
    assert not valid[1, 4] and not valid[1, 5]
    assert (sigma[:, 4:] == 0).all()
    known[0, 4] = True
    with pytest.raises(ValueError, match="posterior"):
        fixed_coordinates(value, error, observed, known, center, scale)


def test_reconstruction_blocks_cover_cache_and_replay_independent_rng():
    from masked_stellar_autoencoder.training.fullfit_records import (
        reconstruction_block_start,
    )

    rng = np.random.default_rng(1901)
    state = rng.bit_generator.state
    starts = [reconstruction_block_start(150000000, 256, rng) for _ in range(100)]
    assert min(starts) >= 0 and max(starts) <= 150000000 - 256
    assert any(x > 140000000 for x in starts)
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    assert starts == [
        reconstruction_block_start(150000000, 256, restored) for _ in range(100)
    ]
