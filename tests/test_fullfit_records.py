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


def test_whole_xp_dropout_uses_the_missing_token_and_never_mutates_inputs():
    from masked_stellar_autoencoder.training.fullfit_records import (
        xp_dropout_inputs,
        xp_input_columns,
    )

    names = ["W1", "G"] + [f"{a}_{j}" for a in ("bp", "rp") for j in range(1, 56)]
    names += ["PARALLAX"]
    cols = xp_input_columns(names)
    x = np.arange(4 * len(names), dtype="f4").reshape(4, len(names))
    original = x.copy()
    same = xp_dropout_inputs(x, cols)
    assert same is x
    hidden = xp_dropout_inputs(x, cols, hide_all=True)
    assert (hidden[:, cols] == -9999).all()
    np.testing.assert_array_equal(hidden[:, [0, 1, 112]], x[:, [0, 1, 112]])
    np.testing.assert_array_equal(x, original)
    a = xp_dropout_inputs(np.tile(x, (250, 1)), cols, rate=0.3, seed=[1, 2, 3])
    b = xp_dropout_inputs(np.tile(x, (250, 1)), cols, rate=0.3, seed=[1, 2, 3])
    np.testing.assert_array_equal(a, b)  # deterministic for resume
    rows = (a[:, cols] == -9999).all(1)
    assert 0.2 < rows.mean() < 0.4
    assert ((a[:, cols] == -9999).any(1) == rows).all()  # whole-XP, never partial


def test_noxp_view_matches_real_no_xp_gaia_inputs():
    from masked_stellar_autoencoder.training.fullfit_records import (
        noxp_view_spec,
        xp_dropout_inputs,
    )

    names = ["W1", "W2", "G", "BP", "RP"]
    names += [f"{a}_{j}" for a in ("bp", "rp") for j in range(1, 56)]
    names += ["J", "G_PS1", "PARALLAX", "EBV", "pmra", "pmdec", "RA", "DEC"]
    view = noxp_view_spec(names, bprp_drop=0.5, astrometry_drop=0.5)
    x = np.ones((4000, len(names)), dtype="f4")
    out = xp_dropout_inputs(x, view["xp"], hide_all=True, seed=[7], view=view)
    keep = {"G", "BP", "RP", "PARALLAX", "pmra", "pmdec", "RA", "DEC"}
    for j, n in enumerate(names):
        if n not in keep:
            assert (out[:, j] == -9999).all(), n  # XP and non-Gaia inputs hidden
    assert (out[:, names.index("G")] == 1).all()  # G, RA, DEC always kept
    assert (out[:, names.index("RA")] == 1).all()
    bp, rp = names.index("BP"), names.index("RP")
    assert ((out[:, bp] == -9999) == (out[:, rp] == -9999)).all()  # BP/RP together
    assert 0.45 < (out[:, bp] == -9999).mean() < 0.55
    plx = names.index("PARALLAX")
    assert 0.45 < (out[:, plx] == -9999).mean() < 0.55
    again = xp_dropout_inputs(x, view["xp"], hide_all=True, seed=[7], view=view)
    np.testing.assert_array_equal(out, again)  # reproducible views
    partial = xp_dropout_inputs(x, view["xp"], rate=0.3, seed=[1], view=view)
    hidden = (partial[:, view["xp"]] == -9999).all(1)
    assert (partial[~hidden] == 1).all()  # untouched rows keep every input
