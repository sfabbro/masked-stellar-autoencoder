"""Production input semantics, independent RNG and physical noise-scale checks."""

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.experiment_augmentation import (
    aggregate_point_ensemble,
    canonical_xp_columns,
    supervised_augmentation,
)
from masked_stellar_autoencoder.training.input_measurement import standardized_xp_errors


def fixture(rows=32):
    names = ["G", *[f"{band}_{i}" for band in ("bp", "rp") for i in range(1, 56)], "RA"]
    values = torch.linspace(-1, 1, rows * len(names)).reshape(rows, len(names))
    visible = torch.ones_like(values, dtype=torch.bool)
    visible[:, -1] = False
    values[:, -1] = -9999
    return values, visible, canonical_xp_columns(names)


def generator(seed=42):
    return torch.Generator().manual_seed(seed)


def test_zero_policy_is_bitwise_identity_and_does_not_mutate_or_share_storage():
    values, visible, columns = fixture()
    values[0, 1] = -0.0
    original = values.clone()
    output, observed, audit = supervised_augmentation(
        values, visible, columns, generator=generator()
    )
    assert output.numpy().tobytes() == original.numpy().tobytes()
    assert values.numpy().tobytes() == original.numpy().tobytes()
    assert output.data_ptr() != values.data_ptr()
    assert observed.data_ptr() != visible.data_ptr()
    assert torch.equal(observed, visible)
    assert audit["hidden_xp"] == 0


def test_per_star_noise_is_reproducible_and_keeps_missing_and_non_xp_unchanged():
    values, visible, columns = fixture()
    visible[0, 3] = False
    values[0, 3] = -9999
    sigma = torch.full((len(values), 110), 0.2)
    sigma[0, 2] = float("nan")  # invalid unobserved error never restores a value
    before = torch.random.get_rng_state().clone()
    a, mask_a, _ = supervised_augmentation(
        values, visible, columns, sigma_xp=sigma, noise_scale=1, generator=generator()
    )
    b, mask_b, _ = supervised_augmentation(
        values, visible, columns, sigma_xp=sigma, noise_scale=1, generator=generator()
    )
    assert torch.equal(a, b) and torch.equal(mask_a, mask_b)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert a[0, 3] == -9999
    assert torch.equal(a[:, [0, -1]], values[:, [0, -1]])
    assert torch.equal(mask_a, visible)


def test_mask_probability_one_hides_only_visible_xp_and_blocks_its_gradient():
    values, visible, columns = fixture()
    values.requires_grad_()
    output, observed, audit = supervised_augmentation(
        values, visible, columns, mask_probability=1, generator=generator()
    )
    assert (output[:, columns] == -9999).all()
    assert not observed[:, columns].any()
    assert torch.equal(output[:, [0, -1]], values[:, [0, -1]])
    output.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert (values.grad[:, columns] == 0).all()
    assert (values.grad[:, 0] == 1).all()
    assert audit["hidden_xp"] == len(values) * 110


def test_structured_band_mask_selects_one_whole_band_per_row():
    values, visible, columns = fixture()
    output, observed, _ = supervised_augmentation(
        values,
        visible,
        columns,
        mask_probability=1,
        mask_mode="band",
        generator=generator(),
    )
    bp, rp = observed[:, columns[:55]], observed[:, columns[55:]]
    assert torch.all(bp.all(1) ^ rp.all(1))
    assert torch.all(bp.any(1) ^ rp.any(1))
    assert torch.equal(output[:, [0, -1]], values[:, [0, -1]])


def test_noise_units_and_empirical_variance_match_per_star_reported_sigma():
    values, visible, columns = fixture(rows=4096)
    raw = np.ones(values.shape, dtype="f8") * 0.8
    raw_scale = np.ones(values.shape[1], dtype="f8") * 2
    g = np.where(np.arange(len(values)) % 2 == 0, 8.5, 6.0)
    sigma_np = standardized_xp_errors(
        raw, raw_scale, np.asarray(columns), g_magnitude=g, error_convention="raw_xp"
    )
    np.testing.assert_allclose(sigma_np[::2], 0.4)
    np.testing.assert_allclose(sigma_np[1::2], 0.04)
    sigma = torch.tensor(sigma_np, dtype=values.dtype)
    output, _, _ = supervised_augmentation(
        values, visible, columns, sigma_xp=sigma, noise_scale=0.5, generator=generator()
    )
    standardized_noise = (output[:, columns] - values[:, columns]) / (0.5 * sigma)
    assert abs(float(standardized_noise.mean())) < 0.01
    assert abs(float(standardized_noise.std()) - 1) < 0.01


def test_policy_variants_consume_paired_independent_augmentation_draws():
    values, visible, columns = fixture()
    generators = [generator(), generator(), generator()]
    for gen, noise, mask in zip(generators, [0, 1, 1], [0, 0, 0.1], strict=True):
        supervised_augmentation(
            values,
            visible,
            columns,
            sigma_xp=torch.ones(len(values), 110),
            noise_scale=noise,
            mask_probability=mask,
            generator=gen,
        )
    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    assert torch.equal(generators[1].get_state(), generators[2].get_state())


@pytest.mark.parametrize(
    "options",
    [
        {"noise_scale": -1},
        {"noise_scale": float("nan")},
        {"mask_probability": 1.1},
        {"mask_probability": -0.1},
        {"mask_mode": "arbitrary"},
    ],
)
def test_invalid_policies_are_rejected(options):
    values, visible, columns = fixture()
    with pytest.raises(ValueError):
        supervised_augmentation(
            values, visible, columns, generator=generator(), **options
        )


def test_visible_mask_and_noise_alignment_are_explicit():
    values, visible, columns = fixture()
    with pytest.raises(ValueError, match="110"):
        supervised_augmentation(values, visible, columns[:-1], generator=generator())
    with pytest.raises(ValueError, match="Per-star"):
        supervised_augmentation(
            values,
            visible,
            columns,
            sigma_xp=torch.ones(110),
            noise_scale=1,
            generator=generator(),
        )
    with pytest.raises(ValueError, match="positive"):
        supervised_augmentation(
            values,
            visible,
            columns,
            sigma_xp=torch.zeros(len(values), 110),
            noise_scale=1,
            generator=generator(),
        )
    values[0, -1] = 2
    with pytest.raises(ValueError, match="Invisible"):
        supervised_augmentation(values, visible, columns, generator=generator())


def test_ensemble_uses_source_id_pairing_float64_and_uncalibrated_member_spread():
    predictions = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    ids = torch.tensor([[9007199254740993, 9007199254740995]] * 3)
    mean, spread = aggregate_point_ensemble(predictions, ids)
    torch.testing.assert_close(mean, torch.tensor([2.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(
        spread, torch.tensor([np.sqrt(2 / 3), np.sqrt(8 / 3)], dtype=torch.float64)
    )
    bad_ids = ids.clone()
    bad_ids[1] = bad_ids[1].flip(0)
    with pytest.raises(ValueError, match="order"):
        aggregate_point_ensemble(predictions, bad_ids)
    with pytest.raises(ValueError, match="integer"):
        aggregate_point_ensemble(predictions, ids.double())
