"""Synthetic-only checks; no stellar checkpoint or native data is loaded."""

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.multisurvey_masked_energy import (
    masked_energy_score,
    masked_noise_aware_energy_score,
)


def independent_numpy_reference(samples, targets, masks, scales):
    result = []
    samples, targets, masks, scales = map(np.asarray, (samples, targets, masks, scales))
    count = samples.shape[0]
    for row, mask in enumerate(masks):
        values = samples[:, row, :][:, mask] / scales[mask]
        truth = targets[row, mask] / scales[mask]
        first = sum(np.linalg.norm(x - truth) for x in values) / count
        pair_sum = sum(
            np.linalg.norm(values[i] - values[j])
            for i in range(count)
            for j in range(count)
            if i != j
        )
        result.append(first - pair_sum / (2 * count * (count - 1)))
    return np.asarray(result)


def example():
    generator = torch.Generator().manual_seed(82)
    samples = torch.randn((6, 4, 6), generator=generator, dtype=torch.float64)
    target = torch.randn((4, 6), generator=generator, dtype=torch.float64)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    target[~mask] = float("nan")
    scale = torch.tensor([2.0, 0.5, 1.3, 1.0, 0.75, 2.5], dtype=torch.float64)
    return samples, target, mask, scale


def test_grouped_mask_score_matches_independent_formula_and_weight_reductions():
    samples, target, mask, scale = example()
    expected = independent_numpy_reference(samples, target, mask, scale)
    got = masked_energy_score(samples, target, mask, scale=scale, reduction="none")
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-12, atol=1e-12)
    weights = torch.tensor([0.5, 2.0, 1.2, 0.7], dtype=torch.float64)
    weighted = masked_energy_score(
        samples,
        target,
        mask,
        scale=scale,
        weights=weights,
        reduction="none",
        chunk_size=1,
    )
    torch.testing.assert_close(weighted, got * weights)
    total = masked_energy_score(
        samples, target, mask, scale=scale, weights=weights, reduction="sum"
    )
    mean = masked_energy_score(samples, target, mask, scale=scale, weights=weights)
    torch.testing.assert_close(total, weighted.sum())
    torch.testing.assert_close(mean, weighted.sum() / weights.sum())


def test_full_mask_matches_torchregress_energy_score():
    from torchregress.metrics.distribution import energy_score

    samples, target, _, scale = example()
    target = torch.nan_to_num(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    expected = energy_score(samples / scale, target / scale, reduction="none")
    actual = masked_energy_score(samples, target, mask, scale=scale, reduction="none")
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_noisy_one_dimensional_score_matches_normal_crps():
    from scipy.special import ndtr

    sample_count = 2048
    draws = 0.3 + 1.2 * torch.randn(
        (sample_count, 1, 1),
        generator=torch.Generator().manual_seed(740),
        dtype=torch.float64,
    )
    target = torch.tensor([[0.7]], dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    score = masked_noise_aware_energy_score(
        draws,
        target,
        mask,
        torch.tensor([[0.4]], dtype=torch.float64),
        mask,
        scale=[1.0],
        generator=torch.Generator().manual_seed(741),
        assume_independent_errors=True,
    )
    sigma = np.sqrt(1.2**2 + 0.4**2)
    z = (0.7 - 0.3) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    expected = sigma * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    assert abs(float(score) - expected) < 0.04


def test_all_63_nonempty_six_dimensional_masks_and_coordinate_permutation():
    torch.manual_seed(3)
    masks = torch.tensor(
        [[bool(value & (1 << axis)) for axis in range(6)] for value in range(1, 64)]
    )
    samples = torch.randn((5, 63, 6), dtype=torch.float64)
    target = torch.randn((63, 6), dtype=torch.float64)
    scale = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64)
    one = masked_energy_score(
        samples, target, masks, scale=scale, reduction="none", chunk_size=1
    )
    order = torch.tensor([5, 1, 3, 0, 4, 2])
    two = masked_energy_score(
        samples[:, :, order],
        target[:, order],
        masks[:, order],
        scale=scale[order],
        reduction="none",
        chunk_size=7,
    )
    torch.testing.assert_close(one, two, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(one).all()


def test_noise_uses_seeded_independent_gaussians_and_explicit_unknown_floor():
    samples, target, mask, scale = example()
    known = mask.clone()
    known[0, 2] = False
    error = torch.where(known, torch.full_like(target, 0.2), float("nan"))
    floor = torch.tensor([0.1, 0.1, 0.3, 0.1, 0.1, 0.1], dtype=torch.float64)
    kwargs = dict(
        scale=scale,
        unknown_error_floor=floor,
        assume_independent_errors=True,
        reduction="none",
    )
    first = masked_noise_aware_energy_score(
        samples,
        target,
        mask,
        error,
        known,
        generator=torch.Generator().manual_seed(123),
        **kwargs,
    )
    again = masked_noise_aware_energy_score(
        samples,
        target,
        mask,
        error,
        known,
        generator=torch.Generator().manual_seed(123),
        **kwargs,
    )
    other = masked_noise_aware_energy_score(
        samples,
        target,
        mask,
        error,
        known,
        generator=torch.Generator().manual_seed(124),
        **kwargs,
    )
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    assert not torch.equal(first, other)
    sigma = torch.where(mask, torch.where(known, error, floor), 0.0)
    noise = torch.randn(
        samples.shape, generator=torch.Generator().manual_seed(123), dtype=samples.dtype
    )
    reference = masked_energy_score(
        samples + noise * sigma.unsqueeze(0),
        target,
        mask,
        scale=scale,
        reduction="none",
    )
    torch.testing.assert_close(first, reference, rtol=0, atol=0)


def test_exact_zero_distances_have_finite_gradients():
    samples = torch.zeros((4, 2, 6), dtype=torch.float64, requires_grad=True)
    target = torch.zeros((2, 6), dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 0]], dtype=torch.bool)
    score = masked_energy_score(
        samples, target, mask, scale=torch.ones(6, dtype=torch.float64)
    )
    assert score.item() == 0.0
    score.backward()
    assert samples.grad is not None and torch.isfinite(samples.grad).all()
    assert not samples.grad.any()


def test_shared_latent_draws_allow_toy_source_map_gradients():
    # A caller constructs each observed coordinate from the same latent draw;
    # this algebraic toy is not a calibrated astrophysical observation map.
    latent = torch.randn(
        (9, 3, 2),
        generator=torch.Generator().manual_seed(231),
        dtype=torch.float64,
        requires_grad=True,
    )
    coefficient = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    mapped = torch.stack(
        (latent[..., 0], latent[..., 1], latent[..., 0] + coefficient * latent[..., 1]),
        dim=-1,
    )
    target = torch.tensor(
        [[0.2, 0.3, 0.4], [0.4, -0.1, 0.2], [0.0, 0.0, -0.3]], dtype=torch.float64
    )
    mask = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=torch.bool)
    unknown = torch.zeros_like(mask)
    loss = masked_noise_aware_energy_score(
        mapped,
        target,
        mask,
        torch.full_like(target, float("nan")),
        unknown,
        scale=[1.0, 1.0, 1.0],
        unknown_error_floor=[0.1, 0.1, 0.1],
        generator=torch.Generator().manual_seed(232),
        assume_independent_errors=True,
    )
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert coefficient.grad is not None and torch.isfinite(coefficient.grad)
    assert coefficient.grad.abs() > 0


def test_off_diagonal_u_statistic_is_used_without_a_clamp():
    samples = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.float64)
    target = torch.tensor([[0.0]], dtype=torch.float64)
    score = masked_energy_score(
        samples, target, torch.ones_like(target, dtype=torch.bool), scale=[1.0]
    )
    assert score.item() == 0.0
    target[0, 0] = -1.0
    score = masked_energy_score(
        samples, target, torch.ones_like(target, dtype=torch.bool), scale=[1.0]
    )
    assert score.item() == 0.0
    samples = torch.tensor([[[-1.0]], [[1.0]], [[1.0]]], dtype=torch.float64)
    target[0, 0] = 0.0
    # Identical first term, off-diagonal statistic exceeds the biased pair mean.
    torch.testing.assert_close(
        masked_energy_score(
            samples, target, torch.ones_like(target, dtype=torch.bool), scale=[1.0]
        ),
        torch.tensor(1.0 / 3.0, dtype=torch.float64),
    )


def test_rejects_missing_unknown_and_untrusted_noise_inputs():
    samples, target, mask, scale = example()
    known = mask.clone()
    known[0, 2] = False
    error = torch.where(known, torch.full_like(target, 0.2), float("nan"))
    kwargs = dict(
        scale=scale,
        generator=torch.Generator().manual_seed(9),
        assume_independent_errors=True,
    )
    with pytest.raises(ValueError, match="Unknown observed errors"):
        masked_noise_aware_energy_score(samples, target, mask, error, known, **kwargs)
    with pytest.raises(ValueError, match="positive floor"):
        masked_noise_aware_energy_score(
            samples, target, mask, error, known, unknown_error_floor=0.0, **kwargs
        )
    with pytest.raises(ValueError, match="numeric, not boolean"):
        masked_noise_aware_energy_score(
            samples, target, mask, error, known, unknown_error_floor=True, **kwargs
        )
    with pytest.raises(ValueError, match="numeric, not boolean"):
        masked_noise_aware_energy_score(
            samples,
            target,
            mask,
            error,
            known,
            unknown_error_floor=[True] * 6,
            **kwargs,
        )
    with pytest.raises(ValueError, match="scalar, \\[D\\] or \\[B,D\\]"):
        masked_noise_aware_energy_score(
            samples,
            target,
            mask,
            error,
            known,
            unknown_error_floor=torch.ones((4, 1)),
            **kwargs,
        )
    with pytest.raises(ValueError, match="independent-error"):
        masked_noise_aware_energy_score(
            samples,
            target,
            mask,
            error,
            known,
            scale=scale,
            generator=torch.Generator(),
            assume_independent_errors=False,
            unknown_error_floor=0.2,
        )
    with pytest.raises(ValueError, match="Generator"):
        masked_noise_aware_energy_score(
            samples,
            target,
            mask,
            error,
            known,
            scale=scale,
            generator=None,
            assume_independent_errors=True,
            unknown_error_floor=0.2,
        )
    bad_error = error.clone()
    bad_error[0, 0] = 0.0
    with pytest.raises(ValueError, match="Known observed errors"):
        masked_noise_aware_energy_score(
            samples, target, mask, bad_error, known, unknown_error_floor=0.2, **kwargs
        )
    bad_known = known.clone()
    bad_known[1, 0] = True
    with pytest.raises(ValueError, match="require observed"):
        masked_noise_aware_energy_score(
            samples, target, mask, error, bad_known, unknown_error_floor=0.2, **kwargs
        )
    bad_mask = mask.clone()
    bad_mask[0] = False
    with pytest.raises(ValueError, match="Every star"):
        masked_energy_score(samples, target, bad_mask, scale=scale)
    bad_target = target.clone()
    bad_target[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite observed"):
        masked_energy_score(samples, bad_target, mask, scale=scale)
    with pytest.raises(ValueError, match="positive per-coordinate scales"):
        masked_energy_score(samples, target, mask, scale=torch.zeros(6))
    with pytest.raises(ValueError, match="weights"):
        masked_energy_score(
            samples,
            target,
            mask,
            scale=scale,
            weights=torch.tensor([1.0, 0.0, 1.0, 1.0]),
        )


def test_joint_six_nsf_has_finite_flow_and_context_gradients_after_startup():
    from masked_stellar_autoencoder.training.remediation_joint6 import JointSixHead

    torch.manual_seed(913)
    head = JointSixHead("nsf", latent_dim=3, width=16, particles=16).double()
    assert head.flow.target_dim == 6
    context = torch.randn((4, 3), dtype=torch.float64, requires_grad=True)
    target = torch.randn((4, 6), dtype=torch.float64)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    sigma = torch.full_like(target, 0.05)
    optimizer = torch.optim.Adam(head.flow.parameters(), lr=0.002)
    context_gradients = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        context.grad = None
        draws = head.sample(
            context, 16, generator=torch.Generator().manual_seed(20 + step)
        )
        assert draws.shape == (16, 4, 6) and draws.requires_grad
        score = masked_noise_aware_energy_score(
            draws,
            target,
            mask,
            sigma,
            mask,
            scale=torch.ones(6, dtype=torch.float64),
            generator=torch.Generator().manual_seed(100 + step),
            assume_independent_errors=True,
        )
        score.backward()
        flow_gradients = [p.grad for p in head.flow.parameters() if p.grad is not None]
        assert flow_gradients and all(torch.isfinite(g).all() for g in flow_gradients)
        assert sum(g.abs().sum().item() for g in flow_gradients) > 0
        assert all(p.grad is None for p in head.gaussian.parameters())
        assert context.grad is not None and torch.isfinite(context.grad).all()
        context_gradients.append(context.grad.abs().sum().item())
        optimizer.step()
    assert context_gradients[1] > 0
