"""VIR adapter invariants independent of the continuation optimizer."""

import torch

from masked_stellar_autoencoder.training.prediction_vir import VIRPredictionHead


def make(mode="vir"):
    labels = torch.tensor([-3.0, -2.0, -1.0, -0.8, -0.5, 0.0, 0.1, 0.2])
    torch.manual_seed(42)
    return VIRPredictionHead(
        labels, mode=mode, input_dim=4, latent_dim=3, n_bins=5
    ), labels


def test_matched_modes_start_at_same_mean_and_finite_variance():
    x = torch.randn(8, 4)
    for mode in sorted(VIRPredictionHead.MODES):
        head, _ = make(mode)
        out = head(x)
        assert out.shape == (8, 1, 2)
        assert torch.equal(out[:, :, 0], torch.zeros(8, 1))
        assert torch.isfinite(out).all()


def test_objective_has_finite_gradient_in_all_ablations():
    for mode in sorted(VIRPredictionHead.MODES):
        head, labels = make(mode)
        x = torch.randn(8, 4)
        head.refresh_statistics(x, labels)
        objective = head.objective(x, labels)
        objective.backward()
        assert torch.isfinite(objective)
        assert all(torch.isfinite(p.grad).all() for p in head.parameters())


def test_statistics_refresh_and_checkpoint_reload_are_exact():
    head, labels = make()
    x = torch.randn(8, 4)
    with torch.no_grad():
        head.nig_head.weight.normal_(std=0.1)
    head.refresh_statistics(x, labels, chunk_size=3)
    expected = head(x)
    replacement, _ = make()
    replacement.load_state_dict(head.state_dict(), strict=True)
    assert torch.equal(expected, replacement(x))
    assert int(replacement.bin_counts.sum()) == len(labels)
    assert bool(replacement.statistics_ready)


def test_prediction_needs_no_test_labels_and_is_row_permutation_equivariant():
    head, labels = make()
    fit = torch.randn(8, 4)
    head.refresh_statistics(fit, labels)
    test = torch.randn(5, 4)
    permutation = torch.tensor([4, 0, 3, 1, 2])
    torch.testing.assert_close(head(test)[permutation], head(test[permutation]))


def test_kl_uses_raw_gaussian_and_library_dimension_mean_convention():
    head, labels = make()
    x = torch.randn(8, 4)
    head.refresh_statistics(x, labels)
    mean, variance = head.encode(x)
    expected_kl = 0.5 * (variance + mean.square() - 1 - variance.log()).mean()
    # With a zero NIG head, predictive loss cannot depend on the latent sample;
    # subtraction therefore isolates the KL even when recolouring is active.
    raw = torch.zeros(len(labels), 3)
    predictive = head.loss_fn(raw, labels)
    torch.testing.assert_close(
        head.objective(x, labels), predictive + head.kl_weight * expected_kl
    )
