import pytest

torch = pytest.importorskip("torch")

from masked_stellar_autoencoder.models.model import MaskedGaussianNLLLoss


def test_masked_gaussian_nll_loss_negative_float_target_var():
    loss_fn = MaskedGaussianNLLLoss()
    pred_mean = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    pred_var = torch.tensor([0.1, 0.1])
    target_var = -1.0  # float with negative value

    with pytest.raises(ValueError, match="var has negative entry/entries"):
        loss_fn(pred_mean, target, pred_var, target_var)


def test_masked_gaussian_nll_loss_negative_tensor_target_var():
    loss_fn = MaskedGaussianNLLLoss()
    pred_mean = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    pred_var = torch.tensor([0.1, 0.1])
    target_var = torch.tensor([0.1, -0.2])  # tensor with negative value

    with pytest.raises(ValueError, match="var has negative entry/entries"):
        loss_fn(pred_mean, target, pred_var, target_var)
