import pytest
import torch
import torch.nn as nn
from rtdl_num_embeddings import PeriodicEmbeddings

from masked_stellar_autoencoder.models.blocks import (
    DenseResnet,
    ResBlock,
    TabDenseEncoder,
)


def test_resblock_activation_elu():
    block = ResBlock(in_features=10, out_features=10, active="elu")
    assert isinstance(block.active, nn.ELU)
    assert block.active.inplace is True


def test_resblock_activation_gelu():
    block = ResBlock(in_features=10, out_features=10, active="gelu")
    assert isinstance(block.active, nn.GELU)


def test_resblock_activation_relu():
    block = ResBlock(in_features=10, out_features=10, active="relu")
    assert isinstance(block.active, nn.ReLU)
    assert block.active.inplace is True


def test_resblock_unsupported_activation():
    with pytest.raises(
        ValueError,
        match="Unsupported activation type: foo. Use 'elu', 'gelu', or 'relu'",
    ):
        ResBlock(in_features=10, out_features=10, active="foo")


def test_resblock_norm_batch():
    block = ResBlock(in_features=10, out_features=10, norm="batch")
    assert isinstance(block.normal, nn.BatchNorm1d)


def test_resblock_norm_layer():
    block = ResBlock(in_features=10, out_features=10, norm="layer")
    assert isinstance(block.normal, nn.LayerNorm)


def test_resblock_unsupported_norm():
    with pytest.raises(
        ValueError, match="Unsupported norm type: foo. Use 'batch' or 'layer'"
    ):
        ResBlock(in_features=10, out_features=10, norm="foo")


@pytest.mark.parametrize("encoder_type", ["resnet", "densenet"])
def test_missing_periodic_inputs_do_not_train_frequencies(encoder_type):
    torch.manual_seed(42)
    encoder = (
        DenseResnet(3, [8], pe=True, d_embedding=4).dense_resnet[0]
        if encoder_type == "resnet"
        else TabDenseEncoder(3, 8, d_embedding=4).pe
    )
    encoder(torch.full((2, 3), -9999.0)).sum().backward()
    assert torch.count_nonzero(encoder.periodic.weight.grad) == 0
    assert torch.count_nonzero(encoder.linear.weight.grad) > 0
    encoder.zero_grad(set_to_none=True)
    encoder(torch.tensor([[-9999.0, 0.25, 1.0]])).sum().backward()
    assert torch.count_nonzero(encoder.periodic.weight.grad[0]) == 0
    assert torch.count_nonzero(encoder.periodic.weight.grad[1:]) > 0


def test_old_periodic_checkpoint_preserves_predictions_and_strict_loading():
    old = PeriodicEmbeddings(3, d_embedding=4, lite=False)
    embedding = DenseResnet(3, [8], pe=True, d_embedding=4).dense_resnet[0]
    embedding.load_state_dict(old.state_dict(), strict=True)
    values = torch.tensor([[-9999.0, 0.25, 1.0], [0.0, -9999.0, -2.0]])
    torch.testing.assert_close(embedding(values), old(values), rtol=0, atol=0)
    broken = old.state_dict()
    del broken["linear.weight"]
    with pytest.raises(RuntimeError, match="linear.weight"):
        embedding.load_state_dict(broken, strict=True)


def test_missing_periodic_encoding_survives_updates_and_checkpoint_roundtrip():
    embedding = DenseResnet(3, [8], pe=True, d_embedding=4).dense_resnet[0]
    missing = torch.full((2, 3), -9999.0)
    expected = embedding(missing).detach().clone()
    with torch.no_grad():
        embedding.periodic.weight.add_(0.00001)
    torch.testing.assert_close(embedding(missing), expected, rtol=0, atol=0)
    resumed = DenseResnet(3, [8], pe=True, d_embedding=4).dense_resnet[0]
    resumed.load_state_dict(embedding.state_dict(), strict=True)
    torch.testing.assert_close(resumed(missing), expected, rtol=0, atol=0)
    observed = torch.tensor([[0.0, 0.25, -2.0]])
    torch.testing.assert_close(
        resumed(observed),
        resumed.activation(resumed.linear(resumed.periodic(observed))),
        rtol=0,
        atol=0,
    )
