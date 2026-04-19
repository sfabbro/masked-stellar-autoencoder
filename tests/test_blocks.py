import pytest

torch = pytest.importorskip("torch")
from torch import nn

from models.blocks import ResBlock, DenseResnet, TabResnetEncoder, TabResnet


def test_resblock_shape():
    batch_size = 4
    in_features = 10
    out_features = 20
    x = torch.randn(batch_size, in_features)

    # Test with change in dimensions
    model = ResBlock(in_features, out_features)
    out = model(x)
    assert out.shape == (batch_size, out_features)

    # Test with same dimensions
    model_same = ResBlock(in_features, in_features)
    out_same = model_same(x)
    assert out_same.shape == (batch_size, in_features)


def test_resblock_activations_and_norms():
    batch_size = 4
    in_features = 10
    x = torch.randn(batch_size, in_features)

    for norm in ['batch', 'layer']:
        for activ in ['elu', 'gelu', 'relu']:
            model = ResBlock(in_features, in_features, norm=norm, activ=activ)
            out = model(x)
            assert out.shape == (batch_size, in_features)

    # Test invalid inputs
    with pytest.raises(ValueError, match="Unsupported norm type"):
        ResBlock(in_features, in_features, norm="invalid")

    with pytest.raises(ValueError, match="Unsupported activation type"):
        ResBlock(in_features, in_features, activ="invalid")


def test_dense_resnet_shape():
    batch_size = 4
    input_dim = 15
    blocks_dims = [32, 64]
    x = torch.randn(batch_size, input_dim)

    # Test without periodic embeddings
    model = DenseResnet(input_dim, blocks_dims, pe=False)
    out = model(x)
    assert out.shape == (batch_size, blocks_dims[-1])

    # Test with periodic embeddings
    model_pe = DenseResnet(input_dim, blocks_dims, pe=True, d_embedding=8)
    out_pe = model_pe(x)
    assert out_pe.shape == (batch_size, blocks_dims[-1])


def test_tabresnet_encoder_shape():
    batch_size = 4
    continuous_cols = 10
    blocks_dims = [16, 32]
    x = torch.randn(batch_size, continuous_cols)

    model = TabResnetEncoder(continuous_cols, blocks_dims)
    out = model(x)
    assert out.shape == (batch_size, blocks_dims[-1])


def test_tabresnet_shape():
    batch_size = 4
    continuous_cols = 10
    blocks_dims = [16, 32]
    x = torch.randn(batch_size, continuous_cols)

    # Test symmetric decoder
    model = TabResnet(continuous_cols, blocks_dims)
    out, encoded = model(x)
    assert encoded.shape == (batch_size, blocks_dims[-1])
    assert out.shape == (batch_size, continuous_cols)

    # Test asymmetric decoder
    decoder_dims = [16, 8]
    output_cols = 5
    model_asym = TabResnet(continuous_cols, blocks_dims, output_cols=output_cols, decoder_dims=decoder_dims)
    out_asym, encoded_asym = model_asym(x)
    assert encoded_asym.shape == (batch_size, blocks_dims[-1])
    assert out_asym.shape == (batch_size, output_cols)
