import pytest
import torch
from models.blocks import ResBlock

def test_resblock_invalid_norm():
    """Test that ResBlock raises ValueError for unsupported norm type."""
    with pytest.raises(ValueError, match="Unsupported norm type: invalid. Use 'batch' or 'layer'"):
        ResBlock(in_features=16, out_features=16, norm='invalid')

def test_resblock_invalid_activ():
    """Test that ResBlock raises ValueError for unsupported activation type."""
    with pytest.raises(ValueError, match="Unsupported activation type: invalid. Use 'elu', 'gelu', or 'relu'"):
        ResBlock(in_features=16, out_features=16, activ='invalid')

def test_resblock_valid_instantiation():
    """Test ResBlock instantiates correctly with valid parameters."""
    # Test batch norm (default)
    block_batch = ResBlock(in_features=16, out_features=32, norm='batch', activ='relu')
    assert isinstance(block_batch.normal, torch.nn.BatchNorm1d)
    assert isinstance(block_batch.activ, torch.nn.ReLU)

    # Test layer norm
    block_layer = ResBlock(in_features=16, out_features=32, norm='layer', activ='gelu')
    assert isinstance(block_layer.normal, torch.nn.LayerNorm)
    assert isinstance(block_layer.activ, torch.nn.GELU)

    # Test elu (default)
    block_elu = ResBlock(in_features=16, out_features=32, activ='elu')
    assert isinstance(block_elu.activ, torch.nn.ELU)
