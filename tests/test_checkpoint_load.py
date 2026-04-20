"""torch.load compatibility for training checkpoints (NumPy in dict, etc.)."""
import os
import sys

from unittest import mock
import numpy as np
import torch

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo)

from models.checkpoint_load import torch_load_trusted


def test_torch_load_trusted_accepts_numpy_in_dict(tmp_path):
    path = tmp_path / "ckpt_np.pth"
    torch.save({"meta": np.float64(1.5), "w": torch.ones(2)}, str(path))
    out = torch_load_trusted(str(path), map_location="cpu")
    assert isinstance(out["meta"], np.floating)
    assert out["w"].shape == (2,)


def test_torch_load_trusted_state_dict_only(tmp_path):
    path = tmp_path / "ckpt_sd.pth"
    lin = torch.nn.Linear(2, 3)
    torch.save({"autoencoder_state_dict": lin.state_dict()}, str(path))
    out = torch_load_trusted(str(path), map_location="cpu")
    assert "autoencoder_state_dict" in out


def test_torch_load_trusted_with_weights_only():
    # Simulate PyTorch 2.6+ where weights_only is an argument
    mock_sig = mock.MagicMock()
    mock_sig.parameters = {"weights_only": mock.MagicMock()}

    with mock.patch("models.checkpoint_load.inspect.signature", return_value=mock_sig):
        with mock.patch("models.checkpoint_load.torch.load", return_value={"mock": "data"}) as mock_torch_load:
            res = torch_load_trusted("dummy.pth", map_location="cpu")

            assert res == {"mock": "data"}
            mock_torch_load.assert_called_once_with("dummy.pth", map_location="cpu", weights_only=False)


def test_torch_load_trusted_without_weights_only():
    # Simulate older PyTorch where weights_only is not an argument
    mock_sig = mock.MagicMock()
    mock_sig.parameters = {"f": mock.MagicMock()}

    with mock.patch("models.checkpoint_load.inspect.signature", return_value=mock_sig):
        with mock.patch("models.checkpoint_load.torch.load", return_value={"mock": "data"}) as mock_torch_load:
            res = torch_load_trusted("dummy.pth", map_location="cpu")

            assert res == {"mock": "data"}
            mock_torch_load.assert_called_once_with("dummy.pth", map_location="cpu")
