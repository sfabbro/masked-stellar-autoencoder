"""torch.load compatibility for training checkpoints (NumPy in dict, etc.)."""

import inspect
import os
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "masked_stellar_autoencoder"))
sys.path.insert(0, _repo)

from models.checkpoint_load import torch_load_trusted


def test_torch_load_trusted_accepts_numpy_in_dict(tmp_path):
    path = tmp_path / "ckpt_np.pth"
    torch.save({"meta": np.float64(1.5), "w": torch.ones(2)}, str(path))
    out = torch_load_trusted(str(path), map_location="cpu", weights_only=False)
    assert isinstance(out["meta"], np.floating)
    assert out["w"].shape == (2,)


def test_torch_load_trusted_state_dict_only(tmp_path):
    path = tmp_path / "ckpt_sd.pth"
    lin = torch.nn.Linear(2, 3)
    torch.save({"autoencoder_state_dict": lin.state_dict()}, str(path))
    out = torch_load_trusted(str(path), map_location="cpu")
    assert "autoencoder_state_dict" in out


def test_torch_load_trusted_saves_and_loads_tensor(tmp_path):
    path = tmp_path / "dummy.pth"
    dummy_tensor = torch.ones(1)
    torch.save(dummy_tensor, str(path))
    loaded = torch_load_trusted(str(path))
    assert torch.equal(loaded, dummy_tensor)


def test_torch_load_trusted_with_weights_only_support(mocker, tmp_path):
    path = tmp_path / "dummy_weights.pth"
    torch.save({"test": torch.ones(1)}, str(path))

    # Mock inspect.signature to simulate torch.load supporting weights_only
    mock_signature = mocker.patch("inspect.signature")
    mock_sig_obj = mocker.Mock()
    mock_sig_obj.parameters = {"weights_only": mocker.Mock()}
    mock_signature.return_value = mock_sig_obj

    # Spy on torch.load to verify arguments without breaking functionality
    spy_torch_load = mocker.spy(torch, "load")

    out = torch_load_trusted(str(path), map_location="cpu", weights_only=True)

    assert "test" in out
    assert out["test"].shape == (1,)
    spy_torch_load.assert_called_once_with(
        str(path), map_location="cpu", weights_only=True
    )


def test_torch_load_trusted_without_weights_only_support(mocker, tmp_path):
    path = tmp_path / "dummy_no_weights.pth"
    torch.save({"test": torch.ones(1)}, str(path))

    # Mock inspect.signature to simulate torch.load NOT supporting weights_only
    mock_signature = mocker.patch("inspect.signature")
    mock_sig_obj = mocker.Mock()
    mock_sig_obj.parameters = {"f": mocker.Mock()}
    mock_signature.return_value = mock_sig_obj

    # Spy on torch.load to verify arguments without breaking functionality
    spy_torch_load = mocker.spy(torch, "load")

    out = torch_load_trusted(str(path), map_location="cpu", weights_only=True)

    assert "test" in out
    assert out["test"].shape == (1,)
    spy_torch_load.assert_called_once_with(
        str(path), map_location="cpu"
    )


def test_torch_load_trusted_default_weights_only_is_true(mocker, tmp_path):
    path = tmp_path / "dummy_default.pth"
    torch.save({"test": torch.ones(1)}, str(path))

    mock_signature = mocker.patch("inspect.signature")
    mock_sig_obj = mocker.Mock()
    mock_sig_obj.parameters = {"weights_only": mocker.Mock()}
    mock_signature.return_value = mock_sig_obj

    spy_torch_load = mocker.spy(torch, "load")

    out = torch_load_trusted(str(path), map_location="cpu")

    assert "test" in out
    assert out["test"].shape == (1,)
    spy_torch_load.assert_called_once_with(
        str(path), map_location="cpu", weights_only=True
    )
