"""Check screen budget/parent binding and strict feature/state reload."""

import sys
from pathlib import Path

import torch
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from launch_stellar_improvement import bind_prediction_policy
from run_frozen_head_screen import flow_features, save_host_npz, stream, train_head


def test_screen_binding_never_uses_legacy_parent():
    suite = yaml.safe_load(Path("configs/frozen-head-screen.yaml").read_text())
    for job in suite["jobs"]:
        result = bind_prediction_policy(
            dict(task="frozen-head-screen", run_id=job["run_id"]), {}
        )
        assert result["MSA_PRETRAIN_RESUME"] == suite["checkpoint"]
        assert result["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert sum(len(x) for x in stream(11, 5, 7)) == 35


def test_frozen_smoke_nig_refresh_and_all_method_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.randn(32, 256)
    y = torch.linspace(-3, 2, 32)
    test = torch.randn(9, 256)
    policy = dict(head_lr=0.001, batch_size=32, head_updates=3000)
    for mode in ["raw_mse", "unweighted_nig", "density_nig", "vir"]:
        result = train_head(x, y, test, mode, policy, tmp_path, True)
        assert result["updates"] == 4 and result["strict_disk_reload_all_test_exact"]


def test_cnf_fitonly_projection_and_full_feature_disk_reload(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    x = torch.randn(32, 256)
    test = torch.randn(7, 256)
    y = torch.cat((torch.full((8,), -1.5), torch.zeros(24)))
    for mode in ["nll", "cnf"]:
        a, b = flow_features(
            x,
            test,
            y,
            mode,
            dict(pca_dimensions=8, cnf_cut_dex=-1.0, flow_lr=0.0003, flow_updates=1000),
            tmp_path,
            True,
        )
        assert a.shape == (32, 258) and b.shape == (7, 258)
        torch.testing.assert_close(a[:, -2:].mean(0), torch.zeros(2), atol=1e-5, rtol=0)


def test_numpy_serialization_never_implicitly_coerces_device_tensor(
    tmp_path, monkeypatch
):
    import numpy as np

    def reject_implicit_array(self, dtype=None):
        raise TypeError("device tensor cannot implicitly convert to numpy")

    monkeypatch.setattr(torch.Tensor, "__array__", reject_implicit_array)
    values = torch.arange(6.0).reshape(2, 3).requires_grad_()
    save_host_npz(tmp_path / "host.npz", values=values, scale=np.ones(3))
    with np.load(tmp_path / "host.npz") as saved:
        np.testing.assert_array_equal(saved["values"], [[0, 1, 2], [3, 4, 5]])
