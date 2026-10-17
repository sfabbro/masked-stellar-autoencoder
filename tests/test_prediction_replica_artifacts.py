"""Artifact path invariants for multi-model SIMEX, independent of GPU runtime."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_prediction_comparison import replica_directory  # noqa: E402


def test_single_replica_preserves_flat_control_layout(tmp_path):
    actual = replica_directory(tmp_path, {"lambda_value": 0.0, "simulation": 0}, 0, 1)
    assert actual == tmp_path
    assert not list(tmp_path.iterdir())


def test_simex_all17_models_have_distinct_nonoverwriting_directories(tmp_path):
    (tmp_path / "initial_reconstruction.npz").write_bytes(
        b"common actual initial arrays"
    )
    (tmp_path / "dropout_policy.json").write_text('{"dropout": "preserved"}')
    (tmp_path / "optimizer_policy.json").write_text('{"parent_moments": true}')
    replicas = [
        {"lambda_value": lam, "simulation": sim}
        for lam in [0.0, 0.5, 1.0, 1.5, 2.0]
        for sim in range(1 if lam == 0 else 4)
    ]
    outputs = []
    for index, replica in enumerate(replicas):
        output = replica_directory(tmp_path, replica, index, len(replicas))
        outputs.append(output)
        for name in [
            "initial_reconstruction.npz",
            "dropout_policy.json",
            "optimizer_policy.json",
        ]:
            assert (output / name).read_bytes() == (tmp_path / name).read_bytes()
        # Each checkpoint really occupies a distinct directory; no last-model
        # root checkpoint can accidentally stand in for the SIMEX aggregate.
        (output / "checkpoint.pth").write_text(f"actual replica placeholder {index}")
    assert len(outputs) == len(set(outputs)) == 17
    assert not (tmp_path / "checkpoint.pth").exists()
    for index, output in enumerate(outputs):
        assert (
            output / "checkpoint.pth"
        ).read_text() == f"actual replica placeholder {index}"
    with pytest.raises(FileExistsError):
        replica_directory(tmp_path, replicas[0], 0, len(replicas))
