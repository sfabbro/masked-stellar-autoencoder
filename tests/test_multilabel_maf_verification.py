"""Independent MAF mathematics and completed-audit provenance reuse."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


def load_verifier():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/xp/monitor/verify_multilabel_maf.py"
    )
    spec = importlib.util.spec_from_file_location("independent_maf_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def triangular_state():
    state = {}
    for k in range(3):
        prefix = f"loss_fn.flow.transform.transforms.{k}."
        state[prefix + "order"] = torch.tensor([0, 1, 2] if k != 1 else [2, 1, 0])
        for i, inputs, outputs in [(0, 19, 64), (2, 64, 64), (4, 64, 6)]:
            key = prefix + f"hyper.{i}."
            state[key + "weight"] = torch.zeros(outputs, inputs, dtype=torch.float64)
            state[key + "bias"] = torch.zeros(outputs, dtype=torch.float64)
            state[key + "mask"] = torch.ones(outputs, inputs, dtype=torch.bool)
    # First map is exactly (x0, x1+0.4*x0, x2) on the test domain.
    prefix = "loss_fn.flow.transform.transforms.0.hyper."
    state[prefix + "0.weight"][0, 0] = 1
    state[prefix + "0.bias"][0] = 10
    state[prefix + "2.weight"][0, 0] = 1
    state[prefix + "4.weight"][2, 0] = 0.4
    state[prefix + "4.bias"][2] = -4
    return state


def test_independent_maf_triangular_map_density_jacobian():
    ref = load_verifier()
    state = triangular_state()
    x = torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 0.3]], dtype=torch.float64)
    context = torch.zeros(2, 16, dtype=torch.float64)
    z, logdet = ref.transform(x, context, state)
    expected = x.clone()
    expected[:, 1] += 0.4 * x[:, 0]
    torch.testing.assert_close(z, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(logdet, torch.zeros(2, dtype=x.dtype), rtol=0, atol=0)
    torch.testing.assert_close(
        ref.inverse(z, context, state), x, rtol=1e-12, atol=1e-12
    )
    jac = torch.autograd.functional.jacobian(
        lambda v: ref.transform(v[None], context[:1], state)[0][0], x[0]
    )
    torch.testing.assert_close(
        jac,
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=x.dtype
        ),
    )


def test_completed_maf_audit_reuses_exact_provenance_without_replay(
    tmp_path, monkeypatch
):
    ref = load_verifier()
    monkeypatch.setattr(ref, "P", tmp_path)
    manifest = {"status": "complete", "arm": {"kind": "flow", "flow_type": "maf"}}
    files = ["checkpoint.pth", "paired-cohort.npz", "predictions.npz", "conformal.npz"]
    for name in files:
        (tmp_path / name).write_bytes(b"fixture-not-a-model")
    manifest["checkpoint_sha256"] = ref.point.digest(tmp_path / "checkpoint.pth")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    archive = tmp_path / "independent-maf-verifier.py"
    archive.write_text("# exact historical verifier\n")
    bindings = {
        name: ref.point.digest(tmp_path / name) for name in ["manifest.json", *files]
    }
    bindings["verifier"] = ref.point.digest(archive)
    report = {"status": "complete_strict_pass", "bindings": bindings}
    (tmp_path / "independent-maf-report.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("Completed model was loaded/replayed"),
    )
    assert ref.main() == report
    (tmp_path / "predictions.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding differs"):
        ref.main()
