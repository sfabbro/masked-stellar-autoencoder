"""Versioned rescoring must never accept unbound or misidentified inputs."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rescore_density_checkpoint as r


def policy_fixture(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    parent = tmp_path / "parent.pth"
    config = tmp_path / "model.yaml"
    parent.write_bytes(b"parent")
    config.write_bytes(b"config")
    (directory / "checkpoint.pth").write_bytes(b"checkpoint")
    (directory / "paired-cohort.npz").write_bytes(b"paired")
    checkpoint = r.digest(directory / "checkpoint.pth")
    old = dict(
        run_id="run",
        status="complete",
        source_sha="a" * 40,
        policy=dict(library_revision="b" * 40),
        checkpoint_sha256=checkpoint,
        arm=dict(kind="flow", flow_type="nsf"),
    )
    (directory / "manifest.json").write_text(json.dumps(old))
    p = dict(
        run_id="run",
        source_sha="a" * 40,
        library_revision="b" * 40,
        checkpoint_sha256=checkpoint,
        head_kwargs=dict(kind="flow", flow_type="nsf"),
        paired_sha256=r.digest(directory / "paired-cohort.npz"),
        manifest_sha256=r.digest(directory / "manifest.json"),
        parent_sha256=r.digest(parent),
        model_config_sha256=r.digest(config),
        scoring_code_sha256=r.digest(r.scoring.__file__),
        head_code_sha256=r.digest(r.heads.__file__),
        architecture_code_sha256=r.digest(r.architecture.__file__),
    )
    return p, directory, parent, config


def test_all_bound_inputs_accepted(tmp_path):
    p, *args = policy_fixture(tmp_path)
    assert r.validate_policy(p, *args, installed_revision="b" * 40)["run_id"] == "run"


@pytest.mark.parametrize(
    "key",
    [
        "paired_sha256",
        "manifest_sha256",
        "checkpoint_sha256",
        "parent_sha256",
        "model_config_sha256",
        "scoring_code_sha256",
        "head_code_sha256",
        "architecture_code_sha256",
        "run_id",
        "source_sha",
        "library_revision",
    ],
)
def test_tamper_rejected(tmp_path, key):
    p, *args = policy_fixture(tmp_path)
    p[key] = "0" * 64 if key.endswith("_sha256") else "wrong"
    with pytest.raises(ValueError):
        r.validate_policy(p, *args, installed_revision="b" * 40)


def test_missing_pin_and_head_semantics_rejected(tmp_path):
    p, *args = policy_fixture(tmp_path)
    p["head_kwargs"]["noise_aware"] = True
    with pytest.raises(ValueError, match="head policy"):
        r.validate_policy(p, *args, installed_revision="b" * 40)
    del p["paired_sha256"]
    with pytest.raises(ValueError, match="missing"):
        r.validate_policy(p, *args, installed_revision="b" * 40)


def test_roles_preserve_order_and_reject_overlap():
    data = dict(
        source_id=np.array([101, 102, 103]),
        test_rows=np.array([2, 0]),
        calibration_rows=np.array([1]),
    )
    np.testing.assert_array_equal(r.validate_roles(data)["test"], [2, 0])
    data["calibration_rows"] = np.array([2])
    with pytest.raises(ValueError, match="overlap"):
        r.validate_roles(data)


def test_scoring_orchestration_full_roles_distinct_output(tmp_path, monkeypatch):
    p, directory, parent, config = policy_fixture(tmp_path)
    np.savez(
        directory / "paired-cohort.npz",
        source_id=np.array([101, 102, 103]),
        test_rows=np.array([2, 0]),
        calibration_rows=np.array([1]),
    )
    config.write_text("{}")
    p["paired_sha256"] = r.digest(directory / "paired-cohort.npz")
    p["model_config_sha256"] = r.digest(config)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(p))
    originals = {path.name: r.digest(path) for path in directory.iterdir()}

    class Distribution:
        def read_text(self, name):
            return json.dumps({"vcs_info": {"commit_id": "b" * 40}})

    class Module:
        def load_state_dict(self, state, strict):
            assert strict

    monkeypatch.setattr(
        r.importlib.metadata, "distribution", lambda name: Distribution()
    )
    monkeypatch.setattr(
        r,
        "torch_load_trusted",
        lambda *a, **kw: dict(model_state_dict={}, head_state_dict={}),
    )
    monkeypatch.setattr(r.architecture, "load_model", lambda *a: Module())
    monkeypatch.setattr(r.heads, "MultiLabelHead", lambda **kw: Module())
    seen = []

    def score(model, head, data, rows, device):
        seen.append(rows.tolist())
        return dict(
            mean=np.zeros((len(rows), 3), dtype="f8"),
            intrinsic_log_prob_native=np.zeros(len(rows), dtype="f8"),
        )

    monkeypatch.setattr(r.scoring, "predictions", score)
    monkeypatch.setattr(r.subprocess, "check_output", lambda *a, **kw: "new-source\n")
    output = tmp_path / "corrected"
    argv = [
        "--policy",
        str(policy_path),
        "--directory",
        str(directory),
        "--parent",
        str(parent),
        "--model-config",
        str(config),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    r.main(argv)
    assert seen == [[2, 0], [1]]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_updates"] == 0
    assert manifest["original_failures_cleared"] is False
    assert manifest["independent_verification"] == "pending"
    with np.load(output / "predictions-fp64.npz") as arrays:
        np.testing.assert_array_equal(arrays["test_source_id"], [103, 101])
    assert originals == {path.name: r.digest(path) for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        r.main(argv)


def test_explicit_original_library_separate_from_scoring_revision(tmp_path):
    p, directory, parent, config = policy_fixture(tmp_path)
    old = json.loads((directory / "manifest.json").read_text())
    old["policy"]["library_revision"] = "c" * 40
    (directory / "manifest.json").write_text(json.dumps(old))
    p["manifest_sha256"] = r.digest(directory / "manifest.json")
    with pytest.raises(ValueError, match="original library"):
        r.validate_policy(p, directory, parent, config, installed_revision="b" * 40)
    p["original_library_revision"] = "c" * 40
    r.validate_policy(p, directory, parent, config, installed_revision="b" * 40)
