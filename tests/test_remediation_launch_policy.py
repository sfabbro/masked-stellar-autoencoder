import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import remediation_launch_policy as policy


@pytest.mark.parametrize(
    "task,gpu",
    [
        ("remediation-input-audit", 0),
        ("remediation-source", 0),
        ("remediation-joint-preflight", 1),
        ("remediation-pilot", 1),
    ],
)
def test_remediation_explicit_config_resources_and_pin(
    tmp_path, monkeypatch, task, gpu
):
    f = tmp_path / "policy.yaml"
    f.write_text(
        yaml.safe_dump(
            dict(
                run_id="test",
                arms=[dict(run_id="test")],
                library_revision=policy.LIBRARY,
            )
        )
    )
    monkeypatch.setitem(policy.CONFIGS, task, f)
    plan = dict(
        run_id="test",
        task=task,
        resources=dict(gpu=gpu),
        image="images.canfar.net/astroai/terminal:latest",
    )
    env = policy.bind_remediation_policy(plan, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert env["MSA_REMEDIATION_POLICY"] == str(f)
    for bad in [
        dict(run_id="unknown"),
        dict(resources=dict(gpu=1 - gpu)),
        dict(image="private/image"),
        dict(env=dict(MSA_EXPERIMENT_PRESENTATIONS="450000000")),
    ]:
        with pytest.raises(ValueError):
            policy.bind_remediation_policy(dict(plan, **bad), {})


def test_project_storage_gate_blocks_actual_over_quota_and_missing_properties():
    props = {
        "ivo://ivoa.net/vospace/core#quota": "20000000000000",
        "ivo://ivoa.net/vospace/core#length": "20000005095420",
    }
    with pytest.raises(ValueError, match="no job attempted"):
        policy.assert_project_storage_budget(dict(properties=str(props)))
    props["ivo://ivoa.net/vospace/core#length"] = "19950000000000"
    from types import MappingProxyType

    receipt = policy.assert_project_storage_budget(
        dict(properties=MappingProxyType(props))
    )
    assert receipt["available_bytes"] == 50_000_000_000
    with pytest.raises(KeyError):
        policy.assert_project_storage_budget(dict(properties={}))


def test_post_cleanup_recovery_plan_is_distinct_and_matches_bound_configs():
    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load(
        (root / "configs/remediation-recovery-launch-20261026-v3.yaml").read_text()
    )["jobs"]
    assert len(jobs) == 3
    for job in jobs:
        assert job["run_id"].endswith("_v3")
        if job["task"] != "remediation-input-audit":
            assert (
                yaml.safe_load((root / policy.CONFIGS[job["task"]]).read_text())[
                    "run_id"
                ]
                == job["run_id"]
            )
            policy.bind_remediation_policy(job, {})


def test_gradient_recovery_uses_distinct_run_ids_and_original_inputs():
    root = Path(__file__).parents[1]
    original = yaml.safe_load(
        (root / "configs/remediation-pilots-20261026.yaml").read_text()
    )
    recovery = yaml.safe_load(
        (
            root / "configs/remediation-pilots-gradient-recovery-20261026.yaml"
        ).read_text()
    )
    launch = yaml.safe_load(
        (
            root / "configs/remediation-pilot-gradient-recovery-launch-20261026.yaml"
        ).read_text()
    )
    assert len(recovery["arms"]) == len(launch["jobs"]) == 3
    assert {x["run_id"] for x in recovery["arms"]} == {
        x["run_id"] for x in launch["jobs"]
    }
    assert not {x["run_id"] for x in recovery["arms"]} & {
        x["run_id"] for x in original["arms"]
    }
    for key in (
        "checkpoint_sha256",
        "source_manifest",
        "library_revision",
        "seed",
        "updates",
    ):
        assert recovery[key] == original[key]
    for job in launch["jobs"]:
        env = policy.bind_remediation_policy(job, {})
        assert env["MSA_REMEDIATION_POLICY"].endswith(
            "remediation-pilots-gradient-recovery-20261026.yaml"
        )


def test_nsf_probe_launch_uses_distinct_inference_only_config():
    root = Path(__file__).parents[1]
    cfg = yaml.safe_load(
        (root / "configs/remediation-nsf-qmc-probe-20261026.yaml").read_text()
    )
    launch = yaml.safe_load(
        (root / "configs/remediation-nsf-qmc-probe-launch-20261026.yaml").read_text()
    )
    assert len(cfg["arms"]) == len(launch["jobs"]) == 1
    arm = cfg["arms"][0]
    assert (
        arm["integration_probe"]
        and arm["encoder"] == "frozen"
        and arm["reconstruction_weight"] == 0
    )
    assert (
        arm["run_id"] == launch["jobs"][0]["run_id"] == "remediation_nsf6_qmc_probe_v1"
    )
    env = policy.bind_remediation_policy(launch["jobs"][0], {})
    assert env["MSA_REMEDIATION_POLICY"].endswith(
        "remediation-nsf-qmc-probe-20261026.yaml"
    )


def test_nsf_qmc256_recoveries_bind_passing_probe_and_new_runs():
    root = Path(__file__).parents[1]
    original = yaml.safe_load(
        (root / "configs/remediation-pilots-20261026.yaml").read_text()
    )
    cfg = yaml.safe_load(
        (root / "configs/remediation-nsf-qmc256-recovery-20261026.yaml").read_text()
    )
    plan = yaml.safe_load(
        (
            root / "configs/remediation-nsf-qmc256-recovery-launch-20261026.yaml"
        ).read_text()
    )
    assert cfg["particles"] == cfg["integration_gate_particles"] == 256
    assert cfg["nsf_micro_batch_size"] == 16 and cfg["integration_chunk_rows"] == 8
    assert cfg["integration_gate"] == original["integration_gate"]
    assert len(cfg["arms"]) == len(plan["jobs"]) == 4
    assert not {x["run_id"] for x in cfg["arms"]} & {
        x["run_id"] for x in original["arms"]
    }
    for job in plan["jobs"]:
        assert "remediation_nsf6_qmc_probe_v1" in job["depends_on"]
        assert policy.bind_remediation_policy(job, {})[
            "MSA_REMEDIATION_POLICY"
        ].endswith("remediation-nsf-qmc256-recovery-20261026.yaml")
