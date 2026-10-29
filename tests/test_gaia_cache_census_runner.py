"""Synthetic admission, output receipt and fixed launch-policy tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).parent


def load(name):
    path = HERE / (name + ".py")
    if not path.exists():
        path = HERE.parent / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config(tmp_path, runner):
    base_manifest = tmp_path / "base-manifest.json"
    base_path = tmp_path / "base.h5"
    data = {
        "run_id": runner.RUN,
        "source_path": str(tmp_path / "gaia.fits"),
        "source_sha": "a" * 64,
        "expected_source_rows": 3,
        "base_manifest": str(base_manifest),
        "base_path": str(base_path),
        "base_sha": "b" * 64,
        "cache_path": str(tmp_path / "cache"),
        "original_path": str(tmp_path / "original.h5"),
        "expected_cache_rows": 4,
        "chunk_rows": 2,
    }
    base_manifest.write_text(
        json.dumps(
            {
                "run_id": "remediation_source_v3",
                "status": "remediation_source_complete",
                "source_commit": runner.BASE_COMMIT,
                "prepared_sha256": data["base_sha"],
                "prepared_path": data["base_path"],
            }
        )
    )
    return data


def fake_execute(
    source,
    base,
    cache,
    *,
    original_path,
    source_sha,
    base_sha,
    output,
    expected_source_rows,
    expected_cache_rows,
    chunk_rows,
    progress,
):
    assert expected_cache_rows == 4 and chunk_rows == 2
    assert source_sha == "a" * 64 and base_sha == "b" * 64
    assert source.endswith("gaia.fits") and base.endswith("base.h5")
    assert cache.endswith("cache") and original_path.endswith("original.h5")
    progress(2, 4)
    progress(4, 4)
    output.mkdir()
    arrays = output / "identities.npz"
    np.savez_compressed(arrays, source_id=np.array([2, 3, 4], dtype="i8"))
    result = {
        "schema": "multisurvey-gaia-cache-census-v1",
        "source_only": True,
        "model_loaded": False,
        "training_performed": False,
        "source_sha256": source_sha,
        "base_sha256": base_sha,
        "source_rows": expected_source_rows,
        "identities_path": str(arrays.resolve()),
        "identities_sha256": sha(arrays),
        "identities_bytes": arrays.stat().st_size,
        "identities_crc": "pass",
        "identities_roundtrip": "pass",
    }
    (output / "summary.json").write_text(json.dumps(result))
    return result


def test_success_binds_base_progress_and_verified_artifacts(tmp_path):
    module = load("run_multisurvey_gaia_cache_census")
    data = config(tmp_path, module)
    out = tmp_path / "out"
    result = module.run(data, out, source_commit="c" * 40, execute=fake_execute)
    assert result["status"] == "multisurvey_gaia_cache_census_complete"
    assert result["source_only"] and not result["model_loaded"]
    assert not result["training_performed"]
    assert result["summary_sha256"] == sha(out / "census" / "summary.json")
    assert result["result"]["identities_sha256"] == sha(
        out / "census" / "identities.npz"
    )
    assert json.loads((out / "manifest.json").read_text()) == result
    progress = json.loads((out / "progress.json").read_text())
    assert progress == {
        "run_id": module.RUN,
        "source_commit": "c" * 40,
        "phase": "cache_id_scan",
        "rows_scanned": 4,
        "rows_total": 4,
    }
    with pytest.raises(FileExistsError):
        module.run(data, out, source_commit="c" * 40, execute=fake_execute)


def test_runner_executes_real_source_only_census_on_synthetic_files(
    tmp_path, monkeypatch
):
    module = load("run_multisurvey_gaia_cache_census")
    fixtures = load("test_gaia_cache_census")
    source = fixtures.source_data.__wrapped__(tmp_path, monkeypatch)
    data = config(tmp_path, module)
    data.update(
        source_path=str(source["source"]),
        source_sha=source["source_sha"],
        expected_source_rows=source["expected_source_rows"],
        base_path=str(source["base"]),
        base_sha=source["base_sha"],
        cache_path=str(source["cache"]),
        original_path=str(source["original_path"]),
        expected_cache_rows=source["expected_cache_rows"],
        chunk_rows=source["chunk_rows"],
    )
    Path(data["base_manifest"]).write_text(
        json.dumps(
            {
                "run_id": "remediation_source_v3",
                "status": "remediation_source_complete",
                "source_commit": module.BASE_COMMIT,
                "prepared_sha256": data["base_sha"],
                "prepared_path": data["base_path"],
            }
        )
    )
    output = tmp_path / "actual"
    result = module.run(data, output, source_commit="c" * 40)
    assert result["status"] == "multisurvey_gaia_cache_census_complete"
    assert result["result"]["counts"]["outside_base"]["stars"] == 2
    assert result["result"]["counts"]["complete_xp_in_cache_outside_base"]["stars"] == 1
    assert result["summary_sha256"] == sha(output / "census" / "summary.json")


@pytest.mark.parametrize(
    "field", ["run_id", "status", "source_commit", "prepared_sha256", "prepared_path"]
)
def test_wrong_completed_base_is_preserved_as_failure(tmp_path, field):
    module = load("run_multisurvey_gaia_cache_census")
    data = config(tmp_path, module)
    base = Path(data["base_manifest"])
    doc = json.loads(base.read_text())
    doc[field] = "wrong"
    base.write_text(json.dumps(doc))
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="source binding"):
        module.run(data, out, source_commit="c" * 40, execute=fake_execute)
    failure = json.loads((out / "manifest.json").read_text())
    assert failure["status"] == "failed" and not failure["training_performed"]
    assert not (out / "census").exists()


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "source_only",
        "model_loaded",
        "training_performed",
        "source_sha256",
        "base_sha256",
        "source_rows",
        "identities_path",
        "identities_sha256",
        "identities_bytes",
        "identities_crc",
        "identities_roundtrip",
    ],
)
def test_wrong_output_binding_fails_closed(tmp_path, field):
    module = load("run_multisurvey_gaia_cache_census")
    data = config(tmp_path, module)
    out = tmp_path / "bad"

    def wrong(*args, **kwargs):
        result = fake_execute(*args, **kwargs)
        result[field] = "wrong"
        (kwargs["output"] / "summary.json").write_text(json.dumps(result))
        return result

    with pytest.raises(ValueError, match="result binding"):
        module.run(data, out, source_commit="c" * 40, execute=wrong)
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"


def test_bad_progress_and_input_rejected(tmp_path):
    module = load("run_multisurvey_gaia_cache_census")
    data = config(tmp_path, module)

    def bad(*args, **kwargs):
        kwargs["progress"](4, 4)
        kwargs["progress"](2, 4)

    with pytest.raises(ValueError, match="progress"):
        module.run(data, tmp_path / "bad", source_commit="c" * 40, execute=bad)
    assert (
        json.loads((tmp_path / "bad/manifest.json").read_text())["status"] == "failed"
    )
    for changed in (
        dict(data, extra=True),
        dict(data, run_id="wrong"),
        dict(data, expected_source_rows=True),
        dict(data, source_sha="x" * 64),
    ):
        with pytest.raises(ValueError, match="configuration"):
            module.run(
                changed,
                tmp_path / "uncreated",
                source_commit="c" * 40,
                execute=fake_execute,
            )
        assert not (tmp_path / "uncreated").exists()
    with pytest.raises(ValueError, match="configuration"):
        module.run(
            data,
            tmp_path / "uncreated",
            source_commit="not-a-sha",
            execute=fake_execute,
        )


def test_policy_fixed_source_resources_image_and_output(tmp_path, monkeypatch):
    module = load("multisurvey_gaia_cache_census_launch_policy")
    cfg = HERE / "multisurvey-gaia-cache-census-20261028.yaml"
    plan_path = HERE / "multisurvey-gaia-cache-census-launch-20261028.yaml"
    if not cfg.exists():
        cfg = HERE.parent / "configs" / cfg.name
        plan_path = HERE.parent / "configs" / plan_path.name
    plan = yaml.safe_load(plan_path.read_text())["jobs"][0]
    monkeypatch.setattr(module, "CONFIG", cfg)
    env = {"MSA_EXPERIMENT_OUTPUT": module.ROOT + module.RUN}
    bound = module.bind_gaia_cache_census_policy(plan, dict(env))
    assert bound["MSA_MULTISURVEY_POLICY"] == str(cfg)
    assert bound["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    for bad in (
        dict(plan, resources={"cores": 8, "ram": 128, "gpu": 1}),
        dict(plan, image="private"),
        dict(plan, run_id="wrong"),
        dict(plan, task="wrong"),
        dict(plan, depends_on=["other"]),
        dict(plan, env={"OTHER": "1"}),
    ):
        with pytest.raises(ValueError):
            module.bind_gaia_cache_census_policy(bad, dict(env))
    with pytest.raises(ValueError, match="distinct"):
        module.bind_gaia_cache_census_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": "/arc/other"}
        )
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(dict(module.EXPECTED, source_sha="0" * 64)))
    monkeypatch.setattr(module, "CONFIG", changed)
    with pytest.raises(ValueError, match="source contract"):
        module.bind_gaia_cache_census_policy(plan, dict(env))
