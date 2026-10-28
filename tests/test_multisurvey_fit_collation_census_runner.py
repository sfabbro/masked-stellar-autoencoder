"""Source-only census runner: fake completed producer and synthetic overlay."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from masked_stellar_autoencoder.training.multisurvey_overlay import file_sha256


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_config(tmp_path):
    fixture = load_module(
        "overlay_fixture", Path(__file__).with_name("test_multisurvey_overlay.py")
    )
    _, _, overlay = fixture.fixture(tmp_path)
    producer = tmp_path / "producer.json"
    sha = file_sha256(overlay)
    overlay_data = json.loads(overlay.read_text())
    producer.write_text(
        json.dumps(
            {
                "run_id": "multisurvey_desi_unique_overlay_v1",
                "status": "multisurvey_desi_overlay_complete",
                "source_commit": "cf40e37c87d4e724b84fd1dffcac6d9e1363560d",
                "result": {
                    "overlay_manifest_path": str(overlay),
                    "overlay_manifest_sha256": sha,
                    "overlay_sha256": overlay_data["overlay"]["sha256"],
                },
            }
        )
    )
    ids = np.asarray([11, 33], dtype="<i8")
    return {
        "run_id": "multisurvey_fit_collation_census_v1",
        "overlay_producer_manifest": str(producer),
        "overlay_manifest": str(overlay),
        "overlay_manifest_sha256": sha,
        "overlay_sha256": overlay_data["overlay"]["sha256"],
        "expected_fit_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
    }


def runner():
    return load_module(
        "fit_collation_census_runner",
        Path(__file__).parents[1] / "scripts/run_multisurvey_fit_collation_census.py",
    )


def test_runner_binds_completed_overlay_writes_verified_arrays_and_no_model(tmp_path):
    module = runner()
    config = make_config(tmp_path)
    out = tmp_path / "good"
    manifest = module.run(config, out, source_commit="a" * 40)
    assert manifest["status"] == "multisurvey_fit_collation_census_complete"
    assert manifest["source_only"] and not manifest["model_loaded"]
    assert not manifest["training_performed"]
    receipt = manifest["result"]
    assert (
        receipt["source_binding"]
        == config["overlay_sha256"] + ":" + config["overlay_manifest_sha256"]
    )
    assert receipt["fit_ids_sha256"] == config["expected_fit_ids_sha256"]
    assert receipt["star_arrays_crc"] == receipt["star_arrays_roundtrip"] == "pass"
    assert receipt["summary_sha256"] == file_sha256(out / "summary.json")
    assert receipt["star_arrays_sha256"] == file_sha256(out / "star-availability.npz")
    with zipfile.ZipFile(out / "star-availability.npz") as archive:
        assert archive.testzip() is None
    with np.load(out / "star-availability.npz", allow_pickle=False) as arrays:
        assert arrays["source_id"].tolist() == [11, 33]
        assert arrays["selected_observed"].shape == (2, 1, 3)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["selected_targets_any_source"] == {"teff": 0, "logg": 0, "fe_h": 1}
    assert json.loads((out / "manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        module.run(config, out, source_commit="a" * 40)


@pytest.mark.parametrize(
    "field",
    [
        "run_id",
        "status",
        "source_commit",
        "overlay_manifest_path",
        "overlay_manifest_sha256",
        "overlay_sha256",
    ],
)
def test_runner_preserves_failure_for_wrong_producer_binding(tmp_path, field):
    module = runner()
    config = make_config(tmp_path)
    path = Path(config["overlay_producer_manifest"])
    data = json.loads(path.read_text())
    if field in data:
        data[field] = "wrong"
    else:
        data["result"][field] = "wrong"
    path.write_text(json.dumps(data))
    out = tmp_path / "bad-producer"
    with pytest.raises(ValueError, match="producer binding"):
        module.run(config, out, source_commit="a" * 40)
    failure = json.loads((out / "manifest.json").read_text())
    assert failure["status"] == "failed" and not failure["training_performed"]
    assert not (out / "star-availability.npz").exists()


def test_runner_rejects_wrong_fit_binding_and_missing_producer(tmp_path):
    module = runner()
    config = make_config(tmp_path)
    with pytest.raises(ValueError, match="identity binding"):
        module.run(
            dict(config, expected_fit_ids_sha256="0" * 64),
            tmp_path / "bad-fit",
            source_commit="a" * 40,
        )
    assert (
        json.loads((tmp_path / "bad-fit/manifest.json").read_text())["status"]
        == "failed"
    )
    Path(config["overlay_producer_manifest"]).unlink()
    with pytest.raises(FileNotFoundError):
        module.run(config, tmp_path / "missing", source_commit="a" * 40)
    assert (
        json.loads((tmp_path / "missing/manifest.json").read_text())["status"]
        == "failed"
    )


def test_runner_rejects_changed_overlay_or_crc_failure(tmp_path, monkeypatch):
    module = runner()
    config = make_config(tmp_path)
    Path(config["overlay_manifest"]).write_text("{}")
    with pytest.raises(ValueError, match="Manifest SHA256 mismatch"):
        module.run(config, tmp_path / "bad-overlay", source_commit="a" * 40)
    assert (
        json.loads((tmp_path / "bad-overlay/manifest.json").read_text())["status"]
        == "failed"
    )

    other = tmp_path / "other"
    other.mkdir()
    config2 = make_config(other)
    monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda self: "corrupt.npy")
    with pytest.raises(ValueError, match="CRC verification"):
        module.run(config2, tmp_path / "bad-crc", source_commit="a" * 40)
    assert (
        json.loads((tmp_path / "bad-crc/manifest.json").read_text())["status"]
        == "failed"
    )
    assert not (tmp_path / "bad-crc/star-availability.npz").exists()


def test_runner_rejects_bad_config_before_creating_output(tmp_path):
    module = runner()
    config = make_config(tmp_path)
    out = tmp_path / "bad-config"
    with pytest.raises(ValueError, match="configuration"):
        module.run(dict(config, extra=True), out, source_commit="a" * 40)
    assert not out.exists()
    with pytest.raises(ValueError, match="configuration"):
        module.run(config, out, source_commit="not-a-revision")
    assert not out.exists()


def test_launch_policy_guards_source_resource_output_and_overrides(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).parents[1] / "scripts/multisurvey_fit_collation_launch_policy.py"
    )
    module = load_module("fit_collation_launch_policy", script)
    plan = yaml.safe_load(
        Path(
            "configs/multisurvey-fit-collation-census-launch-20261028.yaml"
        ).read_text()
    )["jobs"][0]
    good = {"MSA_EXPERIMENT_OUTPUT": module.ROOT + module.RUN}
    actual = module.bind_fit_collation_census_policy(plan, dict(good))
    assert actual["MSA_MULTISURVEY_POLICY"] == str(module.CONFIG)
    assert actual["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    for bad in (
        dict(plan, resources={"cores": 8, "ram": 128, "gpu": 1}),
        dict(plan, image="other"),
        dict(plan, run_id="different"),
        dict(plan, task="different"),
        dict(plan, depends_on=["other"]),
        dict(plan, env={"OTHER": "1"}),
    ):
        with pytest.raises(ValueError):
            module.bind_fit_collation_census_policy(bad, dict(good))
    with pytest.raises(ValueError, match="distinct"):
        module.bind_fit_collation_census_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": "/arc/other"}
        )
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(dict(module.EXPECTED, overlay_sha256="0" * 64)))
    monkeypatch.setattr(module, "CONFIG", changed)
    with pytest.raises(ValueError, match="source contract"):
        module.bind_fit_collation_census_policy(plan, dict(good))
