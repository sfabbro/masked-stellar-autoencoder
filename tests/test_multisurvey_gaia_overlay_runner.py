"""Synthetic four-survey overlay runner and launch contract."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from masked_stellar_autoencoder.training.multisurvey_overlay import file_sha256


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    prior = load(
        "overlay_fixture", Path(__file__).with_name("test_multisurvey_overlay.py")
    )
    parent, desi, _ = prior.fixture(tmp_path)
    gaia = tmp_path / "gaia.fits"
    columns = [
        fits.Column(name="source_id", format="K", array=np.array([11, 99], dtype="i8"))
    ]
    values = {
        "teff_gspspec": [5000, 4900],
        "teff_gspspec_lower": [4900, 4800],
        "teff_gspspec_upper": [5100, 5000],
        "logg_gspspec": [4, 3],
        "logg_gspspec_lower": [3.9, 2.9],
        "logg_gspspec_upper": [4.1, 3.1],
        "mh_gspspec": [-0.5, -1],
        "mh_gspspec_lower": [-0.6, -1.1],
        "mh_gspspec_upper": [-0.4, -0.9],
        "alphafe_gspspec": [0.2, 0.3],
        "alphafe_gspspec_lower": [0.1, 0.2],
        "alphafe_gspspec_upper": [0.3, 0.4],
    }
    columns.extend(
        fits.Column(name=k, format="E", array=np.array(v, dtype="f4"))
        for k, v in values.items()
    )
    columns.append(
        fits.Column(
            name="flags_gspspec",
            format="41A",
            array=np.array(["0" * 41] * 2, dtype="S41"),
        )
    )
    fits.BinTableHDU.from_columns(columns).writeto(gaia)
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    producer = tmp_path / "desi.json"
    producer.write_text(
        json.dumps(
            {
                "run_id": "multisurvey_desi_unique_overlay_v1",
                "status": "multisurvey_desi_overlay_complete",
                "source_commit": "cf40e37c87d4e724b84fd1dffcac6d9e1363560d",
                "result": {"selected": desi},
            }
        )
    )
    fit_ids = np.array([11, 33], dtype="<i8")
    return {
        "run_id": "multisurvey_gaia_gspspec_overlay_v1",
        "gaia_path": str(gaia),
        "gaia_sha256": file_sha256(gaia),
        "gaia_rows": 2,
        "base_path": str(parent),
        "base_sha256": file_sha256(parent),
        "parent_registry": str(registry),
        "desi_producer_manifest": str(producer),
        "desi_sidecar": desi["path"],
        "desi_sidecar_sha256": desi["sha256"],
        "expected_fit_ids_sha256": hashlib.sha256(fit_ids.tobytes()).hexdigest(),
    }


def test_overlay_runner_preserves_sources_roles_and_partial_abundances(tmp_path):
    module = load(
        "gaia_overlay_runner",
        Path(__file__).parents[1] / "scripts/run_multisurvey_gaia_overlay.py",
    )
    config = fixture(tmp_path)
    out = tmp_path / "output"
    manifest = module.run(config, out, source_commit="a" * 40)
    assert manifest["status"] == "multisurvey_gaia_gspspec_overlay_complete"
    assert (
        manifest["source_only"]
        and not manifest["model_loaded"]
        and not manifest["training_performed"]
    )
    result = manifest["result"]
    assert result["fit_arrays_crc"] == result["fit_arrays_roundtrip"] == "pass"
    assert result["summary_sha256"] == file_sha256(out / "summary.json")
    with zipfile.ZipFile(out / "fit-star-availability.npz") as z:
        assert z.testzip() is None
    summary = json.loads((out / "summary.json").read_text())
    assert summary["gaia_by_existing_role"]["outside_base_ids"] == 1
    assert summary["gaia_by_existing_role"]["roles"]["fit"]["logg"] == 1
    assert (
        summary["gaia_by_existing_role"]["roles"]["fit"]["alpha_source_specific"] == 1
    )
    assert summary["gaia_by_existing_role"]["roles"]["fit"]["fe_h_direct"] == 0
    assert summary["fit_union"]["eligible_fit_stars"] == 2
    assert set(summary["fit_union"]["sources"]) == {"extra", "gaia_dr3_gspspec"}
    with pytest.raises(FileExistsError):
        module.run(config, out, source_commit="a" * 40)


def test_runner_rejects_wrong_producer_and_fit_identity(tmp_path):
    module = load(
        "gaia_overlay_runner",
        Path(__file__).parents[1] / "scripts/run_multisurvey_gaia_overlay.py",
    )
    config = fixture(tmp_path)
    p = Path(config["desi_producer_manifest"])
    x = json.loads(p.read_text())
    x["status"] = "failed"
    p.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="producer binding"):
        module.run(config, tmp_path / "bad", source_commit="a" * 40)
    assert (
        json.loads((tmp_path / "bad/manifest.json").read_text())["status"] == "failed"
    )
    x["status"] = "multisurvey_desi_overlay_complete"
    p.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="FIT identities"):
        module.run(
            dict(config, expected_fit_ids_sha256="0" * 64),
            tmp_path / "bad-ids",
            source_commit="a" * 40,
        )
    assert (
        json.loads((tmp_path / "bad-ids/manifest.json").read_text())["status"]
        == "failed"
    )


def test_gaia_launch_policy_is_fixed():
    module = load(
        "gaia_overlay_policy",
        Path(__file__).parents[1] / "scripts/multisurvey_gaia_overlay_launch_policy.py",
    )
    plan = yaml.safe_load(
        Path(
            "configs/multisurvey-gaia-gspspec-overlay-launch-20261028.yaml"
        ).read_text()
    )["jobs"][0]
    env = {"MSA_EXPERIMENT_OUTPUT": module.ROOT + module.RUN}
    assert module.bind_gaia_overlay_policy(plan, dict(env))[
        "MSA_MULTISURVEY_POLICY"
    ] == str(module.CONFIG)
    for changed in (
        dict(plan, resources={"cores": 8, "ram": 128, "gpu": 1}),
        dict(plan, image="private"),
        dict(plan, env={"EXTRA": "1"}),
        dict(plan, run_id="other"),
    ):
        with pytest.raises(ValueError):
            module.bind_gaia_overlay_policy(changed, dict(env))
