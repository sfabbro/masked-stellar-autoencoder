import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_alpha_target_wave as alpha


def policy():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/desi-alpha-source-policy-20261023.json"
        ).read_text()
    )


def test_blank_header_requires_actual_definition_policy_and_rejects_wrong_unit():
    columns = {"ALPHAFE": {"unit": None}, "ALPHAFE_ERR": {"unit": None}}
    document = policy()
    assert alpha.validate_unit_policy(document, columns) == {
        "ALPHAFE": None,
        "ALPHAFE_ERR": None,
    }
    document["native_header_blank_allowed_with_this_explicit_policy"] = False
    with pytest.raises(ValueError, match="Blank"):
        alpha.validate_unit_policy(document, columns)
    with pytest.raises(ValueError, match="conflicts"):
        alpha.validate_unit_policy(
            policy(),
            {"ALPHAFE": {"unit": "linear_ratio"}, "ALPHAFE_ERR": {"unit": "dex"}},
        )
    document = policy()
    document["definition_is_apogee_alpha_m"] = True
    with pytest.raises(ValueError, match="definition"):
        alpha.validate_unit_policy(document, columns)


def test_independent_allrow_audit_detects_projected_label_tampering(
    tmp_path, monkeypatch
):
    import hashlib

    import h5py
    from astropy.io import fits
    from prepare_alpha_source_evidence import prepare

    fields = {
        "TARGETID": ("K", [21]),
        "ALPHAFE": ("D", [-0.3]),
        "ALPHAFE_ERR": ("D", [0.03]),
        "FEH": ("D", [-1.2]),
        "FEH_ERR": ("D", [0.1]),
        "TEFF": ("D", [5000.0]),
        "LOGG": ("D", [2.0]),
        "RVS_WARN": ("K", [0]),
        "SUCCESS": ("L", [True]),
        "PRIMARY": ("L", [True]),
        "SN_R": ("D", [20.0]),
        "RR_SPECTYPE": ("4A", ["STAR"]),
        "TARGET_RA": ("D", [10.0]),
        "TARGET_DEC": ("D", [20.0]),
    }
    columns = [
        fits.Column(
            name=k,
            format=fmt,
            array=np.array(v),
            unit="dex" if k.startswith("ALPHAFE") else None,
        )
        for k, (fmt, v) in fields.items()
    ]
    gaia = [
        fits.Column(
            name="SOURCE_ID", array=np.array([2**60 + 1], dtype="i8"), format="K"
        ),
        fits.Column(name="RA", array=np.array([10.0]), format="D"),
        fits.Column(name="DEC", array=np.array([20.0]), format="D"),
    ]
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(columns, name="RVTAB"),
        fits.BinTableHDU.from_columns(gaia, name="GAIA"),
    ]
    hdus += [
        fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="TARGETID", array=np.array([21], dtype="i8"), format="K"
                )
            ],
            name=n,
        )
        for n in ("FIBERMAP", "SPTAB")
    ]
    raw = tmp_path / "raw.fits"
    fits.HDUList(hdus).writeto(raw)
    expected = hashlib.sha256(raw.read_bytes()).hexdigest()
    prepare(raw, tmp_path / "projected", expected_sha=expected)
    monkeypatch.setattr(alpha, "RAW_SHA", expected)
    document = policy()
    document["source_sha256"] = expected
    path = tmp_path / "projected/desi-alpha-native.h5"
    result = alpha.independently_audit_projection(raw, path, policy=document)
    assert result["counts"]["all_rows_checked"] == 1
    with h5py.File(path, "r+") as h5:
        record = h5["measurements"][0]
        record["alpha_fe"] = 0.4
        h5["measurements"][0] = record
    with pytest.raises(AssertionError, match="alpha_fe"):
        alpha.independently_audit_projection(raw, path, policy=document)


def test_native_alpha_affine_density_preserves_negative_signed_coordinate():
    from run_alpha_target_wave import alpha_native_coordinates

    mean, variance, density = alpha_native_coordinates(
        np.array([[-2.0], [1.0]]),
        np.ones((2, 1)),
        np.array([-1.0, -2.0]),
        np.array([-0.1]),
        np.array([0.2]),
    )
    np.testing.assert_allclose(mean[:, 0], [-0.5, 0.1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(variance, 0.04, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        density, np.array([-1.0, -2.0]) - np.log(0.2), rtol=0, atol=1e-15
    )


def test_parent_is_explicit_and_preflighted_before_original_row_audit(
    tmp_path, monkeypatch
):
    projection = tmp_path / "projection"
    projection.mkdir()
    (projection / "manifest.json").write_text(
        json.dumps(
            dict(
                status="native_source_projection_complete_not_training_certificate",
                source_sha256=alpha.RAW_SHA,
                sidecar_sha256="sidecar",
            )
        )
    )
    evidence = tmp_path / "policy.json"
    evidence.write_text(json.dumps(policy()))
    checkpoint = tmp_path / "canonical-arc-parent-86.pth"
    seen = []

    def fake_sha(path):
        path = Path(path)
        seen.append(path)
        return {
            "desi-alpha-native.h5": "sidecar",
            "enriched-feature-cohort.h5": alpha.FEATURE_SHA,
            "distance-feature-cohort.h5": alpha.REFERENCE_SHA,
            checkpoint.name: alpha.PARENT_SHA,
        }[path.name]

    def audit_reached(*args, **kwargs):
        raise RuntimeError("input contracts checked before expensive audit")

    monkeypatch.setattr(alpha, "sha", fake_sha)
    monkeypatch.setattr(alpha, "independently_audit_projection", audit_reached)
    with pytest.raises(RuntimeError, match="input contracts checked"):
        alpha.prepare(projection, tmp_path, tmp_path / "output", evidence, checkpoint)
    assert checkpoint in seen
    assert not any("main-checkpoints" in str(path) for path in seen)
    assert json.loads((tmp_path / "output/manifest.json").read_text())[
        "checkpoint_path"
    ] == str(checkpoint)
