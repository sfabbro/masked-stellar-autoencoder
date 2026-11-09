import numpy as np
import pytest

from masked_stellar_autoencoder.training.survey_map_arrays import (
    inverse_map_axis,
    map_axis,
)


@pytest.mark.parametrize("kind", ["offset", "affine", "spline"])
def test_inverse_mapping_roundtrip_all_targets_knots_and_extrapolation(kind):
    rng = np.random.default_rng(41)
    coefficients = dict(
        kind=kind,
        center=np.array([5000, 3, -0.5, 0.1, 1, 0]),
        scale=np.array([1000, 1, 1, 0.2, 0.7, 1]),
        bias=rng.normal(size=(3, 6)),
        slopes=rng.uniform(0.2, 3, size=(3, 6, 8)),
    )
    for j in range(6):
        x = np.r_[np.linspace(-8, 8, 101), np.arange(-4, 5)]
        x = x * coefficients["scale"][j] + coefficients["center"][j]
        if j >= 4:
            x = np.exp(x)
        y = map_axis(x, j, 2, coefficients)
        np.testing.assert_allclose(
            inverse_map_axis(y, j, 2, coefficients), x, rtol=1e-12, atol=1e-10
        )


def test_inverse_rejects_missing_and_invalid_log_coordinates():
    c = dict(
        center=np.zeros(6),
        scale=np.ones(6),
        bias=np.zeros((1, 6)),
        slopes=np.ones((1, 6, 8)),
        kind="spline",
    )
    with pytest.raises(ValueError, match="finite"):
        inverse_map_axis([np.nan], 1, 0, c)
    with pytest.raises(ValueError, match="positive"):
        inverse_map_axis([0], 4, 0, c)


def test_full_union_report_on_saved_synthetic_arrays(tmp_path, monkeypatch):
    """Exercise masks, all four roles, inverse labels, file bindings and clusters.

    Comparator catalogues are independently scheduled read-only joins; this check
    uses no astronomical input or neural checkpoint and never loads a model.
    """
    import hashlib
    import importlib.util
    import json
    import sys
    import zipfile

    import h5py

    spec = importlib.util.spec_from_file_location(
        "union_report", "scripts/report_full_label_union_qa.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "compare_catalogues", lambda *args: None)
    archive = tmp_path / "expanded_six_source_20261107_v2/clusters.zip"
    archive.parent.mkdir()
    ids = np.arange(60, dtype="i8") + 6000000000000000000
    roles = np.repeat(
        np.array([b"fit", b"tune", b"cal", b"devtest", b"external"], dtype="S8"), 12
    )
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "catalogues/testcluster.txt",
            "# source_id qflag memberprob\n"
            + "\n".join(f"{sid} 2 0.99" for sid in ids[-12:]),
        )
    realsha = report.sha
    monkeypatch.setattr(
        report,
        "sha",
        lambda path, kind="sha256": (
            "9eafa2e10afc031790cfe36ed8545aa6336cc87c317fab167e9cea9dacbd0748"
            if str(path) == str(archive)
            else realsha(path, kind)
        ),
    )
    rng = np.random.default_rng(31)
    values = rng.normal(size=(60, 6)) * [800, 0.7, 0.6, 0.1, 1, 0.5] + [
        5000,
        3,
        -0.4,
        0.1,
        5,
        2,
    ]
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as f:
        f["source_id"] = ids
        f["role"] = roles
        g = f.create_group("native_observations/survey")
        g["star_row"] = np.arange(60)
        g["value"] = values
        g["observed"] = np.ones((60, 6), bool)
    sourcehash = realsha(source)
    sm = tmp_path / "source-manifest.json"
    sm.write_text(
        json.dumps(
            dict(
                prepared_path=str(source),
                prepared_sha256=sourcehash,
                source_registry=["survey"],
            )
        )
    )
    producer = tmp_path / "producer"
    producer.mkdir()
    with h5py.File(producer / "predictions.h5", "w") as f:
        f["source_id"] = ids
        f["role"] = roles
        eligible = np.ones(60, bool)
        eligible[0] = False
        f["eligible"] = eligible
        f["median"] = values + 0.01
        f["quantiles"] = values[:, None, :] + np.linspace(-0.2, 0.2, 9)[None, :, None]
        f["cluster_row"] = np.arange(48, 60)
        f["cluster_draws"] = np.repeat(values[-12:, None, :], 1024, axis=1)
    np.savez(
        producer / "map-coefficients.npz",
        source_registry=np.array(["survey"]),
        kind="spline",
        center=np.zeros(6),
        scale=np.ones(6),
        bias=np.zeros((1, 6)),
        slopes=np.ones((1, 6, 8)),
    )
    (producer / "manifest.json").write_text(
        json.dumps(
            dict(
                status="full_label_union_inference_complete",
                source_manifest_path=str(sm),
                source_sha256=sourcehash,
                predictions_sha256=realsha(producer / "predictions.h5"),
                map_sha256=realsha(producer / "map-coefficients.npz"),
                checkpoint_sha256="synthetic",
                checkpoint_update=2000,
                draws=128,
            )
        )
    )
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys, "argv", ["report", "--producer", str(producer), "--output", str(output)]
    )
    report.main()
    result = json.loads((output / "manifest.json").read_text())
    assert result["status"] == "full_label_union_report_complete"
    assert result["spectroscopic_union_stars"] == 59
    assert result["spectroscopic_label_union_stars_including_no_prediction"] == 60
    assert len(result["residual_statistics"]) == 24
    assert result["kiel_counts"]["all/survey"]["all_reference"]["n"] == 60
    assert result["kiel_counts"]["all/survey"]["prediction"]["n"] == 59
    assert result["clusters"][0]["n"] == 12
    assert (output / "residual-statistics.csv").exists()
