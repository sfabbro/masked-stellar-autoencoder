import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.seismic_ablation import (
    SeismicChannel,
    matched_schedule,
)


@pytest.mark.parametrize("kind", ["identity", "affine", "context"])
def test_seismic_map_is_identity_at_initialization_and_invertible(kind):
    center = [5000, 3, -0.5, 0.1, 1, 0]
    scale = [1000, 0.8, 0.7, 0.2, 1, 1]
    channel = SeismicChannel(center, scale, kind)
    x = torch.randn(32, 7, 6)
    torch.testing.assert_close(channel(x), x)
    if kind != "identity":
        with torch.no_grad():
            channel.raw.copy_(torch.tensor([0.7, -0.5, 0.4, -0.6]))
    y = channel(x)
    g = y[..., 1] * scale[1] + center[1]
    torch.testing.assert_close(
        channel.inverse_gravity(g, x[..., 0], x[..., 2]),
        x[..., 1] * scale[1] + center[1],
    )
    assert torch.exp(channel.coefficients()[1]) > 0
    torch.testing.assert_close(y[..., [0, 2, 3, 4, 5]], x[..., [0, 2, 3, 4, 5]])


def test_off_treatment_keeps_native_gradient_scale_and_schedule():
    schedules = []
    for treatment in [0.0, 0.1, 1.0]:
        rows = []
        for s, p, rng in matched_schedule([13, 0, 19], 5, 17):
            aux = rng.choice(53, 32, replace=False)
            rows.append((s.copy(), p.copy(), aux))
        schedules.append(rows)
        native = torch.tensor(2.0, requires_grad=True)
        seismic = torch.tensor(3.0, requires_grad=True)
        ((288 * native + 32 * treatment * seismic) / 320).backward()
        assert native.grad.item() == pytest.approx(0.9)
        assert seismic.grad.item() == pytest.approx(0.1 * treatment)
    for a, b in zip(schedules[0], schedules[2], strict=True):
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)


def test_incremental_report_uses_paired_heldout_arrays_and_checks_batches(
    tmp_path, monkeypatch
):
    import importlib.util
    import json
    import sys

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "seis_report", "scripts/report_seismic_supervision_experiments.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "SUITES", ["suite"])
    import zipfile

    archive = tmp_path / "expanded_six_source_20261107_v2/clusters.zip"
    archive.parent.mkdir()
    real_digest = report.digest
    monkeypatch.setattr(
        report,
        "digest",
        lambda path: (
            "9eafa2e10afc031790cfe36ed8545aa6336cc87c317fab167e9cea9dacbd0748"
            if str(path) == str(archive)
            else real_digest(path)
        ),
    )
    suite = tmp_path / "suite"
    suite.mkdir()
    ids = np.arange(12, dtype="i8") + 6100000000000000000
    roles = np.full(12, b"devtest", dtype="S8")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "catalogues/testcluster.txt",
            "# source_id qflag memberprob\n" + "\n".join(f"{sid} 2 .99" for sid in ids),
        )
    np.savez(
        suite / "panel.npz",
        source_id=ids,
        tess_source_id=ids[2:],
        tess_reference=np.full(10, 2.5),
        tess_error=np.full(10, 0.01),
    )
    arms = {}
    for name, bias in [("off", 0.1), ("identity", 0.05), ("affine", 0.02)]:
        d = suite / name
        d.mkdir()
        median = np.ones((12, 6))
        median[:, 1] = 2.5 + bias
        np.savez(
            d / "common.npz",
            source_id=ids,
            role=roles,
            median=median,
            logg_draws=np.repeat(median[None, :, 1], 256, axis=0),
        )
        reference = np.zeros((2, 6))
        reference[:, 1] = 2.5
        np.savez(
            d / "devtest-APOKASC3_loggSeis.npz",
            source_id=ids[:2],
            reference=reference,
            center=np.zeros(6),
            scale=np.ones(6),
        )
        (d / "readout.json").write_text(
            json.dumps(
                dict(
                    source_native_scores=[
                        dict(
                            role="tune",
                            source=0,
                            per_target_crps=[0.2, 0.3, None, None, None, None],
                        )
                    ],
                    seismic_coefficients=[0, 0, 0, 0],
                )
            )
        )
        arms[name] = dict(
            status="complete",
            completed_updates=1500,
            readout_sha256=report.digest(d / "common.npz"),
            batch_identity_sha256="matched",
            reconstruction_sequence_sha256="matched",
        )
    (suite / "manifest.json").write_text(
        json.dumps(
            dict(
                status="seismic_experiments_complete",
                source_registry=["apogee", "APOKASC3_loggSeis"],
                panel_sha256=report.digest(suite / "panel.npz"),
                arms=arms,
            )
        )
    )
    benchmark = tmp_path / "bench.csv"
    pd.DataFrame(
        dict(
            source_id=ids,
            Teff=np.ones(12),
            logg=np.full(12, 2.5),
            **{"[FeI/H]": np.ones(12), "[FeII/H]": np.ones(12)},
        )
    ).to_csv(benchmark, index=False)
    benchmark.write_text(benchmark.read_text() + ",1,2.5,1,1\n")
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys, "argv", ["report", "--output", str(output), "--benchmarks", str(benchmark)]
    )
    report.main()
    result = json.loads((output / "manifest.json").read_text())
    assert result["status"] == "seismic_comparison_complete"
    assert result["benchmark_rows_without_gaia_id"] == 1
    assert result["suites"]["suite"]["readouts"]["off"][
        "tune_nonseismic_macro_crps"
    ] == pytest.approx(0.25)
    comparison = next(
        v
        for v in result["comparisons"]
        if v["arm"] == "affine" and v["reference"] == "tess"
    )
    assert comparison["mean"] == pytest.approx(-0.08)
    assert (
        result["suites"]["suite"]["readouts"]["off"]["clusters"]["testcluster"]["n"]
        == 12
    )
    assert (output / "report.md").exists()
