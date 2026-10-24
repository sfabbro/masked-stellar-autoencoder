"""Synthetic math and routing checks; never local native-data execution."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_five_wave_training as runner

from masked_stellar_autoencoder.training.five_wave_heads import (
    QUANTILES,
    QuantileThreeHead,
    build_head,
    masked_native_median_score,
    modality_dropout,
    native_fit_point_scale,
    native_quantiles,
    standardized_quantiles,
)


def test_quantile_baseline_ordered_masked_pinball_not_a_density():
    torch.manual_seed(71)
    head = QuantileThreeHead(latent_dim=4).double()
    x = torch.randn(5, 4, dtype=torch.double)
    y = torch.randn(5, 6, dtype=torch.double)
    mask = torch.rand(5, 6) > 0.4
    q = head(x)
    assert q.shape == (5, 3, 6) and (q[:, 1:] > q[:, :-1]).all()
    levels = q.new_tensor(QUANTILES)[None, :, None]
    delta = y[:, None] - q
    expected = torch.maximum(levels * delta, (levels - 1) * delta).masked_fill(
        ~mask[:, None], 0
    ).sum() / (mask.sum() * 3)
    actual = head.criterion(x, y, torch.zeros_like(y), mask, torch.arange(5))
    torch.testing.assert_close(actual, expected)
    assert not head.has_density
    with pytest.raises(ValueError, match="arbitrary"):
        standardized_quantiles(head, x, (0.1, 0.9))
    zero = head.criterion(
        x, y, torch.zeros_like(y), torch.zeros_like(mask), torch.arange(5)
    )
    zero.backward()
    assert float(zero.detach()) == 0 and all(
        torch.isfinite(p.grad).all() for p in head.parameters()
    )


@pytest.mark.parametrize("kind", ["gaussian", "factorized_nsf"])
def test_actual_distribution_quantiles_and_native_positive_coordinates(kind):
    torch.manual_seed(73)
    head = build_head(dict(kind=kind), latent_dim=4).double()
    x = torch.randn(3, 4, dtype=torch.double)
    q = standardized_quantiles(head, x)
    assert q.shape == (3, 3, 6) and (q[:, 1:] > q[:, :-1]).all()
    if kind == "factorized_nsf":
        for column, scalar in enumerate(head.heads):
            z = scalar.distribution(x).transform(
                q[:, :, column].transpose(0, 1)[..., None]
            )
            cdf = torch.special.ndtr(z).squeeze(-1).transpose(0, 1)
            torch.testing.assert_close(
                cdf, q.new_tensor(QUANTILES).expand_as(cdf), atol=1e-10, rtol=1e-10
            )
    center = torch.arange(6, dtype=torch.double) / 10
    scale = torch.arange(1, 7, dtype=torch.double) / 10
    native = native_quantiles(head, x, center, scale)
    torch.testing.assert_close(native[..., :4], (center + scale * q)[..., :4])
    torch.testing.assert_close(native[..., 4:], (center + scale * q)[..., 4:].exp())
    assert (native[..., 4:] > 0).all()


def test_whole_xp_dropout_preserves_photometry_and_natural_flags():
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    visible = torch.ones_like(x, dtype=torch.bool)
    visible[1, [1, 3]] = False
    x[1, [1, 3]] = -9999
    original = x.clone()
    out, after, flags = modality_dropout(
        x, visible, [1, 3], 1, torch.Generator().manual_seed(2)
    )
    torch.testing.assert_close(x, original)
    torch.testing.assert_close(out[:, [0, 2, 4]], original[:, [0, 2, 4]])
    assert (out[:, [1, 3]] == -9999).all() and not after[:, [1, 3]].any()
    assert flags["input_xp_before"].tolist() == [True, False, True, True]
    assert flags["xp_dropped"].tolist() == [True, False, True, True]
    assert not flags["input_xp_after"].any()
    keep, _, _ = modality_dropout(
        x, visible, [1, 3], 0, torch.Generator().manual_seed(2)
    )
    torch.testing.assert_close(keep, x)


def test_native_fit_scale_and_common_tune_selection_masked_labels():
    value = np.tile(np.arange(6), (40, 1)) + np.arange(40)[:, None] * 0.1
    mask = np.ones((40, 6), bool)
    fit = np.arange(20)
    scale = native_fit_point_scale(value, mask, fit)
    np.testing.assert_allclose(scale, value[:20].std(0))
    held = value[20:] + scale
    observed = mask[20:].copy()
    observed[:4, 3] = False
    target = value[20:].copy()
    target[:4, 3] = np.nan
    score, labels = masked_native_median_score(held, target, observed, scale)
    np.testing.assert_allclose(labels, np.ones(6))
    assert score == pytest.approx(1)
    value[20:] += 10000
    np.testing.assert_allclose(native_fit_point_scale(value, mask, fit), scale)


def test_budget_reuses_three_complete150m_passes_no_staging(tmp_path):
    cfg = yaml.safe_load(Path("configs/five-wave-training-20261024.yaml").read_text())
    contract = runner.budget_contract(cfg)
    assert contract == dict(
        distinct_original_train_rows=150000000,
        epochs=3,
        presentations=450000000,
        expected_encoder_updates=13734,
    )
    with pytest.raises(ValueError, match="three"):
        runner.budget_contract(dict(cfg, reconstruction_training_rows=9000000))
    with pytest.raises(FileNotFoundError, match="no restaging"):
        runner.reuse_existing_cache(
            dict(cfg, reconstruction_cache=str(tmp_path / "absent")), {}
        )
    tiny = np.arange(5)[:, None]
    visits = list(runner.shared.replay_batches(tiny, 15, 3))
    assert sum(len(rows) for rows, _ in visits) == 15
    assert [rows.tolist() for rows, _ in visits] == [[0, 1, 2], [3, 4]] * 3


def test_xp_source_flags_not_inferred_from_input_missingness():
    visible = np.ones((3, 4), bool)
    visible[1:, [1, 2]] = False
    data = dict(
        visible=visible,
        real_xp_present=np.array([True, False, False]),
        real_no_xp=np.array([False, True, False]),
    )
    result = runner.availability(data, [1, 2])
    assert result["input_has_xp"].tolist() == [True, False, False]
    assert result["real_no_xp"].tolist() == [False, True, False]
    with pytest.raises(ValueError, match="Conflicting"):
        runner.availability(dict(data, real_xp_present=np.ones(3, bool)), [1, 2])


def test_native_runner_rejects_cpu_before_loading_data(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CANFAR"):
        runner.run(
            SimpleNamespace(
                config=tmp_path / "missing", run_id="notrun", output=tmp_path / "out"
            )
        )
    assert not (tmp_path / "out").exists()


def test_evaluation_batch_cache_preserves_identity_and_rejects_tamper(tmp_path):
    import json

    path = tmp_path / "batch.npz"
    ids = np.array([12, 13], dtype="i8")
    assert runner.load_evaluation_batch(path, ids) is None
    np.savez_compressed(path, _source_id=ids, point_native=np.zeros((2, 6)))
    with pytest.raises(ValueError, match="Incomplete"):
        runner.load_evaluation_batch(path, ids)
    receipt = path.with_suffix(".sha256.json")
    receipt.write_text(json.dumps(dict(sha256=runner.shared.sha(path))))
    assert runner.load_evaluation_batch(path, ids)["point_native"].shape == (2, 6)
    with pytest.raises(AssertionError):
        runner.load_evaluation_batch(path, ids[::-1])
    np.savez_compressed(path, _source_id=ids, point_native=np.ones((2, 6)))
    with pytest.raises(ValueError, match="Changed"):
        runner.load_evaluation_batch(path, ids)
