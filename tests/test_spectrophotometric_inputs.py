import pytest
import torch

from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)


def test_alias_counterfactual_cannot_change_any_safe_input():
    names = [
        "G",
        "bp_1",
        "rp_55",
        "PARALLAX",
        "parallax_snr",
        "e_parallax",
        "r_med_photogeo",
        "distance_modulus",
        "age",
        "teff",
        "fe_h",
        "RA",
        "pmra",
        "new_target_alias",
    ]
    x = torch.arange(2 * len(names), dtype=torch.float32).reshape(2, -1)
    original = x.clone()
    observed = torch.ones_like(x, dtype=torch.bool)
    safe, visible = spectrophotometric_inputs(x, observed, names)
    changed = x.clone()
    changed[:, 3:] = float("nan")
    other, other_visible = spectrophotometric_inputs(changed, observed, names)
    assert torch.equal(safe, other) and torch.equal(visible, other_visible)
    assert torch.equal(x, original)
    assert torch.equal(safe[:, :3], x[:, :3])
    assert torch.all(safe[:, 3:] == -9999)


def test_missingness_and_blocked_feature_gradients():
    x = torch.tensor([[12.0, float("nan"), 5.0, 0.1, -9999.0]], requires_grad=True)
    observed = torch.tensor([[True, True, False, True, True]])
    safe, visible = spectrophotometric_inputs(
        x, observed, ["G", "bp_1", "J", "PARALLAX", "H"]
    )
    assert torch.equal(visible, torch.tensor([[True, False, False, False, False]]))
    assert torch.isfinite(safe).all()
    safe.sum().backward()
    assert torch.equal(x.grad, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))


def test_no_random_inference_mask_and_explicit_reddening_policy():
    x = torch.tensor([[1.0, 0.2, 3.0]])
    observed = torch.ones_like(x, dtype=torch.bool)
    for seed in (1, 2, 99):
        torch.manual_seed(seed)
        safe, visible = spectrophotometric_inputs(x, observed, ["G", "EBV", "bp_55"])
        assert torch.equal(safe, x) and visible.all()
    safe, visible = spectrophotometric_inputs(
        x, observed, ["G", "EBV", "bp_55"], include_reddening=False
    )
    assert safe[0, 1] == -9999 and not visible[0, 1]


def test_unknown_xp_orders_and_ambiguous_schema_rejected_or_hidden():
    x = torch.ones(1, 3)
    safe, visible = spectrophotometric_inputs(
        x, torch.ones_like(x, dtype=torch.bool), ["bp_0", "bp_56", "bp_1_error"]
    )
    assert torch.all(safe == -9999) and not visible.any()
    with pytest.raises(ValueError, match="duplicate"):
        spectrophotometric_inputs(
            torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), ["G", "g"]
        )
    with pytest.raises(ValueError, match="boolean"):
        spectrophotometric_inputs(x, torch.ones_like(x), ["G", "BP", "RP"])
