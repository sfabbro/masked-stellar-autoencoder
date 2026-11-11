import numpy as np
import pandas as pd
import pytest
import torch

from masked_stellar_autoencoder.training import full_input_assembly as fia
from masked_stellar_autoencoder.training.context_latent import ContextLatent


def test_context_latent_is_identity_at_init_and_learns_from_metadata():
    torch.manual_seed(0)
    module = ContextLatent(latent_dim=8, context_dim=3)
    z = torch.randn(5, 8)
    values = torch.tensor([[1.0, float("nan"), 2.0]] * 5)
    visible = torch.isfinite(values)
    torch.testing.assert_close(module(z, values, visible), z)
    out = module(z, values, visible).sum()
    out.backward()
    assert module.net[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        module(z, values[:, :2], visible[:, :2])


def test_missing_metadata_is_masked_not_zero():
    module = ContextLatent(latent_dim=4, context_dim=2)
    with torch.no_grad():
        module.net[-1].weight.normal_()
    z = torch.zeros(2, 4)
    a = module(z, torch.tensor([[0.0, 0.0]] * 2), torch.tensor([[True, True]] * 2))
    b = module(
        z, torch.tensor([[float("nan")] * 2] * 2), torch.tensor([[False] * 2] * 2)
    )
    assert not torch.allclose(a, b)  # observed zero differs from missing


def test_full_input_assembly_scaling_merging_and_context():
    names = ["G", "BP", "PARALLAX", "EBV", "pmra", "pmdec", "RA", "DEC"]
    center = np.zeros(len(names))
    scale = np.full(len(names), 2.0)
    raw = np.array([[15.0, -9999.0, 1.0, 0.1, 5.0, -3.0, 10.0, 20.0]])
    std = fia.standardize(raw, center, scale)
    assert np.isnan(std[0, 1]) and std[0, 0] == 7.5
    base = np.array(
        [[np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], "f4"
    )
    gaia = fia.gaia_frame_to_matrix(
        pd.DataFrame(
            dict(phot_g_mean_mag=[14.0], phot_bp_mean_mag=[14.5], parallax=[2.0])
        ),
        names,
    )
    filled = fia.merge_rows(base, fia.standardize(gaia, center, scale), overwrite=False)
    assert filled[0, 0] == 7.0 and filled[0, 1] == 1.0 and filled[0, 2] == 1.0
    over = fia.merge_rows(base, fia.standardize(gaia, center, scale), overwrite=True)
    assert over[0, 1] == 7.25
    enc = fia.to_encoder(filled)
    assert enc[0, 3] == -9999.0 and enc[0, 0] == 7.0
    ctx = fia.context_raw(
        parallax=[1.0],
        parallax_error=[0.1],
        ebv=[np.nan],
        pmra=[5.0],
        pmdec=[-3.0],
        pmra_error=[0.0],
        pmdec_error=[0.2],
        ra=[0.0],
        dec=[90.0],
    )
    assert ctx.shape == (1, 10) and ctx[0, 1] == pytest.approx(-1.0)
    assert np.isnan(ctx[0, 2]) and np.isnan(ctx[0, 5])  # missing EBV, zero error
    assert ctx[0, 9] == pytest.approx(1.0)
    c, s = fia.context_scaler(np.vstack([ctx, ctx * 2]), np.array([0, 1]))
    assert s[7:].tolist() == [1, 1, 1] and c[7:].tolist() == [0, 0, 0]
