import math

import numpy as np
import torch

from masked_stellar_autoencoder.training import likelihood_head as lh
from masked_stellar_autoencoder.training import likelihood_validation as V


def _linear_gaussian(n, seed=0):
    """z = W theta + noise; theta from a skewed population (10 % at -2)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(8, 2, generator=torch.Generator().manual_seed(123))
    rare = torch.rand(n, generator=g) < 0.1
    t1 = torch.where(rare, -2.0, 0.0) + 0.3 * torch.randn(n, generator=g)
    t2 = torch.randn(n, generator=g)
    theta = torch.stack([t1, t2], 1)
    z = theta @ W.T + 0.3 * torch.randn(n, 8, generator=g)
    return z, theta, rare, W


def test_score_fn_matches_forward_and_standardises():
    head = lh.RatioHead(8, 2, embed_dims=(16,), embed_out=4, hidden=(16,))
    head.set_standardisation(np.ones(8), 2 * np.ones(8), [1.0, -1.0], [2.0, 0.5])
    z, theta = torch.randn(5, 8), torch.randn(5, 2)
    e = head.embed(z)
    torch.testing.assert_close(head.score_fn()(theta, e), head(theta, z))
    torch.testing.assert_close(head.standardise(theta)[:, 0], (theta[:, 0] - 1.0) / 2.0)
    torch.testing.assert_close(e, head.embedder((z - 1.0) / 2.0))


def test_ratio_head_mle_is_unbiased_for_the_rare_population(tmp_path):
    torch.manual_seed(0)
    z, theta, rare, W = _linear_gaussian(8000)
    head = lh.RatioHead(8, 2, embed_dims=(64,), embed_out=16, hidden=(64, 64))
    train, valid = torch.arange(6000), torch.arange(6000, 7000)
    hist = lh.train_ratio_head(head, z, theta, train, valid, max_epochs=25,
                               patience=25, lr=3e-3, log=lambda m: None)  # fmt: skip
    assert hist["best_valid"] < hist["valid"][0]
    lh.save_head(head, tmp_path / "h.pt", history=hist)
    head, blob = lh.load_head(tmp_path / "h.pt")
    test = np.arange(7000, 8000)
    emb = lh.embed_rows(head, z, test, device="cpu")
    bounds = [[-4.0, 2.0], [-4.0, 4.0]]
    mle, smax, atb = lh.maximize(head, emb, bounds, init=np.zeros((1000, 2)),
                                 n_candidates=64, log=lambda m: None)  # fmt: skip
    ls = torch.linalg.lstsq(W, z[test].T).solution.T.numpy()  # analytic MLE
    assert np.corrcoef(mle[:, 0], ls[:, 0])[0, 1] > 0.95
    r = rare[test].numpy()
    t = theta[test].numpy()
    assert abs(np.median(mle[r, 0] - t[r, 0])) < 0.15
    lo, hi, lob, hib = lh.profiles(
        head, emb, mle, smax, 0.5, bounds, log=lambda m: None
    )
    assert np.all(lo <= mle + 1e-4) and np.all(hi >= mle - 1e-4)
    cover = np.mean((lo[:, 0] <= t[:, 0]) & (t[:, 0] <= hi[:, 0]))
    assert 0.4 < cover < 0.95
    lam = lh.lr_statistic(head, emb, t, smax)
    assert np.all(lam >= 0) and np.median(lam) < 2.0
    np.testing.assert_allclose(lh.lr_statistic(head, emb, mle, smax), 0.0, atol=1e-3)
    # Profile statistic: 0 at the MLE, never above the joint statistic.
    lp = lh.profile_statistic(head, emb, 0, t[:, 0], smax, mle, bounds, n_candidates=16)
    assert np.all(lp <= lam + 1e-3) and np.all(lp >= 0)
    l0 = lh.profile_statistic(
        head, emb, 0, mle[:, 0], smax, mle, bounds, n_candidates=16
    )
    np.testing.assert_allclose(l0, 0.0, atol=1e-3)


def test_offset_selection_and_decision_tables():
    rng = np.random.default_rng(0)
    n = 4000
    ref = rng.uniform(-4, 0.5, n)
    G = rng.uniform(10, 17.5, n)
    good = ref + rng.normal(0, 0.05, n)
    biased = good + np.where(ref < -2, 0.3, 0.0)
    tg, tb = V.offset_table(good, ref, G), V.offset_table(biased, ref, G)
    row = next(r for r in tb if r["feh_lo"] == -3.0 and r["g"] == "G<=16")
    assert abs(row["bias"] - 0.3) < 0.02 and row["n"] > 50
    sel = V.selection_table(ref, G, {"good": good < -2.5, "biased": biased < -2.5})
    assert sel["all"]["good"]["purity"] > 0.9
    assert sel["all"]["biased"]["completeness"] < sel["all"]["good"]["completeness"]
    covered = rng.uniform(size=n) < 0.68
    theta = np.column_stack([rng.uniform(4000, 6000, n), rng.uniform(1, 5, n), ref])
    cov = V.coverage_table(theta[G <= 16], G[G <= 16], covered[G <= 16], 0.68)
    assert abs(cov["all"]["coverage"] - 0.68) < 0.03
    anchors = {"teff_anchor": {"mle": dict(n=100, bias=5.0, nmad=40.0),
                               "posterior": dict(n=100, bias=0.0, nmad=40.0)}}  # fmt: skip
    d = V.decide(tg, tg, cov, anchors, "mle")
    assert d["adopt"], d
    d = V.decide(tb, tg, cov, anchors, "mle")
    assert not d["adopt"] and not d["criteria"]["bias"]["passed"]
    assert d["criteria"]["coverage"]["passed"] and d["criteria"]["anchors"]["passed"]
    bad = {"teff_anchor": {"mle": dict(n=100, bias=30.0, nmad=40.0),
                           "posterior": dict(n=100, bias=0.0, nmad=40.0)}}  # fmt: skip
    assert not V.decide(tg, tg, cov, bad, "mle")["criteria"]["anchors"]["passed"]
    assert math.isfinite(V.nmad(good - ref))
