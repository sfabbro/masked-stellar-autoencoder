import numpy as np
import pytest

from masked_stellar_autoencoder.training.decision_pdf_metrics import sample_scores


def test_empirical_crps_and_masks_are_not_gaussianized():
    draws = np.array([-1.0, 0.0, 1.0])[:, None, None]
    out = sample_scores(draws, np.zeros((1, 1)), np.ones((1, 1), bool))
    np.testing.assert_allclose(out["crps"], 2 / 9)
    np.testing.assert_allclose(out["pit"], 2 / 3)
    missing = sample_scores(draws, np.full((1, 1), np.nan), np.zeros((1, 1), bool))
    assert np.isnan(missing["crps"]).all() and np.isnan(missing["energy"]).all()
    with pytest.raises(ValueError):
        sample_scores(draws[:1], np.zeros((1, 1)), np.ones((1, 1), bool))
