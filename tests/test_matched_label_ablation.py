import numpy as np

from masked_stellar_autoencoder.training.matched_label_ablation import (
    desi_logg_masks,
    gravity_pool,
)


def test_dropping_logg_cannot_change_record_eligibility_or_sample_sequence():
    observed = np.zeros((30, 6), bool)
    observed[:25, :3] = True
    observed[25, 1] = True
    observed[26, 0] = True
    observed[27, 3] = True
    a = desi_logg_masks(observed, observed, False)
    b = desi_logg_masks(observed, observed, True)
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3], b[3])
    assert not a[3][25] and not b[3][25]
    assert a[0][0, 1] and not b[0][0, 1]
    np.testing.assert_array_equal(a[0][:, [0, 2, 3, 4, 5]], b[0][:, [0, 2, 3, 4, 5]])
    ga = np.random.default_rng(4)
    gb = np.random.default_rng(4)
    physical = np.linspace(0, 5, 30)
    for update in range(1, 60):
        pa = gravity_pool(a[2][a[3]], physical[a[3]], update, 7)
        pb = gravity_pool(b[2][b[3]], physical[b[3]], update, 7)
        np.testing.assert_array_equal(
            ga.choice(pa, size=min(8, len(pa)), replace=False),
            gb.choice(pb, size=min(8, len(pb)), replace=False),
        )
