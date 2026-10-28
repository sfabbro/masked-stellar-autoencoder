import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stellar_catalog_jobs import identity_split
from stellar_enriched_views import fit_masks


def test_validation_never_fitted_even_when_identity_train_and_large_ids_exact():
    ids = np.arange(2**60, 2**60 + 100, dtype="i8")
    eligible = ids[identity_split(ids) == 0]
    train = eligible[:3]
    validation = eligible[3:6]
    split, membership, broad, strict = fit_masks(ids, train, validation)
    assert not broad[np.isin(ids, validation)].any()
    assert not strict[np.isin(ids, validation)].any()
    assert strict.sum() == 3
    assert broad.sum() == len(eligible) - 3
    assert (membership[np.isin(ids, train)] == 0).all()
    assert (membership[np.isin(ids, validation)] == 1).all()
    assert (membership[~np.isin(ids, np.r_[train, validation])] == 255).all()
    np.testing.assert_array_equal(split, identity_split(ids))
