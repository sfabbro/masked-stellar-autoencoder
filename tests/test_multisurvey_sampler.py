"""Finite-population and checkpoint contracts for survey-stratified sampling."""

import itertools
import json

import numpy as np
import pytest

from masked_stellar_autoencoder.training.multisurvey_sampler import (
    StratifiedStarSampler,
)

IDS = [11, 12, 13, 14, 21, 22]
MASKS = [1, 1, 1, 1, 3, 3]  # Bitmask 3 denotes one overlap stratum, not two stars.


def test_exact_stratified_weighted_expectation_matches_population_mean():
    sampler = StratifiedStarSampler(IDS, MASKS, {1: 1, 3: 1}, seed=7)
    batch = sampler.sample()
    weights = dict(zip(batch["presence_mask"], batch["weight"], strict=True))
    assert weights[1] == pytest.approx(4 / 3)
    assert weights[3] == pytest.approx(2 / 3)
    assert sum(batch["weight"]) == pytest.approx(2)
    values = {11: 2.0, 12: -3.0, 13: 20.0, 14: 8.0, 21: 100.0, 22: -1.0}
    possible = itertools.product(IDS[:4], IDS[4:])
    estimates = [
        (weights[1] * values[a] + weights[3] * values[b]) / 2 for a, b in possible
    ]
    assert np.mean(estimates) == pytest.approx(np.mean(list(values.values())))


def test_each_batch_has_unique_sources_and_fixed_presence_strata():
    sampler = StratifiedStarSampler(IDS, MASKS, {1: 2, 3: 2}, seed=38)
    for _ in range(8):
        batch = sampler.sample()
        assert len(np.unique(batch["source_id"])) == 4
        assert np.count_nonzero(batch["presence_mask"] == 1) == 2
        assert np.count_nonzero(batch["presence_mask"] == 3) == 2
        assert np.array_equal(
            np.asarray(IDS)[batch["input_position"]], batch["source_id"]
        )
        assert batch["weight"].sum() == pytest.approx(4)
    # Allocating all of the second stratum forces it to recur in every batch.
    assert {21, 22}.issubset(set(batch["source_id"]))


def test_state_restores_rng_and_is_independent_of_input_order():
    original = StratifiedStarSampler(IDS, MASKS, {1: 1, 3: 2}, seed=10)
    original.sample()
    state = json.loads(json.dumps(original.state_dict()))
    expected = original.sample()
    order = [5, 1, 3, 0, 4, 2]
    reordered_ids = np.asarray(IDS)[order]
    reordered_masks = np.asarray(MASKS)[order]
    restored = StratifiedStarSampler(
        reordered_ids, reordered_masks, {3: 2, 1: 1}, seed=55
    )
    restored.load_state_dict(state)
    actual = restored.sample()
    for key in ("source_id", "presence_mask", "weight"):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_array_equal(
        reordered_ids[actual["input_position"]], actual["source_id"]
    )
    assert original.contract_sha256 == restored.contract_sha256
    assert (
        original.state_dict()["batches_drawn"]
        == restored.state_dict()["batches_drawn"]
        == 2
    )


def test_restore_rejects_changed_population_strata_allocations_and_rng_without_mutating_state():
    original = StratifiedStarSampler(IDS, MASKS, {1: 1, 3: 1}, seed=10)
    state = original.state_dict()
    changed = [
        StratifiedStarSampler([10, *IDS[1:]], MASKS, {1: 1, 3: 1}, seed=10),
        StratifiedStarSampler(IDS, [1, 1, 1, 3, 1, 3], {1: 1, 3: 1}, seed=10),
        StratifiedStarSampler(IDS, MASKS, {1: 2, 3: 1}, seed=10),
    ]
    for sampler in changed:
        before = sampler.state_dict()
        with pytest.raises(ValueError, match="contract differs"):
            sampler.load_state_dict(state)
        assert sampler.state_dict() == before
    bad = json.loads(json.dumps(state))
    bad["rng_state"]["bit_generator"] = "MT19937"
    with pytest.raises(ValueError, match="random state"):
        original.load_state_dict(bad)
    assert original.state_dict() == state
    bad = json.loads(json.dumps(state))
    bad["batches_drawn"] = True
    with pytest.raises(ValueError, match="batch count"):
        original.load_state_dict(bad)
    assert original.state_dict() == state


def test_returned_state_and_contract_are_detached():
    sampler = StratifiedStarSampler(IDS, MASKS, {1: 1, 3: 1}, seed=5)
    expected = sampler.state_dict()
    state = sampler.state_dict()
    state["contract"]["strata"][0]["allocation"] = 42
    state["rng_state"]["state"]["state"] = 0
    assert sampler.state_dict() == expected
    contract = sampler.contract
    contract["strata"][0]["population"] = 0
    assert sampler.state_dict() == expected


@pytest.mark.parametrize(
    "ids,masks,allocation,seed",
    [
        ([], [], {}, 1),
        ([1, 1], [1, 2], {1: 1, 2: 1}, 1),
        ([1.0], [1], {1: 1}, 1),
        ([True], [1], {1: 1}, 1),
        ([0], [1], {1: 1}, 1),
        ([-1], [1], {1: 1}, 1),
        (np.array([2**63], dtype=np.uint64), [1], {1: 1}, 1),
        ([1], [1.0], {1: 1}, 1),
        ([1], [0], {1: 1}, 1),
        ([1], [-1], {1: 1}, 1),
        ([1, 2], [1], {1: 1}, 1),
        ([1, 2], [1, 2], {1: 1}, 1),
        ([1], [1], {1: 1, 2: 1}, 1),
        ([1], [1], {1: 2}, 1),
        ([1], [1], {1: 0}, 1),
        ([1], [1], {1: 1.0}, 1),
        ([1], [1], {True: 1}, 1),
        ([1], [1], {1: True}, 1),
        ([1], [1], {2**64: 1}, 1),
        ([1], [1], {1: 1}, True),
        ([1], [1], {1: 1}, -1),
    ],
)
def test_invalid_sampling_contracts_fail(ids, masks, allocation, seed):
    with pytest.raises(ValueError):
        StratifiedStarSampler(ids, masks, allocation, seed=seed)


def test_full_uint64_presence_mask_is_exact():
    masks = np.array([1, 2**63], dtype=np.uint64)
    sampler = StratifiedStarSampler([1, 2], masks, {1: 1, 2**63: 1}, seed=2)
    assert set(map(int, sampler.sample()["presence_mask"])) == {1, 2**63}


def test_validate_batch_accepts_joint_reordering_and_restored_last_draw_without_rng_change():
    sampler = StratifiedStarSampler(IDS, MASKS, {1: 2, 3: 1}, seed=90)
    batch = sampler.sample()
    before = sampler.state_dict()
    order = [2, 0, 1]
    sampler.validate_batch(
        batch["source_id"][order], batch["presence_mask"][order], batch["weight"][order]
    )
    assert sampler.state_dict() == before
    restored = StratifiedStarSampler(IDS[::-1], MASKS[::-1], {1: 2, 3: 1}, seed=3)
    restored.load_state_dict(json.loads(json.dumps(before)))
    restored.validate_batch(batch["source_id"], batch["presence_mask"], batch["weight"])
    assert restored.state_dict() == before
    for key in ("source_id", "presence_mask", "weight"):
        np.testing.assert_array_equal(sampler.sample()[key], restored.sample()[key])
        # Both generators advance once per comparison; their states remain equal.


def test_validate_batch_rejects_forged_values_or_misalignment_without_rng_change():
    sampler = StratifiedStarSampler(IDS, MASKS, {1: 1, 3: 1}, seed=9)
    batch = sampler.sample()
    ids, masks, weights = (
        batch[k].copy() for k in ("source_id", "presence_mask", "weight")
    )
    before = sampler.state_dict()
    with pytest.raises(ValueError, match="mapping differs"):
        sampler.validate_batch(ids, masks[::-1], weights)
    with pytest.raises(ValueError, match="sampling weights"):
        sampler.validate_batch(ids, masks, weights[::-1])
    changed = weights.copy()
    changed[0] *= 1.000001
    with pytest.raises(ValueError, match="sampling weights"):
        sampler.validate_batch(ids, masks, changed)
    changed = weights.copy()
    changed[0] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        sampler.validate_batch(ids, masks, changed)
    changed = weights.copy()
    changed[0] = 0
    with pytest.raises(ValueError, match="finite and positive"):
        sampler.validate_batch(ids, masks, changed)
    changed_ids = ids.copy()
    changed_ids[0] = 999
    with pytest.raises(ValueError, match="outside the bound population"):
        sampler.validate_batch(changed_ids, masks, weights)
    with pytest.raises(ValueError, match="unique"):
        sampler.validate_batch([ids[0], ids[0]], masks, weights)
    with pytest.raises(ValueError, match="stratum counts"):
        sampler.validate_batch([11, 12], [1, 1], [4 / 3, 4 / 3])
    h1_index = int(np.flatnonzero(masks == 1)[0])
    other_in_same_stratum = next(i for i in IDS[:4] if i != ids[h1_index])
    changed_ids = ids.copy()
    changed_ids[h1_index] = other_in_same_stratum
    with pytest.raises(ValueError, match="most recent sampled identities"):
        sampler.validate_batch(changed_ids, masks, weights)
    with pytest.raises(ValueError, match="must align"):
        sampler.validate_batch(ids, masks, weights[:1])
    assert sampler.state_dict() == before


def test_validate_batch_allows_double_roundoff_but_not_float32_rounding():
    sampler = StratifiedStarSampler(IDS, MASKS, {1: 1, 3: 1}, seed=4)
    batch = sampler.sample()
    close = np.nextafter(batch["weight"], np.inf)
    sampler.validate_batch(batch["source_id"], batch["presence_mask"], close)
    with pytest.raises(ValueError, match="sampling weights"):
        sampler.validate_batch(
            batch["source_id"],
            batch["presence_mask"],
            batch["weight"].astype(np.float32),
        )


def test_restore_rejects_invalid_last_draw_identity_receipt():
    sampler = StratifiedStarSampler(IDS, MASKS, {1: 1, 3: 1}, seed=3)
    sampler.sample()
    state = sampler.state_dict()
    for bad_last in (None, [11, 11], [11, 12], [11, 999]):
        altered = json.loads(json.dumps(state))
        altered["last_source_ids"] = bad_last
        with pytest.raises(ValueError, match="last sampled source identities"):
            sampler.load_state_dict(altered)
        assert sampler.state_dict() == state
