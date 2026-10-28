"""Reproducible, population-weighted sampling of unique survey-labeled stars.

A stratum is the complete nonzero survey-presence bitmask of one eligible star;
bitmasks such as 1, 2 and 3 are *disjoint categories*, not separate source pools.
All nonempty strata participate in every batch. Labels, roles, and provenance
must already have been validated by the caller; this module cannot verify them.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

import numpy as np


def _positive_integer_vector(values, *, name: str, signed: bool) -> np.ndarray:
    """Reject floating or lossy identities, masks, and empty vectors."""
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty exact integer vector")
    if np.any(array == 0) or (array.dtype.kind == "i" and np.any(array < 0)):
        raise ValueError(f"{name} must contain positive integers")
    if signed and array.dtype.kind == "u" and np.any(array > np.iinfo("i8").max):
        raise ValueError(f"{name} exceeds signed int64")
    return np.array(array, dtype="<i8" if signed else "<u8", copy=True)


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an exact positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be an exact positive integer")
    return result


class StratifiedStarSampler:
    """Draw unique stars from fixed survey-presence strata in each batch.

    Within each stratum, a batch is sampled uniformly without replacement. Batches
    are independent draws, so a star can reappear in later batches. For stratum
    h, every selected star receives w_h = (N_h/N)/(n_h/B). Therefore the batch
    mean of w_h times any fixed per-star loss is an unbiased estimator of the
    uniform eligible-star mean. The weights sum to B in every batch; they must
    be applied exactly once to per-star losses (not separately to measurements).
    """

    def __init__(self, source_ids, presence_masks, allocations: Mapping, *, seed: int):
        ids = _positive_integer_vector(source_ids, name="source_ids", signed=True)
        masks = _positive_integer_vector(
            presence_masks, name="presence_masks", signed=False
        )
        if masks.shape != ids.shape:
            raise ValueError("Source IDs and presence masks must align")
        order = np.argsort(ids, kind="stable")
        ids, masks = ids[order], masks[order]
        if np.any(np.diff(ids) == 0):
            raise ValueError("Source IDs must be unique")
        if not isinstance(allocations, Mapping) or not allocations:
            raise ValueError("A nonempty per-stratum allocation mapping is required")
        counts = {}
        for raw_mask, raw_count in allocations.items():
            mask = _positive_integer(raw_mask, "Allocation stratum")
            count = _positive_integer(raw_count, "Allocation count")
            if mask > np.iinfo("u8").max:
                raise ValueError("Allocation stratum exceeds uint64")
            if mask in counts:
                raise ValueError("Allocation strata must be unique")
            counts[mask] = count
        groups = {int(mask): np.flatnonzero(masks == mask) for mask in np.unique(masks)}
        if set(counts) != set(groups):
            raise ValueError("Every nonempty stratum needs exactly one allocation")
        if any(counts[mask] > len(indices) for mask, indices in groups.items()):
            raise ValueError("Allocation exceeds its stratum population")
        if (
            isinstance(seed, bool | np.bool_)
            or not isinstance(seed, int | np.integer)
            or seed < 0
        ):
            raise ValueError("Seed must be an exact nonnegative integer")

        self._ids = ids
        self._masks = masks
        self._input_positions = order
        self._groups = groups
        self._allocations = counts
        self._batch_size = sum(counts.values())
        self._rng = np.random.Generator(np.random.PCG64(int(seed)))
        self._batches_drawn = 0
        self._last_source_ids = None
        self._contract = {
            "schema": "multisurvey-stratified-star-sampler-v2",
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "population": len(ids),
            "source_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
            "presence_masks_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
            "strata": [
                {
                    "presence_mask": mask,
                    "population": len(groups[mask]),
                    "allocation": counts[mask],
                }
                for mask in sorted(groups)
            ],
        }
        self._contract_sha256 = hashlib.sha256(
            json.dumps(self._contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for array in (
            self._ids,
            self._masks,
            self._input_positions,
            *self._groups.values(),
        ):
            array.setflags(write=False)

    @property
    def contract(self) -> dict:
        """Return a detached, JSON-serializable sampling contract."""
        return copy.deepcopy(self._contract)

    @property
    def contract_sha256(self) -> str:
        return self._contract_sha256

    def sample(self) -> dict[str, np.ndarray]:
        """Return aligned original-input positions, IDs, strata and per-star weights."""
        indices, weights = [], []
        for mask in sorted(self._groups):
            pool = self._groups[mask]
            count = self._allocations[mask]
            indices.append(self._rng.choice(pool, size=count, replace=False))
            weight = (len(pool) * self._batch_size) / (len(self._ids) * count)
            weights.append(np.full(count, weight, dtype=np.float64))
        indices = np.concatenate(indices)
        weights = np.concatenate(weights)
        order = self._rng.permutation(self._batch_size)
        self._last_source_ids = np.sort(self._ids[indices]).copy()
        self._batches_drawn += 1
        return {
            "input_position": self._input_positions[indices[order]].copy(),
            "source_id": self._ids[indices[order]].copy(),
            "presence_mask": self._masks[indices[order]].copy(),
            "weight": weights[order],
        }

    def _validate_sample_ids(self, ids: np.ndarray) -> np.ndarray:
        """Find source rows and check the configured unique stratum counts."""
        if len(ids) != self._batch_size:
            raise ValueError("Batch size differs from the sampling allocation")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Batch source IDs must be unique")
        positions = np.searchsorted(self._ids, ids)
        if np.any(positions >= len(self._ids)) or not np.array_equal(
            self._ids[positions], ids
        ):
            raise ValueError("Batch contains a source ID outside the bound population")
        masks = self._masks[positions]
        for mask, count in self._allocations.items():
            if np.count_nonzero(masks == mask) != count:
                raise ValueError("Batch stratum counts differ from the allocation")
        return positions

    def validate_batch(self, source_ids, presence_masks, weights) -> None:
        """Validate aligned arrays against the latest draw without advancing RNG.

        A common permutation of all arrays is permitted. IDs must be the latest
        sampled set, because substituting another valid ID can bias the sample.
        A permutation only within a single stratum is indistinguishable when its
        masks and weights are identical; these are equivalent aligned inputs.
        """
        ids = _positive_integer_vector(source_ids, name="source_ids", signed=True)
        masks = _positive_integer_vector(
            presence_masks, name="presence_masks", signed=False
        )
        raw_weights = np.asarray(weights)
        if (
            ids.shape != masks.shape
            or raw_weights.shape != ids.shape
            or raw_weights.dtype.kind not in "iuf"
        ):
            raise ValueError("Batch IDs, masks and numeric weights must align")
        weight = np.asarray(raw_weights, dtype=np.float64)
        if not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError("Sampling weights must be finite and positive")
        positions = self._validate_sample_ids(ids)
        if not np.array_equal(masks, self._masks[positions]):
            raise ValueError("Batch ID and presence-stratum mapping differs")
        if self._last_source_ids is None or not np.array_equal(
            np.sort(ids), self._last_source_ids
        ):
            raise ValueError("Batch differs from the most recent sampled identities")
        expected = np.empty(len(ids), dtype=np.float64)
        for mask, count in self._allocations.items():
            expected[masks == mask] = (len(self._groups[mask]) * self._batch_size) / (
                len(self._ids) * count
            )
        if not np.allclose(weight, expected, rtol=8 * np.finfo(np.float64).eps, atol=0):
            raise ValueError("Batch sampling weights differ from the bound allocation")

    def state_dict(self) -> dict:
        """Return a detached portable contract and the exact current PCG64 state."""
        return {
            "contract": self.contract,
            "contract_sha256": self._contract_sha256,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "batches_drawn": self._batches_drawn,
            "last_source_ids": None
            if self._last_source_ids is None
            else self._last_source_ids.tolist(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore only an exact population, allocation, version and RNG binding."""
        if not isinstance(state, dict) or set(state) != {
            "contract",
            "contract_sha256",
            "rng_state",
            "batches_drawn",
            "last_source_ids",
        }:
            raise ValueError("Invalid sampler checkpoint schema")
        if (
            state["contract"] != self._contract
            or state["contract_sha256"] != self._contract_sha256
        ):
            raise ValueError("Sampler checkpoint contract differs")
        batches = state["batches_drawn"]
        if (
            isinstance(batches, bool | np.bool_)
            or not isinstance(batches, int | np.integer)
            or batches < 0
        ):
            raise ValueError("Invalid sampled-batch count")
        last = state["last_source_ids"]
        if batches == 0:
            if last is not None:
                raise ValueError("Unsampled checkpoint cannot have last source IDs")
            last_ids = None
        else:
            try:
                last_ids = _positive_integer_vector(
                    last, name="last_source_ids", signed=True
                )
                self._validate_sample_ids(last_ids)
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid last sampled source identities") from error
            last_ids = np.sort(last_ids)
        try:
            bit_generator = np.random.PCG64()
            bit_generator.state = copy.deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            raise ValueError("Invalid PCG64 random state") from error
        self._rng = np.random.Generator(bit_generator)
        self._batches_drawn = int(batches)
        self._last_source_ids = last_ids
