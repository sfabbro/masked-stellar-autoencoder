"""``msa-split-v1`` partitions for any set of Gaia DR3 source ids.

Spec: ``docs/programme/data.md`` section 3, ``docs/programme/datasets.yaml``;
frozen at ``/arc/projects/astroai/msa/runs/msa_split_v1`` by
``scripts/build_msa_split_v1.py``. The base partition is a pure function of the
key (salted splitmix64 of ``source_id``); the freeze stores only the objects it
moved, in ``overrides.parquet`` (co-natal group members, which take their
group's partition, and reserved QA objects). So a star's frozen partition is
its override when it has one, else its hash partition: exactly what the
per-pixel ``assignment/`` files hold, without reading 1.8B rows.

Partition codes: 0 FIT, 1 DEV, 2 CAL, 3 TEST, 4 QA_RESERVED. Input classes
(from the freeze where listed): 0 X, 1 P, 2 G, 3 N.

This is the one implementation of the hash in the package (the v11 data
streamer and the evaluator import it); the frozen builder
``scripts/build_msa_split_v1.py`` keeps its own copy, whose script checksum is
part of the freeze, and ``tests/test_msa_split.py`` checks the two agree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SPEC = "msa-split-v1"
FREEZE = Path("/arc/projects/astroai/msa/runs/msa_split_v1")
EDGES = (0.75, 0.80, 0.88)
PARTITIONS = ("FIT", "DEV", "CAL", "TEST", "QA_RESERVED")
FIT, DEV, CAL, TEST, RESERVED = range(5)
INPUT_CLASSES = ("X", "P", "G", "N")
# datasets.yaml hash.test_vectors (integer keys)
TEST_VECTORS = (
    (4185355646116022528, 0.662500593453, FIT),
    (4398046511104, 0.798229566688, DEV),
    (4295806720, 0.852028592613, CAL),
    (4295806753, 0.882257348434, TEST),
)


def salt64(salt: bytes = b"msa-split-v1") -> int:
    return int.from_bytes(hashlib.sha256(salt).digest()[:8], "little")


SALT64 = salt64()
REC_SALT64 = salt64(b"msa-rec-v1")  # nested REC samples: unit(key, REC_SALT64) < rate


def unit(source_id, salt: int = SALT64) -> np.ndarray:
    """Salted splitmix64 finaliser: Gaia ids -> float64 in [0, 1)."""
    a = np.asarray(source_id)
    if a.dtype.kind == "f":
        raise TypeError("source ids stored as float lose precision")
    a = a.astype(np.int64)
    if np.any(a < 0):
        raise ValueError("negative source id")
    z = a.view(np.uint64) ^ np.uint64(salt)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


def partition_of(u) -> np.ndarray:
    """Partition code (0 FIT .. 3 TEST) of unit-hash values."""
    return np.searchsorted(np.asarray(EDGES), np.asarray(u), side="right").astype(
        np.uint8
    )


def hash_partition(source_id) -> np.ndarray:
    """Partition from the hash alone (no groups, no reservations)."""
    return partition_of(unit(source_id))


class MsaSplit:
    """Frozen ``msa-split-v1`` assignment: hash partition + freeze overrides."""

    def __init__(self, override_ids, override_partition, override_group=None,
                 override_class=None, reason=None):  # fmt: skip
        ids = np.asarray(override_ids, np.int64)
        order = np.argsort(ids, kind="stable")
        self._ids = ids[order]
        self._part = np.asarray(override_partition, np.uint8)[order]
        n = len(ids)
        self._group = (
            np.asarray(override_group, np.int64)[order]
            if override_group is not None
            else np.full(n, -1, np.int64)
        )
        self._cls = (
            np.asarray(override_class, np.int16)[order]
            if override_class is not None
            else np.full(n, -1, np.int16)
        )
        self._reason = (
            np.asarray(reason, object)[order]
            if reason is not None
            else np.full(n, "", object)
        )

    @classmethod
    def from_freeze(cls, freeze=FREEZE) -> MsaSplit:
        import pandas as pd

        o = pd.read_parquet(
            Path(freeze) / "overrides.parquet",
            columns=[
                "source_id",
                "partition",
                "group_id",
                "input_class",
                "reserved_reason",
            ],
        )
        return cls(
            o["source_id"].to_numpy(np.int64), o["partition"].to_numpy(np.uint8),
            o["group_id"].to_numpy(np.int64), o["input_class"].to_numpy(np.int16),
            o["reserved_reason"].fillna("").to_numpy(object),
        )  # fmt: skip

    def _lookup(self, source_id):
        sid = np.asarray(source_id, np.int64)
        if not len(self._ids):
            return sid, np.zeros(len(sid), np.int64), np.zeros(len(sid), bool)
        pos = np.searchsorted(self._ids, sid).clip(max=len(self._ids) - 1)
        return sid, pos, self._ids[pos] == sid

    def partition(self, source_id) -> np.ndarray:
        """Frozen partition code (0-4) per id."""
        sid, pos, hit = self._lookup(source_id)
        out = hash_partition(sid)
        out[hit] = self._part[pos[hit]]
        return out

    def group_id(self, source_id) -> np.ndarray:
        """Row of ``groups.parquet`` (-1 when the star is in no group)."""
        sid, pos, hit = self._lookup(source_id)
        out = np.full(len(sid), -1, np.int64)
        out[hit] = self._group[pos[hit]]
        return out

    def reserved_reason(self, source_id) -> np.ndarray:
        sid, pos, hit = self._lookup(source_id)
        out = np.full(len(sid), "", object)
        out[hit] = self._reason[pos[hit]]
        return out


def check_test_vectors() -> None:
    for key, u_spec, part in TEST_VECTORS:
        u = float(unit(np.array([key]))[0])
        if abs(u - u_spec) > 1e-12 or int(hash_partition(np.array([key]))[0]) != part:
            raise AssertionError(f"msa-split-v1 test vector failed for {key}")
