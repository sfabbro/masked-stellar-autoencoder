"""Streaming v11 training rows from HATS / parquet partitions with the frozen
``msa-split-v1`` assignment.

Layout. Every input is a tree with the HATS partition layout of
``/arc/projects/hats/gaia/dr3`` (``partition_info.csv`` and
``dataset/Norder=k/Dir=d/Npix=n.parquet`` or an ``Npix=n/`` directory). The first tree
is the *anchor* (one row per object: Gaia DR3, or the xmatcher master); the others
(``dr3_xp_continuous``, re-partitioned Gaia-keyed survey tables, ...) are joined to it
on ``source_id`` inside the same partition. The assignment
(``/arc/projects/astroai/msa/runs/msa_split_v1/assignment/Norder=k/Dir=d/Npix=n.parquet``:
``source_id, partition, group_id, input_class``) is joined the same way, so nothing
depends on row order.

Selection. Rows whose assignment partition is in ``train_partitions`` (REC = FIT
objects; DEV for REC-DEV) are kept; reserved QA objects (partition 4) never are. An
optional nested REC sample keeps ``unit(source_id, salt "msa-rec-v1") < rate``
(``datasets.yaml``). There is no row cap: every selected row of every partition is
streamed once per epoch.

Streaming. Each worker takes a disjoint, epoch-shuffled share of the partitions and
interleaves ``interleave_partitions`` of them. A partition is read in chunks of
``read_batch_rows`` anchor rows; the matching rows of each secondary tree are read as a
contiguous row range (HATS trees share the ``_healpix_29`` order, so ranges advance;
an out-of-order tree is still read correctly, with a re-open counted as a rewind).
Rows go to a shuffle buffer, then to buckets by token count, and leave as micro-batches
of about ``tokens_per_microbatch`` tokens (many short photometry-only rows, fewer long
XP rows), which bounds activation memory independently of the row mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from masked_stellar_autoencoder.data.msa_split import (  # noqa: F401  (re-exported)
    REC_SALT64,
    SALT64,
    partition_of,
    salt64,
    unit,
)

from .filters import FilterOperator, exact_band_variance
from .schema import N_XP, BandRegistry, ContractArrays, ContractReader, xp_normalisation

PARTITION_CODE = {"FIT": 0, "DEV": 1, "CAL": 2, "TEST": 3}
RESERVED = 4
INPUT_CLASSES = ("X", "P", "G", "N")


# ---------------------------------------------------------------- split hash
# One implementation of the msa-split-v1 hash: data/msa_split.py (names kept).
SPLIT_SALT64 = SALT64  # 0xb6f6e5353645106b
split_unit = unit
split_partition = partition_of


# ---------------------------------------------------------------- HATS layout


def hats_partitions(root: str | Path) -> list[tuple[int, int]]:
    lines = (Path(root) / "partition_info.csv").read_text().split()
    header = lines[0].split(",")
    io, ip = header.index("Norder"), header.index("Npix")
    out = []
    for ln in lines[1:]:
        f = ln.split(",")
        out.append((int(f[io]), int(f[ip])))
    return out


def partition_files(root: str | Path, order: int, npix: int) -> list[Path]:
    """Parquet files of one partition (``dataset/`` prefix optional, file or directory)."""
    root = Path(root)
    rel = Path(f"Norder={order}") / f"Dir={npix // 10000 * 10000}"
    for base in (root / "dataset" / rel, root / rel):
        d = base / f"Npix={npix}"
        if d.is_dir():
            return sorted(d.glob("*.parquet"))
        f = base / f"Npix={npix}.parquet"
        if f.is_file():
            return [f]
    return []


class ForwardReader:
    """Row-range reads over the parquet files of one partition, mostly forward."""

    def __init__(self, files: list[Path], columns: list[str] | None, batch_rows: int):
        import pyarrow.parquet as pq

        self.files = files
        self.columns = columns
        self.batch_rows = batch_rows
        self.groups = []  # (file index, row group, start, rows)
        start = 0
        for fi, f in enumerate(files):
            md = pq.ParquetFile(f).metadata
            for g in range(md.num_row_groups):
                n = md.row_group(g).num_rows
                self.groups.append((fi, g, start, n))
                start += n
        self.n_rows = start
        self.starts = np.asarray([g[2] for g in self.groups] + [start], np.int64)
        self.rewinds = 0
        self.jumps = 0
        self._reset(0)

    def _gen(self, gi0: int):
        import pyarrow.parquet as pq

        open_file, pf = -1, None
        for gi in range(gi0, len(self.groups)):
            fi, g, start, _ = self.groups[gi]
            if fi != open_file:
                pf = pq.ParquetFile(
                    self.files[fi], buffer_size=8 << 20, pre_buffer=False
                )
                open_file = fi
            pos = start
            for batch in pf.iter_batches(
                batch_size=self.batch_rows, row_groups=[g], columns=self.columns
            ):
                yield pos, batch, gi
                pos += batch.num_rows

    def _reset(self, row: int):
        gi = max(int(np.searchsorted(self.starts, row, side="right")) - 1, 0)
        gi = min(gi, max(len(self.groups) - 1, 0))
        self._it = self._gen(gi) if self.groups else iter(())
        self._buf = None
        self._buf_start = int(self.starts[gi]) if self.groups else 0
        self._gi = gi

    def read(self, lo: int, hi: int):
        import pyarrow as pa

        if not 0 <= lo < hi <= self.n_rows:
            raise IndexError(f"row range [{lo}, {hi}) outside [0, {self.n_rows})")
        if lo < self._buf_start:
            self.rewinds += 1
            self._reset(lo)
        elif (
            self._buf is not None and lo >= self.starts[self._gi + 1] + self.batch_rows
        ):
            # far ahead of the current row group: jump instead of decoding the gap
            nxt = int(np.searchsorted(self.starts, lo, side="right")) - 1
            if nxt > self._gi:
                self.jumps += 1
                self._reset(lo)
        parts = []
        while True:
            if self._buf is not None and self._buf_start + self._buf.num_rows > lo:
                a = max(lo - self._buf_start, 0)
                b = min(hi - self._buf_start, self._buf.num_rows)
                if b > a:
                    parts.append(pa.Table.from_batches([self._buf.slice(a, b - a)]))
                if self._buf_start + self._buf.num_rows >= hi:
                    break
            nxt = next(self._it, None)
            if nxt is None:
                raise IndexError("unexpected end of partition")
            self._buf_start, self._buf, self._gi = nxt
        return pa.concat_tables(parts) if len(parts) > 1 else parts[0]


def _lookup(
    sorted_ids: np.ndarray, sorted_pos: np.ndarray, query: np.ndarray
) -> np.ndarray:
    if not len(sorted_ids):
        return np.full(len(query), -1, np.int64)
    i = np.searchsorted(sorted_ids, query).clip(max=len(sorted_ids) - 1)
    return np.where(sorted_ids[i] == query, sorted_pos[i], -1)


def _sorted_index(ids: np.ndarray):
    order = np.argsort(ids, kind="stable")
    s = ids[order]
    if len(s) > 1 and np.any(s[1:] == s[:-1]):
        raise ValueError("duplicate source_id inside a partition")
    return s, order.astype(np.int64)


# ---------------------------------------------------------------- row blocks


@dataclass
class RowBlock:
    source_id: np.ndarray  # int64 [n]
    input_class: np.ndarray  # uint8 [n]
    values: np.ndarray  # float32 [n, n_bands], NaN absent
    errors: np.ndarray  # float32 [n, n_bands]
    xp_syn: np.ndarray  # float32 [n, K]: a' . c_norm of the observed XP (NaN none)
    xp_var: np.ndarray  # float32 [n, K]: its exact variance

    def __len__(self) -> int:
        return len(self.source_id)

    def take(self, idx) -> RowBlock:
        return RowBlock(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))

    @staticmethod
    def concat(blocks: list[RowBlock]) -> RowBlock:
        return RowBlock(
            *(
                np.concatenate([getattr(b, f) for b in blocks])
                for f in RowBlock.__dataclass_fields__
            )
        )

    def n_tokens(self) -> np.ndarray:
        return np.isfinite(self.values).sum(1)

    def to_torch(self) -> dict[str, torch.Tensor]:
        return {
            f: torch.from_numpy(np.ascontiguousarray(getattr(self, f)))
            for f in self.__dataclass_fields__
        }


def assemble(
    arrays: ContractArrays,
    g_mag: np.ndarray | None,
    source_id: np.ndarray,
    input_class: np.ndarray,
    registry: BandRegistry,
    operator: FilterOperator | None,
    xp_covariance: str = "exact",
) -> RowBlock:
    """Normalise XP to G = 8.5, write it into the XP slots, compute the observed-XP
    synthetic relative fluxes and their exact variances for the filter-integral term."""
    n = len(source_id)
    values, errors = arrays.values, arrays.errors
    k = 0 if operator is None else operator.matrix.shape[1]
    xp_syn = np.full((n, k), np.nan, np.float32)
    xp_var = np.full((n, k), np.nan, np.float32)
    if arrays.xp_coefficients is not None and g_mag is not None:
        c, e = arrays.xp_coefficients, arrays.xp_errors
        has = (
            np.isfinite(c).all((1, 2)) & np.isfinite(e).all((1, 2)) & np.isfinite(g_mag)
        )
        if has.any():
            f = xp_normalisation(g_mag[has])
            cn = c[has].reshape(-1, 2 * N_XP) / f[:, None]
            en = e[has].reshape(-1, 2 * N_XP) / f[:, None]
            xi = registry.xp_indices
            v = values[has]
            v[:, xi] = cn
            values[has] = v
            er = errors[has]
            er[:, xi] = en
            errors[has] = er
            if operator is not None:
                a = torch.as_tensor(operator.matrix)
                xp_syn[has] = (torch.as_tensor(cn) @ a).numpy()
                corr = None
                if xp_covariance == "exact":
                    if arrays.xp_correlations is None:
                        raise ValueError(
                            "data.xp_covariance is exact but the XP correlation columns "
                            "are not in the trees"
                        )
                    corr = torch.as_tensor(
                        np.nan_to_num(arrays.xp_correlations[has]), dtype=torch.float64
                    )
                var = exact_band_variance(a, torch.as_tensor(e[has]), corr).numpy()
                xp_var[has] = var / (f * f)[:, None]
    return RowBlock(
        source_id=np.asarray(source_id, np.int64),
        input_class=np.asarray(input_class, np.uint8),
        values=values,
        errors=errors,
        xp_syn=xp_syn,
        xp_var=xp_var,
    )


# ---------------------------------------------------------------- micro-batches


class MicroBatcher:
    """Shuffle buffer + token-count buckets -> micro-batches of ~``tokens`` tokens and
    at most ``max_rows`` rows (the latent stack costs the same per row whatever its
    token count, so rows bound memory too)."""

    def __init__(
        self,
        edges: list[int],
        tokens: int,
        shuffle_rows: int,
        rng: np.random.Generator,
        n_bands: int,
        max_rows: int | None = None,
    ):
        self.edges = np.asarray(list(edges) + [max(n_bands, edges[-1])], np.int64)
        self.caps = np.maximum(tokens // self.edges, 1)
        if max_rows is not None:
            self.caps = np.minimum(self.caps, max_rows)
        self.shuffle_rows = shuffle_rows
        self.rng = rng
        self.buffer: list[RowBlock] = []
        self.buffered = 0
        self.buckets: list[list[RowBlock]] = [[] for _ in self.edges]
        self.counts = np.zeros(len(self.edges), np.int64)

    def add(self, block: RowBlock):
        self.buffer.append(block)
        self.buffered += len(block)
        if self.buffered >= self.shuffle_rows:
            yield from self._drain()

    def _drain(self):
        if not self.buffer:
            return
        blk = RowBlock.concat(self.buffer)
        self.buffer, self.buffered = [], 0
        blk = blk.take(self.rng.permutation(len(blk)))
        b = np.searchsorted(self.edges, blk.n_tokens(), side="left").clip(
            max=len(self.edges) - 1
        )
        for i in np.unique(b):
            self.buckets[i].append(blk.take(np.nonzero(b == i)[0]))
            self.counts[i] += int((b == i).sum())
            while self.counts[i] >= self.caps[i]:
                allb = RowBlock.concat(self.buckets[i])
                cap = int(self.caps[i])
                yield allb.take(np.arange(cap))
                rest = allb.take(np.arange(cap, len(allb)))
                self.buckets[i] = [rest] if len(rest) else []
                self.counts[i] = len(rest)

    def flush(self):
        yield from self._drain()
        for i, parts in enumerate(self.buckets):
            if parts:
                yield RowBlock.concat(parts)
            self.buckets[i], self.counts[i] = [], 0


# ---------------------------------------------------------------- the stream


@dataclass
class StreamSpec:
    trees: list[tuple[str, list[str] | None]]  # (root, columns or None)
    assignment: str
    partitions: list[str]
    reader: ContractReader
    registry: BandRegistry
    operator: FilterOperator | None
    xp_covariance: str = "exact"
    rec_sample_rate: float = 1.0
    read_batch_rows: int = 16384
    interleave: int = 4
    shuffle_rows: int = 65536
    tokens_per_microbatch: int = 1_048_576
    max_rows_per_microbatch: int | None = None
    bucket_edges: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 160, 224])
    limit_partitions: int | None = None
    seed: int = 0
    shuffle: bool = True


class HatsStream(torch.utils.data.IterableDataset):
    """Iterable of micro-batches (dicts of CPU tensors) over one epoch."""

    def __init__(self, spec: StreamSpec):
        super().__init__()
        self.spec = spec
        self.epoch = 0
        self.wanted = np.asarray([PARTITION_CODE[p] for p in spec.partitions], np.uint8)
        parts = hats_partitions(spec.trees[0][0])
        if spec.limit_partitions:
            parts = parts[: spec.limit_partitions]
        self.partitions = parts
        self.stats = {"rows_read": 0, "rows_selected": 0, "rewinds": 0, "partitions": 0}
        self._columns = self._resolve_columns()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve_columns(self) -> list[list[str] | None]:
        """Per tree: the contract columns it holds (from the parquet schema of its first
        partition file), plus source_id."""
        import pyarrow.parquet as pq

        need = set(self.spec.reader.columns())
        out: list[list[str] | None] = []
        seen: set[str] = set()
        for root, cols in self.spec.trees:
            if cols is None:
                files = []
                for o, p in hats_partitions(root):
                    files = partition_files(root, o, p)
                    if files:
                        break
                if not files:
                    raise FileNotFoundError(f"no parquet partition under {root}")
                names = set(pq.ParquetFile(files[0]).schema_arrow.names)
                if "source_id" not in names:
                    raise ValueError(f"{root}: no source_id column")
                cols = sorted((names & need) - seen)
            dup = seen & set(cols)
            if dup:
                raise ValueError(f"columns {sorted(dup)} appear in more than one tree")
            seen |= set(cols)
            out.append(["source_id"] + [c for c in cols if c != "source_id"])
        return out

    # -- one partition
    def iter_partition(self, order: int, npix: int):
        import pyarrow as pa
        import pyarrow.parquet as pq

        spec = self.spec
        afiles = partition_files(spec.assignment, order, npix)
        if not afiles:
            raise FileNotFoundError(
                f"no assignment for Norder={order} Npix={npix} in {spec.assignment}"
            )
        at = pa.concat_tables(
            [
                pq.read_table(f, columns=["source_id", "partition", "input_class"])
                for f in afiles
            ]
        )
        a_ids, a_ord = _sorted_index(at.column("source_id").to_numpy())
        a_part = at.column("partition").to_numpy()[a_ord]
        a_cls = at.column("input_class").to_numpy()[a_ord]
        anchor_files = partition_files(spec.trees[0][0], order, npix)
        if not anchor_files:
            return
        anchor = ForwardReader(anchor_files, self._columns[0], spec.read_batch_rows)
        secondaries = []
        for (root, _), cols in zip(spec.trees[1:], self._columns[1:], strict=True):
            files = partition_files(root, order, npix)
            if not files:
                continue
            ids = (
                pa.concat_tables(
                    [pq.read_table(f, columns=["source_id"]) for f in files]
                )
                .column("source_id")
                .to_numpy()
            )
            s_ids, s_pos = _sorted_index(ids)
            secondaries.append(
                (ForwardReader(files, cols, spec.read_batch_rows), s_ids, s_pos)
            )
        g_col = spec.reader.g_mag_column()
        for lo in range(0, anchor.n_rows, spec.read_batch_rows):
            hi = min(lo + spec.read_batch_rows, anchor.n_rows)
            t = anchor.read(lo, hi)
            sid = t.column("source_id").to_numpy()
            pos = _lookup(a_ids, np.arange(len(a_ids)), sid)
            if np.any(pos < 0):
                raise ValueError(
                    f"Norder={order} Npix={npix}: {int((pos < 0).sum())} anchor rows missing "
                    "from the msa-split-v1 assignment"
                )
            sel = np.isin(a_part[pos], self.wanted)
            if spec.rec_sample_rate < 1.0:
                sel &= split_unit(sid, REC_SALT64) < spec.rec_sample_rate
            self.stats["rows_read"] += len(sid)
            if not sel.any():
                continue
            idx = np.nonzero(sel)[0]
            t = t.take(pa.array(idx))
            sid = sid[idx]
            cls = a_cls[pos[idx]]
            cols = {n: t.column(n) for n in t.column_names}
            for reader, s_ids, s_pos in secondaries:
                p = _lookup(s_ids, s_pos, sid)
                have = p >= 0
                if not have.any():
                    continue
                lo2, hi2 = int(p[have].min()), int(p[have].max()) + 1
                st = reader.read(lo2, hi2)
                take = pa.array(np.where(have, p - lo2, 0), mask=~have)
                st = st.take(take)
                got = st.column("source_id").to_numpy(zero_copy_only=False)
                if np.any(got[have] != sid[have]):
                    raise ValueError("secondary join mismatch")
                for n in st.column_names:
                    if n != "source_id":
                        cols[n] = st.column(n)
            table = pa.table(cols)
            arrays = spec.reader.read(table)
            g = None
            if g_col is not None and g_col in cols:
                g = np.asarray(
                    table.column(g_col).to_numpy(zero_copy_only=False), np.float64
                )
            self.stats["rows_selected"] += len(sid)
            yield assemble(
                arrays, g, sid, cls, spec.registry, spec.operator, spec.xp_covariance
            )
        self.stats["rewinds"] += anchor.rewinds + sum(
            r.rewinds for r, _, _ in secondaries
        )
        self.stats["partitions"] += 1

    # -- the epoch
    def _my_partitions(self, worker: int, n_workers: int):
        parts = list(self.partitions)
        if self.spec.shuffle:
            rng = np.random.default_rng([self.spec.seed, self.epoch])
            parts = [parts[i] for i in rng.permutation(len(parts))]
        return parts[worker::n_workers]

    def blocks(self, worker: int = 0, n_workers: int = 1):
        """Row blocks of this worker's partitions, ``interleave`` partitions at a time."""
        parts = self._my_partitions(worker, n_workers)
        k = self.spec.interleave
        for i in range(0, len(parts), k):
            gens = [self.iter_partition(o, p) for o, p in parts[i : i + k]]
            while gens:
                alive = []
                for g in gens:
                    blk = next(g, None)
                    if blk is not None:
                        yield blk
                        alive.append(g)
                gens = alive

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker, n_workers = (0, 1) if info is None else (info.id, info.num_workers)
        rng = np.random.default_rng([self.spec.seed, self.epoch, worker, 7])
        mb = MicroBatcher(
            self.spec.bucket_edges,
            self.spec.tokens_per_microbatch,
            self.spec.shuffle_rows if self.spec.shuffle else 1,
            rng,
            len(self.spec.registry),
            self.spec.max_rows_per_microbatch,
        )
        for blk in self.blocks(worker, n_workers):
            for out in mb.add(blk):
                yield out.to_torch()
        for out in mb.flush():
            yield out.to_torch()


# ---------------------------------------------------------------- scalers


DEFAULT_SCALE = {"mag": (18.0, 2.0), "xp": (0.0, 1.0), "cond": (0.0, 1.0)}


def fit_scalers(
    blocks, registry: BandRegistry, n_rows: int, min_count: int = 50
) -> dict:
    """Robust per-band centre (median) and scale (IQR / 1.349) from up to ``n_rows``
    rows of an iterable of :class:`RowBlock` (or micro-batch dicts)."""
    vals: list[list[np.ndarray]] = [[] for _ in range(len(registry))]
    seen = 0
    for blk in blocks:
        v = blk["values"].numpy() if isinstance(blk, dict) else blk.values
        v = v[: n_rows - seen]
        for j in range(v.shape[1]):
            col = v[:, j]
            col = col[np.isfinite(col)]
            if col.size:
                vals[j].append(col)
        seen += len(v)
        if seen >= n_rows:
            break
    center = np.zeros(len(registry))
    scale = np.ones(len(registry))
    counts = np.zeros(len(registry), np.int64)
    for b in registry.bands:
        x = np.concatenate(vals[b.index]) if vals[b.index] else np.zeros(0)
        counts[b.index] = x.size
        c0, s0 = DEFAULT_SCALE[b.kind]
        if x.size >= min_count:
            q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            s = (q3 - q1) / 1.349
            if not (s > 0 and math.isfinite(s)):
                s = max(float(np.std(x)), 1e-6)
            center[b.index], scale[b.index] = q2, s
        else:
            center[b.index], scale[b.index] = c0, s0
    return {
        "bands": list(registry.names),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "count": counts.tolist(),
        "rows": int(seen),
        "rule": "median and IQR/1.349 of finite values; bands with fewer than "
        f"{min_count} values use defaults {DEFAULT_SCALE}",
    }


def scaler_arrays(
    scalers: dict, registry: BandRegistry
) -> tuple[np.ndarray, np.ndarray]:
    """Centre/scale in registry order; bands missing from the saved scaler (a longer
    band list) get defaults."""
    by = {n: i for i, n in enumerate(scalers["bands"])}
    c = np.zeros(len(registry))
    s = np.ones(len(registry))
    for b in registry.bands:
        if b.name in by:
            c[b.index], s[b.index] = (
                scalers["center"][by[b.name]],
                scalers["scale"][by[b.name]],
            )
        else:
            c[b.index], s[b.index] = DEFAULT_SCALE[b.kind]
    return c, s
