"""Bounded, projected reads for pretraining HDF5 shard tables."""

from __future__ import annotations

import json
import os
import queue
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np


def prefetch_batches(iterable, depth: int = 2) -> Iterator:
    """Prefetch a bounded number of CPU batches while the GPU works."""
    if depth < 1:
        raise ValueError("prefetch depth must be positive")
    items: queue.Queue = queue.Queue(maxsize=depth)
    stopped = threading.Event()
    finished = object()

    def put(item) -> bool:
        while not stopped.is_set():
            try:
                items.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for batch in iterable:
                if not put(("batch", batch)):
                    return
        except BaseException as exc:
            put(("error", exc))
        else:
            put(("done", finished))

    worker = threading.Thread(target=produce, name="msa-batch-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            kind, value = items.get()
            if kind == "done":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()
        worker.join(timeout=5.0)


def _clean_column(name: str, values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "U"}:
        try:
            return np.array(
                [np.nan if value in {b"", ""} else float(value) for value in values],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Error processing column {name}: {exc}") from exc
    try:
        return values.astype(np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Error processing column {name}: {exc}") from exc


class ProjectedHDF5Store:
    """Read configured fields sequentially and optionally stage a compact cache."""

    def __init__(
        self,
        datafile: h5py.File,
        feature_cols: list[str],
        error_cols: list[str | None],
        *,
        scratch_dir: str | Path | None = None,
        cache_fraction: float = 0.8,
        chunk_rows: int = 65_536,
        shuffle_buffer_bytes: int = 64 * 1024 * 1024,
        scratch_free_bytes: int | None = None,
        progress_file: str | Path | None = None,
        run_id: str | None = None,
        cache_max_bytes: int | None = None,
        cache_keys: list[str] | None = None,
        cache_seed: int = 42,
    ):
        if not feature_cols:
            raise ValueError("At least one feature column is required")
        if len(feature_cols) != len(error_cols):
            raise ValueError("error_cols must align one-to-one with feature_cols")
        if not 0.0 <= cache_fraction <= 1.0:
            raise ValueError("cache_fraction must be between zero and one")
        if chunk_rows < 1 or shuffle_buffer_bytes < 1:
            raise ValueError("chunk_rows and shuffle_buffer_bytes must be positive")
        if cache_max_bytes is not None and (
            isinstance(cache_max_bytes, bool)
            or not isinstance(cache_max_bytes, int)
            or cache_max_bytes < 1
        ):
            raise ValueError("cache_max_bytes must be a positive integer when set")

        self.datafile = datafile
        self.feature_cols = list(feature_cols)
        self.error_cols = list(error_cols)
        self.fields = list(
            dict.fromkeys(
                [*self.feature_cols, *(c for c in self.error_cols if c is not None)]
            )
        )
        self._field_index = {name: i for i, name in enumerate(self.fields)}
        self._feature_count = len(self.feature_cols)
        self.chunk_rows = chunk_rows
        self.shuffle_buffer_bytes = shuffle_buffer_bytes
        self.scratch_dir = self._resolve_scratch(scratch_dir)
        self._scratch_free_bytes = scratch_free_bytes
        self.scratch_free_bytes = scratch_free_bytes
        self._cache_fraction = cache_fraction
        self.progress_file = Path(progress_file) if progress_file else None
        self.run_id = run_id
        self.cache_max_bytes = cache_max_bytes
        self.cache_keys = list(cache_keys) if cache_keys is not None else None
        self.cache_seed = int(cache_seed)
        self._cache_tmp: tempfile.TemporaryDirectory | None = None
        self._cache_arrays: dict[str, np.memmap] = {}
        self._keys: list[str] = []
        self._source_row_counts: dict[str, int] = {}
        self._row_counts: dict[str, int] = {}
        self._cache_row_counts: dict[str, int] = {}
        self._cached_keys: set[str] = set()
        self._error_maxima: dict[str, np.ndarray] = {}
        self.scaler_sample = np.empty((0, self._feature_count), dtype=np.float32)
        self.cache_bytes = 0
        self.mode = "unprepared"
        self.shuffle_policy = "bounded_chunk_shuffle_v1"
        self.read_seconds = 0.0
        self.conversion_seconds = 0.0
        self.transform_seconds = 0.0
        self.prepare_seconds = 0.0

    def _write_progress(self, stage: str, **fields) -> None:
        if self.progress_file is None:
            return
        entry = {
            "run_id": self.run_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            **fields,
        }
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with self.progress_file.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")

    @staticmethod
    def _resolve_scratch(scratch_dir: str | Path | None) -> Path:
        if scratch_dir is not None:
            return Path(scratch_dir)
        configured = Path(tempfile.gettempdir())
        scratch = Path("/scratch")
        if value := os.environ.get("SCRATCH"):
            configured = Path(value)
        elif scratch.is_dir():
            configured = scratch
        return configured

    def prepare(
        self,
        keys: list[str],
        scaler_keys: list[str],
        *,
        scaler_max_rows: int = 1_000_000,
        scaler_seed: int = 42,
        max_rows_per_key: int | None = None,
    ) -> None:
        if not keys:
            raise ValueError("No HDF5 shards were selected")
        if scaler_max_rows < 1:
            raise ValueError("scaler_max_rows must be positive")
        if max_rows_per_key is not None and max_rows_per_key < 1:
            raise ValueError("max_rows_per_key must be positive when set")
        if any(key not in keys for key in scaler_keys):
            raise ValueError("scaler_keys must be among the prepared shards")
        if not scaler_keys:
            raise ValueError("No training shards are available for scaler fitting")

        self._keys = list(keys)
        self._source_row_counts = {
            key: min(
                len(self.datafile[key]),
                max_rows_per_key
                if max_rows_per_key is not None
                else len(self.datafile[key]),
            )
            for key in self._keys
        }
        self._row_counts = dict(self._source_row_counts)
        cache_keys = (
            list(self.cache_keys) if self.cache_keys is not None else list(keys)
        )
        if len(set(cache_keys)) != len(cache_keys):
            raise ValueError("cache_keys must not contain duplicates")
        unknown_cache_keys = [
            key for key in cache_keys if key not in self._source_row_counts
        ]
        if unknown_cache_keys:
            raise ValueError(
                f"cache_keys must be prepared shards: {unknown_cache_keys}"
            )
        nonempty_cache_keys = [
            key for key in cache_keys if self._source_row_counts[key]
        ]
        row_bytes = len(self.fields) * np.dtype(np.float32).itemsize
        full_cache_rows = sum(
            self._source_row_counts[key] for key in nonempty_cache_keys
        )
        cache_row_limit = full_cache_rows
        if self.cache_max_bytes is not None:
            cache_row_limit = min(full_cache_rows, self.cache_max_bytes // row_bytes)

        # ponytail: if the cap cannot retain at least one row from every train shard,
        # skip caching and stream the complete dataset instead of biasing the run.
        if cache_row_limit < len(nonempty_cache_keys):
            cache_row_limit = 0
        self._cache_row_counts = {key: 0 for key in cache_keys}
        if cache_row_limit:
            remaining = cache_row_limit - len(nonempty_cache_keys)
            additional_capacity = sum(
                self._source_row_counts[key] - 1 for key in nonempty_cache_keys
            )
            exact_extra = {
                key: (
                    remaining * (self._source_row_counts[key] - 1) / additional_capacity
                    if additional_capacity
                    else 0
                )
                for key in nonempty_cache_keys
            }
            self._cache_row_counts.update(
                {key: 1 + int(exact_extra[key]) for key in nonempty_cache_keys}
            )
            left = cache_row_limit - sum(self._cache_row_counts.values())
            remainder_order = sorted(
                (
                    key
                    for key in nonempty_cache_keys
                    if self._cache_row_counts[key] < self._source_row_counts[key]
                ),
                key=lambda key: (-(exact_extra[key] % 1), key),
            )
            for key in remainder_order[:left]:
                self._cache_row_counts[key] += 1

        self.cache_bytes = sum(self._cache_row_counts.values()) * row_bytes

        rng = np.random.default_rng(scaler_seed)
        sample_keys = list(scaler_keys)
        if scaler_max_rows < len(sample_keys):
            sample_keys = rng.choice(
                sample_keys, size=scaler_max_rows, replace=False
            ).tolist()
            rows_per_key = 1
        else:
            rows_per_key = scaler_max_rows // len(sample_keys)
        sample_indices = {
            key: np.sort(
                rng.choice(
                    self._row_counts[key],
                    size=min(self._row_counts[key], rows_per_key),
                    replace=False,
                )
            )
            for key in sample_keys
            if self._row_counts[key]
        }

        use_cache = bool(cache_row_limit) and self._cache_fits()
        started = time.perf_counter()
        cache_limit = (
            int(self.scratch_free_bytes * self._cache_fraction)
            if self.scratch_free_bytes is not None
            else None
        )
        print(
            "Pretraining scan starting: "
            f"mode={'cache' if use_cache else 'stream'}, shards={len(self._keys)}, "
            f"projected_cache_bytes={self.cache_bytes}, "
            f"scratch_free_bytes={self.scratch_free_bytes}, "
            f"cache_limit_bytes={cache_limit}, "
            f"cached_training_rows={sum(self._cache_row_counts.values())}"
        )
        total_rows = sum(self._source_row_counts.values())
        self._write_progress(
            "scan_started",
            shard_count=len(self._keys),
            rows_total=total_rows,
            projected_cache_bytes=self.cache_bytes,
            cache_max_bytes=self.cache_max_bytes,
            cached_training_rows=sum(self._cache_row_counts.values()),
            selected_loader_mode="cache" if use_cache else "stream",
            scratch_capacity_bytes=self.scratch_free_bytes,
        )
        try:
            if use_cache:
                try:
                    self._cache_tmp = tempfile.TemporaryDirectory(
                        prefix="msa-pretrain-", dir=self.scratch_dir
                    )
                    self._cached_keys = set(nonempty_cache_keys)
                    self._row_counts.update(
                        {
                            key: self._cache_row_counts[key]
                            for key in nonempty_cache_keys
                        }
                    )
                    all_keys_cached = self._cached_keys == set(self._keys)
                    all_cached_rows = all(
                        self._cache_row_counts[key] == self._source_row_counts[key]
                        for key in self._cached_keys
                    )
                    self.mode = (
                        "cache"
                        if all_keys_cached and all_cached_rows
                        else (
                            "cache_subset" if not all_cached_rows else "cache_partial"
                        )
                    )
                    self.shuffle_policy = "cached_block_shuffle_v1"
                    samples = self._scan(sample_indices, write_cache=True)
                except OSError as exc:
                    self.close_cache()
                    self.read_seconds = self.conversion_seconds = 0.0
                    self._cached_keys.clear()
                    self._row_counts = dict(self._source_row_counts)
                    self.shuffle_policy = "bounded_chunk_shuffle_v1"
                    print(
                        f"Pretraining scratch cache unavailable ({exc}); streaming HDF5"
                    )
                    self.mode = "stream"
                    samples = self._scan(sample_indices, write_cache=False)
            else:
                self.mode = "stream"
                samples = self._scan(sample_indices, write_cache=False)
            if not samples:
                raise ValueError("Training shards contain no rows for scaler fitting")
            self.scaler_sample = np.concatenate(samples, axis=0)
            self.prepare_seconds = time.perf_counter() - started
            print(
                "Pretraining input: "
                f"mode={self.mode}, projected_cache_bytes={self.cache_bytes}, "
                f"prepare_s={self.prepare_seconds:.1f}, "
                f"source_read_s={self.read_seconds:.1f}, "
                f"conversion_s={self.conversion_seconds:.1f}"
            )
            self._write_progress(
                "loader_ready",
                loader_mode=self.mode,
                shuffle_policy=self.shuffle_policy,
                rows_total=total_rows,
                projected_cache_bytes=self.cache_bytes,
                cache_max_bytes=self.cache_max_bytes,
                cached_training_rows=(
                    sum(self._row_counts[key] for key in self._cached_keys)
                ),
                prepare_seconds=round(self.prepare_seconds, 3),
                source_read_seconds=round(self.read_seconds, 3),
                conversion_seconds=round(self.conversion_seconds, 3),
            )
        except Exception as exc:
            self._write_progress(
                "scan_failed", error_type=type(exc).__name__, error=str(exc)
            )
            self.close_cache()
            raise

    def _cache_fits(self) -> bool:
        if self._cache_fraction <= 0 or not self.scratch_dir.is_dir():
            return False
        try:
            free = (
                self._scratch_free_bytes
                if self._scratch_free_bytes is not None
                else shutil.disk_usage(self.scratch_dir).free
            )
        except OSError:
            return False
        self.scratch_free_bytes = free
        return self.cache_bytes <= int(free * self._cache_fraction)

    def _read_matrix(self, dataset: h5py.Dataset, start: int, stop: int) -> np.ndarray:
        read_started = time.perf_counter()
        records = dataset.fields(self.fields)[start:stop]
        self.read_seconds += time.perf_counter() - read_started
        conversion_started = time.perf_counter()
        matrix = np.column_stack(
            [_clean_column(name, records[name]) for name in self.fields]
        )
        self.conversion_seconds += time.perf_counter() - conversion_started
        return matrix

    def _scan(
        self, sample_indices: dict[str, np.ndarray], *, write_cache: bool
    ) -> list[np.ndarray]:
        samples: list[np.ndarray] = []
        overall_rows_completed = 0
        for key_index, key in enumerate(self._keys):
            key_started = time.perf_counter()
            read_before = self.read_seconds
            conversion_before = self.conversion_seconds
            dataset = self.datafile[key]
            n_rows = self._source_row_counts[key]
            names = dataset.dtype.names or ()
            missing = [name for name in self.fields if name not in names]
            if missing:
                raise ValueError(f"Missing projected fields in '{key}': {missing}")
            cache = None
            cache_rows = self._cache_row_counts.get(key, 0)
            if write_cache and key in self._cached_keys:
                cache_path = Path(self._cache_tmp.name) / f"shard-{key_index}.npy"
                cache = np.lib.format.open_memmap(
                    cache_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(cache_rows, len(self.fields)),
                )
                self._cache_arrays[key] = cache
            cache_indices = None
            if cache is not None and cache_rows < n_rows:
                cache_indices = np.sort(
                    np.random.default_rng(
                        np.random.SeedSequence([self.cache_seed, key_index])
                    ).choice(n_rows, size=cache_rows, replace=False)
                )
            cache_rows_written = 0

            error_maxima = np.full(self._feature_count, -np.inf, dtype=np.float32)
            chosen = sample_indices.get(key, np.empty(0, dtype=np.int64))
            chunk_rows = self._chunk_rows_for(dataset)
            n_chunks = (n_rows + chunk_rows - 1) // chunk_rows
            for chunk_index, start in enumerate(range(0, n_rows, chunk_rows), start=1):
                stop = min(start + self._chunk_rows_for(dataset), n_rows)
                matrix = self._read_matrix(dataset, start, stop)
                if cache is not None:
                    if cache_indices is None:
                        cache[start:stop] = matrix
                    else:
                        first = int(np.searchsorted(cache_indices, start, side="left"))
                        last = int(np.searchsorted(cache_indices, stop, side="left"))
                        if first < last:
                            rows = cache_indices[first:last] - start
                            cache[
                                cache_rows_written : cache_rows_written + len(rows)
                            ] = matrix[rows]
                            cache_rows_written += len(rows)
                for i, error in enumerate(self.error_cols):
                    if error is None:
                        continue
                    values = matrix[:, self._field_index[error]]
                    valid = np.isfinite(values) & (values > 0)
                    if valid.any():
                        error_maxima[i] = max(
                            error_maxima[i], float(values[valid].max())
                        )
                if chosen.size:
                    first = int(np.searchsorted(chosen, start, side="left"))
                    last = int(np.searchsorted(chosen, stop, side="left"))
                    if first < last:
                        rows = chosen[first:last] - start
                        samples.append(matrix[rows, : self._feature_count].copy())
                if (
                    chunk_index == 1
                    or chunk_index == n_chunks
                    or chunk_index % max(1, n_chunks // 4) == 0
                ):
                    self._write_progress(
                        "scan_progress",
                        key=key,
                        key_index=key_index + 1,
                        shard_count=len(self._keys),
                        rows_completed=stop,
                        rows_total=n_rows,
                        overall_rows_completed=overall_rows_completed + stop,
                        overall_rows_total=sum(self._row_counts.values()),
                        elapsed_seconds=round(time.perf_counter() - key_started, 3),
                        source_read_seconds=round(self.read_seconds - read_before, 3),
                        conversion_seconds=round(
                            self.conversion_seconds - conversion_before, 3
                        ),
                    )
                    print(
                        "Pretraining scan progress: "
                        f"key={key}, rows={stop}/{n_rows}, "
                        f"elapsed_s={time.perf_counter() - key_started:.1f}, "
                        f"source_read_s={self.read_seconds - read_before:.1f}, "
                        f"conversion_s={self.conversion_seconds - conversion_before:.1f}"
                    )
            overall_rows_completed += n_rows
            error_maxima[~np.isfinite(error_maxima)] = 1.0
            self._error_maxima[key] = error_maxima
            if cache is not None:
                cache.flush()
        return samples

    def _chunk_rows_for(self, dataset: h5py.Dataset) -> int:
        physical_rows = dataset.chunks[0] if dataset.chunks else 0
        return max(self.chunk_rows, physical_rows)

    def error_maxima(self, key: str) -> np.ndarray:
        return self._error_maxima[key]

    def _iter_raw(
        self,
        key: str,
        max_rows: int | None = None,
        *,
        shuffle_chunks: bool = False,
        seed: int = 0,
    ) -> Iterator[np.ndarray]:
        if key not in self._row_counts:
            raise KeyError(f"Shard '{key}' was not prepared")
        count = min(
            self._row_counts[key],
            max_rows if max_rows is not None else self._row_counts[key],
        )
        dataset = self.datafile[key]
        cached = self._cache_arrays.get(key)
        chunk_rows = self._chunk_rows_for(dataset)
        starts = range(0, count, chunk_rows)
        if cached is not None and shuffle_chunks:
            chunks = np.arange((count + chunk_rows - 1) // chunk_rows)
            if len(chunks) > 1:
                starts = np.random.default_rng(seed).permutation(chunks) * chunk_rows
        for start in starts:
            start = int(start)
            stop = min(start + chunk_rows, count)
            started = time.perf_counter()
            if cached is not None:
                matrix = np.asarray(cached[start:stop])
                self.read_seconds += time.perf_counter() - started
            else:
                matrix = self._read_matrix(dataset, start, stop)
            yield matrix

    def _prepare_arrays(
        self, matrix: np.ndarray, key: str, scaler, scale_factors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        started = time.perf_counter()
        x = matrix[:, : self._feature_count].copy()
        x[~np.isfinite(x)] = np.nan
        errors = np.empty_like(x)
        maxima = self.error_maxima(key)
        for i, error in enumerate(self.error_cols):
            if error is None:
                errors[:, i] = scale_factors[i]
            else:
                values = matrix[:, self._field_index[error]]
                valid = np.isfinite(values) & (values > 0)
                errors[:, i] = np.where(valid, values, maxima[i])
        x = scaler.transform(x).astype(np.float32, copy=False)
        errors /= scale_factors
        self.transform_seconds += time.perf_counter() - started
        return x, errors

    def iter_batches(
        self,
        key: str,
        scaler,
        scale_factors: np.ndarray,
        *,
        batch_rows: int,
        shuffle: bool,
        seed: int,
        max_rows: int | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        rng = np.random.default_rng(seed)
        if not shuffle:
            pending_x = pending_e = None
            for matrix in self._iter_raw(key, max_rows):
                x, e = self._prepare_arrays(matrix, key, scaler, scale_factors)
                if pending_x is not None:
                    x = np.concatenate((pending_x, x), axis=0)
                    e = np.concatenate((pending_e, e), axis=0)
                stop = len(x) - len(x) % batch_rows
                for start in range(0, stop, batch_rows):
                    yield x[start : start + batch_rows], e[start : start + batch_rows]
                pending_x, pending_e = x[stop:].copy(), e[stop:].copy()
            if pending_x is not None and len(pending_x):
                yield pending_x, pending_e
            return

        batch_bytes = (
            batch_rows * self._feature_count * 2 * np.dtype(np.float32).itemsize
        )
        buffer_limit = max(self.shuffle_buffer_bytes, batch_bytes)
        parts_x: list[np.ndarray] = []
        parts_e: list[np.ndarray] = []
        buffered_bytes = 0

        def emit() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            nonlocal parts_x, parts_e, buffered_bytes
            if not parts_x:
                return
            x = np.concatenate(parts_x, axis=0)
            e = np.concatenate(parts_e, axis=0)
            order = rng.permutation(len(x))
            x, e = x[order], e[order]
            for start in range(0, len(x), batch_rows):
                yield x[start : start + batch_rows], e[start : start + batch_rows]
            parts_x, parts_e, buffered_bytes = [], [], 0

        for matrix in self._iter_raw(
            key,
            max_rows,
            shuffle_chunks=key in self._cache_arrays,
            seed=seed ^ 0x9E3779B9,
        ):
            x, e = self._prepare_arrays(matrix, key, scaler, scale_factors)
            part_bytes = x.nbytes + e.nbytes
            if parts_x and buffered_bytes + part_bytes > buffer_limit:
                yield from emit()
            parts_x.append(x)
            parts_e.append(e)
            buffered_bytes += part_bytes
        yield from emit()

    def sample_batches(
        self,
        key: str,
        scaler,
        scale_factors: np.ndarray,
        *,
        sample_rows: int,
        seed: int,
        batch_rows: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        count = self._row_counts[key]
        sample_count = min(count, sample_rows)
        dataset = self.datafile[key]
        cached = self._cache_arrays.get(key)
        if cached is None:
            # ponytail: stream one random contiguous window to avoid scattered
            # HDF5 reads on /arc; use reservoir sampling if window bias matters.
            rng = np.random.default_rng(seed)
            start = (
                int(rng.integers(0, count - sample_count + 1))
                if sample_count < count
                else 0
            )
            stop = start + sample_count
            for row_start in range(start, stop, batch_rows):
                row_stop = min(row_start + batch_rows, stop)
                matrix = self._read_matrix(dataset, row_start, row_stop)
                yield self._prepare_arrays(matrix, key, scaler, scale_factors)
            return

        indices = np.sort(
            np.random.default_rng(seed).choice(count, size=sample_count, replace=False)
        )
        for start in range(0, len(indices), batch_rows):
            chosen = indices[start : start + batch_rows]
            read_started = time.perf_counter()
            matrix = np.asarray(cached[chosen])
            self.read_seconds += time.perf_counter() - read_started
            yield self._prepare_arrays(matrix, key, scaler, scale_factors)

    def close_cache(self) -> None:
        for array in self._cache_arrays.values():
            try:
                array._mmap.close()
            except (AttributeError, OSError):
                pass
        self._cache_arrays.clear()
        if self._cache_tmp is not None:
            self._cache_tmp.cleanup()
            self._cache_tmp = None

    def close(self) -> None:
        self.close_cache()
