"""Data-parallel training across GPU slices (MIG) of one session.

CUDA lets a process see a single MIG slice, so each slice runs its own
process. Gradients are summed over Gloo through host memory (default), or
over NCCL (``MSA_SLICE_BACKEND=nccl``), which needs
``NCCL_MULTI_RANK_GPU_ENABLE=1`` to accept several ranks on one physical GPU. Every rank builds the same global batch from
the same sampler seed and trains on its contiguous shard; summing the shard
gradients, each weighted by shard size over the global batch, gives the
single-process gradient. All ranks then apply the same optimizer step, so the
parameters stay identical; ``check_in_sync`` verifies that.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SliceGroup:
    rank: int = 0
    world: int = 1

    @property
    def active(self) -> bool:
        return self.world > 1

    @property
    def leader(self) -> bool:
        return self.rank == 0

    def shard(self, n: int) -> tuple[int, int]:
        """Contiguous ``[start, stop)`` of ``n`` global rows for this rank."""
        return shard_bounds(n, self.rank, self.world)

    def barrier(self) -> None:
        if self.active:
            torch.distributed.barrier()

    def close(self, *, sync: bool = False) -> None:
        """Leave the group; ``sync`` first waits for every rank to get here."""
        if self.active and torch.distributed.is_initialized():
            if sync:
                torch.distributed.barrier()
            torch.distributed.destroy_process_group()


def shard_bounds(n: int, rank: int, world: int) -> tuple[int, int]:
    if not (0 <= rank < world) or n < 0:
        raise ValueError("Require 0 <= rank < world and n >= 0")
    bounds = np.linspace(0, n, world + 1).round().astype(int)
    return int(bounds[rank]), int(bounds[rank + 1])


def init_from_env(timeout_hours: float = 12.0) -> SliceGroup:
    """Join the Gloo group described by ``RANK``/``WORLD_SIZE``, if any.

    The timeout covers the longest time ranks wait for the leader (epoch-end
    evaluation and checkpoint export run on rank 0 only).
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world < 1 or not (0 <= rank < world):
        raise ValueError("Invalid RANK/WORLD_SIZE")
    if world == 1:
        return SliceGroup()
    backend = os.environ.get("MSA_SLICE_BACKEND", "gloo")
    if backend not in ("gloo", "nccl"):
        raise ValueError("MSA_SLICE_BACKEND must be gloo or nccl")
    options = {}
    if backend == "nccl":
        # Each rank sees only its own slice, as device 0.
        torch.cuda.set_device(0)
        options["device_id"] = torch.device("cuda", 0)
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world,
        timeout=dt.timedelta(hours=timeout_hours),
        **options,
    )
    return SliceGroup(rank, world)


def _reduce(tensor: torch.Tensor, op=None) -> torch.Tensor:
    """All-reduce on the backend's device (CUDA for NCCL, host for Gloo)."""
    on = "cuda" if torch.distributed.get_backend() == "nccl" else "cpu"
    work = tensor.detach().to(on, copy=True)  # never reduce the caller's tensor
    torch.distributed.all_reduce(work, op=op or torch.distributed.ReduceOp.SUM)
    return work.to(tensor.device)


def allreduce_gradients(params, group: SliceGroup, extras=()) -> list[float]:
    """Sum gradients (missing ones count as zero) and scalar ``extras``.

    Returns the summed extras. Gradients become identical on every rank. A
    parameter with no gradient on any rank keeps ``grad`` None.
    """
    params = list(params)
    extras = [float(v) for v in extras]
    if not group.active:
        return extras
    present = _reduce(
        torch.tensor([p.grad is not None for p in params], dtype=torch.float32)
    )
    pieces = []
    for p, seen in zip(params, present.tolist(), strict=True):
        if seen:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            pieces.append(g.reshape(-1))
    device = params[0].device if params else torch.device("cpu")
    pieces.append(torch.tensor(extras, dtype=torch.float32, device=device))
    total = _reduce(torch.cat([q.to(torch.float32) for q in pieces]))
    offset = 0
    for p, seen in zip(params, present.tolist(), strict=True):
        if not seen:
            continue
        n = p.numel()
        value = total[offset : offset + n].view_as(p).to(p.dtype)
        if p.grad is None:
            p.grad = value.clone()
        else:
            p.grad.copy_(value)
        offset += n
    return total[offset:].tolist()


def check_in_sync(params, group: SliceGroup) -> None:
    """Fail closed when any rank's parameters differ from the others."""
    if not group.active:
        return
    flat = torch.cat([p.detach().reshape(-1).to("cpu", torch.float64) for p in params])
    digest = torch.stack([flat.sum(), (flat * flat).sum(), flat.abs().max()])
    high = _reduce(digest, torch.distributed.ReduceOp.MAX)
    low = _reduce(-digest, torch.distributed.ReduceOp.MAX)
    if not torch.equal(high, -low):
        raise RuntimeError("Slice ranks hold different parameters")
