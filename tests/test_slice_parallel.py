import os
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from masked_stellar_autoencoder.training.slice_parallel import (
    SliceGroup,
    allreduce_gradients,
    check_in_sync,
    init_from_env,
    shard_bounds,
)


def test_shard_bounds_partition_rows():
    for n in [0, 1, 7, 256, 320]:
        for world in [1, 2, 3, 4, 7]:
            parts = [shard_bounds(n, r, world) for r in range(world)]
            assert parts[0][0] == 0 and parts[-1][1] == n
            assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
            sizes = [hi - lo for lo, hi in parts]
            assert max(sizes) - min(sizes) <= 1


def test_single_process_is_a_no_op():
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.full((3,), 2.0)
    assert allreduce_gradients([p], SliceGroup(), extras=[1.5]) == [1.5]
    assert torch.equal(p.grad, torch.full((3,), 2.0))
    check_in_sync([p], SliceGroup())


def model_and_data():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    )
    unused = torch.nn.Parameter(torch.zeros(4))  # no grad on any rank
    x = torch.randn(10, 5)
    y = torch.randn(10, 2)
    return model, unused, x, y


def shard_step(model, x, y, lo, hi):
    loss = ((model(x[lo:hi]) - y[lo:hi]) ** 2).mean(1).sum() / len(x)
    loss.backward()
    return float(loss.detach())


def worker(rank, world, port, out):
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    group = init_from_env(timeout_hours=0.05)
    model, unused, x, y = model_and_data()
    lo, hi = group.shard(len(x))
    local = shard_step(model, x, y, lo, hi)
    params = [*model.parameters(), unused]
    (total,) = allreduce_gradients(params, group, extras=[local])
    out[rank] = dict(
        total=total,
        grads=[None if p.grad is None else p.grad.clone() for p in params],
    )
    torch.optim.SGD(params, lr=0.1).step()
    check_in_sync(params, group)
    if rank == 1:
        with torch.no_grad():
            params[0].add_(1e-3)
    try:
        check_in_sync(params, group)
        out[f"diverged{rank}"] = False
    except RuntimeError:
        out[f"diverged{rank}"] = True
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3])
def test_sharded_gradients_match_single_process(world):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(worker, args=(world, port, out), nprocs=world, join=True)
    model, unused, x, y = model_and_data()
    expected_total = shard_step(model, x, y, 0, len(x))
    expected = [p.grad for p in [*model.parameters(), unused]]
    for rank in range(world):
        r = out[rank]
        np.testing.assert_allclose(r["total"], expected_total, rtol=1e-5)
        for got, want in zip(r["grads"], expected, strict=True):
            if want is None:
                assert got is None
            else:
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
        assert out[f"diverged{rank}"]
    for rank in range(1, world):  # identical bits on every rank
        for a, b in zip(out[0]["grads"], out[rank]["grads"], strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
