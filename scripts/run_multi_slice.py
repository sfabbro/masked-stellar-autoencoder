"""Run one training process per GPU slice of this session.

Usage: ``python scripts/run_multi_slice.py [--slices N] -- python -u train.py ...``

Each process sees one slice (``CUDA_VISIBLE_DEVICES`` set to its MIG or GPU
UUID) and joins a Gloo group through ``RANK``/``WORLD_SIZE``. If any rank
fails, the others are stopped and the launcher exits with that rank's code;
non-leader ranks may finish early while rank 0 completes evaluation/export.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time


def slice_uuids():
    out = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
    ).stdout
    mig = [
        line.split("UUID: ")[1].rstrip(")").strip()
        for line in out.splitlines()
        if "UUID: MIG-" in line
    ]
    if mig:
        return mig
    return [
        line.split("UUID: ")[1].rstrip(")").strip()
        for line in out.splitlines()
        if line.startswith("GPU ") and "UUID: GPU-" in line
    ]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def relay(rank, stream):
    for line in stream:
        sys.stdout.write(line if rank == 0 else f"[rank{rank}] {line}")
        sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slices", type=int, default=0, help="0 = every visible slice")
    p.add_argument(
        "--threads-per-rank",
        type=int,
        default=0,
        help="torch/OpenMP threads per rank (0 = visible cores / slices; the "
        "session's core quota can be lower than the visible cores)",
    )
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        raise SystemExit("A training command is required after --")
    uuids = slice_uuids()
    world = a.slices or len(uuids)
    if not 1 <= world <= len(uuids):
        raise SystemExit(f"Requested {world} slices; {len(uuids)} visible")
    threads = a.threads_per_rank or max(1, len(os.sched_getaffinity(0)) // world)
    port = str(free_port())
    print(
        f"multi-slice: world={world} threads_per_rank={threads} slices={uuids[:world]}",
        flush=True,
    )
    procs = []
    for rank in range(world):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=uuids[rank],
            RANK=str(rank),
            WORLD_SIZE=str(world),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            OMP_NUM_THREADS=str(threads),
            MSA_THREADS_PER_RANK=str(threads),
        )
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=relay, args=(rank, proc.stdout), daemon=True).start()
        procs.append(proc)
    code = 0
    while True:
        codes = [proc.poll() for proc in procs]
        failed = [(r, c) for r, c in enumerate(codes) if c not in (None, 0)]
        if failed:
            rank, code = failed[0]
            print(f"multi-slice: rank {rank} exited {code}; stopping", flush=True)
            for proc in procs:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
            deadline = time.monotonic() + 60
            for proc in procs:
                try:
                    proc.wait(max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    proc.kill()
            break
        if codes[0] == 0 and all(c is not None for c in codes):
            break
        time.sleep(5)
    time.sleep(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
