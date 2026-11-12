"""Probe how a multi-slice CANFAR GPU session exposes its devices.

Records the visible GPUs (MIG slices) and times a two-process all-reduce with
each process pinned to one slice, for the NCCL and Gloo backends. This decides
whether data-parallel training across slices is practical.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKER = r"""
import os, sys, time, json, torch, torch.distributed as dist
backend, rank, world, port, numel = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port)
dist.init_process_group(backend, rank=rank, world_size=world)
x = torch.ones(numel, device="cuda")
out = {}
for i in range(6):
    torch.cuda.synchronize(); t = time.perf_counter()
    if backend == "gloo":
        y = x.cpu(); dist.all_reduce(y); x.copy_(y)
    else:
        dist.all_reduce(x)
    torch.cuda.synchronize(); out.setdefault("seconds", []).append(time.perf_counter() - t)
    x.fill_(1.0)
out.update(rank=rank, device=torch.cuda.get_device_name(), count=torch.cuda.device_count())
if rank == 0:
    print("RESULT " + json.dumps(out), flush=True)
dist.destroy_process_group()
"""


def run(cmd, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, **kw)
        return dict(code=p.returncode, out=p.stdout[-4000:], err=p.stderr[-4000:])
    except Exception as exc:  # the probe records failures instead of raising
        return dict(code=None, err=repr(exc))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--numel", type=int, default=5_000_000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    r = dict(
        env={k: v for k, v in os.environ.items() if "CUDA" in k or "NVIDIA" in k},
        nvidia_smi_L=run(["nvidia-smi", "-L"]),
        nproc=os.cpu_count(),
        affinity=len(os.sched_getaffinity(0)),
    )
    uuids = [
        line.split("UUID: ")[1].rstrip(")").strip()
        for line in r["nvidia_smi_L"].get("out", "").splitlines()
        if "MIG" in line and "UUID: MIG-" in line
    ]
    r["mig_uuids"] = uuids
    import torch

    r["torch_device_count_default"] = torch.cuda.device_count()
    r["torch_devices"] = [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]
    r["all_reduce"] = {}
    if len(uuids) >= 2:
        for k, backend in enumerate(["nccl", "gloo"]):
            procs = []
            for rank, uuid in enumerate(uuids[:2]):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, NCCL_DEBUG="WARN")
                procs.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            WORKER,
                            backend,
                            str(rank),
                            "2",
                            str(29500 + k),
                            str(a.numel),
                        ],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            started = time.perf_counter()
            res = []
            for proc in procs:
                try:
                    out, err = proc.communicate(timeout=300)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    out, err = proc.communicate()
                res.append(dict(code=proc.returncode, out=out[-3000:], err=err[-3000:]))
            r["all_reduce"][backend] = dict(
                numel=a.numel, wall=time.perf_counter() - started, processes=res
            )
    (a.output / "probe.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
