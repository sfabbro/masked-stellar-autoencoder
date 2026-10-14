"""Launch one bounded CANFAR headless session per XP reconstruction experiment."""

import argparse
import base64
import json
import re
import socket
import zlib
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--git-ref", required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--photometry-preflight", action="store_true")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=Path("configs/pretrain.xp-reconstruction.yaml"),
    )
    parser.add_argument("--benchmark")
    parser.add_argument("--train-stars", type=int, default=10000000)
    parser.add_argument("--micro-batch-size", type=int, default=16384)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--resume-submissions", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", args.git_ref):
        parser.error("Invalid Git ref")
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_sha):
        parser.error("Expected a complete Git commit SHA")
    if args.record.exists() and not args.resume_submissions:
        parser.error(
            "Use a new launch record; existing sessions must not be duplicated"
        )
    specification = yaml.safe_load(args.experiments.read_text())
    github_ip = (
        socket.gethostbyname("github.com")
        if not args.dry_run
        else "DNS_RESOLVED_AT_SUBMISSION"
    )
    addresses = {}
    if not args.dry_run:
        from urllib.parse import urlparse

        lock = yaml.safe_load(Path("experiments/xp/pixi.lock").read_text())
        hosts = {
            "github.com",
            "pypi.org",
            "files.pythonhosted.org",
            "conda.anaconda.org",
            "download.pytorch.org",
            "download-r2.pytorch.org",
        }

        def collect(value):
            if isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
            elif isinstance(value, str) and value.startswith("https://"):
                hosts.add(urlparse(value).hostname)

        collect(lock)
        for host in sorted(hosts):
            addresses[host] = socket.gethostbyname(host)
    proxy_source = Path("scripts/canfar_dns_proxy.py").read_text()
    remote = (
        proxy_source
        + "\n"
        + f"dns_proxy=start_dns_proxy({addresses!r})\n"
        + "import os,subprocess\n"
        "root='/scratch/src/sfabbro/masked-stellar-autoencoder'\n"
        "os.makedirs('/scratch/src/sfabbro',exist_ok=True)\n"
        f"os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.curloptResolve',GIT_CONFIG_VALUE_0='github.com:443:{github_ip}')\n"
        f"subprocess.run(['git','clone','--depth','50','--branch',{args.git_ref!r},'https://github.com/sfabbro/masked-stellar-autoencoder.git',root],check=True)\n"
        f"subprocess.run(['git','checkout','--detach',{args.git_sha!r}],cwd=root,check=True)\n"
        "subprocess.run(['bash',root+'/batch_scripts/canfar_entrypoint.sh','pretrain-experiments'],check=True)\n"
    )
    payload = (
        "exec(__import__('zlib').decompress(__import__('base64').b64decode('"
        + base64.b64encode(zlib.compress(remote.encode())).decode()
        + "')))"
    )
    jobs = []
    record = {
        "checkpoint": args.checkpoint,
        "git_ref": args.git_ref,
        "git_sha": args.git_sha,
        "output_root": args.output_root,
        "train_stars": args.train_stars,
        "benchmark": args.benchmark,
        "experiments_config": str(args.experiments),
        "micro_batch_size": args.micro_batch_size,
        "photometry_preflight": args.photometry_preflight,
        "validation_stars": 1_000_000,
        "presentations": 20_000_000,
        "seed": 42,
        "jobs": jobs,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_submissions and args.record.exists():
        existing = json.loads(args.record.read_text())
        for key in (
            "checkpoint",
            "git_sha",
            "output_root",
            "benchmark",
            "train_stars",
            "micro_batch_size",
            "experiments_config",
            "photometry_preflight",
        ):
            if (
                existing.get(key, False)
                if key == "photometry_preflight"
                else existing.get(key)
            ) != record.get(key):
                parser.error(f"Existing submission record differs: {key}")
        jobs.extend(existing["jobs"])
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    session = None
    if not args.dry_run:
        from canfar.sessions import Session

        session = Session()
    for experiment in specification["experiments"]:
        run_id = experiment["run_id"]
        if args.runs and run_id not in args.runs:
            continue
        if any(job["run_id"] == run_id and job["session_ids"] for job in jobs):
            continue
        output = f"{args.output_root}/{run_id}"
        env = {
            "WORK": "/scratch/src",
            "PYTHONNOUSERSITE": "1",
            "MSA_XP_EXPERIMENTS": "1",
            "MSA_EXPERIMENTS_CONFIG": str(args.experiments),
            "MSA_PRETRAIN_RESUME": args.checkpoint,
            "MSA_EXPERIMENT_RUNS": run_id,
            "MSA_EXPERIMENT_OUTPUT": output,
            "MSA_EXPERIMENT_TRAIN_ROWS": str(args.train_stars),
            "MSA_EXPERIMENT_VALIDATION_ROWS": "1000000",
            "MSA_EXPERIMENT_PRESENTATIONS": "20000000",
            "MSA_EXPERIMENT_LOG_ROWS": "5000000",
            "MSA_EXPERIMENT_DIAGNOSTIC_ROWS": "10000",
            "MSA_EXPERIMENT_INDEPENDENT_RNG": "1",
            "MSA_EXPERIMENT_MICRO_BATCH": str(args.micro_batch_size),
        }
        if args.photometry_preflight:
            env["MSA_EXPERIMENT_PHOTOMETRY_PREFLIGHT"] = "1"
        if args.benchmark:
            env["MSA_EXPERIMENT_BENCHMARK"] = args.benchmark
        request = {
            "name": f"msa-xp-{run_id.replace('_', '-')}",
            "image": "images.canfar.net/astroai/base:latest",
            "kind": "headless",
            "cmd": "python",
            "args": "-u -c " + payload,
            "cores": 4,
            "ram": 64,
            "gpu": 0 if args.photometry_preflight else 1,
            "env": env,
        }
        ids = [] if args.dry_run else session.create(**request)
        job = {
            "run_id": run_id,
            "session_ids": ids,
            "output": output,
            "log": f"{output}.session.log",
            "resources": {
                "cpu": 4,
                "memory_gib": 64,
                "gpu": 0 if args.photometry_preflight else 1,
            },
            "status": "dry_run"
            if args.dry_run
            else "submitted"
            if ids
            else "submission_failed",
        }
        jobs[:] = [prior for prior in jobs if prior["run_id"] != run_id]
        jobs.append(job)
        args.record.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(job), flush=True)
        if not args.dry_run and not ids:
            raise RuntimeError(
                f"Submission failed for {run_id}; preceding IDs are recorded"
            )


if __name__ == "__main__":
    main()
