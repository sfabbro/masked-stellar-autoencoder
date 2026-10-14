"""Idempotent headless submission of the authorized XP follow-up suite."""

import argparse
import base64
import datetime as dt
import hashlib
import json
import socket
import zlib
from pathlib import Path
from urllib.parse import urlparse

import yaml


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def bootstrap(git_ref, git_sha, addresses):
    proxy = Path("scripts/canfar_dns_proxy.py").read_text()
    source = proxy + f"\ndns_proxy=start_dns_proxy({addresses!r})\n"
    source += (
        "import os, subprocess\n"
        "root='/scratch/src/sfabbro/masked-stellar-autoencoder'\n"
        "os.makedirs('/scratch/src/sfabbro',exist_ok=True)\n"
        f"os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='http.curloptResolve',GIT_CONFIG_VALUE_0='github.com:443:{addresses.get('github.com', 'DRY_RUN')}')\n"
        f"subprocess.run(['git','clone','--depth','100','--branch',{git_ref!r},'https://github.com/sfabbro/masked-stellar-autoencoder.git',root],check=True)\n"
        f"subprocess.run(['git','checkout','--detach',{git_sha!r}],cwd=root,check=True)\n"
        "print('PINNED_SOURCE',subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),flush=True)\n"
        "subprocess.run(['bash',root+'/batch_scripts/canfar_entrypoint.sh','pretrain-experiments'],cwd=root,check=True)\n"
    )
    encoded = base64.b64encode(zlib.compress(source.encode())).decode()
    return (
        "-u -c exec(__import__('zlib').decompress(__import__('base64').b64decode('"
        + encoded
        + "')))"
    )


def dns_addresses():
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

    collect(yaml.safe_load(Path("experiments/xp/pixi.lock").read_text()))
    return {host: socket.gethostbyname(host) for host in sorted(hosts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--git-ref", default="codex/xp-reconstruction-experiments")
    parser.add_argument(
        "--record",
        type=Path,
        default=Path("experiments/xp/launch-followup-e47-20261014.json"),
    )
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(args.git_sha) != 40 or any(
        char not in "0123456789abcdef" for char in args.git_sha
    ):
        parser.error("Require a complete immutable Git SHA")
    root = "/arc/projects/k-pop/msa_runs/xp-followup-e47-20261014"
    checkpoint = "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_47.pth"
    benchmark = (
        "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5"
    )
    config = "configs/pretrain.xp-followup.yaml"
    specifications = [
        {**x, "task": "training"}
        for x in yaml.safe_load(Path(config).read_text())["experiments"]
    ]
    specifications += [
        {**x, "task": "analysis"}
        for x in yaml.safe_load(Path("configs/xp-followup-analysis.yaml").read_text())[
            "analyses"
        ]
    ]
    identifiers = [x["run_id"] for x in specifications]
    if len(set(identifiers)) != len(identifiers):
        parser.error("Duplicate run IDs")
    if args.runs and not set(args.runs).issubset(identifiers):
        parser.error("Unknown experiment requested")
    fingerprint = hashlib.sha256(
        json.dumps(specifications, sort_keys=True).encode()
    ).hexdigest()
    record = {
        "git_sha": args.git_sha,
        "git_ref": args.git_ref,
        "output_root": root,
        "checkpoint": checkpoint,
        "benchmark": benchmark,
        "specification_sha256": fingerprint,
        "jobs": [],
    }
    if args.record.exists():
        existing = json.loads(args.record.read_text())
        for field in record.keys() - {"jobs"}:
            if record[field] != existing[field]:
                parser.error(f"Immutable launch record differs: {field}")
        record = existing
    atomic_json(args.record, record)
    session = None
    if not args.dry_run:
        from canfar.sessions import Session

        session = Session(timeout=30)
    payload = bootstrap(
        args.git_ref, args.git_sha, {} if args.dry_run else dns_addresses()
    )
    for settings in specifications:
        run_id = settings["run_id"]
        if args.runs and run_id not in args.runs:
            continue
        if settings.get("depends_on") and not args.dry_run:
            prerequisite = root + "/" + settings["depends_on"] + "/manifest.json"
            import subprocess

            response = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--max-time",
                    "30",
                    "--cert",
                    "/Users/fabbros/.ssl/cadcproxy.pem",
                    "https://ws-uv.canfar.net/arc/files"
                    + prerequisite.removeprefix("/arc"),
                ],
                capture_output=True,
                text=True,
            )
            try:
                completed = (
                    response.returncode == 0
                    and json.loads(response.stdout)["status"] == "complete"
                )
            except (ValueError, KeyError):
                completed = False
            if not completed:
                print(
                    json.dumps(
                        {"run_id": run_id, "status": "waiting_for_completed_checkpoint"}
                    ),
                    flush=True,
                )
                continue
        prior = next((x for x in record["jobs"] if x["run_id"] == run_id), None)
        if prior and prior.get("session_ids"):
            continue
        name = "msa-xpf14-" + run_id.replace("_", "-")
        # Any ambiguous request MUST be reconciled before another create.
        if session and prior and prior.get("attempted_at"):
            try:
                matches = [x for x in session.fetch() if x.get("name") == name]
            except Exception as error:
                print(
                    json.dumps({"run_id": run_id, "reconciliation_error": str(error)}),
                    flush=True,
                )
                continue
            if matches:
                prior.update(
                    session_ids=[x["id"] for x in matches],
                    status="accepted_recovered",
                    recovered_sessions=matches,
                )
                atomic_json(args.record, record)
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "session_ids": prior["session_ids"],
                            "recovered": True,
                        }
                    ),
                    flush=True,
                )
                continue
            elapsed = (
                dt.datetime.now(dt.UTC)
                - dt.datetime.fromisoformat(prior["attempted_at"])
            ).total_seconds()
            if elapsed < 90 or prior.get("attempts", 0) >= 3:
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "status": "awaiting_reconciliation",
                            "elapsed": elapsed,
                        }
                    ),
                    flush=True,
                )
                continue
        output = root + "/" + run_id
        env = {
            "WORK": "/scratch/src",
            "PYTHONNOUSERSITE": "1",
            "MSA_XP_EXPERIMENTS": "1",
            "MSA_EXPERIMENT_TASK": settings["task"],
            "MSA_ANALYSIS_SPEC": "configs/xp-followup-analysis.yaml",
            "MSA_EXPERIMENTS_CONFIG": config,
            "MSA_PRETRAIN_RESUME": checkpoint,
            "MSA_EXPERIMENT_RUNS": run_id,
            "MSA_EXPERIMENT_OUTPUT": output,
            "MSA_EXPERIMENT_TRAIN_ROWS": "9000000",
            "MSA_EXPERIMENT_VALIDATION_ROWS": "1000000",
            "MSA_EXPERIMENT_PRESENTATIONS": "20000000",
            "MSA_EXPERIMENT_LOG_ROWS": "5000000",
            "MSA_EXPERIMENT_DIAGNOSTIC_ROWS": "10000",
            "MSA_EXPERIMENT_INDEPENDENT_RNG": "1",
            "MSA_EXPERIMENT_MICRO_BATCH": "512",
            "MSA_EXPERIMENT_BENCHMARK": benchmark,
            "MSA_EXPERIMENT_SEED": str(settings.get("seed", 42)),
            "MSA_EXPERIMENT_VALIDATION_SEED": "42",
        }
        job = {
            "run_id": run_id,
            "name": name,
            "settings": settings,
            "session_ids": [],
            "output": output,
            "resources": {"cpu": 4, "memory_gib": 64, "gpu": 1},
            "status": "dry_run" if args.dry_run else "submission_pending",
            "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
            "attempts": (prior or {}).get("attempts", 0) + 1,
        }
        record["jobs"] = [x for x in record["jobs"] if x["run_id"] != run_id] + [job]
        atomic_json(args.record, record)
        if session:
            try:
                job["session_ids"] = (
                    session.create(
                        name=name,
                        image="images.canfar.net/astroai/base:latest",
                        kind="headless",
                        cmd="python",
                        args=payload,
                        cores=4,
                        ram=64,
                        gpu=1,
                        env=env,
                    )
                    or []
                )
                job["status"] = (
                    "submitted" if job["session_ids"] else "submission_ambiguous"
                )
            except Exception as error:
                job.update(status="submission_ambiguous", submission_error=str(error))
            atomic_json(args.record, record)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "session_ids": job["session_ids"],
                    "status": job["status"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
