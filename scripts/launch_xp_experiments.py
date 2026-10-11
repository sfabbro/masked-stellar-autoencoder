"""Launch one bounded CANFAR headless session per XP reconstruction experiment."""

import argparse
import base64
import json
import re
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
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/-]*", args.git_ref):
        parser.error("Invalid Git ref")
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_sha):
        parser.error("Expected a complete Git commit SHA")
    if args.record.exists():
        parser.error(
            "Use a new launch record; existing sessions must not be duplicated"
        )
    specification = yaml.safe_load(
        Path("configs/pretrain.xp-reconstruction.yaml").read_text()
    )
    remote = (
        "import os,subprocess\n"
        "root='/scratch/src/sfabbro/masked-stellar-autoencoder'\n"
        "os.makedirs('/scratch/src/sfabbro',exist_ok=True)\n"
        f"subprocess.run(['git','clone','--depth','50','--branch',{args.git_ref!r},'https://github.com/sfabbro/masked-stellar-autoencoder.git',root],check=True)\n"
        f"subprocess.run(['git','checkout','--detach',{args.git_sha!r}],cwd=root,check=True)\n"
        "subprocess.run(['bash',root+'/batch_scripts/canfar_entrypoint.sh','pretrain-experiments'],check=True)\n"
    )
    payload = (
        "exec(__import__('base64').b64decode('"
        + base64.b64encode(remote.encode()).decode()
        + "'))"
    )
    jobs = []
    record = {
        "checkpoint": args.checkpoint,
        "git_ref": args.git_ref,
        "git_sha": args.git_sha,
        "output_root": args.output_root,
        "train_stars": 10_000_000,
        "validation_stars": 1_000_000,
        "presentations": 20_000_000,
        "seed": 42,
        "jobs": jobs,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    session = None
    if not args.dry_run:
        from canfar.sessions import Session

        session = Session()
    for experiment in specification["experiments"]:
        run_id = experiment["run_id"]
        output = f"{args.output_root}/{run_id}"
        env = {
            "WORK": "/scratch/src",
            "PYTHONNOUSERSITE": "1",
            "MSA_XP_EXPERIMENTS": "1",
            "MSA_EXPERIMENTS_CONFIG": "configs/pretrain.xp-reconstruction.yaml",
            "MSA_PRETRAIN_RESUME": args.checkpoint,
            "MSA_EXPERIMENT_RUNS": run_id,
            "MSA_EXPERIMENT_OUTPUT": output,
            "MSA_EXPERIMENT_TRAIN_ROWS": "10000000",
            "MSA_EXPERIMENT_VALIDATION_ROWS": "1000000",
            "MSA_EXPERIMENT_PRESENTATIONS": "20000000",
            "MSA_EXPERIMENT_LOG_ROWS": "5000000",
            "MSA_EXPERIMENT_DIAGNOSTIC_ROWS": "10000",
        }
        request = {
            "name": f"msa-xp-{run_id.replace('_', '-')}",
            "image": "images.canfar.net/astroai/base:latest",
            "kind": "headless",
            "cmd": "python",
            "args": "-u -c " + payload,
            "cores": 4,
            "ram": 64,
            "gpu": 1,
            "env": env,
        }
        ids = [] if args.dry_run else session.create(**request)
        job = {
            "run_id": run_id,
            "session_ids": ids,
            "output": output,
            "log": f"{output}.session.log",
            "resources": {"cpu": 4, "memory_gib": 64, "gpu": 1},
            "status": "dry_run"
            if args.dry_run
            else "submitted"
            if ids
            else "submission_failed",
        }
        jobs.append(job)
        args.record.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(job), flush=True)
        if not args.dry_run and not ids:
            raise RuntimeError(
                f"Submission failed for {run_id}; preceding IDs are recorded"
            )


if __name__ == "__main__":
    main()
