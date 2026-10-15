"""Idempotent staged headless launches; accepted IDs never get replaced."""

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import yaml
from launch_xp_followup import atomic_json, bootstrap, dns_addresses

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015"
PARENT = "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_57.pth"
BENCH = "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5"


def eligible(plan, now, dependencies):
    if plan.get("not_before_utc") and now < dt.datetime.fromisoformat(
        plan["not_before_utc"]
    ):
        return False
    if plan.get("require_native_registry") and not dependencies.get(
        "native_labels_ready", False
    ):
        return False
    return all(dependencies.get(name, False) for name in plan.get("depends_on", []))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--git-sha", required=True)
    p.add_argument(
        "--plan", type=Path, default=Path("configs/stellar-improvement-plan.yaml")
    )
    p.add_argument(
        "--record",
        type=Path,
        default=Path("experiments/xp/launch-stellar-improvement-20261015.json"),
    )
    p.add_argument("--runs", nargs="+")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-submit", type=int, default=12)
    args = p.parse_args()
    if len(args.git_sha) != 40:
        p.error("Require immutable source SHA")
    plans = yaml.safe_load(args.plan.read_text())["jobs"]
    record = (
        json.loads(args.record.read_text())
        if args.record.exists()
        else {"output_root": ROOT, "jobs": []}
    )
    client = None
    existing = []
    if not args.dry_run:
        from canfar.sessions import Session

        client = Session(timeout=30)
        existing = client.fetch()
    dependencies = {}
    for name in {x for job in plans for x in job.get("depends_on", [])}:
        response = (
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--max-time",
                    "20",
                    "--cert",
                    "/Users/fabbros/.ssl/cadcproxy.pem",
                    "https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/"
                    + name
                    + "/manifest.json",
                ],
                capture_output=True,
                text=True,
            )
            if not args.dry_run
            else None
        )
        try:
            content = (
                json.loads(response.stdout)
                if response and response.returncode == 0
                else {}
            )
            dependencies[name] = content.get("status") == "complete"
            if name == "label_registry_v1":
                result = content.get("result", {})
                dependencies["native_labels_ready"] = bool(
                    dependencies[name]
                    and result.get("native_galah_apogee_complete")
                    and all(
                        result.get("sources", {}).get(s, {}).get("quality_pass", 0) > 0
                        for s in ("galah", "apogee")
                    )
                )
        except (ValueError, KeyError):
            dependencies[name] = False
    payload = bootstrap(
        "codex/xp-reconstruction-experiments",
        args.git_sha,
        {} if args.dry_run else dns_addresses(),
    )
    submitted = 0
    for plan in plans:
        run = plan["run_id"]
        if args.runs and run not in args.runs:
            continue
        prior = next((x for x in record["jobs"] if x["run_id"] == run), None)
        if prior and prior.get("session_ids"):
            continue
        if not eligible(plan, dt.datetime.now(dt.UTC), dependencies):
            print(
                json.dumps(
                    {"run_id": run, "status": "waiting_for_time_or_dependencies"}
                ),
                flush=True,
            )
            continue
        name = "msa-si15-" + run.replace("_", "-")
        matches = [x for x in existing if x.get("name") == name]
        if matches:
            if prior is None:
                prior = {"run_id": run, "name": name, "output": ROOT + "/" + run}
                record["jobs"].append(prior)
            prior.update(
                session_ids=[x["id"] for x in matches],
                status="accepted_recovered",
                recovered=matches,
            )
            atomic_json(args.record, record)
            continue
        if prior and prior.get("attempted_at") and not args.dry_run:
            elapsed = (
                dt.datetime.now(dt.UTC)
                - dt.datetime.fromisoformat(prior["attempted_at"])
            ).total_seconds()
            if elapsed < 120:
                print(
                    json.dumps({"run_id": run, "status": "awaiting_reconciliation"}),
                    flush=True,
                )
                continue
        if submitted >= args.max_submit:
            break
        env = {
            "WORK": "/scratch/src",
            "PYTHONNOUSERSITE": "1",
            "MSA_XP_EXPERIMENTS": "1",
            "MSA_PRETRAIN_RESUME": PARENT,
            "MSA_EXPERIMENT_TASK": plan["task"],
            "MSA_EXPERIMENT_RUNS": run,
            "MSA_EXPERIMENT_OUTPUT": ROOT + "/" + run,
            "MSA_EXPERIMENT_BENCHMARK": BENCH,
            "MSA_EXPERIMENT_TRAIN_ROWS": "9000000",
            "MSA_EXPERIMENT_VALIDATION_ROWS": "1000000",
            "MSA_EXPERIMENT_PRESENTATIONS": "20000000",
            "MSA_EXPERIMENT_LOG_ROWS": "5000000",
            "MSA_EXPERIMENT_DIAGNOSTIC_ROWS": "10000",
            "MSA_EXPERIMENT_INDEPENDENT_RNG": "1",
            "MSA_EXPERIMENT_MICRO_BATCH": "512",
            "MSA_EXPERIMENT_SEED": "42",
            "MSA_EXPERIMENT_VALIDATION_SEED": "42",
        }
        env.update(plan.get("env", {}))
        resources = (
            {"cores": 4, "ram": 64, "gpu": 1}
            if plan["task"] != "catalog"
            else {"cores": 4, "ram": 32}
        )
        if plan["task"] == "catalog":
            env.update(MSA_EXPERIMENT_CPU_ONLY="1", MSA_CATALOG_MODE=plan["mode"])
        job = {
            "run_id": run,
            "name": name,
            "source_sha": args.git_sha,
            "output": ROOT + "/" + run,
            "plan": plan,
            "resources": resources,
            "session_ids": [],
            "status": "dry_run" if args.dry_run else "submission_pending",
            "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
            "attempts": (prior or {}).get("attempts", 0) + 1,
        }
        record["jobs"] = [x for x in record["jobs"] if x["run_id"] != run] + [job]
        atomic_json(args.record, record)
        if client:
            try:
                job["session_ids"] = (
                    client.create(
                        name=name,
                        image="images.canfar.net/astroai/base:latest",
                        kind="headless",
                        cmd="python",
                        args=payload,
                        env=env,
                        **resources,
                    )
                    or []
                )
                job["status"] = (
                    "accepted" if job["session_ids"] else "submission_ambiguous"
                )
            except Exception as error:
                job.update(status="submission_ambiguous", error=str(error))
            atomic_json(args.record, record)
        print(
            json.dumps(
                {
                    "run_id": run,
                    "status": job["status"],
                    "session_ids": job["session_ids"],
                }
            ),
            flush=True,
        )
        submitted += 1


if __name__ == "__main__":
    main()
