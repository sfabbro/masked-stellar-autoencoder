"""Submit two read-only comparisons of a pinned completed standard checkpoint."""

import argparse
import datetime as dt
import json
from pathlib import Path

import yaml
from launch_xp_followup import atomic_json, bootstrap, dns_addresses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/xp-latest-standard-analysis.yaml")
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if len(args.git_sha) != 40 or any(
        x not in "0123456789abcdef" for x in args.git_sha
    ):
        parser.error("Require a complete immutable source SHA")
    analysis = yaml.safe_load(args.spec.read_text())
    epoch = analysis.get("checkpoint_epoch", 55)
    if args.record is None:
        args.record = Path(
            f"experiments/xp/launch-latest-standard-e{epoch}-20261014.json"
        )
    pinned = analysis["analyses"][0]
    record = {
        "git_sha": args.git_sha,
        "checkpoint": pinned["checkpoint"],
        "checkpoint_sha256": pinned["checkpoint_sha256"],
        "checkpoint_epoch": epoch,
        "read_only_evaluation": True,
        "benchmark": analysis["benchmark"],
        "output_root": f"/arc/projects/k-pop/msa_runs/xp-latest-standard-e{epoch}-20261014",
        "jobs": [],
    }
    if args.record.exists():
        previous = json.loads(args.record.read_text())
        for field in record.keys() - {"jobs"}:
            if record[field] != previous[field]:
                parser.error(f"Immutable record differs: {field}")
        record = previous
    atomic_json(args.record, record)
    client = None
    if not args.dry_run:
        from canfar.sessions import Session

        client = Session(timeout=30)
    payload = bootstrap(
        "codex/xp-reconstruction-experiments",
        args.git_sha,
        {} if args.dry_run else dns_addresses(),
    )
    for run_id, task in [
        (pinned["run_id"], "analysis"),
        (f"evaluate_latest_standard{epoch}", "evaluation"),
    ]:
        prior = next((x for x in record["jobs"] if x["run_id"] == run_id), None)
        if prior and prior["session_ids"]:
            continue
        name = f"msa-xp-standard{epoch}-" + task
        output = record["output_root"] + "/" + run_id
        if client:
            matches = [x for x in client.fetch() if x.get("name") == name]
            if matches:
                job = {
                    "run_id": run_id,
                    "name": name,
                    "session_ids": [x["id"] for x in matches],
                    "output": output,
                    "status": "accepted_recovered",
                    "task": task,
                }
                record["jobs"] = [
                    x for x in record["jobs"] if x["run_id"] != run_id
                ] + [job]
                atomic_json(args.record, record)
                print(json.dumps(job), flush=True)
                continue
            if prior:
                elapsed = (
                    dt.datetime.now(dt.UTC)
                    - dt.datetime.fromisoformat(prior["attempted_at"])
                ).total_seconds()
                if elapsed < 90:
                    print(
                        json.dumps(
                            {"run_id": run_id, "status": "awaiting_reconciliation"}
                        ),
                        flush=True,
                    )
                    continue
        env = {
            "WORK": "/scratch/src",
            "PYTHONNOUSERSITE": "1",
            "MSA_XP_EXPERIMENTS": "1",
            "MSA_EXPERIMENT_TASK": "analysis" if task == "analysis" else "training",
            "MSA_ANALYSIS_SPEC": str(args.spec),
            "MSA_EXPERIMENTS_CONFIG": analysis.get(
                "experiments_config", "configs/pretrain.xp-latest-standard.yaml"
            ),
            "MSA_PRETRAIN_RESUME": record["checkpoint"],
            "MSA_EXPERIMENT_RUNS": run_id,
            "MSA_EXPERIMENT_OUTPUT": output,
            "MSA_EXPERIMENT_BENCHMARK": record["benchmark"],
            "MSA_EXPERIMENT_TRAIN_ROWS": "9000000",
            "MSA_EXPERIMENT_VALIDATION_ROWS": "1000000",
            "MSA_EXPERIMENT_PRESENTATIONS": "20000000",
            "MSA_EXPERIMENT_MICRO_BATCH": "512",
            "MSA_EXPERIMENT_DIAGNOSTIC_ROWS": "10000",
            "MSA_EXPERIMENT_INDEPENDENT_RNG": "1",
            "MSA_EXPERIMENT_EVALUATION_ONLY": "1" if task == "evaluation" else "0",
        }
        job = {
            "run_id": run_id,
            "name": name,
            "task": task,
            "output": output,
            "session_ids": [],
            "status": "dry_run" if args.dry_run else "submission_pending",
            "attempted_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        record["jobs"] = [x for x in record["jobs"] if x["run_id"] != run_id] + [job]
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
                job.update(status="submission_ambiguous", error=str(error))
            atomic_json(args.record, record)
        print(json.dumps(job), flush=True)


if __name__ == "__main__":
    main()
