"""Idempotent staged headless launches; accepted IDs never get replaced."""

import argparse
import datetime as dt
import json
import socket
import subprocess
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse

import yaml
from launch_xp_followup import atomic_json, bootstrap, dns_addresses
from targeted_policy import resolve_policy

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015"
PARENT = "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_57.pth"
BENCH = "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5"


def launch_addresses():
    """Extend the existing TLS-preserving tunnel to explicitly configured data hosts."""
    addresses = dns_addresses()
    sources = yaml.safe_load(Path("configs/stellar-catalog-sources.yaml").read_text())[
        "sources"
    ]
    hosts = {urlparse(source["url"]).hostname for source in sources}
    hosts.update({"cdsarc.u-strasbg.fr", "vizier.cds.unistra.fr", "data.sdss.org"})
    for host in sorted(hosts):
        try:
            addresses[host] = socket.gethostbyname(host)
        except OSError as error:
            print(
                json.dumps({"catalogue_host_unresolved": host, "error": str(error)}),
                flush=True,
            )
    return addresses


def eligible(plan, now, dependencies):
    if plan.get("not_before_utc") and now < dt.datetime.fromisoformat(
        plan["not_before_utc"]
    ):
        return False
    if plan.get("require_native_registry"):
        registries = [
            name
            for name in plan.get("depends_on", [])
            if name.startswith("label_registry_")
        ]
        if not registries or not all(
            dependencies.get(name + ":native_labels_ready", False)
            for name in registries
        ):
            return False
    return all(dependencies.get(name, False) for name in plan.get("depends_on", []))


def bind_prediction_policy(
    plan, env, config_path=Path("configs/prediction-comparison-10m.yaml")
):
    """Resolve the actual prediction parent/budget, never the legacy default."""
    if plan["task"] == "density-precision-certificate":
        suite = yaml.safe_load(
            Path("configs/density-precision-certificate-20261022.yaml").read_text()
        )
        if plan["run_id"] != suite["run_id"]:
            raise ValueError("Certificate run differs from pinned policy")
        expected = {
            "MSA_PRETRAIN_RESUME": suite["checkpoint"],
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError(
                    "Certificate parent/budget override differs from policy"
                )
            env[key] = value
        return env
    if plan["task"] in {
        "multilabel-screen",
        "multilabel-long",
        "multilabel-density-screen",
        "multilabel-w2",
        "multilabel-w1",
    }:
        suite_path = {
            "multilabel-screen": "configs/multilabel-screen-20261020.yaml",
            "multilabel-long": "configs/multilabel-long-20261020.yaml",
            "multilabel-density-screen": "configs/multilabel-density-20261021.yaml",
            "multilabel-w2": "configs/multilabel-w2-20261022.yaml",
            "multilabel-w1": "configs/multilabel-w1-20261022.yaml",
        }[plan["task"]]
        suite = yaml.safe_load(Path(suite_path).read_text())
        arm = next((a for a in suite["arms"] if a["run_id"] == plan["run_id"]), None)
        if arm is None:
            raise ValueError("Multilabel screen absent from numerical policy")
        expected = {
            "MSA_PRETRAIN_RESUME": suite["checkpoint"],
            "MSA_EXPERIMENT_PRESENTATIONS": str(
                0
                if arm.get("kind") == "forward"
                else suite["reconstruction_presentations"]
            ),
        }
        for key, value in expected.items():
            explicit = plan.get("env", {}).get(key)
            if explicit is not None and str(explicit) != value:
                raise ValueError(
                    f"Multilabel override {key} differs from numerical policy"
                )
            env[key] = value
        return env
    if plan["task"] == "test-time-screen":
        suite = yaml.safe_load(Path("configs/test-time-screen.yaml").read_text())
        if plan["run_id"] != suite["run_id"]:
            raise ValueError("Test-time screen differs from numerical policy")
        explicit = plan.get("env", {}).get("MSA_PRETRAIN_RESUME")
        if explicit is not None and explicit != suite["checkpoint"]:
            raise ValueError("Test-time checkpoint differs")
        env["MSA_PRETRAIN_RESUME"] = suite["checkpoint"]
        env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
        return env
    if plan["task"] == "forward-eiv-screen":
        suite = yaml.safe_load(Path("configs/forward-eiv-screen.yaml").read_text())
        if plan["run_id"] != suite["run_id"]:
            raise ValueError("Forward EIV screen differs from numerical policy")
        explicit = plan.get("env", {}).get("MSA_PRETRAIN_RESUME")
        if explicit is not None and explicit != suite["checkpoint"]:
            raise ValueError("Forward EIV parent differs")
        env["MSA_PRETRAIN_RESUME"] = suite["checkpoint"]
        env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
        return env
    if plan["task"] == "frozen-head-screen":
        suite = yaml.safe_load(Path("configs/frozen-head-screen.yaml").read_text())
        if not any(a["run_id"] == plan["run_id"] for a in suite["jobs"]):
            raise ValueError("Frozen screen absent from numerical policy")
        explicit = plan.get("env", {}).get("MSA_PRETRAIN_RESUME")
        if explicit is not None and explicit != suite["checkpoint"]:
            raise ValueError("Frozen screen parent differs from numerical policy")
        env["MSA_PRETRAIN_RESUME"] = suite["checkpoint"]
        env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
        return env
    if plan["task"] == "targeted-prediction":
        config_path = Path("configs/targeted-10m-20261019.yaml")
    elif plan["task"] == "large-huber":
        config_path = Path("configs/residual-huber-150m.yaml")
    elif plan["task"] == "prediction-continuation":
        config_path = Path("configs/residual-huber-long.yaml")
    elif plan["task"] != "prediction-comparison":
        return env
    suite = yaml.safe_load(config_path.read_text())
    if not any(a["run_id"] == plan["run_id"] for a in suite["arms"]):
        raise ValueError("Prediction run absent from numerical policy")
    suite = resolve_policy(
        suite, next(a for a in suite["arms"] if a["run_id"] == plan["run_id"])
    )
    expected = {
        "MSA_PRETRAIN_RESUME": suite["checkpoint"],
        "MSA_EXPERIMENT_PRESENTATIONS": str(suite["reconstruction_presentations"]),
    }
    for key, value in expected.items():
        explicit = plan.get("env", {}).get(key)
        if explicit is not None and str(explicit) != value:
            raise ValueError(f"Launch override {key} differs from numerical policy")
        env[key] = value
    return env


def encode_creation_form(parameters):
    """Keep repeated env fields intact in a standard URL-encoded POST body."""
    content = urlencode(parameters).encode()
    if parse_qsl(content.decode(), keep_blank_values=True) != [
        (str(k), str(v)) for k, v in parameters
    ]:
        raise ValueError("Submission fields changed during form encoding")
    return content


def create_form_session(client, **fields):
    from canfar.helpers.session import _response_session_id
    from canfar.utils import build

    payloads = build.create_parameters(**fields)
    if len(payloads) != 1:
        raise ValueError("This bounded launcher requires one session per job")
    response = client.client.post(
        url="session", content=encode_creation_form(payloads[0])
    )
    return [_response_session_id(response)]


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
    p.add_argument("--request-timeout", type=int, default=30)
    p.add_argument("--form-body", action="store_true")
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

        client = Session(timeout=args.request_timeout)
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
            if content.get("mode") == "collect":
                sources = content.get("result", {}).get("sources", [])
                dependencies[name] &= bool(sources) and all(
                    "error" not in source for source in sources
                )
            if name.startswith("label_registry_"):
                result = content.get("result", {})
                dependencies[name + ":native_labels_ready"] = bool(
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
        {} if args.dry_run else launch_addresses(),
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
        env = bind_prediction_policy(plan, env)
        resources = (
            {"cores": 4, "ram": 64, "gpu": 1}
            if plan["task"] != "catalog"
            else {"cores": 4, "ram": 32}
        )
        if plan.get("resources"):
            resources = plan["resources"]
            if set(resources) - {"cores", "ram", "gpu"} or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in resources.values()
            ):
                raise ValueError(
                    "Resource override must contain positive integer cores/ram/gpu"
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
            "submission_transport": "form_body" if args.form_body else "native_query",
        }
        record["jobs"] = [x for x in record["jobs"] if x["run_id"] != run] + [job]
        atomic_json(args.record, record)
        if client:
            try:
                job["session_ids"] = (
                    create_form_session
                    if args.form_body
                    else lambda client, **fields: client.create(**fields)
                )(
                    client,
                    name=name,
                    image="images.canfar.net/astroai/base:latest",
                    kind="headless",
                    cmd="python",
                    args=payload,
                    env=env,
                    **resources,
                ) or []
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
        if client and not job["session_ids"]:
            break


if __name__ == "__main__":
    main()
