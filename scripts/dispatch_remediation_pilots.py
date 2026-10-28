"""Finite first-attempt metadata gate for an immutable remediation pilot matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = "https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015"
REQUIRED = {
    "remediation_inputs_v4": "remediation_input_audit_complete",
    "remediation_source_v3": "remediation_source_complete",
    "remediation_joint6_preflight_v3": "remediation_joint6_preflight_pass",
}


def ready(manifests, expected_source, head_sha):
    """Missing metadata is pending; an actual terminal failure is a closed gate."""
    if set(manifests) != set(REQUIRED):
        return "pending"
    for run, status in REQUIRED.items():
        manifest = manifests[run]
        actual = manifest.get("status", "")
        if actual in {"failed", "incomplete", "remediation_joint6_preflight_failed"}:
            return "blocked"
        if actual != status:
            return "pending"
        code = manifest.get("audit_code_commit", manifest.get("source_commit"))
        expected = (
            expected_source[run]
            if isinstance(expected_source, dict)
            else expected_source
        )
        if code != expected or manifest.get("run_id") != run:
            raise ValueError("Prerequisite run/source binding differs")
    audit = manifests["remediation_inputs_v4"]["readiness"]
    source = manifests["remediation_source_v3"]
    preflight = manifests["remediation_joint6_preflight_v3"]
    if not audit.get("phot_xp_input_pass") or not audit.get(
        "native_label_association_pass"
    ):
        return "blocked"
    if not source.get("training_source_ready") or not source.get("prepared_sha256"):
        return "blocked"
    if preflight.get("head_module_sha256") != head_sha:
        raise ValueError("Pilot head differs from actual tested preflight")
    if not all(
        preflight.get("checks", {}).get(kind, {}).get("passed")
        for kind in ("gaussian", "mdn", "nsf")
    ):
        return "blocked"
    return "ready"


def read_manifest(run, context):
    with urllib.request.urlopen(
        ROOT + "/" + run + "/manifest.json", context=context, timeout=25
    ) as response:
        data = response.read(5_000_001)
    if len(data) > 5_000_000:
        raise ValueError("Manifest exceeds bounded metadata contract")
    return json.loads(data)


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--prerequisite-sha", required=True)
    parser.add_argument("--audit-source-sha")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    if (
        len(args.source_sha) != 40
        or len(args.prerequisite_sha) != 40
        or not 0 < args.hours <= 24
    ):
        raise ValueError("Require immutable SHAs and finite bounded lifetime")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive state creation: never start a second writer or replay a stopped attempt.
    with args.state.open("x") as stream:
        json.dump({"status": "initializing"}, stream)
    state = dict(
        status="waiting",
        pilot_source=args.source_sha,
        prerequisite_source=args.prerequisite_sha,
        started=dt.datetime.now(dt.UTC).isoformat(),
        submission_attempted=False,
    )
    head_sha = hashlib.sha256(
        Path(
            "src/masked_stellar_autoencoder/training/remediation_joint6.py"
        ).read_bytes()
    ).hexdigest()
    context = ssl.create_default_context()
    context.load_cert_chain("/Users/fabbros/.ssl/cadcproxy.pem")
    deadline = time.monotonic() + args.hours * 3600
    try:
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != args.source_sha
        ):
            raise ValueError("Active runner checkout differs")
        # A successful push must precede this driver; never publish while it owns the ledger.
        published = subprocess.check_output(
            [
                "git",
                "ls-remote",
                "origin",
                "refs/heads/codex/xp-reconstruction-experiments",
            ],
            text=True,
        ).split()[0]
        if published != args.source_sha:
            raise ValueError("Pilot source is not the published origin head")
        while time.monotonic() < deadline:
            manifests = {}
            unavailable = []
            for run in REQUIRED:
                try:
                    manifests[run] = read_manifest(run, context)
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    json.JSONDecodeError,
                ) as error:
                    unavailable.append(
                        dict(run_id=run, error_type=type(error).__name__)
                    )
            expected_sources = {run: args.prerequisite_sha for run in REQUIRED}
            if args.audit_source_sha:
                expected_sources["remediation_inputs_v4"] = args.audit_source_sha
            gate = ready(manifests, expected_sources, head_sha)
            state.update(
                status=gate,
                manifests=manifests,
                unavailable=unavailable,
                checked=dt.datetime.now(dt.UTC).isoformat(),
            )
            atomic(args.state, state)
            if gate == "blocked":
                return 2
            if gate == "ready":
                state.update(
                    status="first_submission_attempt", submission_attempted=True
                )
                atomic(args.state, state)
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/launch_stellar_improvement.py",
                        "--git-sha",
                        args.source_sha,
                        "--plan",
                        str(args.plan),
                        "--max-submit",
                        "20",
                        "--form-body",
                        "--request-timeout",
                        "45",
                    ],
                    check=False,
                )
                state.update(
                    status="launcher_returned_no_automatic_retry",
                    launcher_returncode=result.returncode,
                )
                ledger = json.loads(
                    Path(
                        "experiments/xp/launch-stellar-improvement-20261015.json"
                    ).read_text()
                )
                import yaml

                wanted = {
                    job["run_id"]
                    for job in yaml.safe_load(args.plan.read_text())["jobs"]
                }
                receipts = [job for job in ledger["jobs"] if job["run_id"] in wanted]
                state["receipts"] = receipts
                if (
                    result.returncode == 0
                    and len(receipts) == len(wanted)
                    and all(job.get("session_ids") for job in receipts)
                ):
                    state["status"] = "all_pilots_accepted"
                atomic(args.state, state)
                return 0 if state["status"] == "all_pilots_accepted" else 3
            time.sleep(45)
        state["status"] = "deadline_no_submission"
        atomic(args.state, state)
        return 4
    except Exception as error:
        state.update(
            status="stopped_no_retry", error_type=type(error).__name__, error=str(error)
        )
        atomic(args.state, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
