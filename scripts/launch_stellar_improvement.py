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
    if plan.get("require_unseen_ancestry") and not dependencies.get(
        "final_six_large_feh_ancestry_v1:unseen_heldout", False
    ):
        return False
    if plan.get("require_unseen_feh_test") and not dependencies.get(
        "final_six_large_feh_ancestry_v1:unseen_test", False
    ):
        return False
    return all(dependencies.get(name, False) for name in plan.get("depends_on", []))


def bind_prediction_policy(
    plan, env, config_path=Path("configs/prediction-comparison-10m.yaml")
):
    """Resolve the actual prediction parent/budget, never the legacy default."""
    if plan["task"] == "multisurvey-consensus-preflight":
        from multisurvey_consensus_preflight_launch_policy import (
            bind_consensus_preflight_policy,
        )

        return bind_consensus_preflight_policy(plan, env)
    if plan["task"] == "multisurvey-desi-overlay":
        from multisurvey_desi_overlay_launch_policy import bind_desi_overlay_policy

        return bind_desi_overlay_policy(plan, env)
    if plan["task"] == "multisurvey-desi-census":
        from multisurvey_desi_census_launch_policy import bind_desi_census_policy

        return bind_desi_census_policy(plan, env)
    if plan["task"] == "multisurvey-desi-source":
        from multisurvey_desi_launch_policy import bind_multisurvey_desi_policy

        return bind_multisurvey_desi_policy(plan, env)
    if plan["task"] in {
        "remediation-input-audit",
        "remediation-source",
        "remediation-joint-preflight",
        "remediation-pilot",
    }:
        from remediation_launch_policy import bind_remediation_policy

        return bind_remediation_policy(plan, env)
    if plan["task"] == "expanded-report-recovery":
        from expanded_report_launch_policy import bind_report_policy

        return bind_report_policy(plan, env)
    if plan["task"] in {
        "full-label-evaluation-preparation",
        "expanded-source-contract-audit",
        "expanded-checkpoint-evaluation",
    }:
        from expanded_evaluation_launch_policy import bind_expanded_evaluation_policy

        return bind_expanded_evaluation_policy(plan, env)
    if plan["task"] in {"five-job-source-preparation", "five-wave-training"}:
        from five_job_launch_policy import bind_five_job_policy

        return bind_five_job_policy(plan, env)
    if plan["task"] in {
        "final-six-frozen",
        "final-six-ancestry",
        "metalpoor-corner-selection",
    }:
        filenames = {
            "final-six-frozen": "configs/final-six-frozen-20261024.yaml",
            "final-six-ancestry": "configs/final-six-ancestry-20261024.yaml",
            "metalpoor-corner-selection": "configs/metalpoor-corner-selection-20261024.yaml",
        }
        suite = yaml.safe_load(Path(filenames[plan["task"]]).read_text())
        if plan["task"] == "final-six-frozen":
            if (
                plan["run_id"] not in {a["run_id"] for a in suite["arms"]}
                or not {
                    "six_target_source_preparation_v3",
                    "final_six_cuda_smoke_v1",
                    "final_six_large_feh_ancestry_v1",
                }.issubset(plan.get("depends_on", []))
                or not plan.get("require_unseen_ancestry")
            ):
                raise ValueError(
                    "Frozen arm or required source/CUDA/ancestry gates differ"
                )
        elif plan["run_id"] != suite["run_id"]:
            raise ValueError("Unknown final source/corner data audit")
        expected = {
            "MSA_PRETRAIN_RESUME": yaml.safe_load(
                Path("configs/final-six-recipes-20261024.yaml").read_text()
            )["checkpoint"],
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError(
                    "Final source/zero-reconstruction budget override differs"
                )
            env[key] = value
        return env
    if plan["task"] in {"final-six-recipes", "final-six-smoke"}:
        suite = yaml.safe_load(
            Path("configs/final-six-recipes-20261024.yaml").read_text()
        )
        if plan["task"] == "final-six-recipes":
            arm = next(
                (a for a in suite["arms"] if a["run_id"] == plan["run_id"]), None
            )
            if arm is None:
                raise ValueError("Six-target arm absent from immutable policy")
            if not {
                "six_target_source_preparation_v3",
                "final_six_cuda_smoke_v1",
                "final_six_large_feh_ancestry_v1",
            }.issubset(plan.get("depends_on", [])):
                raise ValueError(
                    "Six-target training requires source, CUDA and ancestry gates"
                )
            if not plan.get("require_unseen_ancestry"):
                raise ValueError("Six-target training requires ancestry eligibility")
        else:
            if (
                plan["run_id"] != "final_six_cuda_smoke_v1"
                or "six_target_source_preparation_v3" not in plan.get("depends_on", [])
            ):
                raise ValueError("Unknown six-target smoke or missing source gate")
            arm = {}
        expected = {
            "MSA_PRETRAIN_RESUME": suite["reference_checkpoint"]
            if arm.get("parent86_control")
            else suite["checkpoint"],
            "MSA_EXPERIMENT_PRESENTATIONS": str(suite["reconstruction_presentations"])
            if plan["task"] == "final-six-recipes"
            else "2560",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Six-target parent/budget override differs")
            env[key] = value
        return env
    if plan["task"] in {"six-target-preparation", "guarded-ttt"}:
        if plan["task"] == "six-target-preparation":
            config_path = {
                "six_target_source_preparation_v1": "configs/six-target-preparation-20261024.yaml",
                "six_target_source_preparation_v2": "configs/six-target-preparation-20261024-v2.yaml",
                "six_target_source_preparation_v3": "configs/six-target-preparation-20261024-v3.yaml",
            }.get(plan["run_id"])
            if config_path is None:
                raise ValueError("Unknown six-target source preparation")
            suite = yaml.safe_load(Path(config_path).read_text())
            env["MSA_SIX_SOURCE_CONFIG"] = config_path
            if plan["run_id"] != suite["run_id"]:
                raise ValueError("Unknown six-target source preparation")
        else:
            suite = yaml.safe_load(
                Path("configs/guarded-ttt-screen-20261024.yaml").read_text()
            )
            if plan["run_id"] not in {arm["run_id"] for arm in suite["arms"]}:
                raise ValueError("Guarded TTT arm absent from immutable policy")
        expected = {
            "MSA_PRETRAIN_RESUME": "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth",
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Final prediction source/budget override differs")
            env[key] = value
        return env
    if plan["task"] in {
        "density-rescore",
        "conformal-validation",
        "target-head-pilot",
        "catalogue-comparison",
    }:
        expected = {
            "MSA_PRETRAIN_RESUME": "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth",
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Evaluation/pilot parent and replay budget differ")
            env[key] = value
        return env
    if plan["task"] in {"target-predictions", "target-source-evidence", "mae-ttt"}:
        suite = yaml.safe_load(
            Path("configs/posterior-target-wave-20261023.yaml").read_text()
        )
        expected = {
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
            "MSA_PRETRAIN_RESUME": "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth",
        }
        if plan["task"] == "target-predictions":
            arm = next(
                (a for a in suite["runs"] if a["run_id"] == plan["run_id"]), None
            )
            if arm is None:
                raise ValueError("Target run absent from immutable source policy")
            prep = suite["preparation"][arm["preparation"]]
            remote = ROOT + "/" + Path(prep["local_directory"]).name
            expected.update(
                MSA_TARGET_PREPARED=remote + "/prepared.npz",
                MSA_TARGET_PREPARATION_MANIFEST=remote + "/manifest.json",
                MSA_TARGET_KIND=arm["kind"],
                MSA_TARGET_SEED=str(arm["seed"]),
                MSA_TARGET_STEPS=str(suite["training_policy"]["steps"]),
            )
        elif plan["task"] == "target-source-evidence":
            if plan["run_id"] != suite["source_projection"]["run_id"]:
                raise ValueError("Alpha source run absent from immutable policy")
        else:
            mae = yaml.safe_load(
                Path("configs/mae-ttt-screen-20261023.yaml").read_text()
            )
            if plan["run_id"] not in {a["run_id"] for a in mae["arms"]}:
                raise ValueError("TTT run absent from immutable policy")
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Target/TTT source policy override differs")
            env[key] = value
        return env
    if plan["task"] == "final-six-normalization-export":
        if plan["run_id"] != "final6_distribution_normalization_v1":
            raise ValueError("Unknown immutable normalization export")
        if plan.get("env", {}).get("MSA_EXPERIMENT_PRESENTATIONS", "0") != "0":
            raise ValueError("Metadata export cannot train")
        env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
        return env
    if plan["task"] == "final-six-distribution-metrics":
        config_path = "configs/final-six-distribution-metrics-20261024.yaml"
        config = yaml.safe_load(Path(config_path).read_text())
        matches = [a for a in config["arms"] if a["run_id"] == plan["run_id"]]
        if len(matches) != 1 or plan.get("depends_on") != [
            matches[0]["producer_run_id"]
        ]:
            raise ValueError("Distribution diagnostic producer/policy differs")
        expected = {
            "MSA_DISTRIBUTION_METRICS_CONFIG": config_path,
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError(
                    "Distribution diagnostics must not change source or training budget"
                )
            env[key] = value
        return env
    if plan["task"] == "metalpoor-corner-sampling":
        filename = "configs/metalpoor-corner-sampling-20261024.yaml"
        config = yaml.safe_load(Path(filename).read_text())
        if plan["run_id"] != config["run_id"] or len(config["models"]) != 3:
            raise ValueError("Explicit three verified corner descriptors required")
        required = {
            "representative_metalpoor_corner_selection_v1",
            *(m["run_id"] for m in config["models"]),
        }
        if not required.issubset(plan.get("depends_on", [])):
            raise ValueError("Corner source/model dependencies differ")
        expected = {
            "MSA_CORNER_SAMPLING_CONFIG": filename,
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Corner policy override differs")
            env[key] = value
        return env
    if plan["task"] == "final-six-independent":
        producer = plan.get("producer_run_id")
        arms = set()
        for filename in (
            "configs/final-six-recipes-20261024.yaml",
            "configs/final-six-frozen-20261024.yaml",
        ):
            arms.update(
                a["run_id"] for a in yaml.safe_load(Path(filename).read_text())["arms"]
            )
        if (
            producer not in arms
            or plan["run_id"] != "qa_" + producer
            or plan.get("depends_on") != [producer]
        ):
            raise ValueError(
                "Independent six-target audit must bind a known completed producer"
            )
        expected = {
            "MSA_FINAL_SIX_PRODUCER_RUN": producer,
            "MSA_FINAL_SIX_PRODUCER_SHA": "ba3cefdf13939588a5039292d7fb62fa3828dd2a",
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError(
                    "Independent six-target source or zero-training policy differs"
                )
            env[key] = value
        return env
    if plan["task"] == "alpha-head-verification":
        suite = yaml.safe_load(
            Path("configs/alpha-independent-heads-20261023.yaml").read_text()
        )
        if plan["run_id"] != suite["run_id"]:
            raise ValueError("Unknown independent alpha head audit")
        expected = {
            "MSA_PRETRAIN_RESUME": "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth",
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
        }
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Independent alpha audit policy override differs")
            env[key] = value
        return env
    if plan["task"] in {"alpha-feature-preparation", "alpha-target-predictions"}:
        suite = yaml.safe_load(
            Path("configs/alpha-target-wave-20261023.yaml").read_text()
        )
        expected = {
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
            "MSA_PRETRAIN_RESUME": "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth",
        }
        if plan["task"] == "alpha-feature-preparation":
            if plan["run_id"] != suite["feature_preparation"]["run_id"]:
                raise ValueError("Unknown alpha preparation")
        else:
            arm = next(
                (a for a in suite["runs"] if a["run_id"] == plan["run_id"]), None
            )
            if arm is None:
                raise ValueError("Unknown alpha predictive arm")
            expected.update(
                MSA_TARGET_PREPARED=suite["training_policy"]["prepared"],
                MSA_TARGET_PREPARATION_MANIFEST=suite["training_policy"][
                    "preparation_manifest"
                ],
                MSA_TARGET_KIND=arm["kind"],
                MSA_TARGET_SEED=str(arm["seed"]),
                MSA_TARGET_STEPS=str(suite["training_policy"]["steps"]),
                MSA_TARGET_LIBRARY_REV=suite["library_revision"],
            )
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Alpha source/target policy override differs")
            env[key] = value
        return env
    if plan["task"] in {"forward-floweb", "floweb-quadrature-recovery"}:
        suite_path = (
            "configs/floweb-quadrature-recovery-20261023.yaml"
            if plan["task"] == "floweb-quadrature-recovery"
            else "configs/forward-optimizer-wave-20261023.yaml"
        )
        suite = yaml.safe_load(Path(suite_path).read_text())
        arm = next((a for a in suite["runs"] if a["run_id"] == plan["run_id"]), None)
        if arm is None:
            raise ValueError("Inversion run absent from immutable policy")
        expected = {
            "MSA_PRETRAIN_RESUME": "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth",
            "MSA_EXPERIMENT_PRESENTATIONS": "0",
            "MSA_FORWARD_ORIGINAL": suite["original"],
            "MSA_FORWARD_MODE": arm["mode"],
            "MSA_FORWARD_SEED": str(arm["seed"]),
            "MSA_FORWARD_TEST_COUNT": str(
                arm.get("test_count", suite["policy"]["test_count"])
            ),
            "MSA_FORWARD_CAL_COUNT": str(
                arm.get("cal_count", suite["policy"]["calibration_count"])
            ),
        }
        if plan["task"] == "floweb-quadrature-recovery":
            expected.update(
                MSA_FLOWEB_PRIOR=arm["prior"],
                MSA_FLOWEB_PRIOR_SHA=arm["prior_sha256"],
                MSA_FLOWEB_PRIOR_NODE_CAP=str(
                    arm.get("prior_node_cap", suite["policy"]["prior_node_cap"])
                ),
                MSA_FLOWEB_CONVERGENCE_REPORT=arm.get("resume_convergence", ""),
                MSA_FLOWEB_CONVERGENCE_SHA=arm.get("resume_convergence_sha256", ""),
            )
        for key, value in expected.items():
            if key in plan.get("env", {}) and str(plan["env"][key]) != value:
                raise ValueError("Forward optimizer policy override differs")
            env[key] = value
        return env
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
        "multilabel-validation",
        "multilabel-sls",
        "multilabel-noise-contrastive",
    }:
        suite_path = {
            "multilabel-screen": "configs/multilabel-screen-20261020.yaml",
            "multilabel-long": "configs/multilabel-long-20261020.yaml",
            "multilabel-density-screen": "configs/multilabel-density-20261021.yaml",
            "multilabel-w2": "configs/multilabel-w2-20261022.yaml",
            "multilabel-w1": "configs/multilabel-w1-20261022.yaml",
            "multilabel-validation": "configs/multilabel-validation-wave-20261022.yaml",
            "multilabel-sls": "configs/multilabel-sls-20261023.yaml",
            "multilabel-noise-contrastive": "configs/multilabel-noise-contrastive-20261023.yaml",
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


def submission_error_evidence(error, body_limit=8192):
    """Retain bounded server rejection evidence without repeating a request."""
    evidence = dict(error=str(error), error_type=type(error).__name__)
    response = getattr(error, "response", None)
    if response is not None:
        evidence["http_status"] = response.status_code
        body = response.content
        evidence["response_body"] = body[:body_limit].decode("utf-8", errors="replace")
        evidence["response_body_truncated"] = len(body) > body_limit
    return evidence


def launch_image(plan):
    """Allow the published AstroAI runtime explicitly, preserving old defaults."""
    image = plan.get("image", "images.canfar.net/astroai/base:latest")
    if image not in {
        "images.canfar.net/astroai/base:latest",
        "images.canfar.net/astroai/terminal:latest",
    }:
        raise ValueError("Launch image must be an approved AstroAI runtime")
    return image


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
    p.add_argument("--git-ref", default="codex/xp-reconstruction-experiments")
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
    if any(
        p["task"] in {"five-job-source-preparation", "five-wave-training"}
        for p in plans
    ):
        from five_job_launch_policy import enforce_submission_budget

        enforce_submission_budget(plans, record)
    client = None
    existing = []
    if not args.dry_run:
        from canfar.sessions import Session

        if any(
            plan["task"].startswith("remediation-")
            or plan["task"]
            in {
                "multisurvey-desi-source",
                "multisurvey-desi-census",
                "multisurvey-desi-overlay",
                "multisurvey-consensus-preflight",
            }
            for plan in plans
        ):
            from canfar.storage import filesystem
            from remediation_launch_policy import assert_project_storage_budget

            storage = assert_project_storage_budget(
                filesystem("arc").info("projects/k-pop")
            )
            print(json.dumps({"remediation_storage_preflight": storage}), flush=True)
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
            expected_status = {
                "remediation_inputs_v1": "remediation_input_audit_complete",
                "remediation_inputs_v2": "remediation_input_audit_complete",
                "remediation_inputs_v3": "remediation_input_audit_complete",
                "remediation_inputs_v4": "remediation_input_audit_complete",
                "remediation_source_v1": "remediation_source_complete",
                "remediation_source_v2": "remediation_source_complete",
                "remediation_source_v3": "remediation_source_complete",
                "remediation_joint6_preflight_v1": "remediation_joint6_preflight_pass",
                "remediation_joint6_preflight_v2": "remediation_joint6_preflight_pass",
                "remediation_joint6_preflight_v3": "remediation_joint6_preflight_pass",
                "remediation_nsf6_qmc_probe_v1": "remediation_nsf_qmc_probe_complete",
                "alpha_desi_source_projection_v1": "native_source_projection_complete_not_training_certificate",
                "alpha_desi_feature_preparation_v1": "source_audited_alpha_features_complete",
                "alpha_desi_feature_preparation_v2": "source_audited_alpha_features_complete",
                "six_target_source_preparation_v1": "sparse_six_target_source_assembly_complete",
                "six_target_source_preparation_v2": "sparse_six_target_source_assembly_complete",
                "six_target_source_preparation_v3": "sparse_six_target_source_assembly_complete",
                "five_job_broad_test_preparation_v1": "five_job_source_preparation_complete",
                "full_label_evaluation_preparation_v1": "full_label_evaluation_source_complete",
                "full_label_evaluation_preparation_v2": "full_label_evaluation_source_complete",
                "representative_metalpoor_corner_selection_v1": "representative_metalpoor_corner_selection_complete",
            }.get(name, "complete")
            if name == "final_six_large_feh_ancestry_v1":
                expected_status = "source_id_feh_aux_ancestry_verified"
                dependencies[name + ":unseen_heldout"] = (
                    content.get("direct_target_heldout_overlap_detected") is False
                )
                dependencies[name + ":unseen_test"] = (
                    content.get("eligible_role_counts", content.get("counts", {}))
                    .get("fe_h", {})
                    .get("test", {})
                    .get("feh_aux_fit_exposed_count")
                    == 0
                )
            dependencies[name] = content.get("status") == expected_status
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
        args.git_ref,
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
        # H100 MIGs are allocated as indivisible 1g slices (10-12 GiB, 1/7 SMs).
        # CPU-only phases must not consume a slice: route them to CPU sessions and
        # omit the GPU request entirely. ``gpu`` may be 0/omitted; >1 only makes
        # sense when the runner itself uses multiple devices.
        cpu_only_tasks = {
            "catalog",
            "age-data-audit",
            "target-source-evidence",
            "alpha-feature-preparation",
            "alpha-head-verification",
            "final-six-independent",
            "six-target-preparation",
            "five-job-source-preparation",
            "full-label-evaluation-preparation",
            "expanded-source-contract-audit",
            "expanded-report-recovery",
            "remediation-input-audit",
            "remediation-source",
            "multisurvey-desi-source",
            "final-six-ancestry",
            "final-six-normalization-export",
            "metalpoor-corner-selection",
            "conformal-validation",
            "target-head-pilot",
            "catalogue-comparison",
            "analysis",
            "prediction-comparison",
        }
        if plan["task"] in cpu_only_tasks:
            resources = {"cores": 8, "ram": 32}
        else:
            resources = {"cores": 4, "ram": 64, "gpu": 1}
        if plan.get("resources"):
            resources = dict(plan["resources"])
            if set(resources) - {"cores", "ram", "gpu"}:
                raise ValueError("Resource override may only contain cores/ram/gpu")
            for key, value in resources.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError("Resource override must contain integers")
                if key == "gpu" and value < 0:
                    raise ValueError("gpu must be >= 0")
                if key != "gpu" and value < 1:
                    raise ValueError("cores/ram must be positive")
        if resources.get("gpu", 0) == 0:
            resources.pop("gpu", None)
            env["MSA_EXPERIMENT_CPU_ONLY"] = "1"
        if plan["task"] == "catalog":
            env.update(MSA_EXPERIMENT_CPU_ONLY="1", MSA_CATALOG_MODE=plan["mode"])
        job = {
            "run_id": run,
            "name": name,
            "source_sha": args.git_sha,
            "output": ROOT + "/" + run,
            "plan": plan,
            "resources": dict(resources),
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
                    image=launch_image(plan),
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
                job.update(
                    status="submission_ambiguous", **submission_error_evidence(error)
                )
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
