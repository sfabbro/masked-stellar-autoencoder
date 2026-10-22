"""Bounded CUDA/independent-CPU certificate for fixed NSF checkpoint scoring.

No fitting, optimizer steps or old-output replacement. This certifies only the
listed source rows; full-role re-evaluation remains a separate versioned audit.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import run_multilabel_screen as scoring
import run_xp_representation_probes as architecture
import torch
import torch.nn.functional as F
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.multilabel_experiments import MultiLabelHead


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(reference, actual):
    """Absolute error gate is log-unit-shift invariant; never loosen old gates."""
    result = {}
    for key, value in reference.items():
        other = actual[key]
        if (
            value.shape != other.shape
            or value.dtype != np.float64
            or other.dtype != np.float64
        ):
            raise ValueError("Float64 arrays with equal shapes required")
        error = float(np.max(np.abs(value - other)))
        result[key] = {
            "passed": bool(
                np.isfinite(value).all() and np.isfinite(other).all() and error <= 1e-9
            ),
            "max_absolute_difference": error,
        }
    return result


def manual_context(x, state):
    h = F.gelu(F.linear(x, state["input.weight"], state["input.bias"]))
    b = F.layer_norm(
        h, (h.shape[-1],), state["block.0.weight"], state["block.0.bias"], eps=1e-5
    )
    b = F.gelu(F.linear(b, state["block.1.weight"], state["block.1.bias"]))
    b = F.linear(b, state["block.3.weight"], state["block.3.bias"])
    return F.linear(h + b, state["output.weight"], state["output.bias"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/density-precision-certificate-20261022.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.run_id != config["run_id"]:
        parser.error("Run ID differs from certificate policy")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "run_id": args.run_id,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "config_sha256": digest(args.config),
        "scoring_code_sha256": digest(scoring.__file__),
        "training_updates": 0,
        "smoke": args.allow_cpu_smoke,
        "scope": "Bounded fixed-checkpoint scoring only; old failures retained, no full-role clearance",
        "cases": [],
    }

    def save():
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    save()
    try:
        if not args.allow_cpu_smoke and not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA is required for a production certificate")
        installed = importlib.metadata.distribution("torchregress")
        direct = json.loads(installed.read_text("direct_url.json") or "{}")
        revision = direct.get("vcs_info", {}).get("commit_id")
        if revision != config["library_revision"]:
            raise ValueError("Installed library revision differs from certificate pin")
        manifest["library_revision"] = revision
        torch.set_num_threads(4)
        device = torch.device("cpu" if args.allow_cpu_smoke else "cuda")
        manifest["device"] = str(device)
        if device.type == "cuda":
            manifest["gpu_name"] = torch.cuda.get_device_name(0)
        base = args.base or Path(config["base"])
        parent_path = args.parent or Path(config["checkpoint"])
        if digest(parent_path) != config["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint SHA mismatch")
        parent = torch_load_trusted(parent_path, map_location="cpu")
        spec = importlib.util.spec_from_file_location(
            "manual_nsf", Path("experiments/xp/monitor/independent_nsf_kernel.py")
        )
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        manifest["kernel_sha256"] = digest(spec.origin)
        for case in config["cases"]:
            directory = base / case["run_id"]
            old = json.loads((directory / "manifest.json").read_text())
            if old["status"] != "complete" or old["source_sha"] != case["source_sha"]:
                raise ValueError("Original production source/status mismatch")
            if old["policy"]["library_revision"] != config["library_revision"]:
                raise ValueError("Original library pin differs from certificate")
            checkpoint = directory / "checkpoint.pth"
            if (
                digest(checkpoint) != case["checkpoint_sha256"]
                or old["checkpoint_sha256"] != case["checkpoint_sha256"]
            ):
                raise ValueError("Fitted checkpoint SHA mismatch")
            paired = directory / "paired-cohort.npz"
            if digest(paired) != case["paired_sha256"]:
                raise ValueError("Paired source arrays SHA mismatch")
            with np.load(paired) as source:
                rows = source[case["role"] + "_rows"][
                    case["start"] : case["start"] + 128
                ]
                if len(rows) != 128:
                    raise ValueError("Certificate requires exactly128 actual rows")
                data = {
                    name: source[name][rows]
                    for name in ["input", "value", "error", "source_id"]
                }
                data.update(
                    {name: source[name] for name in ["label_center", "label_scale"]}
                )
            state = torch_load_trusted(
                checkpoint, map_location="cpu", weights_only=False
            )
            model = architecture.load_model(
                yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
                parent,
                torch.device("cpu"),
            )
            model.load_state_dict(state["model_state_dict"], strict=True)
            head = MultiLabelHead(
                "flow",
                flow_type="nsf",
                flow_objective=old["arm"].get("flow_objective", "nll"),
                contrastive_weight=old["arm"].get("contrastive_weight", 0.1),
            )
            head.load_state_dict(state["head_state_dict"], strict=True)
            indices = np.arange(128)
            cpu = scoring.predictions(model, head, data, indices, torch.device("cpu"))
            gpu = scoring.predictions(model, head, data, indices, device)
            other_batch = scoring.predictions(
                model, head, data, indices, device, batch=64
            )
            s = {
                k: v.double() if v.is_floating_point() else v
                for k, v in state["head_state_dict"].items()
            }
            encoder = copy.deepcopy(model.encoder).double().eval()
            with torch.no_grad():
                context = manual_context(
                    encoder(torch.tensor(data["input"], dtype=torch.float64)), s
                )
                target = torch.tensor(
                    (data["value"].astype("f8") - data["label_center"].astype("f8"))
                    / data["label_scale"].astype("f8")
                )
                lp = kernel.density(target, context, s).numpy()
                samples = torch.cat(
                    [
                        kernel.inverse(nodes[:, None].expand(-1, 128, -1), context, s)
                        for nodes in s["point_nodes"].split(16)
                    ],
                    0,
                )
                mean = samples.mean(0)
                var = (samples - mean).square().mean(0)
            jac = float(np.log(data["label_scale"].astype("f8")).sum())
            independent = {
                "mean": mean.numpy(),
                "variance": var.numpy(),
                "intrinsic_log_prob_standardized": lp,
                "observed_log_prob_standardized": lp,
                "intrinsic_log_prob_native": lp - jac,
                "observed_log_prob_native": lp - jac,
            }
            checks = {
                "cpu_vs_cuda": compare(cpu, gpu),
                "batch128_vs64": compare(gpu, other_batch),
                "manual_cpu_vs_cuda": compare(independent, gpu),
            }
            passed = all(
                field["passed"] for group in checks.values() for field in group.values()
            )
            np.savez_compressed(
                args.output / (case["run_id"] + ".npz"),
                source_id=data["source_id"],
                original_cohort_rows=rows,
                **{"gpu_" + k: v for k, v in gpu.items()},
                **{"independent_cpu_" + k: v for k, v in independent.items()},
            )
            manifest["cases"].append(
                {"policy": case, "passed": passed, "checks": checks}
            )
            save()
            del model, head, state, encoder, s
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ok = all(case["passed"] for case in manifest["cases"])
        manifest["status"] = (
            (
                "cpu_smoke_passed_not_cuda_certificate"
                if args.allow_cpu_smoke
                else "bounded_cuda_cpu_precision_passed"
            )
            if ok
            else "strict_failed"
        )
        save()
        if not ok:
            sys.exit(1)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
