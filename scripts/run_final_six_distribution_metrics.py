"""New CANFAR-only CDF and sample scores for immutable, verified six-target heads.

No SGD or original metric/model audit replay. Two independent 512-draw score
replicates; energy uses 128 members, variogram uses 512, empirical quantiles
pool 1024. Batch caches bind source, library, checkpoint, input and metric code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import sample_metalpoor_corners as sampling
import torch
import yaml
from torchregress.metrics.distribution import (
    crps_from_samples,
    kolmogorov_smirnov_uniform_statistic,
    vario_score,
)

from masked_stellar_autoencoder.training.distribution_diagnostics import (
    GROUPS,
    marginal_cdf,
    native_coordinates,
    predictive_draws,
    predictive_quantile_metrics,
    sample_metrics,
)
from masked_stellar_autoencoder.training.final_six_recipes import (
    NAMES,
    standardized_arrays,
)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_bound_cache(path, ids):
    receipt = path.with_suffix(".sha256.json")
    if not receipt.exists() or json.loads(receipt.read_text()).get(
        "sha256"
    ) != sampling.sha(path):
        raise ValueError("Missing or changed cached metric hash receipt")
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    if not np.array_equal(values["source_id"], ids):
        raise ValueError("Changed cached batch identity")
    return values


def calculate_batch(head, latent, y, sigma, native_y, mask, center, scale, seed):
    result = {}
    draws_by_mode = {"intrinsic": [], "observed": []}
    for rep in range(2):
        generator = torch.Generator(device=latent.device).manual_seed(
            seed + rep * 100003
        )
        draws = predictive_draws(head, latent, 512, generator)
        noise = torch.randn(
            draws.shape, device=draws.device, dtype=draws.dtype, generator=generator
        )
        draws_by_mode["intrinsic"].append(native_coordinates(draws, center, scale))
        draws_by_mode["observed"].append(
            native_coordinates(draws + noise * sigma, center, scale)
        )
    for mode, members in draws_by_mode.items():
        cdf = marginal_cdf(head, latent, y, None if mode == "intrinsic" else sigma)
        result[mode + "_pit"] = cdf.masked_fill(~mask, float("nan"))
        first = sample_metrics(members[0], native_y, center, scale, mask)
        second = sample_metrics(members[1], native_y, center, scale, mask)
        for key in first:
            result[mode + "_" + key] = (first[key] + second[key]) / 2
            result[mode + "_" + key + "_replica_half_gap"] = (
                first[key] - second[key]
            ).abs() / 2
        for key, value in predictive_quantile_metrics(
            torch.cat(members), native_y, mask
        ).items():
            result[mode + "_" + key] = value
    return result, draws_by_mode


def pilot_contracts(head, latent, y, mask, native_y, sigma, center, scale, samples):
    cdf = marginal_cdf(head, latent, y)
    lower = marginal_cdf(head, latent, y - 0.05)
    upper = marginal_cdf(head, latent, y + 0.05)
    if not ((lower <= cdf + 1e-12).all() and (cdf <= upper + 1e-12).all()):
        raise AssertionError("Intrinsic CDF is not monotone")
    empirical = (torch.cat(samples["intrinsic"]) <= native_y).double().mean(0)
    max_gap = float((empirical - cdf).abs()[mask].max())
    # This is an explicit finite-sampling contract, NOT a relaxed density gate.
    if max_gap > 0.12:
        raise AssertionError(f"1024-draw empirical/intrinsic CDF mismatch {max_gap}")
    column = int(mask.sum(0).argmax())
    rows = mask[:, column].nonzero().squeeze(1)[:8]
    if not len(rows):
        raise ValueError("No observed pilot labels")
    source = samples["intrinsic"][0][:16, rows, column]
    target = native_y[rows, column]
    crps = crps_from_samples(source, target, reduction="none")
    library_vario = vario_score(source, target, rho=1)
    if abs(library_vario + float(crps.mean())) > 1e-8:
        raise AssertionError("Vario rho1 must be negative empirical CRPS")
    normal = torch.distributions.Normal(y.new_tensor(0), y.new_tensor(1))
    roundtrip = 0.0
    if head.kind == "factorized_nsf":
        for scalar in head.heads:
            distribution = scalar.distribution(latent)
            for probability in (0.05, 0.5, 0.95):
                q = normal.icdf(y.new_tensor(probability)).expand(1, len(latent), 1)
                recovered = normal.cdf(
                    distribution.transform(distribution.transform.inv(q))
                )
                roundtrip = max(roundtrip, float((recovered - probability).abs().max()))
        if roundtrip > 1e-10:
            raise AssertionError("Actual scalar NSF CDF/inverse roundtrip failed")
    observed = marginal_cdf(head, latent, y, sigma)
    no_error = sigma == 0
    if not torch.allclose(cdf[no_error], observed[no_error], atol=1e-12, rtol=0):
        raise AssertionError("Unknown/zero error must not change CDF")
    return dict(
        status="passed",
        empirical_cdf_max_gap=max_gap,
        empirical_cdf_threshold=0.12,
        empirical_sample_count=1024,
        cdf_monotone=True,
        actual_scalar_nsf_cdf_roundtrip_max_error=roundtrip,
        original_5e_minus5_density_reports_unchanged=True,
    )


def aggregate(values, role):
    rows = []
    mask, target = values["mask"], values["target_native"]
    tail = mask[:, 2] & (target[:, 2] < -2)
    for scope, keep in (("global", np.ones(len(mask), bool)), ("metal_poor", tail)):
        for column, name in enumerate(NAMES):
            for mode in ("intrinsic", "observed"):
                subset = keep & mask[:, column]
                for survey in [
                    "all_sources",
                    *sorted(set(values["label_source"][subset, column].astype(str))),
                ]:
                    valid = (
                        subset
                        if survey == "all_sources"
                        else subset
                        & (values["label_source"][:, column].astype(str) == survey)
                    )
                    record = dict(
                        role=role,
                        scope=scope,
                        target=name,
                        mode=mode,
                        survey=survey,
                        count=int(valid.sum()),
                        known_measurement_error_count=int(
                            (valid & values["error_known"][:, column]).sum()
                        ),
                    )
                    for key, array in values.items():
                        if key.startswith(mode + "_") and array.ndim == 2:
                            selected = array[valid, column]
                            record[key] = (
                                float(selected.mean()) if len(selected) else None
                            )
                    if valid.any():
                        pit = torch.from_numpy(
                            values[mode + "_pit"][valid, column].copy()
                        )
                        record[mode + "_pit_uniform_ks"] = float(
                            kolmogorov_smirnov_uniform_statistic(pit)
                        )
                    rows.append(record)
        for group, columns in GROUPS.items():
            valid = keep & mask[:, columns].all(-1)
            for mode in ("intrinsic", "observed"):
                record = dict(
                    role=role,
                    scope=scope,
                    group=group,
                    mode=mode,
                    count=int(valid.sum()),
                )
                for key, array in values.items():
                    if (
                        key.startswith(mode + "_")
                        and array.ndim == 1
                        and ("_" + group + "_") in key
                    ):
                        record[key] = (
                            float(array[valid].mean()) if valid.any() else None
                        )
                rows.append(record)
    return rows


def run(args):
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Native distribution diagnostics execute only on CANFAR headless CUDA"
        )
    cfg = yaml.safe_load(args.config.read_text())
    arm = next(a for a in cfg["arms"] if a["run_id"] == args.run_id)
    choices = yaml.safe_load(Path(cfg["selected_models"]).read_text())
    descriptor = next(
        m for m in choices["models"] if m["run_id"] == arm["producer_run_id"]
    )
    suite = yaml.safe_load(Path(cfg["training_policy"]).read_text())
    installed = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if installed.get("vcs_info", {}).get("commit_id") != suite["library_revision"]:
        raise ValueError("Actual library pin differs")
    evidence = json.loads(Path(choices["selection_evidence"]).read_text())
    if (
        sampling.sha(choices["selection_evidence"])
        != choices["selection_evidence_sha256"]
    ):
        raise ValueError("TUNE-only choice evidence differs")
    sampling.validate_descriptor(
        descriptor, choices["selection_sha256"], suite["prepared_sha256"], evidence
    )
    if sampling.sha(suite["prepared"]) != suite["prepared_sha256"]:
        raise ValueError("Prepared source differs")
    args.output.mkdir(parents=True, exist_ok=True)
    singleton = (args.output / "singleton.lock").open("w")
    fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bindings = dict(
        config=sampling.sha(args.config),
        checkpoint=descriptor["checkpoint_sha256"],
        input=suite["prepared_sha256"],
        proof=descriptor["independent_report_sha256"],
        producer_source=descriptor["expected_source_sha"],
        runner=sampling.sha(__file__),
        kernel=sampling.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/distribution_diagnostics.py"
        ),
        torch_version=torch.__version__,
        library=suite["library_revision"],
        model_loader=sampling.sha(sampling.__file__),
        encoder_loader=sampling.sha(sampling.architecture.__file__),
        encoder_implementation=sampling.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/models/model.py"
        ),
        head_implementation=sampling.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/final_six_recipes.py"
        ),
        head_components=sampling.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/multilabel_experiments.py"
        ),
    )
    identity = args.output / "bindings.json"
    if identity.exists() and json.loads(identity.read_text()) != bindings:
        raise ValueError("Changed batch cache bindings; preserve old output")
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get(
        "status"
    ) in {"complete", "failed"}:
        raise ValueError("Terminal diagnostic output exists; no automatic repeat")
    atomic(identity, bindings)
    manifest = dict(
        status="preparing",
        run_id=args.run_id,
        producer_run_id=arm["producer_run_id"],
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        bindings=bindings,
        training_presentations=0,
        optimizer_steps=0,
        old_prediction_and_audit_outputs_repeated=False,
        known_error_scope="Quoted diagonal measurement errors only; age/distance posterior widths never used as measurement errors; unknown sigma zero is a scoring convention, not known zero noise",
        metric_scope="Actual learned distributions. Native CRPS/quantiles; FIT-standardized coordinate energy and variogram (ln age/distance). Intrinsic exact CDF; NSF observed CDF finite fixed32-QMC approximation.",
        sample_policy=dict(
            replicates=2,
            samples_each=512,
            energy_samples_each=128,
            variogram_samples_each=512,
            quantile_pooled_samples=1024,
        ),
        tail_pit_warning="Conditioning on measured FeH<−2 changes the PIT null even for a correct model: tail KS descriptive, not a valid uniformity hypothesis test",
        target_limits="Catalogue posterior age/distance, measured native labels; no physical posterior or missing six-target joint truth claim",
        score_orientation="CRPS/energy/variogram/pinball/interval score lower is better; Vario rho1 higher is better (negative CRPS). Tail PIT KS is descriptive only.",
        full_pipeline_certificate=False,
    )
    atomic(manifest_path, manifest)
    with np.load(suite["prepared"], allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    model, head, center, scale = sampling.load_model_and_head(
        descriptor, "cuda", data["feature_names"]
    )
    atomic(
        args.output / "normalization.json",
        dict(
            center=center.cpu().tolist(),
            scale=scale.cpu().tolist(),
            checkpoint_sha256=descriptor["checkpoint_sha256"],
            coordinates=["K", "dex", "dex", "dex", "ln(Gyr)", "ln(kpc)"],
        ),
    )
    model.requires_grad_(False)
    head.requires_grad_(False)
    standardized, sigma, mask, _ = standardized_arrays(
        data["value"],
        data["observed"] & data["quality"],
        data["error"],
        data["error_observed"],
        center.cpu().numpy(),
        scale.cpu().numpy(),
    )
    cache = args.output / "batches"
    cache.mkdir(exist_ok=True)
    summary = []
    pilot_checks = {}
    manifest.update(status="running", gpu=torch.cuda.get_device_name())
    atomic(manifest_path, manifest)
    for role_index, role in enumerate(("cal", "test")):
        rows = np.flatnonzero(data["role"] == role)
        pieces = []
        for start in range(0, len(rows), 128):
            rr = rows[start : start + 128]
            file = cache / f"{role}-{start:06d}.npz"
            contract_file = cache / f"{role}-{start:06d}-contract.json"
            if file.exists():
                result = load_bound_cache(file, data["source_id"][rr])
                if start == 0:
                    if (
                        not contract_file.exists()
                        or json.loads(contract_file.read_text())["status"] != "passed"
                    ):
                        raise ValueError("Cached pilot is not a passed pilot")
                    pilot_checks[role] = json.loads(contract_file.read_text())
            else:
                seed = cfg["seed"] + role_index * 1000003 + start
                with torch.no_grad():
                    latent = model.encoder(
                        torch.tensor(
                            data["input"][rr], dtype=torch.float64, device="cuda"
                        )
                    )
                    y = torch.tensor(
                        standardized[rr], dtype=torch.float64, device="cuda"
                    )
                    error = torch.tensor(sigma[rr], dtype=torch.float64, device="cuda")
                    target = torch.tensor(
                        data["value"][rr], dtype=torch.float64, device="cuda"
                    )
                    valid = torch.tensor(mask[rr], device="cuda")
                    values, sample_sets = calculate_batch(
                        head, latent, y, error, target, valid, center, scale, seed
                    )
                    result = {k: v.cpu().numpy() for k, v in values.items()}
                    result.update(
                        source_id=data["source_id"][rr],
                        rows=rr,
                        mask=mask[rr],
                        target_native=data["value"][rr],
                        error_known=data["error_observed"][rr] & mask[rr],
                        label_source=data["label_source"][rr],
                    )
                    temp = file.with_suffix(".tmp")
                    with temp.open("wb") as stream:
                        np.savez_compressed(stream, **result)
                    temp.replace(file)
                    atomic(
                        file.with_suffix(".sha256.json"), {"sha256": sampling.sha(file)}
                    )
                    if start == 0:
                        np.savez_compressed(
                            cache / f"{role}-pilot-samples.npz",
                            intrinsic=np.concatenate(
                                [x.cpu().numpy() for x in sample_sets["intrinsic"]]
                            ),
                            observed=np.concatenate(
                                [x.cpu().numpy() for x in sample_sets["observed"]]
                            ),
                            source_id=data["source_id"][rr],
                        )
                        pilot_checks[role] = pilot_contracts(
                            head,
                            latent,
                            y,
                            valid,
                            target,
                            error,
                            center,
                            scale,
                            sample_sets,
                        )
                        atomic(contract_file, pilot_checks[role])
            pieces.append(result)
            atomic(
                args.output / "progress.json",
                dict(
                    role=role,
                    processed=start + len(rr),
                    total=len(rows),
                    completed_metric_batches=len(pieces),
                    bindings=bindings,
                ),
            )
        joined = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        path = args.output / (role + "-distribution-metrics.npz")
        np.savez_compressed(path, **joined)
        summary.extend(aggregate(joined, role))
    atomic(
        args.output / "summary.json",
        dict(
            rows=summary,
            bindings=bindings,
            sample_policy=manifest["sample_policy"],
            pit_tail_null=manifest["tail_pit_warning"],
        ),
    )
    manifest.update(
        status="complete",
        pilot_contracts=pilot_checks,
        metrics_sha256={
            role: sampling.sha(args.output / (role + "-distribution-metrics.npz"))
            for role in ("cal", "test")
        },
        summary_sha256=sampling.sha(args.output / "summary.json"),
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
    )
    atomic(manifest_path, manifest)


def record_failure(args, error):
    """A rejected duplicate or changed binding must not alter another run."""
    if isinstance(error, BlockingIOError):
        return
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    old = manifest.get("bindings", {})
    if manifest.get("status") in {"complete", "failed"}:
        return
    if old and (
        old.get("runner") != sampling.sha(__file__)
        or old.get("config") != sampling.sha(args.config)
    ):
        return
    manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
    atomic(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final-six-distribution-metrics-20261024.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        record_failure(args, error)
        raise


if __name__ == "__main__":
    main()
