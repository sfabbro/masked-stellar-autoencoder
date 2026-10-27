"""CANFAR-only bounded remediation pilots on audited sparse six-target sources.

This is a 1000-update pilot, not a pass over the full FIT population. Every
eligible per-label FIT pool remains eligible for sampling. Natural XP is kept
at supervised training/evaluation; reconstruction masking is a separate stream.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | np.ndarray):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, bytes):
        return obj.decode()
    return obj


def atomic(path, value):
    staged = path.with_suffix(path.suffix + ".building")
    staged.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def admission(cfg, arm):
    """Read prerequisites once and fail immediately; GPU jobs never wait/poll."""
    source = load_json(cfg["source_manifest"])
    audit = load_json(cfg["input_audit_manifest"])
    preflight = load_json(cfg["joint_preflight_manifest"])
    if source.get("status") != "remediation_source_complete" or not source.get(
        "training_source_ready"
    ):
        raise ValueError("Actual remediation source is not complete/ready")
    if (
        audit.get("readiness", {}).get("phot_xp_input_pass") is not True
        or audit.get("readiness", {}).get("native_label_association_pass") is not True
    ):
        raise ValueError("Exact raw photXP/native label associations have not passed")
    if (
        preflight.get("status") != "remediation_joint6_preflight_pass"
        or preflight.get("run_id") != cfg["joint_preflight_run_id"]
    ):
        raise ValueError("Actual coupled six-target preflight has not passed")
    if preflight.get("library_revision") != cfg["library_revision"]:
        raise ValueError("Preflight library differs")
    for kind in ("gaussian", "mdn", "nsf"):
        if preflight.get("checks", {}).get(kind, {}).get("passed") is not True:
            raise ValueError(f"Coupled {kind} preflight gate did not pass")
    if (
        arm.get("input_profile", "phot_xp") not in source.get("feature_profiles", {})
        or source["feature_profiles"][arm.get("input_profile", "phot_xp")].get("ready")
        is not True
    ):
        raise ValueError("Requested source input profile is not verified")
    if arm["reconstruction_weight"] and arm["encoder"] == "frozen":
        raise ValueError("Reconstruction cannot update a frozen encoder control")
    if arm["kind"] not in ("huber", "gaussian", "mdn", "nsf") or arm["encoder"] not in (
        "frozen",
        "finetune",
    ):
        raise ValueError("Undeclared head/encoder arm")
    return source, audit, preflight


def normalize_targets(value, observed, errors, error_observed, fit):
    """FIT-only native3/alpha linear, positive teacher age/distance log scales."""
    value = np.asarray(value, dtype="f8")
    valid = np.asarray(observed, bool) & np.isfinite(value)
    valid[:, 4:] &= value[:, 4:] > 0
    coordinate = np.where(valid, value, 0)
    coordinate[:, 4:] = np.log(np.where(valid[:, 4:], value[:, 4:], 1))
    center, scale = [], []
    for column in range(6):
        v = coordinate[fit & valid[:, column], column]
        if len(v) < 16 or not np.isfinite(v).all() or not v.std() > 0:
            raise ValueError(f"Insufficient nondegenerate FIT pool for {NAMES[column]}")
        center.append(v.mean())
        scale.append(v.std())
    center, scale = np.array(center), np.array(scale)
    known = np.asarray(error_observed, bool) & valid
    errors = np.asarray(errors, dtype="f8")
    if known[:, 4:].any():
        raise ValueError("Posterior teacher widths cannot be quoted measurement errors")
    if np.any(known & (~np.isfinite(errors) | (errors <= 0))):
        raise ValueError("Known measurement errors must be finite positive")
    standardized = np.where(valid, (coordinate - center) / scale, 0)
    sigma = np.where(known, np.where(np.isfinite(errors), errors, 0) / scale, 0)
    return standardized, sigma, valid, center, scale, known


def balanced_pools(fit, mask, sources, raw_logg):
    """Every FIT row eligible; label/source/gravity strata sampled explicitly."""
    pools = []
    bins = np.searchsorted([1, 2, 3, 4, 4.5], raw_logg, side="right")
    bins[~np.isfinite(raw_logg)] = -1
    for column in range(6):
        eligible = np.flatnonzero(fit & mask[:, column])
        if len(eligible) < 16:
            raise ValueError("Too few FIT labels for balanced sampler")
        strata = []
        for source in np.unique(sources[eligible, column]):
            rows = eligible[sources[eligible, column] == source]
            for band in np.unique(bins[rows]) if column == 1 else [-2]:
                subset = rows[bins[rows] == band] if column == 1 else rows
                strata.append((str(source), int(band), subset))
        pools.append(strata)
    return pools


def draw_rows(rng, pools, per_label, target_column=None):
    blocks = []
    for strata in pools if target_column is None else [pools[target_column]]:
        which = rng.integers(len(strata), size=per_label)
        blocks.append(np.array([rng.choice(strata[j][2]) for j in which], dtype="i8"))
    return np.unique(np.concatenate(blocks))


def fixed_role_rows(ids, roles, mask, role, per_label):
    """Fixed model-blind hashed identity panel; this is an explicit pilot cap."""
    hashed = np.asarray(ids, dtype="u8") * np.uint64(11400714819323198485)
    blocks = []
    for j in range(6):
        candidates = np.flatnonzero((roles == role) & mask[:, j])
        if not len(candidates):
            raise ValueError(f"Missing {role} label {NAMES[j]}")
        blocks.append(
            candidates[np.argsort(hashed[candidates], kind="stable")[:per_label]]
        )
    return np.unique(np.concatenate(blocks))


def profile_batch(h5, rows, profile, source_manifest):
    rows = np.asarray(rows, dtype="i8")
    if np.any(np.diff(rows) <= 0):
        raise ValueError("HDF5 rows must be sorted and unique")
    x = h5["input"][rows].astype("f4")
    if profile == "phot_xp_measured_parallax":
        spec = source_manifest["feature_profiles"][profile]
        j = spec["feature_index"]
        if h5["feature_names"][j].decode() != "PARALLAX":
            raise ValueError("Measured parallax feature slot differs")
        values = h5[spec["override_dataset"]][rows]
        available = h5[spec["availability_dataset"]][rows].astype(bool)
        if not np.isfinite(values[available]).all() or not np.all(
            values[~available] == -9999
        ):
            raise ValueError("Measured parallax standardized availability differs")
        x[:, j] = values
    elif profile != "phot_xp":
        raise ValueError("Unknown verified input profile")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite model inputs")
    return x


def integration_gate(diag, thresholds):
    """Strict finite-QMC estimator admission; this is not exact marginalization."""

    def array(name):
        value = diag[name]
        return (
            value.detach().cpu().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )

    ess, replica, refinement = (
        array(n)
        for n in (
            "relative_ess",
            "independent_replica_log_prob_gap",
            "refinement_log_prob_gap",
        )
    )
    metrics = {
        "relative_ess_median": np.median(ess),
        "relative_ess_min": np.min(ess),
        "replica_gap_p95": np.quantile(replica, 0.95),
        "refinement_gap_p95": np.quantile(refinement, 0.95),
    }
    if (
        not all(np.isfinite(v) for v in metrics.values())
        or metrics["relative_ess_median"] < thresholds["median_relative_ess_min"]
        or metrics["relative_ess_min"] < thresholds["minimum_relative_ess_min"]
        or metrics["replica_gap_p95"] > thresholds["gap_p95_max"]
        or metrics["refinement_gap_p95"] > thresholds["gap_p95_max"]
    ):
        raise ValueError(
            "Real sparse/noisy NSF integration gate failed: "
            + json.dumps(jsonable(metrics))
        )
    return metrics


def diagnostics_in_row_chunks(
    head, context, target, sigma, mask, *, particles, chunk_rows
):
    """Evaluate the same fixed-QMC diagnostics in bounded independent row chunks."""
    import torch

    if (
        chunk_rows < 1
        or len(context) < 1
        or any(len(x) != len(context) for x in (target, sigma, mask))
    ):
        raise ValueError("Aligned nonempty inputs and positive chunk size are required")
    chunks = [
        head.diagnostics(
            context[i : i + chunk_rows],
            target[i : i + chunk_rows],
            sigma[i : i + chunk_rows],
            mask[i : i + chunk_rows],
            particles=particles,
        )
        for i in range(0, len(context), chunk_rows)
    ]
    if any(set(chunk) != set(chunks[0]) for chunk in chunks):
        raise ValueError("Chunked diagnostics have inconsistent fields")
    result = {}
    for key in chunks[0]:
        values = [chunk[key] for chunk in chunks]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.cat(values, dim=0)
        else:
            if any(value != values[0] for value in values[1:]):
                raise ValueError("Chunked diagnostic metadata differs")
            result[key] = values[0]
    return result


def validate_integration_probe(cfg, arm):
    """Restrict the inference-only probe to a finite, explicitly versioned design."""
    if (
        arm.get("kind") != "nsf"
        or arm.get("encoder") != "frozen"
        or arm.get("reconstruction_weight") != 0
        or not arm.get("integration_probe")
        or cfg.get("integration_probe_particles") != [128, 256, 512, 1024]
        or cfg.get("integration_probe_chunk_rows") != 8
    ):
        raise ValueError("Integration probe policy differs")


def validate_microbatch_encoder(encoder, *, finetune):
    """Batch-statistics layers change meaning under microbatch accumulation."""
    import torch

    if finetune and any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in encoder.modules()
    ):
        raise ValueError(
            "Finetuned encoder with batch normalization cannot be microbatched"
        )


def backward_supervised_chunks(
    head, latent, target, sigma, observed, rows, *, chunk_rows, tensor
):
    """Accumulate the same observed-coordinate-weighted objective in small graphs."""
    import torch

    if chunk_rows < 1 or len(rows) < 1:
        raise ValueError("Positive chunk size and nonempty rows required")
    total = int(observed[rows].sum())
    if total <= 0:
        raise ValueError("No observed targets in supervised batch")
    value = None
    sums = {}
    for start in range(0, len(rows), chunk_rows):
        rr = rows[start : start + chunk_rows]
        count = int(observed[rr].sum())
        if count == 0:
            continue
        weight = count / total
        loss = head.criterion(
            latent(rr),
            tensor(target[rr]),
            tensor(sigma[rr]),
            tensor(observed[rr], torch.bool),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite chunked supervised objective")
        contribution = loss * weight
        contribution.backward()
        detached = contribution.detach()
        value = detached if value is None else value + detached
        for key, component in getattr(head, "last_loss_components", {}).items():
            amount = (
                float(component.detach())
                if hasattr(component, "detach")
                else float(component)
            )
            sums[key] = sums.get(key, 0.0) + weight * amount
    if value is None:
        raise ValueError("No supervised chunks contributed")
    return value, sums


def accumulated_encoder_gradient_contract(reconstruction, parameters):
    """Compare already-accumulated supervised and unweighted reconstruction gradients."""
    import torch

    params = tuple(p for p in parameters if p.requires_grad)
    if not params:
        return {"scope": "frozen encoder"}
    supervised = tuple(None if p.grad is None else p.grad.detach() for p in params)
    reco = (
        torch.autograd.grad(
            reconstruction, params, retain_graph=True, allow_unused=True
        )
        if reconstruction.requires_grad
        else tuple(None for _ in params)
    )
    aa = sum(float(g.double().square().sum()) for g in supervised if g is not None)
    bb = sum(float(g.detach().double().square().sum()) for g in reco if g is not None)
    ab = sum(
        float(g.double().mul(h.detach().double()).sum())
        for g, h in zip(supervised, reco, strict=True)
        if g is not None and h is not None
    )
    return dict(
        supervised_l2=aa**0.5,
        reconstruction_unweighted_l2=bb**0.5,
        cosine=ab / (aa * bb) ** 0.5 if aa > 0 and bb > 0 else None,
    )


def validate_nsf_recovery(cfg, arm):
    if (
        arm.get("kind") != "nsf"
        or cfg.get("particles") != 256
        or cfg.get("integration_gate_particles") != 256
        or cfg.get("nsf_micro_batch_size") != 16
        or cfg.get("integration_chunk_rows") != 8
    ):
        raise ValueError("NSF recovery particle or microbatch policy differs")
    if (
        sha(cfg["integration_probe_manifest"])
        != cfg["integration_probe_manifest_sha256"]
    ):
        raise ValueError("NSF QMC probe manifest SHA differs")
    probe = load_json(cfg["integration_probe_manifest"])
    if (
        probe.get("status") != "remediation_nsf_qmc_probe_complete"
        or probe.get("run_id") != "remediation_nsf6_qmc_probe_v1"
        or probe.get("source_commit") != cfg["integration_probe_source_commit"]
        or probe.get("first_passing_particles") != 256
        or probe.get("training_performed") is not False
        or probe.get("library_revision") != cfg["library_revision"]
        or probe.get("parent_sha256") != cfg["checkpoint_sha256"]
    ):
        raise ValueError("NSF recovery requires the actual matching passing QMC probe")
    passing = [
        x
        for x in probe.get("integration_probe_outcomes", [])
        if x.get("particles") == 256
    ]
    if len(passing) != 1 or not passing[0].get("passed"):
        raise ValueError("NSF 256-particle gate did not pass")
    return probe


def heldout_corner_rows(ids, roles, common):
    """Reject missing or FIT corner identities before any model load/updates."""
    common = np.asarray(common, dtype="i8")
    rows = np.searchsorted(ids, common)
    if np.any(rows >= len(ids)):
        raise ValueError("Common corner identity missing from source")
    if not np.array_equal(ids[rows], common) or np.any(roles[rows] == "fit"):
        raise ValueError("Common model-blind corner IDs/heldout roles differ")
    return np.sort(rows)


def encoder_gradient_contract(supervised, reconstruction, parameters):
    """Independent objective gradients on the same encoder state and batch."""
    import torch

    params = tuple(p for p in parameters if p.requires_grad)
    if not params:
        return {"scope": "frozen encoder"}
    a = torch.autograd.grad(supervised, params, retain_graph=True, allow_unused=True)
    b = (
        torch.autograd.grad(
            reconstruction, params, retain_graph=True, allow_unused=True
        )
        if reconstruction.requires_grad
        else tuple(None for _ in params)
    )
    aa = sum(float(g.detach().double().square().sum()) for g in a if g is not None)
    bb = sum(float(g.detach().double().square().sum()) for g in b if g is not None)
    ab = sum(
        float(g.detach().double().mul(h.detach().double()).sum())
        for g, h in zip(a, b, strict=True)
        if g is not None and h is not None
    )
    return dict(
        supervised_l2=aa**0.5,
        reconstruction_unweighted_l2=bb**0.5,
        cosine=ab / (aa * bb) ** 0.5 if aa > 0 and bb > 0 else None,
    )


def enforce_encoder_gradient(encoder, head, kind, update, supervised_l2=None):
    """Require supervised encoder gradients after a zero-projection first step.

    A zero-initialized Gaussian/MDN output prevents context gradients on the
    first update. The projection must receive a finite nonzero gradient, and
    the next update must reach the encoder through the supervised objective.
    """
    import torch

    params = tuple(p for p in encoder.parameters() if p.requires_grad)
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise FloatingPointError("Finetuned encoder is disconnected")
    nonzero = any(torch.count_nonzero(grad).item() for grad in grads)
    if update >= 2 and supervised_l2 is not None and supervised_l2 <= 0:
        raise FloatingPointError("Finetuned encoder has no supervised gradient")
    if nonzero:
        return "nonzero"
    if update == 1 and kind in ("gaussian", "mdn"):
        projection = head.gaussian.net[-1].weight
        if (
            torch.count_nonzero(projection).item() == 0
            and projection.grad is not None
            and torch.isfinite(projection.grad).all()
            and torch.count_nonzero(projection.grad).item() > 0
        ):
            return "zero_initialized_output_first_step"
    raise FloatingPointError("Finetuned encoder has no nonzero gradient")


def build_head(kind, latent_dim, cfg):
    import torch

    from masked_stellar_autoencoder.training.remediation_joint6 import JointSixHead

    if kind != "huber":
        return JointSixHead(
            kind,
            latent_dim=latent_dim,
            width=cfg["head_width"],
            particles=cfg["particles"],
            seed=cfg["seed"],
            components=cfg["mixture_components"],
        )

    class HuberHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, cfg["head_width"]),
                torch.nn.GELU(),
                torch.nn.Linear(cfg["head_width"], 6),
            )

        def forward(self, context):
            return self.net(context)

        def criterion(self, context, target, sigma, mask):
            point = self(context)
            values = torch.nn.functional.huber_loss(
                point, target, reduction="none", delta=1.0
            )
            return values.masked_fill(~mask, 0).sum() / mask.sum()

    return HuberHead()


def run(cfg, arm, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "admitting",
        "run_id": arm["run_id"],
        "arm": arm,
        "training_performed": False,
        "completed_updates": 0,
        "declared_updates": cfg["updates"],
        "full_epoch_claim": False,
        "script_sha256": sha(__file__),
        "config_sha256": cfg.get("config_sha256"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "library_revision": cfg["library_revision"],
    }
    atomic(output / "manifest.json", manifest)
    try:
        source, audit, preflight = admission(cfg, arm)
        if cfg.get("nsf_micro_batch_size"):
            probe = validate_nsf_recovery(cfg, arm)
            if probe.get("prepared_sha256") != source.get("prepared_sha256"):
                raise ValueError("NSF probe and training source hashes differ")
        prepared = Path(source["prepared_path"])
        if sha(prepared) != source["prepared_sha256"]:
            raise ValueError("Actual remediation prepared SHA changed")
        # Local native/checkpoint execution is forbidden. No torch/model import
        # occurs until the CANFAR mount and accepted source gates are established.
        if not Path("/arc/projects/k-pop").is_dir():
            raise RuntimeError("Native execution is restricted to CANFAR mounts")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Native execution requires CANFAR CUDA")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Installed torchregress6bd differs")
        import run_multilabel_screen as native

        from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
        from masked_stellar_autoencoder.training import remediation_joint6

        if sha(remediation_joint6.__file__) != preflight.get("head_module_sha256"):
            raise ValueError(
                "Joint head source differs from actual successful preflight"
            )
        for key in ("checkpoint", "reference_checkpoint"):
            if sha(cfg[key]) != cfg[key + "_sha256"]:
                raise ValueError("Immutable parent checkpoint SHA differs")
        reference = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        weights = torch_load_trusted(
            cfg["checkpoint"], map_location="cpu", weights_only=False
        )
        if (
            weights.get("presentations") != 4500000000
            or weights.get("completed_updates") != 137340
            or weights.get("parent_sha256") != cfg["reference_checkpoint_sha256"]
        ):
            raise ValueError("Large c544 training lineage/counters differ")
        disk = dict(reference, model_state_dict=weights["model_state_dict"])
        del weights
        signature = reference["run_signature"]
        device = torch.device("cuda")
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
        with h5py.File(prepared, "r") as data:
            if (
                data.attrs.get("schema") != "remediation-sparse-six-target-source-v1"
                or data.attrs.get("source_commit") != source["source_commit"]
            ):
                raise ValueError("Prepared schema/source manifest binding differs")
            names = data["feature_names"][:].astype("U").tolist()
            if names != signature["feature_cols"] or data["target_names"][:].astype(
                "U"
            ).tolist() != list(NAMES):
                raise ValueError("Parent feature/target order differs")
            for key in ("scaler_center", "scaler_scale"):
                np.testing.assert_array_equal(data[key][:], signature[key])
            ids, roles = data["source_id"][:], data["role"][:].astype("U")
            if (
                ids.dtype.kind not in "iu"
                or np.any(ids <= 0)
                or np.any(np.diff(ids) <= 0)
            ):
                raise ValueError(
                    "Source identities must be sorted distinct positive integers"
                )
            values, errors = data["value"][:], data["error"][:]
            quality = data["observed"][:].astype(bool) & data["quality"][:].astype(bool)
            y, sigma, mask, center, scale, error_known = normalize_targets(
                values, quality, errors, data["error_observed"][:], roles == "fit"
            )
            label_sources = data["label_source"][:].astype("U64")
            pools = balanced_pools(
                roles == "fit",
                mask,
                label_sources,
                np.where(mask[:, 1], values[:, 1], np.nan),
            )
            tune = fixed_role_rows(ids, roles, mask, "tune", cfg["tune_per_label"])
            devtest = fixed_role_rows(
                ids, roles, mask, "devtest", cfg["evaluation_per_label"]
            )
            corner_rows = heldout_corner_rows(ids, roles, cfg["corner_source_ids"])
            model = native.probe.load_model(model_cfg, disk, device)
            for parameter in model.parameters():
                parameter.requires_grad_(arm["encoder"] == "finetune")
            if cfg.get("nsf_micro_batch_size"):
                validate_microbatch_encoder(
                    model.encoder, finetune=arm["encoder"] == "finetune"
                )
            head = build_head(
                arm["kind"], model_cfg["model"]["layer_dims"][-1], cfg
            ).to(device)
            parameters = [{"params": head.parameters(), "lr": cfg["head_lr"]}]
            if arm["encoder"] == "finetune":
                parameters.append(
                    {"params": model.parameters(), "lr": cfg["encoder_lr"]}
                )
            optimizer = torch.optim.Adam(parameters)
            profile = arm.get("input_profile", "phot_xp")
            manifest.update(
                status="running",
                prepared_sha256=source["prepared_sha256"],
                source_manifest_sha256=sha(cfg["source_manifest"]),
                input_audit_sha256=sha(cfg["input_audit_manifest"]),
                joint_preflight_sha256=sha(cfg["joint_preflight_manifest"]),
                parent_sha256=cfg["checkpoint_sha256"],
                fit_per_target=mask[roles == "fit"].sum(0),
                tune_panel_rows=len(tune),
                devtest_panel_rows=len(devtest),
                sampler="round-robin target; uniform nonempty source strata, logg additionally uniform gravity bins <1/1-2/2-3/3-4/4-4.5/>=4.5; every FIT identity eligible",
                unknown_error_scope="Unknown quoted errors are point-teacher targets; sigma=0 is a computational convention, not a known zero error",
                selection="first minimum macro-six native conditional-median absolute error / FIT native standard deviation; common model-blind fixed TUNE identity panel",
                natural_xp_at_inference=True,
                pit_scope="512 whole-vector empirical marginal intrinsic PIT; observed PIT only when quoted target error is known, not an exact CDF; baseline unavailable",
                covariance_scope="Actual whole-vector conditional sample covariance; no manufactured correlations and no physical six-dimensional truth claim",
                supervised_xp_dropout=0.0,
            )
            atomic(output / "manifest.json", manifest)

            def tensor(value, dtype=torch.float32):
                return torch.as_tensor(value, dtype=dtype, device=device)

            def latent(rows):
                x = tensor(profile_batch(data, rows, profile, source))
                if arm["encoder"] == "frozen":
                    with torch.no_grad():
                        return model.encoder(x)
                return model.encoder(x)

            def nsf_check(stage):
                balanced = fixed_role_rows(
                    ids, roles, mask, "tune", max(1, cfg["integration_gate_rows"] // 6)
                )
                remaining = np.setdiff1d(tune, balanced)
                rows = np.sort(
                    np.concatenate(
                        [
                            balanced,
                            remaining[
                                : max(0, cfg["integration_gate_rows"] - len(balanced))
                            ],
                        ]
                    )
                )
                model.eval()
                head.eval()
                with torch.no_grad():
                    inputs = (
                        latent(rows),
                        tensor(y[rows]),
                        tensor(sigma[rows]),
                        tensor(mask[rows], torch.bool),
                    )
                    if arm.get("integration_probe") or cfg.get(
                        "integration_chunk_rows"
                    ):
                        diag = diagnostics_in_row_chunks(
                            head,
                            *inputs,
                            particles=cfg["integration_gate_particles"],
                            chunk_rows=cfg.get(
                                "integration_chunk_rows",
                                cfg.get("integration_probe_chunk_rows"),
                            ),
                        )
                    else:
                        diag = head.diagnostics(
                            *inputs, particles=cfg["integration_gate_particles"]
                        )
                diagnostic_arrays = {
                    name: value.detach().cpu().numpy()
                    for name, value in diag.items()
                    if hasattr(value, "detach")
                }
                np.savez_compressed(
                    output / ("integration-" + stage + ".npz"),
                    source_id=ids[rows],
                    observed=mask[rows],
                    standardized_target=y[rows],
                    standardized_known_error=sigma[rows],
                    error_observed=error_known[rows],
                    **diagnostic_arrays,
                )
                proof = integration_gate(diag, cfg["integration_gate"])
                proof.update(
                    rows=len(rows),
                    source_id_sha256=hashlib.sha256(
                        np.asarray(ids[rows], dtype="<i8").tobytes()
                    ).hexdigest(),
                    particles=cfg["integration_gate_particles"],
                    stage=stage,
                )
                atomic(output / ("integration-" + stage + ".json"), proof)
                return proof

            if arm.get("integration_probe"):
                validate_integration_probe(cfg, arm)
                outcomes = []
                for count in cfg["integration_probe_particles"]:
                    cfg["integration_gate_particles"] = count
                    try:
                        proof = nsf_check(f"probe-{count}")
                        outcomes.append(
                            {"particles": count, "passed": True, "metrics": proof}
                        )
                    except ValueError as exc:
                        if not str(exc).startswith(
                            "Real sparse/noisy NSF integration gate failed:"
                        ):
                            raise
                        outcomes.append(
                            {"particles": count, "passed": False, "error": str(exc)}
                        )
                    manifest["integration_probe_outcomes"] = outcomes
                    atomic(output / "manifest.json", manifest)
                passed = [item["particles"] for item in outcomes if item["passed"]]
                manifest.update(
                    status="remediation_nsf_qmc_probe_complete"
                    if passed
                    else "remediation_nsf_qmc_probe_failed",
                    training_performed=False,
                    completed_updates=0,
                    first_passing_particles=min(passed) if passed else None,
                    scope="Inference-only fixed-panel estimator diagnostics; no optimizer step, no exact-marginal claim",
                    artifacts={
                        path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                        for path in output.glob("integration-probe-*")
                        if path.is_file()
                    },
                )
                atomic(output / "manifest.json", manifest)
                return
            if arm["kind"] == "nsf":
                nsf_check("initial")
            cache = None
            if arm["reconstruction_weight"]:
                from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
                    _cache_maps,
                )

                cache, _, _, cache_manifest = _cache_maps(
                    cfg["reconstruction_cache"],
                    names,
                    data["scaler_center"][:],
                    data["scaler_scale"][:],
                    cfg["reconstruction_request_sha256"],
                    cfg["reconstruction_training_rows"],
                )
                manifest["reconstruction_cache_request_sha256"] = cache_manifest[
                    "request_sha256"
                ]
            xp = [j for j, n in enumerate(names) if n.startswith(("bp_", "rp_"))]
            allowed = np.array(
                [
                    n in {"W1", "W2", "G", "BP", "RP"} or j in xp
                    for j, n in enumerate(names)
                ]
            )
            width = len(model_cfg["data"]["recon_cols"])
            if names[:width] != model_cfg["data"]["recon_cols"]:
                raise ValueError("Decoder reconstruction feature order differs")
            mask_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 1701)
            native_scale = np.array(
                [values[(roles == "fit") & mask[:, j], j].std() for j in range(6)]
            )
            if not np.isfinite(native_scale).all() or not (native_scale > 0).all():
                raise ValueError("Invalid native FIT standard deviations")
            manifest["sampler_strata"] = [
                {
                    "target": NAMES[j],
                    "source": source_name,
                    "gravity_bin": band,
                    "eligible_rows": len(pool),
                }
                for j, strata in enumerate(pools)
                for source_name, band, pool in strata
            ]
            counts = np.zeros(6, dtype="i8")
            source_counts = {}
            seen = np.zeros(mask.shape, dtype=bool)
            finite_gradients = 0
            sequence = hashlib.sha256()
            best = float("inf")
            selected = None
            atomic(output / "manifest.json", manifest)

            def evaluate(rows, path=None):
                model.eval()
                head.eval()
                point_blocks = []
                pit_blocks = []
                observed_pit_blocks = []
                sum_error = np.zeros(6)
                n_error = np.zeros(6, dtype="i8")
                for start in range(0, len(rows), cfg["evaluation_batch"]):
                    rr = rows[start : start + cfg["evaluation_batch"]]
                    with torch.no_grad():
                        z = latent(rr)
                        if arm["kind"] == "huber":
                            pred = head(z).cpu().numpy()
                            pit = np.full(pred.shape, np.nan)
                            observed_pit = np.full(pred.shape, np.nan)
                        else:
                            generator = torch.Generator(device=device).manual_seed(
                                cfg["seed"] + int(start)
                            )
                            draw = (
                                head.sample(z, cfg["evaluation_draws"], generator)
                                .cpu()
                                .numpy()
                            )
                            pred = np.median(draw, axis=0)
                            pit = np.mean(draw <= y[rr][None], axis=0)
                            error_rng = np.random.default_rng(
                                cfg["seed"] + 701 + int(start)
                            )
                            noisy = (
                                draw
                                + error_rng.standard_normal(draw.shape)
                                * sigma[rr][None]
                            )
                            observed_pit = np.where(
                                error_known[rr],
                                np.mean(noisy <= y[rr][None], axis=0),
                                np.nan,
                            )
                    native_pred = pred * scale + center
                    native_pred[:, 4:] = np.exp(native_pred[:, 4:])
                    if not np.isfinite(native_pred).all():
                        raise FloatingPointError(
                            "Nonfinite native predictions during selection"
                        )
                    delta = np.where(
                        mask[rr], np.abs(native_pred - values[rr]) / native_scale, 0
                    )
                    sum_error += delta.sum(0)
                    n_error += mask[rr].sum(0)
                    point_blocks.append(pred)
                    pit_blocks.append(pit)
                    observed_pit_blocks.append(observed_pit)
                if np.any(n_error == 0):
                    raise ValueError("Missing TUNE/evaluation label")
                score = float(np.mean(sum_error / n_error))
                pred = np.concatenate(point_blocks)
                if path:
                    native_point = pred * scale + center
                    native_point[:, 4:] = np.exp(native_point[:, 4:])
                    if not np.isfinite(native_point).all():
                        raise FloatingPointError("Nonfinite native point predictions")
                    np.savez_compressed(
                        path,
                        source_id=ids[rows],
                        row=rows,
                        value=values[rows],
                        observed=mask[rows],
                        error=errors[rows],
                        error_observed=error_known[rows],
                        label_source=label_sources[rows],
                        standardized_median=pred,
                        native_median=native_point,
                        empirical_intrinsic_pit=np.where(
                            mask[rows], np.concatenate(pit_blocks), np.nan
                        ),
                        empirical_observed_pit=np.where(
                            mask[rows], np.concatenate(observed_pit_blocks), np.nan
                        ),
                        role=roles[rows],
                        real_xp_present=data["real_xp_present"][:][rows],
                        real_no_xp=data["real_no_xp"][:][rows],
                        measured_parallax_observed=data["measured_parallax/observed"][
                            :
                        ][rows],
                    )
                return score, (sum_error / n_error).tolist()

            for update in range(1, cfg["updates"] + 1):
                torch.manual_seed(cfg["seed"] + update)
                torch.cuda.manual_seed_all(cfg["seed"] + update)
                model.train(arm["encoder"] == "finetune")
                head.train()
                optimizer.zero_grad(set_to_none=True)
                rr = draw_rows(
                    rng, pools, cfg["batch_size"], target_column=(update - 1) % 6
                )
                sequence.update(np.asarray(ids[rr], dtype="<i8").tobytes())
                chunked = bool(cfg.get("nsf_micro_batch_size"))
                if chunked:
                    supervised_loss, components = backward_supervised_chunks(
                        head,
                        latent,
                        y,
                        sigma,
                        mask,
                        rr,
                        chunk_rows=cfg["nsf_micro_batch_size"],
                        tensor=tensor,
                    )
                    loss = supervised_loss
                else:
                    loss = head.criterion(
                        latent(rr),
                        tensor(y[rr]),
                        tensor(sigma[rr]),
                        tensor(mask[rr], torch.bool),
                    )
                    supervised_loss = loss
                    components = {
                        key: float(value.detach())
                        if hasattr(value, "detach")
                        else value
                        for key, value in getattr(
                            head, "last_loss_components", {}
                        ).items()
                    }
                reco = loss.new_zeros(())
                if cache is not None:
                    start = ((update - 1) * cfg["reconstruction_batch"]) % len(cache)
                    positions = np.arange(
                        start, min(start + cfg["reconstruction_batch"], len(cache))
                    )
                    raw = np.asarray(cache[positions]).copy()
                    raw[(~np.isfinite(raw)) | (raw == -9999)] = np.nan
                    raw[:, ~allowed] = np.nan
                    target = tensor(raw)
                    hidden, hidden_mask, finite = native.pretrain.training_mask(
                        target, xp, 0.5, 0.2, mask_rng
                    )
                    prediction, _ = model(hidden)
                    score = (hidden_mask & finite)[:, :width]
                    if not score.any():
                        raise ValueError("No reconstruction masked targets")
                    reco = native.pretrain.experiment_objective(
                        prediction,
                        target[:, :width],
                        score,
                        xp,
                        {"loss": "mae"},
                        score.sum(),
                    )
                    loss = loss + arm["reconstruction_weight"] * reco
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite pilot objective")
                gradient_contract = None
                if update in (1, 2) or update % cfg["tune_every"] == 0:
                    gradient_contract = dict(
                        update=update,
                        reconstruction_weight=arm["reconstruction_weight"],
                        **(
                            accumulated_encoder_gradient_contract(
                                reco, model.encoder.parameters()
                            )
                            if chunked
                            else encoder_gradient_contract(
                                supervised_loss, reco, model.encoder.parameters()
                            )
                        ),
                    )
                    manifest["encoder_gradient_contract"] = gradient_contract
                if chunked:
                    if reco.requires_grad:
                        (arm["reconstruction_weight"] * reco).backward()
                else:
                    loss.backward()
                gradients = [
                    p.grad
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Missing/nonfinite pilot gradients")
                if not any(
                    p.grad is not None and p.grad.ne(0).any() for p in head.parameters()
                ):
                    raise FloatingPointError("Fresh head has no nonzero gradient")
                if arm["encoder"] == "finetune":
                    gradient_status = enforce_encoder_gradient(
                        model.encoder,
                        head,
                        arm["kind"],
                        update,
                        None
                        if gradient_contract is None
                        else gradient_contract["supervised_l2"],
                    )
                    if update <= 2:
                        manifest.setdefault("encoder_gradient_startup", []).append(
                            {
                                "update": update,
                                "status": gradient_status,
                                **gradient_contract,
                            }
                        )
                finite_gradients += 1
                optimizer.step()
                counts += mask[rr].sum(0)
                seen[rr] |= mask[rr]
                for j in range(6):
                    for label_source, number in zip(
                        *np.unique(
                            label_sources[rr, j][mask[rr, j]], return_counts=True
                        )
                    ):
                        key = NAMES[j] + ":" + label_source
                        source_counts[key] = source_counts.get(key, 0) + int(number)
                manifest.update(
                    completed_updates=update,
                    training_performed=True,
                    per_target_presentations=counts,
                    per_target_unique_identities=seen.sum(0),
                    supervised_loss_components=components,
                    supervised_objective=float(supervised_loss.detach()),
                    encoder_dropout_seed=cfg["seed"] + update,
                    per_target_source_presentations=source_counts,
                    finite_gradient_updates=finite_gradients,
                    last_loss=float(loss.detach()),
                    last_reconstruction_mae=float(reco.detach()),
                )
                if update % cfg["tune_every"] == 0 or update == cfg["updates"]:
                    score, per_label = evaluate(tune)
                    record = {
                        "update": update,
                        "tune_macro_normalized_median_mae": score,
                        "per_label": per_label,
                        "loss": float(loss.detach()),
                        "reconstruction_mae": float(reco.detach()),
                    }
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    if score < best:
                        best = score
                        selected = update
                        state = {
                            "run_id": arm["run_id"],
                            "arm": arm,
                            "model_state_dict": model.state_dict(),
                            "head_state_dict": head.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "update": update,
                            "center": center,
                            "scale": scale,
                            "prepared_sha256": source["prepared_sha256"],
                            "parent_sha256": cfg["checkpoint_sha256"],
                            "tune_score": best,
                        }
                        torch.save(state, output / "selected-checkpoint.pth")
                    manifest.update(selected_update=selected, best_tune_score=best)
                if update % 25 == 0 or update == cfg["updates"]:
                    atomic(output / "manifest.json", manifest)
            state = torch_load_trusted(
                output / "selected-checkpoint.pth",
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(state["model_state_dict"], strict=True)
            head.load_state_dict(state["head_state_dict"], strict=True)
            if arm["kind"] == "nsf":
                manifest["selected_nsf_integration"] = nsf_check("selected")
            tune_score, tune_per_label = evaluate(tune, output / "tune-predictions.npz")
            dev_score, dev_per_label = evaluate(
                devtest, output / "development-predictions.npz"
            )
            if arm["kind"] != "huber":
                model.eval()
                head.eval()
                with torch.no_grad():
                    z = latent(corner_rows)
                    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 9999)
                    standardized_draws = (
                        head.sample(z, cfg["corner_draws"], gen).cpu().numpy()
                    )
                native_draws = standardized_draws * scale + center
                native_draws[:, :, 4:] = np.exp(native_draws[:, :, 4:])
                if not np.isfinite(native_draws).all():
                    raise FloatingPointError(
                        "Native coupled draws overflowed; no clipping"
                    )
                covariance = np.asarray(
                    [
                        np.cov(standardized_draws[:, j, :], rowvar=False)
                        for j in range(len(corner_rows))
                    ]
                )
                native_covariance = np.asarray(
                    [
                        np.cov(native_draws[:, j, :], rowvar=False)
                        for j in range(len(corner_rows))
                    ]
                )
                np.savez_compressed(
                    output / "whole-vector-corners.npz",
                    source_id=ids[corner_rows],
                    row=corner_rows,
                    standardized_draws=standardized_draws,
                    native_draws=native_draws,
                    standardized_covariance=covariance,
                    native_covariance=native_covariance,
                    reference=values[corner_rows],
                    observed=mask[corner_rows],
                )
            np.savez_compressed(
                output / "normalization.npz",
                center=center,
                scale=scale,
                native_fit_scale=native_scale,
                feature_names=np.asarray(names),
                target_names=np.asarray(NAMES),
            )
            manifest.update(
                status="remediation_pilot_complete",
                selected_update=selected,
                selected_checkpoint_sha256=sha(output / "selected-checkpoint.pth"),
                sample_sequence_sha256=sequence.hexdigest(),
                tune_score=tune_score,
                tune_per_label=tune_per_label,
                development_score=dev_score,
                development_per_label=dev_per_label,
                development_scope="Fixed hashed pilot panel from previously consumed development roles; not an untouched test or promotion",
                reconstruction_presentations=cfg["updates"]
                * cfg["reconstruction_batch"]
                if cache is not None
                else 0,
                corner_scope="Whole coupled six-target vectors; posterior-reference age/distance scales are not physical truth"
                if arm["kind"] != "huber"
                else "Point baseline has no PDF/PIT or joint sampler",
            )
            manifest["artifacts"] = {
                p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            }
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic(output / "manifest.json", manifest)
        raise
    atomic(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["config_sha256"] = sha(args.config)
    arms = [a for a in cfg["arms"] if a["run_id"] == args.run_id]
    if len(arms) != 1:
        raise ValueError("Run ID must identify one immutable declared pilot")
    run(cfg, arms[0], args.output)


if __name__ == "__main__":
    main()
