"""CANFAR-only source-expanded scoring of immutable completed prediction heads.

No optimizer is constructed. Original benchmark predictions are reused when
inputs and references are unchanged. New identities and newly available labels
are evaluated in a distinct source-bound namespace, with immutable batch caches.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import run_final_six_recipes as shared
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.distribution_diagnostics import (
    marginal_cdf,
    native_coordinates,
    predictive_draws,
)
from masked_stellar_autoencoder.training.final_six_recipes import (
    NAMES,
    standardized_arrays,
)
from masked_stellar_autoencoder.training.five_wave_heads import (
    QUANTILES,
    build_head,
    native_quantiles,
)
from masked_stellar_autoencoder.training.input_measurement import xp_indices

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015"
POINT_KEYS = {"point_native", "quantiles_native", "mean_native", "variance_native"}
METADATA_KEYS = {
    "source_id",
    "target_native",
    "observed",
    "error",
    "error_observed",
    "label_source",
    "quantile_levels",
    "evaluation_mode",
    "real_xp_present",
    "real_no_xp",
    "input_has_xp",
    "xp_status",
    "official_xp_unknown",
    "andrae_prediction_native",
    "andrae_matched",
    "andrae_membership_known",
    "andrae_training_membership_known",
    "andrae_in_training",
    "ancestor_feh_supervised_fit_exposed",
    "ancestor_150m_reconstruction_exposed",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def decode(values):
    values = np.asarray(values)
    return values.astype(str) if values.dtype.kind in "SO" else values


class Prepared:
    """Small-array metadata and arbitrary-row HDF5 reads without full feature copies."""

    def __init__(self, path):
        self.file = (
            h5py.File(path, "r")
            if Path(path).suffix in {".h5", ".hdf5"}
            else np.load(path, allow_pickle=False)
        )

    def __contains__(self, key):
        return key in self.file

    def array(self, key):
        return decode(self.file[key][...])

    def take(self, key, rows):
        rows = np.asarray(rows, dtype="i8")
        if not len(rows):
            return decode(self.file[key][0:0])
        unique, inverse = np.unique(rows, return_inverse=True)
        return decode(self.file[key][unique])[inverse]

    def close(self):
        self.file.close()


def matched_indices(wanted, available):
    wanted, available = np.asarray(wanted), np.asarray(available)
    if wanted.dtype.kind not in "iu" or available.dtype.kind not in "iu":
        raise ValueError("Exact integer source identities required")
    if len(np.unique(available)) != len(available):
        raise ValueError("Original benchmark source IDs duplicate")
    order = np.argsort(available)
    positions = np.searchsorted(available[order], wanted)
    valid = positions < len(available)
    valid[valid] &= available[order[positions[valid]]] == wanted[valid]
    result = np.full(len(wanted), -1, dtype="i8")
    result[valid] = order[positions[valid]]
    return result


def validate_producer(manifest, state, checkpoint_sha, cfg, arm):
    """Producer completion is required; the last training state is not the selected state."""
    if (
        manifest.get("status") != "complete"
        or manifest.get("source_sha") != cfg["producer_source_sha"]
    ):
        raise ValueError("Actual immutable producer must be complete")
    if (
        manifest.get("completed_presentations") != 450_000_000
        or manifest.get("completed_updates") != 13734
    ):
        raise ValueError("Actual producer training budget differs")
    if manifest.get("full_state_reload_equal") is not True:
        raise ValueError("Producer full-state reload gate did not pass")
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Selected checkpoint SHA differs")
    receipt = manifest.get("selected_checkpoint", {})
    selected = manifest.get("selected_model_counters", {})
    # checkpoint.pth is a reserialization of the selected periodic dictionary.
    # Its bytes/SHA may differ despite exact tensor/RNG/optimizer equality.
    expected_selected = {
        "presentations": state.get("completed_presentations"),
        "updates": state.get("completed_updates"),
    }
    if (
        selected != expected_selected
        or {key: receipt.get(key) for key in expected_selected} != expected_selected
        or receipt.get("all_saved_state_values_reload_equal") is not True
        or len(receipt.get("sha256", "")) != 64
    ):
        raise ValueError("Selected state restoration or counters differ")
    if (
        state.get("source_sha") != cfg["producer_source_sha"]
        or state.get("parent_sha256") != cfg["large_parent_sha256"]
    ):
        raise ValueError("Checkpoint source or large-model ancestry differs")
    if state.get("arm", {}).get("run_id") != arm["producer_run_id"]:
        raise ValueError("Selected state belongs to another head")
    if state.get("input_sha256") != cfg["original_prepared_sha256"]:
        raise ValueError("Original training input identity differs")
    if manifest.get("policy", {}).get("library_revision") != cfg["library_revision"]:
        raise ValueError("Producer library revision differs")


def new_inference_rows(original_index, expanded_reference, head_kind):
    """Quantile baseline backfill uses already saved quantiles, not a new forward pass."""
    original_index, extra = (
        np.asarray(original_index),
        np.asarray(expanded_reference, bool),
    )
    if extra.shape != (len(original_index), 6):
        raise ValueError("Six-coordinate reference additions required")
    return (original_index < 0) | (extra.any(1) & (head_kind != "quantile3"))


def combine_cached(original, original_index, computed, observed, original_observed):
    """Reuse actual old points and existing marginal scores; joint new blocks stay new."""
    index = np.asarray(original_index)
    keep = index >= 0
    result = {key: np.array(value, copy=True) for key, value in computed.items()}
    for key, old in original.items():
        if (
            key in METADATA_KEYS
            or not (
                key in POINT_KEYS or key.startswith(("intrinsic_", "observed_", "pit_"))
            )
            or not isinstance(old, np.ndarray)
            or not old.ndim
        ):
            continue
        if len(old) != len(original["source_id"]):
            continue
        if key not in result:
            shape = (len(index), *old.shape[1:])
            result[key] = np.full(shape, np.nan, dtype="f8")
        if key in POINT_KEYS:
            result[key][keep] = old[index[keep]]
        elif old.shape[1:] == (6,):
            values = result[key][keep]
            mask = original_observed[keep] & observed[keep]
            values[mask] = old[index[keep]][mask]
            result[key][keep] = values
        elif old.shape[1:] == (3, 6):
            values = result[key][keep]
            repeated = np.broadcast_to(
                (original_observed[keep] & observed[keep])[:, None], values.shape
            )
            values[repeated] = old[index[keep]][repeated]
            result[key][keep] = values
        elif old.shape[1:] == ():
            # All joint scores with unchanged observed blocks are reusable. A newly
            # complete block may have been NaN originally and must retain new scores.
            values = old[index[keep]]
            finite = np.isfinite(values)
            slots = np.flatnonzero(keep)[finite]
            result[key][slots] = values[finite]
    return result


def report_xp_status(raw):
    """Keep publication-unknown measurements separate from official no-XP."""
    raw = np.asarray(raw).astype(str)
    supported = {
        "complete_xp",
        "official_no_xp",
        "partial_xp",
        "unknown",
        "official_xp_unknown",
    }
    return np.where(np.isin(raw, list(supported)), raw, "unknown")


def source_reference_masks(data, rows):
    """An object exposed in another target cannot enter clean new primary scores."""
    mask = (
        data.take("observed", rows).astype(bool)
        & data.take("quality", rows).astype(bool)
        & data.take("primary_score_eligible", rows).astype(bool)
    )
    diagnostic = data.take("diagnostic_eligible", rows).astype(bool)
    strict = data.take("strict_unexposed_star", rows).astype(bool)
    exposed = data.take("ancestor_feh_supervised_fit_exposed", rows).astype(bool)
    if strict.shape != (len(rows),) or not np.array_equal(strict, ~exposed):
        raise ValueError("Strict object holdout contradicts known target ancestry")
    raw = data.take("xp_status", rows)
    input_diagnostic = (
        data.take("input_diagnostic_only", rows).astype(bool)
        if "input_diagnostic_only" in data
        else raw == "photometry_only_publication_unknown"
    )
    diagnostic[input_diagnostic] |= mask[input_diagnostic]
    mask[input_diagnostic] = False
    mask &= strict[:, None]
    diagnostic &= strict[:, None]
    return mask, diagnostic, raw, input_diagnostic


def validate_batch(part, mask):
    if not np.isfinite(part["point_native"]).all():
        raise FloatingPointError("Nonfinite point predictions; failed batch preserved")
    q = part["quantiles_native"]
    if (
        not np.isfinite(q).all()
        or (np.diff(q, axis=1) < 0).any()
        or (q[..., 4:] <= 0).any()
    ):
        raise FloatingPointError("Invalid predictive quantiles; failed batch preserved")
    for key in ("intrinsic_pit", "observed_pit"):
        if key in part and (
            not np.isfinite(part[key][mask]).all()
            or ((part[key][mask] < 0) | (part[key][mask] > 1)).any()
        ):
            raise FloatingPointError("Invalid CDF/PIT; failed batch preserved")


class ArrayStore:
    """Write all-row results to disk-backed arrays, not a list of millions of batches."""

    def __init__(self, path, size):
        self.path, self.size, self.arrays = path, size, {}
        path.mkdir(exist_ok=True)

    def put(self, rows, values):
        for key, value in values.items():
            value = np.asarray(value)
            if key not in self.arrays:
                path = self.path / (key + ".npy")
                if path.exists():
                    raise FileExistsError("Terminal/result array already exists")
                self.arrays[key] = np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=value.dtype,
                    shape=(self.size, *value.shape[1:]),
                )
            self.arrays[key][rows] = value

    def finish(self):
        for value in self.arrays.values():
            value.flush()
        return {
            path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
            for path in self.path.glob("*.npy")
        }


def score_new(
    model,
    head,
    inputs,
    values,
    errors,
    known,
    mask,
    center,
    scale,
    device,
    seed,
    capture=None,
    external_age=None,
):
    """Actual normalized family, FP64 encoder/head; no density proxy or optimizer."""
    y, sigma, observed, jacobian = standardized_arrays(
        values, mask, errors, known & mask, center, scale
    )
    ct, st = torch.tensor(center, device=device), torch.tensor(scale, device=device)
    with torch.no_grad():
        latent = model.encoder(torch.tensor(inputs, device=device, dtype=torch.float64))
        part = {
            "point_native": native_quantiles(head, latent, ct, st)[:, 1].cpu().numpy(),
            "quantiles_native": native_quantiles(head, latent, ct, st).cpu().numpy(),
        }
        draws = None
        contract = None
        if head.kind != "quantile3":
            from run_final_six_distribution_metrics import calculate_batch

            mean, variance = head.native_moments(latent, ct, st)
            part.update(
                mean_native=mean.cpu().numpy(), variance_native=variance.cpu().numpy()
            )
            yt, et, mt = (
                torch.tensor(y, device=device),
                torch.tensor(sigma, device=device),
                torch.tensor(observed, device=device),
            )
            intrinsic, convolved = [], []
            for c in range(6):
                one = torch.zeros_like(mt)
                one[:, c] = True
                intrinsic.append(
                    head.row_log_prob(latent, yt, et, one, observed=False).cpu().numpy()
                    - jacobian[:, c]
                )
                convolved.append(
                    head.row_log_prob(latent, yt, et, one).cpu().numpy()
                    - jacobian[:, c]
                )
            part["intrinsic_log_prob_native"] = np.where(
                mask, np.stack(intrinsic, -1), np.nan
            )
            part["observed_log_prob_native"] = np.where(
                mask, np.stack(convolved, -1), np.nan
            )
            if capture is not None:
                capture(part)
            diagnostics, draws = calculate_batch(
                head,
                latent,
                yt,
                et,
                torch.tensor(values, device=device),
                mt,
                ct,
                st,
                seed,
            )
            part.update(
                {key: value.cpu().numpy() for key, value in diagnostics.items()}
            )
            part["observed_quantiles_native"] = np.stack(
                [part[f"observed_quantile_{q:g}_native"] for q in QUANTILES], 1
            )
            part["pit_observed_valid"] = mask & known
            if external_age is not None:
                external_valid = np.isfinite(external_age) & (external_age > 0)
                if external_valid.any():
                    ey = yt.clone()
                    ey[:, 4] = torch.tensor(
                        (np.log(np.where(external_valid, external_age, 1)) - center[4])
                        / scale[4],
                        device=device,
                    )
                    part["external_age_intrinsic_pit"] = (
                        marginal_cdf(head, latent, ey)[:, 4].cpu().numpy()
                    )
                    from torchregress.metrics.distribution import crps_from_samples

                    esafe = torch.tensor(
                        np.where(external_valid, external_age, 1), device=device
                    )
                    crps = [
                        crps_from_samples(rep[:, :, 4], esafe, reduction="none")
                        .cpu()
                        .numpy()
                        for rep in draws["intrinsic"]
                    ]
                    part["external_age_intrinsic_crps_gyr"] = (crps[0] + crps[1]) / 2
                    for key in [
                        "external_age_intrinsic_pit",
                        "external_age_intrinsic_crps_gyr",
                    ]:
                        part[key] = np.where(external_valid, part[key], np.nan)
            # Pilot assertions run only after arrays are captured by the caller.
            contract = dict(
                latent=latent,
                y=yt,
                sigma=et,
                mask=mt,
                native_y=torch.tensor(values, device=device),
                center=ct,
                scale=st,
            )
    return part, draws, contract


def make_corners(model, head, data, center, scale, device, seed, output):
    """Source-selected shared stars/axes and actual predictive samples only."""
    if head.kind == "quantile3" or "corner_rows" not in data:
        return None
    rows = data.array("corner_rows").astype("i8")
    if not len(rows):
        return None
    if (
        rows.ndim != 1
        or len(np.unique(rows)) != len(rows)
        or (rows < 0).any()
        or (rows >= len(data.file["source_id"])).any()
    ):
        raise ValueError("Invalid source-selected corner row identities")
    limits = data.array("corner_plot_limits")
    if (
        limits.shape != (len(rows), 6, 2)
        or not np.isfinite(limits).all()
        or not (limits[..., 1] > limits[..., 0]).all()
    ):
        raise ValueError("Invalid fixed corner axes")
    with torch.no_grad():
        latent = model.encoder(
            torch.tensor(data.take("input", rows), device=device, dtype=torch.float64)
        )
        generator = torch.Generator(device=device).manual_seed(seed + 317029)
        draws = predictive_draws(head, latent, 8192, generator)
        samples = (
            native_coordinates(
                draws,
                torch.tensor(center, device=device),
                torch.tensor(scale, device=device),
            )
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
        )
    plotted = samples.copy()
    plotted[..., 4:] = np.log(plotted[..., 4:])
    result = dict(
        source_id=data.take("source_id", rows),
        samples_native=samples,
        samples_plot_coordinates=plotted,
        target_native=data.take("value", rows),
        observed=data.take("quality", rows).astype(bool)
        & data.take("observed", rows).astype(bool),
        error=data.take("error", rows),
        error_observed=data.take("error_observed", rows).astype(bool),
        label_source=data.take("label_source", rows),
        plot_limits=limits,
    )
    np.savez_compressed(output / "representative-corner-samples.npz", **result)
    atomic(
        output / "representative-corner-samples.sha256.json",
        dict(
            sha256=sha(output / "representative-corner-samples.npz"),
            draw_count=8192,
            source_selected=True,
        ),
    )
    return result


def external_report(predictions, output):
    """APOKASC is an external age scale, never pooled with the teacher posterior."""
    output.mkdir(exist_ok=True)
    if "external_age_observed" not in predictions:
        atomic(
            output / "summary.json",
            dict(status="not_available", pooled_with_teacher=False),
        )
        return
    mask = np.asarray(predictions["external_age_observed"], bool)
    target = np.asarray(predictions["external_age_value"])
    mask &= np.isfinite(target) & (target > 0)
    if not mask.any():
        atomic(
            output / "summary.json",
            dict(status="no_eligible_references", pooled_with_teacher=False),
        )
        return
    predicted = np.asarray(predictions["point_native"])[mask, 4]
    target = target[mask]
    residual = predicted - target
    result = dict(
        status="source_scale_point_comparison",
        count=len(target),
        mae_gyr=float(np.abs(residual).mean()),
        bias_gyr=float(residual.mean()),
        rmse_gyr=float(np.sqrt(np.mean(residual**2))),
        pooled_with_teacher=False,
        scope="APOKASC3 seismic/model-dependent reference age, not physical truth or a StarHorse posterior target",
    )
    if "external_age_intrinsic_pit" in predictions:
        pit = np.asarray(predictions["external_age_intrinsic_pit"])[mask]
        if not np.isfinite(pit).all() or ((pit < 0) | (pit > 1)).any():
            raise FloatingPointError("Invalid external age CDF")
        result["mean_crps_gyr"] = float(
            np.asarray(predictions["external_age_intrinsic_crps_gyr"])[mask].mean()
        )
    atomic(output / "summary.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    take = np.arange(len(target))
    if len(take) > 20000:
        take = take[np.linspace(0, len(take) - 1, 20000, dtype=int)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(target[take], predicted[take], s=3, alpha=0.25)
    lo, hi = np.quantile(np.r_[target, predicted], [0.005, 0.995])
    axes[0].plot([lo, hi], [lo, hi], color="black", lw=1)
    axes[0].set(
        xlabel="APOKASC3 age reference [Gyr]",
        ylabel="Predicted conditional median [Gyr]",
    )
    axes[1].scatter(target[take], residual[take], s=3, alpha=0.25)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(
        xlabel="APOKASC3 age reference [Gyr]", ylabel="Prediction minus reference [Gyr]"
    )
    fig.suptitle(
        "External seismic/model age scale; not pooled with StarHorse teacher labels"
    )
    fig.tight_layout()
    fig.savefig(output / "age-reference.png", dpi=180)
    fig.savefig(output / "age-reference.pdf")
    plt.close(fig)


def run(args):
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Expanded native evaluation requires CANFAR headless CUDA")
    cfg = yaml.safe_load(args.config.read_text())
    if (
        cfg.get("optimization_steps") != 0
        or cfg.get("reconstruction_presentations") != 0
    ):
        raise ValueError("Expanded evaluation forbids optimization/presentations")
    arm = next(a for a in cfg["arms"] if a["run_id"] == args.run_id)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            "Existing attempted evaluation output; no automatic repeat"
        )
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(
        status="preparing",
        run_id=args.run_id,
        source_sha=source,
        zero_training=True,
        optimizer_constructed=False,
        reconstruction_presentations=0,
        arm=arm,
        policy=cfg,
    )
    atomic(manifest_path, manifest)
    data = None
    try:
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Actual installed library pin differs")
        producer = Path(cfg["output_root"]) / arm["producer_run_id"]
        pm = json.loads((producer / "manifest.json").read_text())
        checkpoint = producer / "checkpoint.pth"
        checkpoint_sha = sha(checkpoint)
        state = torch_load_trusted(checkpoint, map_location="cpu", weights_only=False)
        validate_producer(pm, state, checkpoint_sha, cfg, arm)
        prep = json.loads(Path(cfg["preparation_manifest"]).read_text())
        expected_source = (
            source
            if cfg["preparation_source_sha"] == "same_as_runner_source"
            else cfg["preparation_source_sha"]
        )
        if (
            prep.get("status") != "full_label_evaluation_source_complete"
            or prep.get("source_commit") != expected_source
        ):
            raise ValueError("Actual expanded source status/commit differs")
        if prep.get("source_provenance") != yaml.safe_load(
            Path(cfg["preparation_policy"]).read_text()
        ):
            raise ValueError("Expanded source policy differs from the accepted policy")
        if not prep.get("model_ready"):
            raise ValueError(
                "Expanded source has not passed model-input eligibility gates"
            )
        if sha(cfg["prepared"]) != prep.get("input_sha256"):
            raise ValueError("Expanded source content SHA differs")
        if (
            not prep.get("original_benchmark_inputs_preserved")
            or prep.get("original_benchmark_input_sha256")
            != cfg["original_prepared_sha256"]
        ):
            raise ValueError(
                "Original model-input identity reuse is not source certified"
            )
        data = Prepared(cfg["prepared"])
        ids = data.array("source_id")
        if (
            ids.dtype.kind not in "iu"
            or (ids <= 0).any()
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError(
                "Expanded source requires exact unique positive Gaia identities"
            )
        features = data.array("feature_names").tolist()
        if features != state["feature_columns"] or data.array(
            "target_names"
        ).tolist() != list(NAMES):
            raise ValueError("Feature/target coordinate order differs from checkpoint")
        if "role" in data and not (data.array("role") == "test").all():
            raise ValueError("Expanded evaluation includes a non-TEST role")
        original_path = producer / "test-natural-predictions.npz"
        receipt = json.loads(original_path.with_suffix(".sha256.json").read_text())
        if sha(original_path) != receipt["sha256"]:
            raise ValueError("Original prediction cache SHA differs")
        with np.load(original_path, allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        old_index = matched_indices(ids, original["source_id"])
        membership = data.array("original_test_member").astype(bool)
        if not np.array_equal(membership, old_index >= 0):
            raise ValueError("Original benchmark membership changed")
        extra = data.array("expanded_reference").astype(bool)
        head = build_head(state["arm"], 256)
        head.load_state_dict(state["head_state_dict"], strict=True)
        device = torch.device("cuda")
        model_cfg = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        model = shared.native.probe.load_model(model_cfg, state, device).double().eval()
        head = head.to(device=device, dtype=torch.float64).eval()
        for item in [model, head]:
            for parameter in item.parameters():
                parameter.requires_grad_(False)
        xp = xp_indices(features).tolist()
        allowed = np.array(
            [
                name in {"W1", "W2", "G", "BP", "RP"} or name.startswith(("bp_", "rp_"))
                for name in features
            ]
        )
        if len(xp) != 110:
            raise ValueError("Original XP layout must contain all110 coordinates")
        center, scale = (
            np.asarray(state["label_center"]),
            np.asarray(state["label_scale"]),
        )
        if (
            center.shape != (6,)
            or scale.shape != (6,)
            or not np.isfinite(center).all()
            or not (scale > 0).all()
        ):
            raise ValueError("Selected FIT normalization differs")
        external_all = (
            data.array("external_age_observed").astype(bool)
            if "external_age_observed" in data
            else np.zeros(len(ids), bool)
        )
        union_observed = data.array("observed").astype(bool) & data.array(
            "quality"
        ).astype(bool)
        if "diagnostic_eligible" in data:
            union_observed |= data.array("diagnostic_eligible").astype(bool)
        previous_observed = np.zeros_like(union_observed)
        previous_observed[membership] = original["observed"][old_index[membership]]
        derived_reference_additions = union_observed & ~previous_observed
        need = new_inference_rows(
            old_index, extra | derived_reference_additions, head.kind
        )
        if head.kind != "quantile3":
            need |= external_all
        binding = dict(
            source_sha=source,
            runner_sha256=sha(__file__),
            prepared_sha256=prep["input_sha256"],
            producer_source_sha=cfg["producer_source_sha"],
            checkpoint_sha256=checkpoint_sha,
            original_predictions_sha256=receipt["sha256"],
            library=direct,
            target_names=NAMES,
            helper_sha256=sha(
                Path(__file__).resolve().parents[1]
                / "src/masked_stellar_autoencoder/training/distribution_diagnostics.py"
            ),
        )
        atomic(args.output / "bindings.json", binding)
        np.savez_compressed(
            args.output / "normalization.npz",
            center=center,
            scale=scale,
            point_scale_fit_native=state["point_scale_fit_native"],
        )
        store = ArrayStore(args.output / "prediction-arrays", len(ids))
        cache_dir = args.output / "batch-cache"
        cache_dir.mkdir()
        first_pilot = True
        if not 1 <= cfg["batch_size"] <= 128:
            raise ValueError("Scoring batches must be bounded to128 rows")
        for start in range(0, len(ids), cfg["batch_size"]):
            rows = np.arange(start, min(start + cfg["batch_size"], len(ids)))
            path = cache_dir / f"batch-{start:09d}.npz"
            values, errors = data.take("value", rows), data.take("error", rows)
            mask, diagnostic, raw_modality, input_diagnostic = source_reference_masks(
                data, rows
            )
            union = mask | diagnostic
            known = data.take("error_observed", rows).astype(bool)
            local_index = old_index[rows]
            old_present = local_index >= 0
            original_mask = np.zeros_like(mask)
            original_mask[old_present] = original["observed"][local_index[old_present]]
            if old_present.any():
                same = original_mask[old_present]
                if not np.array_equal(
                    values[old_present][same],
                    original["target_native"][local_index[old_present]][same],
                ):
                    raise ValueError(
                        "Original observed reference changed rather than being backfilled"
                    )
            compute = need[rows]
            computed = {}
            if compute.any():
                selected = rows[compute]

                def capture_partial(partial):
                    np.savez_compressed(
                        path.with_suffix(".partial.npz"),
                        source_id=ids[selected],
                        **partial,
                    )

                external = (
                    data.take("external_age_value", selected)
                    if "external_age_value" in data
                    else None
                )
                inputs = data.take("input", selected)
                if np.any(inputs[:, ~allowed] != -9999):
                    raise ValueError(
                        "Astrometric/label inputs entered source-expanded predictor"
                    )
                batch, draws, contract = score_new(
                    model,
                    head,
                    inputs,
                    values[compute],
                    errors[compute],
                    known[compute],
                    union[compute],
                    center,
                    scale,
                    device,
                    cfg["seed"] + start * 31,
                    capture_partial,
                    external,
                )
                for key, value in batch.items():
                    computed[key] = np.full(
                        (len(rows), *value.shape[1:]), np.nan, dtype="f8"
                    )
                    computed[key][compute] = value
                if contract is not None and first_pilot and union[compute].any():
                    pilot = dict(
                        source_id=ids[selected],
                        target_native=values[compute],
                        mask=union[compute],
                        error_known=known[compute],
                        center=center,
                        scale=scale,
                    )
                    for law, replicas in draws.items():
                        for rep, value in enumerate(replicas):
                            pilot[f"{law}_replica{rep}"] = value.cpu().numpy()
                    pilot.update({"score_" + k: v for k, v in batch.items()})
                    np.savez_compressed(
                        args.output / "bounded-metric-pilot.npz", **pilot
                    )
                    from run_final_six_distribution_metrics import pilot_contracts

                    # The full computed batch and actual samples survive a pilot failure.
                    np.savez_compressed(
                        path.with_suffix(".pilot-scored.npz"),
                        source_id=ids[selected],
                        **batch,
                    )
                    result = pilot_contracts(head, samples=draws, **contract)
                    atomic(args.output / "pilot-contracts.json", result)
                    first_pilot = False
            part = combine_cached(original, local_index, computed, union, original_mask)
            part.update(
                source_id=ids[rows],
                target_native=values,
                observed=mask,
                diagnostic_eligible=diagnostic,
                error=errors,
                error_observed=known & union,
                label_source=data.take("label_source", rows),
                evaluation_mode=np.full(len(rows), "natural"),
                original_test_member=membership[rows],
                expanded_reference=extra[rows],
                inference_new_reference_or_identity=compute,
                input_diagnostic_only=input_diagnostic,
                raw_xp_status=raw_modality,
            )
            for name in (
                "xp_status",
                "real_xp_present",
                "real_no_xp",
                "input_has_xp",
                "andrae_prediction_native",
                "andrae_matched",
                "andrae_training_membership_known",
                "andrae_in_training",
                "ancestor_feh_supervised_fit_exposed",
                "strict_unexposed_star",
                "ancestor_150m_reconstruction_exposed",
                "external_age_value",
                "external_age_observed",
                "external_age_source",
                "external_age_bounds",
                "photometry_source",
                "photometry_release",
            ):
                if name in data:
                    part[name] = data.take(name, rows)
            if "xp_status" in part:
                part["xp_status"] = report_xp_status(part["xp_status"])
            # Preserve computed arrays before checking finite/CDF contracts.
            np.savez_compressed(path, **part)
            atomic(
                path.with_suffix(".sha256.json"),
                dict(
                    sha256=sha(path),
                    count=len(rows),
                    binding_sha256=sha(args.output / "bindings.json"),
                ),
            )
            validate_batch(part, union)
            store.put(rows, part)
            manifest.update(
                status="running",
                processed=start + len(rows),
                total=len(ids),
                model_forward_rows=int(need[: start + len(rows)].sum()),
                checkpoint_sha256=checkpoint_sha,
            )
            atomic(manifest_path, manifest)
        predictions = store.finish()
        predictions["quantile_levels"] = np.array(QUANTILES)
        corners = make_corners(
            model, head, data, center, scale, device, cfg["seed"], args.output
        )
        external_report(predictions, args.output / "external-apokasc3-age")
        from five_wave_report import make_report

        make_report(
            predictions,
            args.output / "report",
            run_id=args.run_id,
            head_kind=head.kind,
            corners=corners,
            metadata=dict(
                point_functional="intrinsic conditional median",
                source_expanded_evaluation=True,
                physical_truth_claim=False,
                checkpoint_sha256=checkpoint_sha,
                original_benchmark_report=str(producer / "report"),
                input_sha256=prep["input_sha256"],
                original_benchmark_inputs_unchanged=True,
                no_training=True,
                secondary_reference_diagnostics_not_primary=True,
                primary_object_holdout="No known c544 FeH auxiliary supervised exposure, including original benchmark IDs",
                parent86_full_training_history_certified=False,
                andrae_projection_scope="Existing source-ID-bound cached subset only; unmatched expanded IDs are unqueried, not evidence of absence from the released catalogue",
                andrae_mh_is_feh_proxy=True,
            ),
        )
        if np.asarray(predictions["diagnostic_eligible"]).any():
            diagnostic_predictions = dict(predictions)
            diagnostic_predictions["observed"] = np.asarray(
                predictions["diagnostic_eligible"], bool
            )
            make_report(
                diagnostic_predictions,
                args.output / "diagnostic-reference-report",
                run_id=args.run_id + "-diagnostic",
                head_kind=head.kind,
                metadata=dict(
                    point_functional="intrinsic conditional median",
                    diagnostic_only=True,
                    physical_truth_claim=False,
                    scope="Unrecommended alpha measurements or source-uncertain photometry inputs; never pooled with primary results",
                ),
            )
        manifest.update(
            status="complete",
            processed=len(ids),
            model_forward_rows=int(need.sum()),
            reused_original_point_rows=int(membership.sum()),
            source_input_sha256=prep["input_sha256"],
            original_predictions_sha256=receipt["sha256"],
            report_status="producer figures require independent visual review",
            optimizer_constructed=False,
            proper_score_scope="Actual normalized family; fixed32-QMC observed NSF CDF, two512-draw replicas; no Gaussian proxy",
        )
        atomic(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic(manifest_path, manifest)
        raise
    finally:
        if data is not None:
            data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/expanded-checkpoint-evaluation-20261024.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
