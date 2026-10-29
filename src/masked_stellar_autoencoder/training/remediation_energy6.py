"""Prototype components for a separate 6D NSF sample-score pilot.

The caller must supply one audited fixed-role observation vector per star.
This is not a multi-survey observation-map implementation, an observed-data
NLL, a physical truth certificate, or a relaxation of previous NLL gates.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from masked_stellar_autoencoder.training.multisurvey_masked_energy import (
    masked_noise_aware_energy_score,
)
from masked_stellar_autoencoder.training.remediation_joint6 import JointSixHead


def validate_energy_config(config):
    """Validate the six-target normalized-noise and reproducibility contract."""
    draws = config.get("energy_training_draws")
    chunk = config.get("energy_micro_batch_size")
    sample_seed = config.get("energy_sample_seed")
    noise_seed = config.get("energy_noise_seed")
    integers = (draws, chunk, sample_seed, noise_seed)
    if any(isinstance(x, bool) or not isinstance(x, int) for x in integers):
        raise ValueError("Energy draw, chunk and generator seeds must be integers")
    if draws < 2 or chunk < 1 or min(sample_seed, noise_seed) < 0:
        raise ValueError("Positive energy draws/chunk and nonnegative seeds required")
    if sample_seed == noise_seed:
        raise ValueError("Flow-sample and measurement-noise seeds must differ")
    raw_floors = np.asarray(config.get("unknown_error_floor_normalized"))
    if raw_floors.dtype.kind not in "iuf":
        raise ValueError("Unknown-error floors must be real numeric values")
    floors = np.asarray(raw_floors, dtype="f8")
    if floors.shape != (6,) or not np.isfinite(floors).all() or (floors <= 0).any():
        raise ValueError(
            "Six explicit positive normalized unknown-error floors required"
        )
    if config.get("energy_noise_assumption") != "independent_gaussian":
        raise ValueError("Explicit independent Gaussian observation noise required")
    if config.get("energy_floor_scope") != "illustrative_not_measured":
        raise ValueError("The unvalidated floor scope must be explicit")
    return floors


class EnergyJointSixHead(JointSixHead):
    """An unrestricted 6D NSF trained from vector samples, not marginal NLL."""

    def __init__(
        self,
        *,
        latent_dim,
        width,
        particles,
        seed,
        components,
        energy_training_draws,
        unknown_error_floor_normalized,
    ):
        super().__init__(
            "nsf",
            latent_dim=latent_dim,
            width=width,
            particles=particles,
            seed=seed,
            components=components,
            proposal_weight=0.0,
        )
        if (
            isinstance(energy_training_draws, bool)
            or not isinstance(energy_training_draws, int)
            or energy_training_draws < 2
        ):
            raise ValueError("At least two whole-vector samples required")
        floors = torch.as_tensor(unknown_error_floor_normalized, dtype=torch.float32)
        if (
            floors.shape != (6,)
            or not torch.isfinite(floors).all()
            or (floors <= 0).any()
        ):
            raise ValueError("Six finite positive normalized error floors required")
        self.register_buffer("unknown_error_floor_normalized", floors.clone())
        self.energy_training_draws = energy_training_draws
        for parameter in self.gaussian.parameters():
            parameter.requires_grad_(False)
        self.sample_generator = None
        self.noise_generator = None

    def configure_generators(self, device, sample_seed, noise_seed):
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 0
            for v in (sample_seed, noise_seed)
        ):
            raise ValueError("Nonnegative integer generator seeds required")
        if sample_seed == noise_seed:
            raise ValueError("Independent sample and noise seeds required")
        self.sample_generator = torch.Generator(device=device).manual_seed(sample_seed)
        self.noise_generator = torch.Generator(device=device).manual_seed(noise_seed)

    def generator_state(self):
        if self.sample_generator is None or self.noise_generator is None:
            raise ValueError("Energy generators have not been configured")
        return {
            "sample": self.sample_generator.get_state().cpu().clone(),
            "noise": self.noise_generator.get_state().cpu().clone(),
        }

    def restore_generator_state(self, state, *, device):
        if not isinstance(state, dict) or set(state) != {"sample", "noise"}:
            raise ValueError("Both energy generator states are required")
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            for value in state.values()
        ):
            raise ValueError("Energy generator states must be byte tensors")
        # Validate both states before replacing either live generator.
        sample = torch.Generator(device=device)
        noise = torch.Generator(device=device)
        sample.set_state(state["sample"].cpu())
        noise.set_state(state["noise"].cpu())
        self.sample_generator, self.noise_generator = sample, noise

    def criterion(self, context, target, sigma, mask, error_known):
        if self.sample_generator is None or self.noise_generator is None:
            raise ValueError("Energy generators have not been configured")
        draws = self.sample(context, self.energy_training_draws, self.sample_generator)
        score = masked_noise_aware_energy_score(
            draws,
            target,
            mask,
            sigma,
            error_known,
            scale=torch.ones(6, device=target.device, dtype=target.dtype),
            generator=self.noise_generator,
            assume_independent_errors=True,
            unknown_error_floor=self.unknown_error_floor_normalized,
        )
        self.last_loss_components = {"masked_energy_score": score.detach()}
        return score


def capture_pilot_rng_state(head, mask_generator, sampler):
    """Capture independent score, reconstruction, sampler and torch RNG states."""
    if not isinstance(mask_generator, torch.Generator) or not isinstance(
        sampler, np.random.Generator
    ):
        raise ValueError(
            "Explicit reconstruction and NumPy sampler generators required"
        )
    return {
        "energy": head.generator_state(),
        "reconstruction_mask": mask_generator.get_state().cpu().clone(),
        "sampler": copy.deepcopy(sampler.bit_generator.state),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_pilot_rng_state(head, mask_generator, sampler, state, *, device):
    """Validate the full state before restoring a selected pilot checkpoint."""
    keys = {"energy", "reconstruction_mask", "sampler", "torch_cpu", "torch_cuda"}
    if not isinstance(state, dict) or set(state) != keys:
        raise ValueError("Complete pilot RNG state is required")
    if not isinstance(mask_generator, torch.Generator) or not isinstance(
        sampler, np.random.Generator
    ):
        raise ValueError(
            "Explicit reconstruction and NumPy sampler generators required"
        )
    energy = state["energy"]
    if not isinstance(energy, dict) or set(energy) != {"sample", "noise"}:
        raise ValueError("Both energy generator states are required")
    byte_states = [
        energy["sample"],
        energy["noise"],
        state["reconstruction_mask"],
        state["torch_cpu"],
    ]
    if not isinstance(state["torch_cuda"], list):
        raise ValueError("CUDA RNG topology must be a list")
    byte_states += state["torch_cuda"]
    if any(
        not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.ndim != 1
        for v in byte_states
    ):
        raise ValueError("Pilot RNG states must be byte tensors")
    if len(state["torch_cuda"]) != (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    ):
        raise ValueError("CUDA RNG topology differs from checkpoint")
    for key in ("sample", "noise"):
        torch.Generator(device=device).set_state(energy[key].cpu())
    torch.Generator(device=mask_generator.device).set_state(
        state["reconstruction_mask"].cpu()
    )
    torch.Generator(device="cpu").set_state(state["torch_cpu"].cpu())
    for index, value in enumerate(state["torch_cuda"]):
        torch.Generator(device=f"cuda:{index}").set_state(value.cpu())
    copied_sampler = copy.deepcopy(sampler)
    copied_sampler.bit_generator.state = copy.deepcopy(state["sampler"])
    head.restore_generator_state(energy, device=device)
    mask_generator.set_state(state["reconstruction_mask"].cpu())
    sampler.bit_generator.state = copy.deepcopy(state["sampler"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def backward_energy_chunks(
    head, latent, target, sigma, mask, known, rows, *, chunk_rows, tensor
):
    """Accumulate per-star means; unlike old NLL, never weight by label count."""
    rows = np.asarray(rows)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or not len(rows)
        or np.any(rows < 0)
        or np.any(rows[1:] <= rows[:-1])
        or isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise ValueError("Sorted unique batch rows and positive chunk size required")
    target, sigma, mask, known = map(np.asarray, (target, sigma, mask, known))
    if (
        target.ndim != 2
        or target.shape[1] != 6
        or sigma.shape != target.shape
        or mask.shape != target.shape
        or known.shape != target.shape
        or mask.dtype.kind != "b"
        or known.dtype.kind != "b"
        or (known & ~mask).any()
        or rows[-1] >= len(target)
    ):
        raise ValueError("Aligned six-target arrays and boolean masks required")
    if not mask[rows].any(axis=1).all():
        raise ValueError("Every energy-scored star must have an observed target")
    total = len(rows)
    value = None
    for start in range(0, total, chunk_rows):
        rr = rows[start : start + chunk_rows]
        loss = head.criterion(
            latent(rr),
            tensor(target[rr]),
            tensor(sigma[rr]),
            tensor(mask[rr], torch.bool),
            tensor(known[rr], torch.bool),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite energy objective")
        contribution = loss * (len(rr) / total)
        contribution.backward()
        value = (
            contribution.detach() if value is None else value + contribution.detach()
        )
    return value, {"masked_energy_score": float(value)}


def enforce_energy_encoder_gradient(encoder, head, update, supervised_l2):
    """Guard the flow and supervised encoder, never the unused proposal."""
    if not isinstance(update, int) or isinstance(update, bool) or update < 1:
        raise ValueError("Positive update number required")
    if supervised_l2 is not None and (
        not math.isfinite(supervised_l2) or supervised_l2 < 0
    ):
        raise ValueError("Finite nonnegative supervised encoder norm required")
    gradients = [
        p.grad for p in encoder.parameters() if p.requires_grad and p.grad is not None
    ]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("Finetuned encoder gradients missing or nonfinite")
    flow_gradients = [p.grad for p in head.flow.parameters() if p.grad is not None]
    if (
        not flow_gradients
        or not all(torch.isfinite(g).all() for g in flow_gradients)
        or not any(torch.count_nonzero(g).item() for g in flow_gradients)
    ):
        raise FloatingPointError("Flow gradients missing, zero or nonfinite")
    if update >= 2:
        if supervised_l2 is not None and supervised_l2 <= 0:
            raise FloatingPointError("Finetuned encoder has no supervised gradient")
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise FloatingPointError("Finetuned encoder has no nonzero update gradient")
        return "supervised_nonzero"
    if supervised_l2 is not None and supervised_l2 > 0:
        return "supervised_nonzero"
    projection = head.flow.output.weight
    if (
        torch.count_nonzero(projection).item() == 0
        and projection.grad is not None
        and torch.isfinite(projection.grad).all()
        and torch.count_nonzero(projection.grad).item() > 0
    ):
        return "zero_initialized_flow_projection_first_step"
    raise FloatingPointError("First-step flow projection does not support warmup")


def energy_device_preflight(head, context, target, sigma, mask, known):
    """Two copy-only steps; original parameters, generators and global RNG unchanged."""
    if context.ndim != 2 or not len(context):
        raise ValueError("A nonempty preflight context is required")
    original_rng = head.generator_state()
    original_state = {
        name: value.detach().clone() for name, value in head.state_dict().items()
    }
    global_cpu = torch.get_rng_state().clone()
    global_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        candidate = copy.deepcopy(head)
        candidate.train()
        local = context.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam(candidate.flow.parameters(), lr=0.001)
        metrics = []
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            local.grad = None
            value = candidate.criterion(local, target, sigma, mask, known)
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite device preflight score")
            value.backward()
            gradients = [
                p.grad for p in candidate.flow.parameters() if p.grad is not None
            ]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Device preflight flow gradients invalid")
            flow_norm = (
                sum(float(g.detach().double().square().sum()) for g in gradients) ** 0.5
            )
            if flow_norm <= 0 or not math.isfinite(flow_norm) or local.grad is None:
                raise FloatingPointError(
                    "Device preflight flow or context disconnected"
                )
            context_norm = float(local.grad.detach().double().norm())
            if not math.isfinite(context_norm) or (step == 2 and context_norm <= 0):
                raise FloatingPointError("Device preflight context failed after warmup")
            metrics.append(
                {
                    "step": step,
                    "score": float(value.detach()),
                    "flow_gradient_l2": flow_norm,
                    "context_gradient_l2": context_norm,
                }
            )
            optimizer.step()
    finally:
        torch.set_rng_state(global_cpu)
        if global_cuda:
            torch.cuda.set_rng_state_all(global_cuda)
    if any(
        not torch.equal(head.state_dict()[name], value)
        for name, value in original_state.items()
    ):
        raise ValueError("Device preflight mutated the original head")
    after = head.generator_state()
    if any(not torch.equal(after[key], original_rng[key]) for key in original_rng):
        raise ValueError("Device preflight advanced the original training generators")
    return {
        "copy_only": True,
        "updates": 2,
        "steps": metrics,
        "scope": "Numerical sample-score preflight only; no encoder/reconstruction/physical calibration certificate",
    }
