"""Split-bound consensus training primitives for the multi-survey model.

This module does not create catalogue splits or certify astrophysical labels.
It accepts only a digest-bound FIT reader and refuses modified supervised
inputs. Parent and normalization digests are caller declarations, not proof
that their contents were loaded. The external trainer remains responsible for
verifying those files, sampler/scheduler state, evaluation and physical anchors.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .multisurvey_data import SparseSurveyDataset, collate_multisurvey
from .multisurvey_model import _is_sha256, _is_source_binding, multisurvey_objective


def _static_head_config(model, relative_head) -> dict:
    return {
        "canonical_dims": int(model.head.dims),
        "canonical_min_normalized_scale": float(model.head.min_normalized_scale),
        "relative_dims": int(relative_head.dims),
        "relative_degree": int(relative_head.degree),
        "relative_boundary": str(relative_head.boundary),
    }


def _fixed_buffers_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.named_buffers()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FitIdentityGate:
    """Bind a FIT-only dataset and compare each batch to its source records."""

    def __init__(
        self,
        dataset,
        *,
        source_binding: str,
        fit_ids_sha256: str,
        repeat_policy: str = "reject",
    ):
        if not isinstance(dataset, SparseSurveyDataset):
            raise ValueError("A verified sparse survey reader is required")
        if (
            not _is_source_binding(source_binding)
            or dataset.source_binding != source_binding
        ):
            raise ValueError("FIT source digest binding differs")
        if repeat_policy not in {"reject", "lowest_source_row"}:
            raise ValueError("Unknown repeat resolution policy")
        self.dataset = dataset
        self.repeat_policy = repeat_policy
        self.ids = np.asarray(dataset.ids[dataset.star_rows], dtype="<i8")
        if len(self.ids) == 0 or np.any(np.diff(self.ids) <= 0):
            raise ValueError(
                "FIT source identities must be nonempty, sorted and unique"
            )
        self.ids.setflags(write=False)
        digest = hashlib.sha256(self.ids.tobytes()).hexdigest()
        if not _is_sha256(fit_ids_sha256) or digest != fit_ids_sha256:
            raise ValueError("FIT identity digest differs")
        roles = dataset._handle()["role"][dataset.star_rows].astype("U")
        if np.any(roles != "fit"):
            raise ValueError("FIT reader includes non-FIT identities")
        self.source_binding, self.fit_ids_sha256 = source_binding, digest
        self._source_stat = self._stat()

    def _stat(self):
        stat = os.stat(self.dataset.path)
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def verify_batch(self, batch: dict) -> dict:
        """Rebuild source records to reject forged roles, labels or XP masks."""
        if self._stat() != self._source_stat:
            raise ValueError("Pinned FIT source changed after verification")
        ids = batch.get("source_id")
        if (
            not isinstance(ids, torch.Tensor)
            or ids.ndim != 1
            or ids.dtype != torch.int64
        ):
            raise ValueError("Batch needs int64 source identities")
        raw = ids.detach().cpu().numpy()
        if len(raw) == 0 or len(np.unique(raw)) != len(raw):
            raise ValueError("Batch source identities must be nonempty and unique")
        pos = np.searchsorted(self.ids, raw)
        if np.any(pos >= len(self.ids)) or not np.array_equal(self.ids[pos], raw):
            raise ValueError("Batch contains identities outside the pinned FIT split")
        expected = collate_multisurvey(
            [self.dataset[int(i)] for i in pos], repeat_policy=self.repeat_policy
        )
        tensor_keys = (
            "source_id",
            "input",
            "input_available",
            "value",
            "error",
            "observed",
            "error_known",
            "source_index",
            "source_row",
            "is_anchor",
        )
        for key in tensor_keys:
            actual, reference = batch.get(key), expected[key]
            if not isinstance(actual, torch.Tensor) or actual.shape != reference.shape:
                raise ValueError(f"FIT batch field {key} differs from source")
            actual = actual.detach().cpu()
            equal = (
                torch.allclose(actual, reference, rtol=0, atol=0, equal_nan=True)
                if reference.is_floating_point()
                else torch.equal(actual, reference)
            )
            if not equal:
                raise ValueError(f"FIT batch field {key} differs from source")
        for key in (
            "role",
            "source_registry",
            "registry_sha256",
            "source_binding",
            "provenance",
            "repeat_policy",
            "repeat_resolution",
            "target_names",
        ):
            if batch.get(key) != expected[key]:
                raise ValueError(f"FIT batch field {key} differs from source")
        return expected


class ConsensusTrainer:
    """One verified shared-latent Gaussian consensus update at a time."""

    def __init__(
        self,
        model,
        relative_head,
        optimizer,
        gate: FitIdentityGate,
        source_floor,
        *,
        parent_sha256: str,
        normalization_sha256: str,
        library_revision: str,
        msa_revision: str,
        assume_diagonal_catalog_covariance: bool,
        reconstruction_weight: float = 0.0,
        reconstruction_width: int | None = None,
        calibration_regularization_weight: float = 0.0,
        sampling_contract_sha256: str | None = None,
    ):
        if not all(_is_sha256(x) for x in (parent_sha256, normalization_sha256)):
            raise ValueError("Parent and normalization SHA256 bindings are required")
        if any(
            len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev)
            for rev in (library_revision, msa_revision)
        ):
            raise ValueError("Immutable MSA and torchregress revisions are required")
        if reconstruction_weight < 0 or not np.isfinite(reconstruction_weight):
            raise ValueError("Invalid reconstruction weight")
        if assume_diagonal_catalog_covariance is not True:
            raise ValueError(
                "This baseline requires explicit diagonal catalog-covariance consent"
            )
        if sampling_contract_sha256 is not None and not _is_sha256(
            sampling_contract_sha256
        ):
            raise ValueError("Sampling contract must be a SHA256 digest")
        self.sampling_contract_sha256 = sampling_contract_sha256
        self.model, self.relative_head, self.optimizer = model, relative_head, optimizer
        self.gate = gate
        self.source_floor = torch.as_tensor(source_floor).detach().clone()
        self.reconstruction_weight, self.reconstruction_width = (
            reconstruction_weight,
            reconstruction_width,
        )
        self.regularization = calibration_regularization_weight
        self.steps = 0
        expected_params = {
            id(p)
            for module in (model, relative_head)
            for p in module.parameters()
            if p.requires_grad
        }
        supplied_params = [
            p for group in optimizer.param_groups for p in group["params"]
        ]
        if (
            len({id(p) for p in supplied_params}) != len(supplied_params)
            or {id(p) for p in supplied_params} != expected_params
        ):
            raise ValueError(
                "Optimizer must cover exactly the trainable model and calibration parameters"
            )
        names = {
            id(p): prefix + name
            for prefix, module in (("model.", model), ("relative.", relative_head))
            for name, p in module.named_parameters()
            if p.requires_grad
        }
        optimizer_groups = [
            [names[id(p)] for p in group["params"]] for group in optimizer.param_groups
        ]
        floor = (
            torch.as_tensor(source_floor)
            .detach()
            .cpu()
            .to(torch.float64)
            .contiguous()
            .numpy()
        )
        if floor.ndim != 2 or not np.isfinite(floor).all() or (floor < 0).any():
            raise ValueError(
                "A finite nonnegative source error-floor matrix is required"
            )
        self.contract = {
            "schema": "multisurvey-consensus-trainer-v1",
            "source_binding": gate.source_binding,
            "registry_sha256": gate.dataset.registry_sha256,
            "source_registry": list(gate.dataset.names),
            "fit_ids_sha256": gate.fit_ids_sha256,
            "repeat_policy": gate.repeat_policy,
            "parent_sha256": parent_sha256,
            "normalization_sha256": normalization_sha256,
            "torchregress_revision": library_revision,
            "msa_revision": msa_revision,
            "static_head_config": _static_head_config(model, relative_head),
            "canonical_head_type": type(model.head).__module__
            + "."
            + type(model.head).__qualname__,
            "canonical_head_buffers_sha256": _fixed_buffers_sha256(model.head),
            "relative_head_type": type(relative_head).__module__
            + "."
            + type(relative_head).__qualname__,
            "relative_head_buffers_sha256": _fixed_buffers_sha256(relative_head),
            "optimizer_type": type(optimizer).__module__
            + "."
            + type(optimizer).__qualname__,
            "optimizer_parameter_groups": optimizer_groups,
            "catalog_covariance": "explicit diagonal measurement-error assumption",
            "source_floor_shape": list(floor.shape),
            "source_floor_sha256": hashlib.sha256(
                floor.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "reconstruction_weight": reconstruction_weight,
            "reconstruction_width": reconstruction_width,
            "calibration_regularization_weight": calibration_regularization_weight,
            "scope": "survey consensus only; parent and normalization hashes are caller declarations; audit ancestry and physical anchors",
        }
        if sampling_contract_sha256 is not None:
            self.contract["sampling_contract_sha256"] = sampling_contract_sha256
            self.contract["sampling_scope"] = (
                "Caller verifies per-star weights and sampling state against the "
                "pinned contract; reconstruction sampling is separate"
            )
        self.contract_sha256 = hashlib.sha256(
            json.dumps(self.contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _check_fixed_buffers(self) -> None:
        self.gate.dataset._check_source_binding()
        if self.sampling_contract_sha256 != self.contract.get(
            "sampling_contract_sha256"
        ):
            raise ValueError("Pinned sampling contract changed")
        if self.gate.repeat_policy != self.contract["repeat_policy"]:
            raise ValueError("Repeat resolution policy changed")
        if (
            _static_head_config(self.model, self.relative_head)
            != self.contract["static_head_config"]
        ):
            raise ValueError("Fixed head configuration changed")
        if (
            _fixed_buffers_sha256(self.model.head)
            != self.contract["canonical_head_buffers_sha256"]
            or _fixed_buffers_sha256(self.relative_head)
            != self.contract["relative_head_buffers_sha256"]
        ):
            raise ValueError("Fixed normalization or survey-gauge buffers changed")
        floor = self.source_floor.detach().cpu().to(torch.float64).contiguous().numpy()
        floor_sha = hashlib.sha256(
            floor.astype("<f8", copy=False).tobytes()
        ).hexdigest()
        if (
            list(floor.shape) != self.contract["source_floor_shape"]
            or floor_sha != self.contract["source_floor_sha256"]
        ):
            raise ValueError("Pinned source error floors changed")
        names = {
            id(p): prefix + name
            for prefix, module in (
                ("model.", self.model),
                ("relative.", self.relative_head),
            )
            for name, p in module.named_parameters()
            if p.requires_grad
        }
        try:
            groups = [
                [names[id(p)] for p in group["params"]]
                for group in self.optimizer.param_groups
            ]
        except KeyError as error:
            raise ValueError("Optimizer parameter identities changed") from error
        if groups != self.contract["optimizer_parameter_groups"]:
            raise ValueError("Optimizer parameter order changed")

    def step(
        self,
        batch: dict,
        *,
        reconstruction_mask: torch.Tensor | None = None,
        star_weights: torch.Tensor | None = None,
    ) -> dict:
        if (star_weights is None) != (self.sampling_contract_sha256 is None):
            raise ValueError(
                "Star weights and a pinned sampling contract are both required"
            )
        self._check_fixed_buffers()
        verified = self.gate.verify_batch(batch)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        inputs = verified["input"].to(device=device, dtype=dtype)
        verified["input"] = inputs  # supervised XP is the unchanged natural input
        reconstruction = None
        if self.reconstruction_weight:
            width = self.reconstruction_width
            if (
                not isinstance(width, int)
                or isinstance(width, bool)
                or not 0 < width <= inputs.shape[1]
            ):
                raise ValueError("A valid reconstruction prefix width is required")
            if (
                reconstruction_mask is None
                or reconstruction_mask.dtype != torch.bool
                or reconstruction_mask.shape != inputs.shape
            ):
                raise ValueError("A boolean aligned reconstruction mask is required")
            hidden_mask = reconstruction_mask.to(device=device)
            available = verified["input_available"].to(device=device)
            if (hidden_mask & ~available).any() or not hidden_mask[:, :width].any():
                raise ValueError(
                    "Reconstruction may mask only present, nonempty target features"
                )
            reconstruction = {
                "masked_input": inputs.masked_fill(hidden_mask, -9999),
                "target": inputs[:, :width],
                "mask": hidden_mask[:, :width],
                "source_id": verified["source_id"],
                "role": verified["role"],
                "source_binding": self.gate.source_binding,
            }
        elif reconstruction_mask is not None:
            raise ValueError(
                "Reconstruction mask supplied for zero-weight reconstruction"
            )
        self.model.train()
        if not any(p.requires_grad for p in self.model.mae.encoder.parameters()):
            # A frozen encoder must not update BatchNorm state or use dropout.
            self.model.mae.encoder.eval()
        self.relative_head.train()
        self.optimizer.zero_grad(set_to_none=True)
        result = multisurvey_objective(
            self.model,
            verified,
            self.relative_head,
            self.source_floor,
            expected_registry_sha256=self.gate.dataset.registry_sha256,
            expected_source_registry=self.gate.dataset.names,
            expected_training_source_binding=self.gate.source_binding,
            reconstruction=reconstruction,
            reconstruction_weight=self.reconstruction_weight,
            expected_reconstruction_source_binding=self.gate.source_binding
            if reconstruction
            else None,
            calibration_regularization_weight=self.regularization,
            star_weights=star_weights,
            assume_diagonal_catalog_covariance=True,
        )
        if not torch.isfinite(result["total"]):
            raise FloatingPointError("Nonfinite multi-survey training objective")
        result["total"].backward()
        parameters = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]
        gradients = [p.grad for p in parameters if p.grad is not None]
        if (
            not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
            or not any(torch.count_nonzero(g).item() for g in gradients)
        ):
            raise FloatingPointError("Missing or nonfinite optimizer gradients")
        self.optimizer.step()
        self.steps += 1
        return {
            k: float(result[k].detach())
            for k in (
                "total",
                "supervised",
                "reconstruction",
                "calibration_regularization",
            )
        }

    def save(self, path: str | Path, *, sampler_state: dict) -> str:
        """Atomically create a new checkpoint; caller supplies sampler state."""
        self._check_fixed_buffers()
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        if not isinstance(sampler_state, dict):
            raise ValueError("Explicit sampler state is required")
        try:
            sampler_state = json.loads(json.dumps(sampler_state, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("Sampler state must be finite JSON data") from error
        state = {
            "contract": self.contract,
            "contract_sha256": self.contract_sha256,
            "model": self.model.state_dict(),
            "relative_head": self.relative_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "sampler_state": sampler_state,
        }
        temporary = path.with_name(path.name + ".partial-" + str(os.getpid()))
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("xb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def load(self, path: str | Path, *, expected_sha256: str) -> dict:
        """Restore only a SHA-bound checkpoint for this exact training contract."""
        self._check_fixed_buffers()
        if not _is_sha256(expected_sha256):
            raise ValueError("Expected checkpoint SHA256 is required")
        with Path(path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_sha256:
            raise ValueError("Checkpoint SHA256 differs")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state.get("contract_sha256") != self.contract_sha256
            or state.get("contract") != self.contract
        ):
            raise ValueError("Checkpoint training contract differs")
        if state["cuda_rng"] and (
            not torch.cuda.is_available()
            or len(state["cuda_rng"]) != torch.cuda.device_count()
        ):
            raise ValueError("Checkpoint CUDA RNG topology differs")
        self.model.load_state_dict(state["model"], strict=True)
        self.relative_head.load_state_dict(state["relative_head"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state["steps"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state["sampler_state"]
