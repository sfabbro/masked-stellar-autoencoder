"""One canonical correlated atmosphere prediction with sparse survey likelihood.

The MAE encoder and reconstruction path are supplied by the caller and are not
replaced. Survey and anchor mappings apply only during training/evaluation;
inference produces one common distribution per star.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_source_binding(value: object) -> bool:
    # A direct source uses one digest; an overlay binds both source and manifest.
    return (
        isinstance(value, str)
        and all(_is_sha256(p) for p in value.split(":"))
        and len(value.split(":")) in (1, 2)
    )


class CanonicalGaussianHead(nn.Module):
    """Predict a full-covariance Gaussian and convert it to declared units."""

    def __init__(
        self,
        latent_dim: int,
        center: Tensor,
        scale: Tensor,
        *,
        width: int = 128,
        min_normalized_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        center = torch.as_tensor(center).clone().detach()
        scale = torch.as_tensor(scale).clone().detach()
        if (
            center.ndim != 1
            or scale.shape != center.shape
            or not torch.isfinite(center).all()
        ):
            raise ValueError("Center and scale must be finite [D] vectors")
        if (
            not torch.isfinite(scale).all()
            or (scale <= 0).any()
            or min_normalized_scale <= 0
        ):
            raise ValueError("Scale and minimum normalized diagonal must be positive")
        self.dims = center.numel()
        if self.dims < 1 or latent_dim < 1 or width < 1:
            raise ValueError("Positive latent, output, and hidden dimensions required")
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.min_normalized_scale = float(min_normalized_scale)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.Linear(width, self.dims + self.dims * (self.dims + 1) // 2),
        )
        row, column = torch.tril_indices(self.dims, self.dims)
        self.register_buffer("row", row, persistent=False)
        self.register_buffer("column", column, persistent=False)

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        if latent.ndim != 2:
            raise ValueError("latent must be [B, latent_dim]")
        raw = self.net(latent)
        normalized_mean, packed = raw[:, : self.dims], raw[:, self.dims :]
        diagonal = self.row == self.column
        packed = torch.where(
            diagonal[None], F.softplus(packed) + self.min_normalized_scale, packed
        )
        lower = raw.new_zeros((len(raw), self.dims, self.dims))
        lower[:, self.row, self.column] = packed
        return normalized_mean * self.scale + self.center, lower * self.scale[
            None, :, None
        ]


class CanonicalAtmosphereModel(nn.Module):
    """Wrap the existing MAE without modifying its encoder or decoder."""

    def __init__(self, mae: nn.Module, head: CanonicalGaussianHead) -> None:
        super().__init__()
        if not hasattr(mae, "encoder"):
            raise ValueError("The supplied MAE must expose its original encoder")
        self.mae = mae
        self.head = head

    def forward(self, natural_input: Tensor) -> tuple[Tensor, Tensor]:
        """Use the provided natural-input mask; this method adds no XP masking."""
        latent = self.mae.encoder(natural_input)
        # Physical Teff and dex covariances can be badly conditioned in fp32.
        # Allow the observation head and marginal likelihood to use fp64 while
        # keeping the pretrained encoder and reconstruction path in fp32.
        head_dtype = next(self.head.parameters()).dtype
        return self.head(latent.to(dtype=head_dtype))

    def reconstruct(self, masked_input: Tensor):
        """Call the unchanged original MAE reconstruction path."""
        return self.mae(masked_input)


def pack_multisurvey_observations(
    mean: Tensor,
    batch: dict,
    relative_head: nn.Module,
    source_floor: Tensor,
    *,
    expected_registry_sha256: str,
    expected_source_registry: tuple[str, ...],
    anchor_floor: Tensor | None = None,
    common_head: nn.Module | None = None,
    condition_head: nn.Module | None = None,
    assume_diagonal_catalog_covariance: bool = False,
):
    """Linearize all compatible survey/anchor records around the same mean.

    ``source_floor`` is a declared nonnegative [S,D] standard-deviation floor in
    the native target units. An observed value with no known catalog error needs
    a strictly positive floor. If no joint catalog covariance is supplied,
    callers must explicitly accept the diagonal covariance assumption.

    Common and conditional heads are optional. A common head is identifiable
    only with adequate independent anchors; its use does not certify physical
    calibration. Anchors use their own direct identity observation mapping and
    are not corrected again as survey observations.
    """
    from torchregress.losses import PackedObservations

    if mean.ndim != 2:
        raise ValueError("mean must be [B,D]")
    b, d = mean.shape
    names = ("teff", "logg", "fe_h")
    units = ("K", "dex(cm/s2)", "dex")
    if d != 3 or tuple(batch.get("target_names", ())) != names:
        raise ValueError(
            "The identity observation map requires the canonical three-target contract"
        )
    if (
        not _is_sha256(expected_registry_sha256)
        or batch.get("registry_sha256") != expected_registry_sha256
        or tuple(batch.get("source_registry", ())) != expected_source_registry
    ):
        raise ValueError("Batch and model source registry bindings differ")
    provenance = batch.get("provenance")
    raw_mask = batch["observed"].detach().to("cpu", dtype=torch.bool)
    if raw_mask.ndim != 3 or raw_mask.shape[0] != b or raw_mask.shape[2] != d:
        raise ValueError("Observation mask must be [B,M,D]")
    if provenance is None or len(provenance) != b:
        raise ValueError("Per-record survey and anchor provenance is required")
    for row in range(b):
        if (
            len(provenance[row]) > raw_mask.shape[1]
            or raw_mask[row, len(provenance[row]) :].any()
        ):
            raise ValueError("Every observed record needs provenance")
        seen = set()
        for col, record in enumerate(provenance[row]):
            if col >= raw_mask.shape[1] or not raw_mask[row, col].any():
                continue
            dependency = record.get("dependency_id")
            if (
                not dependency
                or dependency in seen
                or not record.get("release")
                or not record.get("pipeline")
            ):
                raise ValueError("Missing or repeated measurement provenance")
            seen.add(dependency)
            definitions = record.get("target_definitions", {})
            for target, unit, used in zip(
                names, units, raw_mask[row, col], strict=True
            ):
                if not used:
                    continue
                definition = definitions.get(target)
                if (
                    not isinstance(definition, dict)
                    or definition.get("quantity") != target
                    or definition.get("unit") != unit
                    or not definition.get("field")
                ):
                    raise ValueError(
                        f"Missing or incompatible {target} observation definition"
                    )
                if target == "fe_h" and str(definition["field"]).strip().upper() in {
                    "M_H",
                    "MH",
                    "MH_GSPSPEC",
                    "M_H_BDASP",
                }:
                    raise ValueError(
                        "Global metallicity cannot be treated as direct Fe/H"
                    )
    observed = batch["observed"].to(device=mean.device, dtype=torch.bool)
    known = batch["error_known"].to(device=mean.device, dtype=torch.bool)
    values = batch["value"].to(device=mean.device, dtype=mean.dtype)
    errors = batch["error"].to(device=mean.device, dtype=mean.dtype)
    sources = batch["source_index"].to(device=mean.device)
    anchors = batch["is_anchor"].to(device=mean.device, dtype=torch.bool)
    if observed.ndim != 3 or observed.shape[0] != b or observed.shape[2] != d:
        raise ValueError("Observation mask must be [B,M,D]")
    m = observed.shape[1]
    if (
        values.shape != observed.shape
        or errors.shape != observed.shape
        or known.shape != observed.shape
        or sources.shape != (b, m)
        or anchors.shape != (b, m)
    ):
        raise ValueError("Observation tensors do not align")
    if (known & ~observed).any() or not torch.isfinite(values[observed]).all():
        raise ValueError("Observed values and error-known flags are invalid")
    if not torch.isfinite(errors[known]).all() or (errors[known] <= 0).any():
        raise ValueError("Known measurement errors must be positive and finite")
    floor = torch.as_tensor(source_floor, dtype=mean.dtype, device=mean.device)
    if (
        floor.ndim != 2
        or floor.shape[0] < 1
        or floor.shape[1] != d
        or not torch.isfinite(floor).all()
        or (floor < 0).any()
    ):
        raise ValueError("source_floor must be finite nonnegative [S,D]")
    if len(expected_source_registry) != len(floor):
        raise ValueError("Source floor rows must follow the bound source registry")
    active_record = observed.any(-1)
    if (active_record & ~anchors & ((sources < 0) | (sources >= len(floor)))).any():
        raise ValueError("Observed survey record has no valid source")
    if (active_record & anchors & (sources != -1)).any():
        raise ValueError("Anchor records must use source_index=-1")
    safe_source = sources.clamp(min=0, max=max(0, len(floor) - 1))
    anchor_floor = (
        torch.zeros(d, device=mean.device, dtype=mean.dtype)
        if anchor_floor is None
        else torch.as_tensor(anchor_floor, device=mean.device, dtype=mean.dtype)
    )
    if (
        anchor_floor.shape != (d,)
        or not torch.isfinite(anchor_floor).all()
        or (anchor_floor < 0).any()
    ):
        raise ValueError("anchor_floor must be finite nonnegative [D]")
    record_floor = torch.where(anchors[..., None], anchor_floor, floor[safe_source])
    if (observed & ~known & (record_floor <= 0)).any():
        raise ValueError("Unknown observed error requires an explicit positive floor")
    err = torch.where(known, errors, torch.zeros_like(errors))

    offsets, derivatives = relative_head(mean)
    if offsets.shape != (b, len(floor), d) or derivatives.shape != (
        b,
        len(floor),
        d,
        d,
    ):
        raise ValueError("Relative source head dimensions differ")
    if hasattr(relative_head, "weights"):
        comparable = relative_head.weights.to(mean.device)[safe_source] > 0
        if (observed & ~anchors[..., None] & ~comparable).any():
            raise ValueError("Observed survey target is not declared comparable")
    idx = safe_source[..., None].expand(b, m, d)
    relative = torch.gather(offsets, 1, idx)
    jac_idx = safe_source[..., None, None].expand(b, m, d, d)
    relative_jac = torch.gather(derivatives, 1, jac_idx)
    common, common_jac = (
        (torch.zeros_like(mean), mean.new_zeros((b, d, d)))
        if common_head is None
        else common_head(mean)
    )
    if common.shape != (b, d) or common_jac.shape != (b, d, d):
        raise ValueError("Common head dimensions differ")
    correction = relative + common[:, None, :]
    correction_jac = relative_jac + common_jac[:, None, :, :]
    if condition_head is not None:
        q = batch.get("conditions")
        q_known = batch.get("conditions_known")
        if q is None or q_known is None:
            raise ValueError("Explicit measurement conditions and known masks required")
        q = q.to(device=mean.device, dtype=mean.dtype)
        q_known = q_known.to(device=mean.device, dtype=torch.bool)
        ref = condition_head.reference_conditions.to(
            device=mean.device, dtype=mean.dtype
        )
        if (
            ref.shape[0] != len(floor)
            or q.shape != (b, m, ref.shape[1])
            or q_known.shape != q.shape
        ):
            raise ValueError("Condition shapes do not align")
        if ((active_record & ~anchors)[..., None] & ~q_known).any():
            raise ValueError(
                "Active condition-dependent observations need known conditions"
            )
        if not torch.isfinite(q[q_known]).all():
            raise ValueError("Known measurement conditions must be finite")
        for j in range(m):
            ref_batch = ref[None].expand(b, -1, -1).clone()
            qj = torch.where(q_known[:, j], q[:, j], ref[safe_source[:, j]])
            ref_batch.scatter_(
                1,
                safe_source[:, j, None, None].expand(b, 1, ref.shape[1]),
                qj[:, None, :],
            )
            u, uj = condition_head(mean, ref_batch)
            chosen = safe_source[:, j, None, None].expand(b, 1, d)
            chosen_j = safe_source[:, j, None, None, None].expand(b, 1, d, d)
            correction = correction + torch.nn.functional.one_hot(
                torch.tensor(j, device=mean.device), m
            )[None, :, None] * u.gather(1, chosen)
            correction_jac = correction_jac + torch.nn.functional.one_hot(
                torch.tensor(j, device=mean.device), m
            )[None, :, None, None] * uj.gather(1, chosen_j)
    correction = torch.where(
        anchors[..., None], torch.zeros_like(correction), correction
    )
    correction_jac = torch.where(
        anchors[..., None, None], torch.zeros_like(correction_jac), correction_jac
    )
    mapped = mean[:, None, :] + correction
    eye = torch.eye(d, dtype=mean.dtype, device=mean.device)
    jacobian = eye[None, None] + correction_jac

    joint = batch.get("joint_covariance")
    if joint is None:
        if not assume_diagonal_catalog_covariance:
            raise ValueError(
                "Explicitly acknowledge diagonal catalog covariance or supply a joint covariance"
            )
        variance = err.square() + record_floor.square()
        covariance = torch.diag_embed(variance.reshape(b, m * d))
    else:
        if joint.shape != (b, m, d, m, d):
            raise ValueError("joint_covariance must be [B,M,D,M,D]")
        covariance = joint.to(device=mean.device, dtype=mean.dtype).reshape(
            b, m * d, m * d
        )
        diagonal = covariance.diagonal(dim1=-2, dim2=-1).reshape(b, m, d)
        if not torch.isfinite(diagonal[known]).all() or not torch.allclose(
            diagonal[known], errors[known].square(), rtol=1e-5, atol=1e-10
        ):
            raise ValueError(
                "joint_covariance diagonal differs from known catalog errors"
            )
        covariance = covariance + torch.diag_embed(
            record_floor.square().reshape(b, m * d)
        )
    return PackedObservations(
        values.reshape(b, m * d),
        mapped.reshape(b, m * d),
        jacobian.reshape(b, m * d, d),
        covariance,
        observed.reshape(b, m * d),
        block_ids=torch.arange(m, device=mean.device)
        .repeat_interleave(d)[None]
        .expand(b, -1),
    )


def multisurvey_objective(
    model: CanonicalAtmosphereModel,
    batch: dict,
    relative_head: nn.Module,
    source_floor: Tensor,
    *,
    expected_registry_sha256: str,
    expected_source_registry: tuple[str, ...],
    expected_training_source_binding: str,
    anchor_floor: Tensor | None = None,
    common_head: nn.Module | None = None,
    condition_head: nn.Module | None = None,
    anchor_contract: dict | None = None,
    expected_anchor_contract_sha256: str | None = None,
    expected_training_split_sha256: str | None = None,
    reconstruction: dict | None = None,
    reconstruction_weight: float = 0.0,
    expected_reconstruction_source_binding: str | None = None,
    calibration_regularization_weight: float = 0.0,
    star_weights: Tensor | None = None,
    assume_diagonal_catalog_covariance: bool = False,
    require_fit_role: bool = True,
):
    """Combine one shared-latent NLL with the original MAE reconstruction path.

    The labeled and masked reconstruction batches may contain different stars.
    Both streams must come from the caller's pinned training sources and have
    aligned FIT roles. For reconstruction, the caller owns mask construction
    and must validate source IDs and split membership before batching. An
    independently audited anchor declaration is digest-bound before training
    a common correction; hashing a declaration does not verify its contents or
    prove that the physical scale is identified. The optimizer/checkpoint
    code must persist source, split, normalization and anchor provenance.
    Optional positive per-star weights change only the supervised NLL; the
    caller must bind and verify their sampling design and the reconstruction
    stream separately. Weighted batches cannot contain unlabeled stars.
    """
    import hashlib
    import json
    import math

    from torchregress.losses import MarginalizedMultiSourceGaussianNLLLoss

    if not math.isfinite(reconstruction_weight) or reconstruction_weight < 0:
        raise ValueError("Reconstruction weight must be finite and nonnegative")
    if (
        not math.isfinite(calibration_regularization_weight)
        or calibration_regularization_weight < 0
    ):
        raise ValueError(
            "Calibration regularization weight must be finite and nonnegative"
        )
    if require_fit_role and any(role != "fit" for role in batch.get("role", [])):
        raise ValueError("Training objective requires fit-role labeled stars")
    if require_fit_role and len(batch.get("role", [])) != len(batch["input"]):
        raise ValueError("Explicit aligned training roles are required")
    if (
        not _is_source_binding(expected_training_source_binding)
        or batch.get("source_binding") != expected_training_source_binding
    ):
        raise ValueError("Labeled batch differs from the pinned training source")
    if common_head is not None:
        if (
            not isinstance(anchor_contract, dict)
            or anchor_contract.get("independence_audited") is not True
            or not _is_sha256(anchor_contract.get("training_anchor_sha256"))
            or not _is_sha256(anchor_contract.get("independence_evidence_sha256"))
            or not _is_sha256(expected_training_split_sha256)
            or anchor_contract.get("training_split_sha256")
            != expected_training_split_sha256
            or anchor_contract.get("training_source_binding")
            != expected_training_source_binding
            or anchor_contract.get("registry_sha256") != expected_registry_sha256
            or not isinstance(anchor_contract.get("covered_targets"), list)
            or not anchor_contract["covered_targets"]
            or any(
                target not in ("teff", "logg", "fe_h")
                for target in anchor_contract["covered_targets"]
            )
            or not _is_sha256(expected_anchor_contract_sha256)
        ):
            raise ValueError(
                "Common correction requires a source- and split-bound anchor contract"
            )
        digest = hashlib.sha256(
            json.dumps(anchor_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != expected_anchor_contract_sha256:
            raise ValueError("Anchor contract digest differs from its pinned value")
        if calibration_regularization_weight <= 0:
            raise ValueError(
                "Common correction requires positive shrinkage regularization"
            )
    x = batch["input"]
    mean, lower = model(x)
    packed = pack_multisurvey_observations(
        mean,
        batch,
        relative_head,
        source_floor,
        expected_registry_sha256=expected_registry_sha256,
        expected_source_registry=expected_source_registry,
        anchor_floor=anchor_floor,
        common_head=common_head,
        condition_head=condition_head,
        assume_diagonal_catalog_covariance=assume_diagonal_catalog_covariance,
    )
    if star_weights is not None:
        if (
            not isinstance(star_weights, Tensor)
            or not star_weights.is_floating_point()
            or star_weights.requires_grad
        ):
            raise ValueError("Star sampling weights must be fixed tensors")
        if star_weights.shape != (len(mean),):
            raise ValueError("Star sampling weights must align with the batch")
        star_weights = star_weights.to(device=mean.device, dtype=mean.dtype)
        if not torch.isfinite(star_weights).all() or (star_weights <= 0).any():
            raise ValueError("Star sampling weights must be finite and positive")
        if not packed.mask.any(dim=-1).all():
            raise ValueError("Weighted batches require at least one label per star")
    supervised = MarginalizedMultiSourceGaussianNLLLoss()(
        mean, lower, packed, weights=star_weights
    )
    reconstruction_loss = supervised.new_zeros(())
    if reconstruction_weight:
        if reconstruction is None:
            raise ValueError(
                "Positive reconstruction weight requires a separate masked batch"
            )
        if (
            not _is_source_binding(expected_reconstruction_source_binding)
            or reconstruction.get("source_binding")
            != expected_reconstruction_source_binding
        ):
            raise ValueError("Reconstruction batch differs from its pinned source")
        n_reconstruction = len(reconstruction["masked_input"])
        if require_fit_role and (
            len(reconstruction.get("role", [])) != n_reconstruction
            or any(role != "fit" for role in reconstruction["role"])
        ):
            raise ValueError("Reconstruction requires aligned fit-role stars")
        ids = torch.as_tensor(reconstruction.get("source_id", []))
        if (
            ids.ndim != 1
            or len(ids) != n_reconstruction
            or ids.dtype not in (torch.int32, torch.int64)
            or (ids <= 0).any()
            or ids.unique().numel() != len(ids)
        ):
            raise ValueError(
                "Reconstruction requires aligned unique positive source IDs"
            )
        prediction = model.reconstruct(reconstruction["masked_input"])
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        target = reconstruction["target"].to(
            device=prediction.device, dtype=prediction.dtype
        )
        mask = reconstruction["mask"].to(device=prediction.device, dtype=torch.bool)
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("Reconstruction prediction, target and mask must align")
        if (
            not mask.any()
            or not torch.isfinite(target[mask]).all()
            or not torch.isfinite(prediction[mask]).all()
        ):
            raise ValueError("Reconstruction needs finite nonempty masked targets")
        reconstruction_loss = (prediction[mask] - target[mask]).abs().mean()
    regularization = supervised.new_zeros(())
    if calibration_regularization_weight:
        for correction_head in (relative_head, common_head, condition_head):
            if correction_head is not None:
                if not hasattr(correction_head, "regularization"):
                    raise ValueError("Calibration head must expose its regularization")
                regularization = regularization + correction_head.regularization()
    return {
        "total": supervised
        + reconstruction_weight * reconstruction_loss
        + calibration_regularization_weight * regularization,
        "supervised": supervised,
        "calibration_regularization": regularization,
        "reconstruction": reconstruction_loss,
        "mean": mean,
        "scale_tril": lower,
        "packed_observations": packed,
    }
