"""Small matched head candidates; no modification to any running experiment.

The BatchEnsemble head is TabM-style, not the full official TabM architecture.
Member losses are averaged during fitting; member means are averaged for point
prediction. Their spread is not asserted to be calibrated uncertainty.
Gaussian scores describe the *observed label* distribution, including unresolved
survey noise/systematics, not a deconvolved distribution of true abundances.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _PlainHead(nn.Module):
    def __init__(self, input_dim: int, width: int, residual: bool):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.residual = residual
        if residual:
            bottleneck = max(1, width // 4)
            self.block = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, width),
            )
        self.output = nn.Linear(width, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: Tensor) -> Tensor:
        hidden = torch.nn.functional.gelu(self.input(latent))
        if self.residual:
            hidden = hidden + self.block(hidden)
        return self.output(hidden).unsqueeze(1)


class _BatchEnsembleHead(nn.Module):
    def __init__(self, input_dim: int, width: int, members: int):
        super().__init__()
        from torchregress.ensemble.layers import BatchEnsembleLinear

        self.input = BatchEnsembleLinear(input_dim, width, members)
        self.output = BatchEnsembleLinear(width, 2, members)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent: Tensor) -> Tensor:
        return self.output(torch.nn.functional.gelu(self.input(latent)))


def build_prediction_head(
    kind: str, input_dim: int = 256, width: int = 128, *, members: int = 4
) -> nn.Module:
    """Return a head with output [row, member, (mean, raw log variance)]."""
    for name, value in (
        ("input_dim", input_dim),
        ("width", width),
        ("members", members),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if kind in {"mlp", "residual"}:
        return _PlainHead(input_dim, width, kind == "residual")
    if kind == "batchensemble":
        return _BatchEnsembleHead(input_dim, width, members)
    raise ValueError(f"Unknown prediction head: {kind}")


def _validate_output(output: Tensor) -> None:
    if output.ndim != 3 or output.shape[-1] != 2 or output.shape[1] < 1:
        raise ValueError("head output must have shape [row, member, 2]")
    if not output.is_floating_point() or not torch.isfinite(output).all():
        raise ValueError("head output must be finite floating point")


def point_prediction(output: Tensor) -> Tensor:
    """Average member means; this function does not summarize uncertainty."""
    _validate_output(output)
    return output[:, :, 0].mean(dim=1)


def fit_lds_huber(targets: Tensor, spec: dict) -> nn.Module:
    """Fit torchregress LDS once on every standardized native fit label.

    ``kernel_width_bins`` is the library histogram-bin sigma, not a dex bandwidth.
    ``LDSLoss.fit`` uses the declared bin count over the fit-label range.
    """
    expected = {
        "kernel": "gaussian",
        "kernel_width_bins": 2,
        "n_bins": 100,
        "reweight_factor": 1.0,
        "base_loss": "huber",
    }
    actual = {
        "kernel": spec.get("kernel"),
        "kernel_width_bins": spec.get("kernel_width_bins"),
        "n_bins": spec.get("n_bins"),
        "reweight_factor": float(spec.get("reweight_factor", -1)),
        "base_loss": spec.get("base_loss"),
    }
    if actual != expected:
        raise ValueError(f"LDS spec must be {expected}")
    if targets.ndim != 1 or not targets.is_floating_point():
        raise ValueError("LDS fit targets must be a floating-point vector")
    if len(targets) < 2 or not torch.isfinite(targets).all():
        raise ValueError("LDS fit targets must be finite and contain a range")
    from torchregress.losses import LDSLoss

    loss = LDSLoss(
        kernel="gaussian",
        kernel_width=2.0,
        reweight_factor=1.0,
        base_loss="huber",
    )
    loss.fit(targets.detach().reshape(-1).cpu(), n_bins=100)
    return loss


def prediction_loss(
    output: Tensor,
    target: Tensor,
    kind: str,
    *,
    lds: nn.Module | None = None,
    label_variance: Tensor | None = None,
) -> Tensor:
    """Mean member loss with explicit valid, observed-label rows only.

    The caller selects eligible labels first. Missing labels are not imputed.
    NLL/CRPS use exp(raw log variance) with no observational-error floor,
    clipping or arbitrary scatter term; nonfinite/underflow outputs fail.
    Equal-member score averaging is not the score of the Gaussian mixture.
    ``gaussian_nll_label_error`` adds the quoted native variance in the same
    standardized units. That quoted variance is not a total calibrated error.
    ``lds_huber`` applies one LDS fit from all native fit labels, not a batch fit.
    """
    _validate_output(output)
    if target.shape != (len(output),) or target.device != output.device:
        raise ValueError("target must be [row] on the output device")
    if not target.is_floating_point() or not torch.isfinite(target).all():
        raise ValueError("target must contain only finite eligible labels")
    if kind not in {
        "huber",
        "gaussian_nll",
        "gaussian_crps",
        "lds_huber",
        "gaussian_nll_label_error",
    }:
        raise ValueError(f"Unknown prediction loss: {kind}")
    if (lds is None) != (kind != "lds_huber"):
        raise ValueError(
            "LDS Huber requires the fitted LDS module and no other loss does"
        )
    if (label_variance is None) != (kind != "gaussian_nll_label_error"):
        raise ValueError(
            "Quoted label variance belongs only to gaussian_nll_label_error"
        )
    if label_variance is not None and (
        label_variance.shape != (len(output),)
        or label_variance.device != output.device
        or not label_variance.is_floating_point()
        or not torch.isfinite(label_variance).all()
        or not torch.all(label_variance > 0)
    ):
        raise ValueError("label variance must be finite and strictly positive [row]")
    if len(output) == 0:
        return output.sum()
    truth = target[:, None].expand(-1, output.shape[1])
    mean = output[:, :, 0]
    if kind in {"huber", "lds_huber"}:
        if kind == "huber":
            return torch.nn.functional.huber_loss(mean, truth, delta=1.0)
        return lds(mean, truth)
    from torchregress.losses import GaussianCRPSLoss, GaussianNLLLoss

    variance = output[:, :, 1].exp()
    if label_variance is not None:
        variance = variance + label_variance[:, None]
    if not torch.all(torch.isfinite(variance) & (variance > 0)):
        raise ValueError("predicted variance must be finite and strictly positive")
    loss_class = GaussianCRPSLoss if kind == "gaussian_crps" else GaussianNLLLoss
    # Passing variance avoids torchregress's log-variance clipping convention.
    loss = loss_class(log_variance=False, min_variance=0.0, eps=0.0)(
        (mean, variance), truth
    )
    if not torch.isfinite(loss):
        raise ValueError("nonfinite prediction loss; no silent clipping")
    return loss


_CONTRASTIVE_FLOW_SPEC = {
    "n_features": 8,
    "context_dim": 1,
    "flow_type": "nsf",
    "n_transforms": 3,
    "hidden_features": [64, 64],
    "temperature": 1.0,
    "margin": 0.0,
    "negative_shifts": [1, 2, 4, 8, 16, 32, 64, 128],
    "grid_points": 201,
    "grid_margin_standardized": 0.5,
}


class ContrastiveFlowEstimator(nn.Module):
    """p(summary(latent) | FeH) trained by torchregress ContrastiveFlowLoss.

    The point estimate is a grid maximum of that likelihood. It is not
    q(FeH | latent), not a posterior under the training-label frequency, and
    not a Gaussian predictive distribution. Negative contexts are fixed rolls
    of the supervised batch, so the row stream stays matched to the other arms.
    """

    def __init__(self, flow_loss: nn.Module, shifts: tuple[int, ...], n_features: int):
        super().__init__()
        self.summary = nn.Sequential(
            nn.Linear(256, n_features), nn.LayerNorm(n_features)
        )
        self.flow_loss = flow_loss
        self.shifts = shifts

    def criterion(self, latent: Tensor, label: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != 256 or label.shape != (len(latent),):
            raise ValueError(
                "contrastive flow expects [row, 256] latents and [row] labels"
            )
        if label.device != latent.device or not torch.isfinite(latent).all():
            raise ValueError("contrastive flow inputs must be finite and on one device")
        if not label.is_floating_point() or not torch.isfinite(label).all():
            raise ValueError("contrastive flow labels must be finite")
        if len(latent) <= max(self.shifts):
            raise ValueError(
                "contrastive batch must be larger than every negative shift"
            )
        summary = self.summary(latent)
        negatives = torch.stack(
            [torch.roll(label, shift) for shift in self.shifts], dim=1
        ).unsqueeze(-1)
        loss = self.flow_loss(label[:, None], summary, negative_context=negatives)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite contrastive flow loss")
        return loss

    def standardized_mle(self, latent: Tensor, grid: Tensor) -> Tensor:
        if grid.ndim != 1 or grid.numel() < 2 or not torch.isfinite(grid).all():
            raise ValueError(
                "FeH grid must be a finite vector with at least two points"
            )
        if len(latent) == 0:
            return grid.new_empty(0)
        summary = self.summary(latent)
        chosen = []
        for start in range(0, len(summary), 256):
            block = summary[start : start + 256]
            context = grid.view(1, -1, 1).expand(len(block), -1, 1).reshape(-1, 1)
            target = (
                block.unsqueeze(1)
                .expand(-1, grid.numel(), -1)
                .reshape(-1, block.shape[1])
            )
            with torch.enable_grad():
                score = self.flow_loss.log_prob(
                    context.detach(), target.detach()
                ).reshape(-1)
            if score.numel() != len(block) * grid.numel():
                raise ValueError(
                    "contrastive flow log probability is not one value per grid point"
                )
            chosen.append(grid[score.reshape(len(block), grid.numel()).argmax(dim=1)])
        return torch.cat(chosen)


def contrastive_flow_spec(spec: dict) -> dict:
    actual = {
        "n_features": spec.get("n_features"),
        "context_dim": spec.get("context_dim"),
        "flow_type": spec.get("flow_type"),
        "n_transforms": spec.get("n_transforms"),
        "hidden_features": [int(v) for v in spec.get("hidden_features", [])],
        "temperature": float(spec.get("temperature", -1)),
        "margin": float(spec.get("margin", -1)),
        "negative_shifts": [int(v) for v in spec.get("negative_shifts", [])],
        "grid_points": spec.get("grid_points"),
        "grid_margin_standardized": float(spec.get("grid_margin_standardized", -1)),
    }
    if actual != _CONTRASTIVE_FLOW_SPEC:
        raise ValueError(f"contrastive flow spec must be {_CONTRASTIVE_FLOW_SPEC}")
    return actual


def build_contrastive_flow(spec: dict) -> ContrastiveFlowEstimator:
    checked = contrastive_flow_spec(spec)
    from torchregress.losses import create_contrastive_flow_loss

    flow = create_contrastive_flow_loss(
        n_features=checked["n_features"],
        context_dim=checked["context_dim"],
        flow_type=checked["flow_type"],
        n_transforms=checked["n_transforms"],
        hidden_features=checked["hidden_features"],
        temperature=checked["temperature"],
        margin=checked["margin"],
    )
    return ContrastiveFlowEstimator(
        flow,
        tuple(checked["negative_shifts"]),
        checked["n_features"],
    )


def contrastive_grid(labels: Tensor, spec: dict) -> Tensor:
    checked = contrastive_flow_spec(spec)
    margin = checked["grid_margin_standardized"]
    low = float(labels.min()) - margin
    high = float(labels.max()) + margin
    if not high > low:
        raise ValueError("contrastive FeH grid needs a positive range")
    return torch.linspace(
        low, high, checked["grid_points"], device=labels.device, dtype=labels.dtype
    )


def rank_contrast_loss(
    latent: Tensor, target: Tensor, *, temperature: float = 2.0
) -> Tensor:
    """RnC on normalized per-star features, using eligible observed labels.

    One row represents one star. Do not repeat each star for ensemble members:
    the population, pair set and temperature must match between compared arms.
    """
    if latent.ndim != 2 or target.shape != (len(latent),):
        raise ValueError("RnC expects [row, latent] and [row] labels")
    if target.device != latent.device:
        raise ValueError("RnC features and target must use the same device")
    if not latent.is_floating_point() or not target.is_floating_point():
        raise ValueError("RnC inputs must be floating point")
    if not torch.isfinite(latent).all() or not torch.isfinite(target).all():
        raise ValueError("RnC inputs must be finite eligible rows")
    if len(latent) < 2:
        return latent.sum() * 0
    from torchregress.losses import RankNContrastLoss

    features = torch.nn.functional.normalize(latent, dim=-1)
    return RankNContrastLoss(temperature=temperature)(features, target)
