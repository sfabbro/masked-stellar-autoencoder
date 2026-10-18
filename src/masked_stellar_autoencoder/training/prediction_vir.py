"""VIR heads for a caller-owned matched MSA continuation loop.

The standalone VIRRegressor.fit loop is deliberately never invoked. Statistics
must be refreshed from fit-only latents by the caller, between optimizer steps.
This adapter uses the statistic/recolour implementation pinned in torchregress
f9f71f2; its output spread is a Student-t predictive quantity, not Gaussian sigma.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class VIRPredictionHead(nn.Module):
    """Matched probabilistic heads, differing only in density/recolour treatment."""

    MODES = {"unweighted_nig", "density_nig", "vir"}

    def __init__(
        self,
        fit_labels,
        *,
        mode="vir",
        input_dim=256,
        latent_dim=64,
        n_bins=20,
        kernel_sigma=1.0,
        smoothing_shrinkage=5.0,
        kl_weight=1e-3,
        reg_weight=0.01,
    ):
        super().__init__()
        from torchregress.algorithms import VIRLoss

        if mode not in self.MODES:
            raise ValueError(f"Unknown VIR mode: {mode}")
        if fit_labels.ndim != 1 or not torch.isfinite(fit_labels).all():
            raise ValueError("VIR requires finite one-dimensional fit labels")
        if any(
            not math.isfinite(value) or value < 0
            for value in (smoothing_shrinkage, kl_weight)
        ):
            raise ValueError("VIR shrinkage and KL weights must be nonnegative")
        self.mode = mode
        self.latent_dim = latent_dim
        self.smoothing_shrinkage = float(smoothing_shrinkage)
        self.kl_weight = float(kl_weight)
        self.latent_head = nn.Linear(input_dim, 2 * latent_dim)
        self.nig_head = nn.Linear(latent_dim, 3)
        with torch.no_grad():
            self.latent_head.bias[latent_dim:].fill_(-4.0)
            self.nig_head.weight.zero_()
            self.nig_head.bias.zero_()
        self.loss_fn = VIRLoss(
            n_bins=n_bins, kernel_sigma=kernel_sigma, reg_weight=reg_weight
        ).fit(fit_labels)
        self.register_buffer("bin_counts", torch.zeros(n_bins))
        for name in (
            "bin_mean",
            "bin_mean_variance",
            "bin_covariance",
            "smooth_mean",
            "smooth_mean_variance",
            "smooth_covariance",
        ):
            self.register_buffer(name, torch.zeros(n_bins, latent_dim))
        self.register_buffer("statistics_ready", torch.tensor(False))

    def encode(self, latent):
        mean, raw_variance = self.latent_head(latent).chunk(2, dim=-1)
        return mean, F.softplus(raw_variance) + 1e-8

    @torch.no_grad()
    def refresh_statistics(self, fit_latents, fit_labels, *, chunk_size=512):
        """Refresh deterministic sufficient statistics using ONLY strict-fit rows.

        No optimizer, RNG, or encoder mode is altered here. Caller supplies
        current encoder eval-mode latents; caller owns provenance and row bans.
        """
        from torchregress.algorithms import VIRRegressor

        if len(fit_latents) != len(fit_labels) or not len(fit_labels):
            raise ValueError("Need aligned nonempty fit latents and labels")
        if (
            not torch.isfinite(fit_latents).all()
            or not torch.isfinite(fit_labels).all()
        ):
            raise ValueError("Statistics require finite fit rows")
        count = torch.zeros_like(self.bin_counts)
        sum_mean = torch.zeros_like(self.bin_mean)
        sum_var = torch.zeros_like(sum_mean)
        sum_second = torch.zeros_like(sum_mean)
        for start in range(0, len(fit_latents), chunk_size):
            mean, variance = self.encode(fit_latents[start : start + chunk_size])
            index = self.loss_fn.bin_index(fit_labels[start : start + chunk_size])
            sum_mean.index_add_(0, index, mean)
            sum_var.index_add_(0, index, variance)
            sum_second.index_add_(0, index, variance + mean.square())
            count.index_add_(0, index, torch.ones_like(index, dtype=count.dtype))
        stats = VIRRegressor._summarise_bin_statistics(
            sum_mean, sum_var, sum_second, count
        )
        smooth = VIRRegressor._smooth_bin_statistics(
            stats, self.loss_fn.smoothing_matrix().to(sum_mean), count
        )
        for name, value in zip(
            ("bin_mean", "bin_mean_variance", "bin_covariance"), stats, strict=True
        ):
            getattr(self, name).copy_(value)
        for name, value in zip(
            ("smooth_mean", "smooth_mean_variance", "smooth_covariance"),
            smooth,
            strict=True,
        ):
            getattr(self, name).copy_(value)
        self.bin_counts.copy_(count)
        self.statistics_ready.fill_(True)

    def _recolour(self, mean, variance, labels):
        from torchregress.algorithms import VIRRegressor

        return VIRRegressor._recolour(
            mean,
            variance,
            self.loss_fn.bin_index(labels),
            (self.bin_mean, self.bin_mean_variance, self.bin_covariance),
            (self.smooth_mean, self.smooth_mean_variance, self.smooth_covariance),
            self.bin_counts,
            self.smoothing_shrinkage,
        )

    def objective(self, latent, labels):
        """Stochastic fit loss; identical latent/KL architecture for all modes."""
        mean, variance = self.encode(latent)
        raw_mean, raw_variance = mean, variance
        if self.mode == "vir" and bool(self.statistics_ready):
            mean, variance = self._recolour(mean, variance, labels)
        sample = mean + variance.sqrt() * torch.randn_like(mean)
        raw = self.nig_head(sample)
        weight = (
            torch.ones_like(labels)
            if self.mode == "unweighted_nig"
            else self.loss_fn.importance_weights(labels)
        )
        predictive = self.loss_fn(raw, labels, importance_weight=weight)
        kl = 0.5 * (raw_mean.square() + raw_variance - raw_variance.log() - 1).mean()
        return predictive + self.kl_weight * kl

    def posterior(self, latent):
        """Label-free deterministic plug-in inference; no heldout-bin oracle."""
        mean, variance = self.encode(latent)
        raw = self.nig_head(mean)
        posterior = self.loss_fn.posterior_parameters(raw)
        if self.mode == "vir" and bool(self.statistics_ready):
            used_mean, _ = self._recolour(mean, variance, posterior.gamma.detach())
            raw = self.nig_head(used_mean)
            posterior = self.loss_fn.posterior_parameters(raw)
        weight = (
            torch.ones_like(posterior.gamma)
            if self.mode == "unweighted_nig"
            else self.loss_fn.importance_weights(posterior.gamma)
        )
        return self.loss_fn.posterior_parameters(raw, weight)

    def forward(self, latent):
        """Return standardised predictive mean/log variance in member layout.

        The second channel is a Student-t variance, NOT Gaussian log variance;
        scientific scoring must reconstruct the NIG posterior via posterior().
        """
        mean, std = self.loss_fn.predictive_mean_std(self.posterior(latent))
        return torch.stack((mean, 2 * std.log()), dim=-1).unsqueeze(1)
