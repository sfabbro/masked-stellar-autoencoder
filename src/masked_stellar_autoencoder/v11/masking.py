"""Masking of the v11 pretraining: per-token masking, whole-survey dropout, whole-XP
dropout and conditioning hiding, all read from ``config.masking``.

A masked quantity is removed from the encoder input exactly as an absent one is (it is
not a token), so a photometry-only row and an XP row with its XP dropped look the same
to the encoder. Reconstruction targets are the masked, present, reconstructable
entries (conditioning tokens are never targets).

Order of operations per row:

1. whole-XP dropout with probability ``xp_dropout`` (rows with XP);
2. whole-survey dropout: every present magnitude survey independently with
   probability ``survey_dropout``;
3. per-token masking of the remaining magnitudes (``token_ratio``) and, when the XP
   block was kept, of each XP coefficient (``xp_token_ratio``);
4. conditioning groups hidden: astrometry (parallax, pmra, pmdec together) with
   ``astrometry_hide``, SFD with ``sfd_hide``;
5. rows left with fewer than ``min_visible`` visible reconstructable tokens get
   randomly chosen masked tokens back (fewer if the row has fewer present).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import MaskingConfig
from .schema import BandRegistry


@dataclass
class MaskPlan:
    visible: torch.Tensor  # [N, n_bands] bool: tokens the encoder sees
    target: torch.Tensor  # [N, n_bands] bool: reconstruction targets


class BandGroups:
    """Index tensors of a registry, on one device."""

    def __init__(self, registry: BandRegistry, device=None):
        kind = torch.as_tensor(registry.kind_index(), device=device)
        self.n = len(registry)
        self.is_mag = kind == 0
        self.is_xp = (kind == 1) | (kind == 2)
        self.is_cond = kind == 3
        self.reconstruct = ~self.is_cond
        self.survey = torch.as_tensor(registry.survey_index(), device=device)
        self.n_surveys = len(registry.surveys)
        astro = registry.indices(survey="gaia_astrometry")
        sfd = registry.indices(survey="sfd")
        self.is_astrometry = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.is_astrometry[torch.as_tensor(astro, device=device)] = True
        self.is_sfd = torch.zeros(self.n, dtype=torch.bool, device=device)
        self.is_sfd[torch.as_tensor(sfd, device=device)] = True
        mag_surveys = torch.unique(self.survey[self.is_mag])
        self.mag_survey_onehot = (
            self.survey[None, :] == mag_surveys[:, None]
        ) & self.is_mag[None, :]  # [S_mag, n_bands]


def _rand(shape, generator, device):
    return torch.rand(shape, generator=generator, device=device)


def sample_masks(
    values: torch.Tensor,
    groups: BandGroups,
    cfg: MaskingConfig,
    generator: torch.Generator | None = None,
) -> MaskPlan:
    dev = values.device
    n = values.shape[0]
    present = torch.isfinite(values)
    hide = torch.zeros_like(present)
    # 1. whole-XP dropout
    has_xp = (present & groups.is_xp).any(1)
    xp_drop = has_xp & (_rand(n, generator, dev) < cfg.xp_dropout)
    hide |= xp_drop[:, None] & groups.is_xp[None, :]
    # 2. whole-survey dropout (magnitude surveys)
    if cfg.survey_dropout > 0:
        onehot = groups.mag_survey_onehot.to(torch.float32)  # [S, n_bands]
        drop_s = _rand((n, onehot.shape[0]), generator, dev) < cfg.survey_dropout
        hide |= (drop_s.to(torch.float32) @ onehot) > 0
    # 3. per-token masking
    u = _rand(present.shape, generator, dev)
    hide |= groups.is_mag[None, :] & (u < cfg.token_ratio)
    hide |= groups.is_xp[None, :] & (u < cfg.xp_token_ratio)
    # 4. conditioning groups
    hide |= (_rand(n, generator, dev) < cfg.astrometry_hide)[
        :, None
    ] & groups.is_astrometry[None, :]
    hide |= (_rand(n, generator, dev) < cfg.sfd_hide)[:, None] & groups.is_sfd[None, :]
    hide &= present
    # 5. keep at least min_visible visible reconstructable tokens
    recon_present = present & groups.reconstruct[None, :]
    n_vis = (recon_present & ~hide).sum(1)
    deficit = (cfg.min_visible - n_vis).clamp_min(0)
    if bool((deficit > 0).any()):
        cand = recon_present & hide
        score = torch.where(cand, _rand(present.shape, generator, dev), -1.0)
        order = torch.argsort(score, dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1, order, torch.arange(order.shape[1], device=dev).expand_as(order)
        )
        restore = cand & (rank < deficit[:, None])
        hide &= ~restore
    visible = present & ~hide
    # a row whose only present tokens are conditioning ones keeps them visible
    empty = visible.sum(1) == 0
    if bool(empty.any()):
        visible[empty] = present[empty]
        hide[empty] = False
    target = hide & present & groups.reconstruct[None, :]
    return MaskPlan(visible=visible, target=target)


def view(
    values: torch.Tensor, groups: BandGroups, hide: tuple[str, ...] = ()
) -> torch.Tensor:
    """Deterministic encoder view: every present value except the named groups
    (``"xp"``, ``"astrometry"``, ``"sfd"``). S3 needs ``hide=("astrometry",)``."""
    visible = torch.isfinite(values)
    known = {
        "xp": groups.is_xp,
        "astrometry": groups.is_astrometry,
        "sfd": groups.is_sfd,
    }
    for h in hide:
        if h not in known:
            raise ValueError(f"unknown view group {h!r}; have {sorted(known)}")
        visible &= ~known[h][None, :]
    return visible
