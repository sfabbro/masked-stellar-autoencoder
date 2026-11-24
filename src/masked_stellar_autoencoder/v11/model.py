"""v11 encoder: a Perceiver-style set encoder over the measured quantities of a star.

Tokens. One token per *visible* measured quantity: a magnitude, an XP coefficient or a
conditioning value (parallax, pm, SFD E(B-V)). A token is

    h = e_band + W_v phi(v) + [W_e phi(log s) | e_no_error]
    token = h + MLP(LN(h))

with ``v`` the robust-scaled value, ``s`` its scaled error, ``phi`` learned Fourier
features shared by all bands, and ``e_band`` the band embedding: a learned id vector +
survey vector + kind vector (magnitude / XP-BP / XP-RP / conditioning) + a projection of
Fourier features of log(pivot wavelength) (magnitudes) or of the coefficient index (XP).
The wavelength term gives a band never seen in training (DR4, Pensus) a sensible
embedding. Absent and masked quantities are not tokens: rows are padded to the longest
visible set and padding is excluded by the attention mask, so there is no fill value.

Encoder. ``M`` learned latents read the tokens by cross-attention (at the layers in
``cross_attn_layers``, default 0 and n_layers/2) and are refined by pre-LN self-attention
blocks. Cost is linear in the number of tokens (one K/V projection per token and read)
and the latent stack does not depend on it, which suits batches mixing 140-token XP rows
with 5-25-token photometry-only rows. The encoder is permutation invariant over tokens.

Latent. Attention pooling (one learned query) plus the latent mean -> LN -> Linear ->
LayerNorm gives ``z`` (``latent_dim``, 256 by default: the size every existing head
takes).

Decoder. Reconstructs any band from ``z`` alone (``z`` is the bottleneck), queried by
band id: a per-star trunk ``g(z)`` (MLP, width ``decoder_hidden``) and a per-band readout
``w_b, b_b`` generated from the decoder's own band embedding, ``x_hat_b = <g(z), w_b> +
b_b``. A query costs one dot product, so decoding all 110 XP coefficients for the
filter-integral term is cheap, and new bands get readouts from their embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .schema import N_XP, BandRegistry


class Fourier(nn.Module):
    """``[sin(2 pi f x), cos(2 pi f x), x]`` with learnable log-spaced frequencies."""

    def __init__(self, n: int, lo: float, hi: float, learnable: bool = True):
        super().__init__()
        f = torch.logspace(math.log10(lo), math.log10(hi), n)
        self.freq = nn.Parameter(f, requires_grad=learnable)

    @property
    def dim(self) -> int:
        return 2 * self.freq.numel() + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = 2 * math.pi * x.unsqueeze(-1) * self.freq.to(x.dtype)
        return torch.cat([a.sin(), a.cos(), x.unsqueeze(-1)], dim=-1)


class BandEmbedding(nn.Module):
    """Embedding of every registry band: id + survey + kind + position features."""

    def __init__(self, registry: BandRegistry, d: int, n_freq: int):
        super().__init__()
        n = len(registry)
        self.id = nn.Embedding(n, d)
        self.survey = nn.Embedding(len(registry.surveys), d)
        self.kind = nn.Embedding(4, d)
        for m in (self.id, self.survey, self.kind):
            nn.init.normal_(m.weight, std=0.02)
        pos = torch.zeros(n)
        is_mag = torch.zeros(n, dtype=torch.bool)
        for b in registry.bands:
            if b.kind == "mag":
                pos[b.index] = math.log10(b.wavelength_nm / 100.0) / 2.5
                is_mag[b.index] = True
            elif b.kind == "xp":
                pos[b.index] = b.xp_coefficient / (N_XP - 1)
        self.register_buffer(
            "survey_index", torch.as_tensor(registry.survey_index()), persistent=False
        )
        self.register_buffer(
            "kind_index", torch.as_tensor(registry.kind_index()), persistent=False
        )
        self.register_buffer("pos", pos, persistent=False)
        self.register_buffer("is_mag", is_mag, persistent=False)
        self.register_buffer(
            "is_xp", self.kind_index.ge(1) & self.kind_index.le(2), persistent=False
        )
        self.ff = Fourier(n_freq, 0.5, 32.0, learnable=False)
        self.pos_mag = nn.Linear(self.ff.dim, d, bias=False)
        self.pos_xp = nn.Linear(self.ff.dim, d, bias=False)

    def table(self) -> torch.Tensor:
        """``[n_bands, d]``."""
        f = self.ff(self.pos)
        pos = (
            self.pos_mag(f) * self.is_mag[:, None]
            + self.pos_xp(f) * self.is_xp[:, None]
        )
        return (
            self.id.weight
            + self.survey(self.survey_index)
            + self.kind(self.kind_index)
            + pos
        )


class TokenEncoder(nn.Module):
    def __init__(self, d: int, n_freq: int):
        super().__init__()
        self.ff_v = Fourier(n_freq, 0.05, 20.0)
        self.ff_e = Fourier(n_freq, 0.02, 2.0)
        self.lin_v = nn.Linear(self.ff_v.dim, d)
        self.lin_e = nn.Linear(self.ff_e.dim, d)
        self.no_error = nn.Parameter(torch.zeros(d))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, value, error, band_emb):
        has_err = torch.isfinite(error) & (error > 0)
        u = (
            torch.where(has_err, error, torch.ones_like(error)).log().clamp(-12.0, 4.0)
            / 4.0
        )
        e_tok = torch.where(
            has_err.unsqueeze(-1), self.lin_e(self.ff_e(u)), self.no_error
        )
        h = band_emb + self.lin_v(self.ff_v(value)) + e_tok
        return h + self.mlp(h)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.h = n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.dropout = dropout

    def forward(self, x, ctx, mask=None):
        b, tq, d = x.shape
        tk = ctx.shape[1]
        q = self.q(x).view(b, tq, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(b, tk, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        attn_mask = None if mask is None else mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o(out.transpose(1, 2).reshape(b, tq, d))


class MLP(nn.Module):
    def __init__(self, d: int, ratio: float, dropout: float = 0.0):
        super().__init__()
        h = int(d * ratio)
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN block: optional cross-attention to tokens, self-attention, MLP."""

    def __init__(self, d, n_heads, ratio, dropout, cross: bool):
        super().__init__()
        self.cross = cross
        if cross:
            self.ln_q = nn.LayerNorm(d)
            self.ln_kv = nn.LayerNorm(d)
            self.xattn = Attention(d, n_heads, dropout)
            self.ln_xm = nn.LayerNorm(d)
            self.xmlp = MLP(d, ratio, dropout)
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d, ratio, dropout)

    def forward(self, lat, tok, mask):
        if self.cross:
            lat = lat + self.xattn(self.ln_q(lat), self.ln_kv(tok), mask)
            lat = lat + self.xmlp(self.ln_xm(lat))
        h = self.ln1(lat)
        lat = lat + self.attn(h, h)
        return lat + self.mlp(self.ln2(lat))


class QueryDecoder(nn.Module):
    def __init__(
        self,
        registry: BandRegistry,
        latent_dim: int,
        hidden: int,
        n_layers: int,
        n_freq: int,
    ):
        super().__init__()
        self.bands = BandEmbedding(registry, hidden, n_freq)
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden), nn.GELU()]
        for _ in range(n_layers):
            layers.append(_Residual(hidden))
        layers.append(nn.LayerNorm(hidden))
        self.trunk = nn.Sequential(*layers)
        self.query = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden + 1),
        )
        nn.init.normal_(self.query[-1].weight, std=0.02 / math.sqrt(hidden))
        nn.init.zeros_(self.query[-1].bias)

    def readout(self):
        q = self.query(self.bands.table())
        return q[:, :-1], q[:, -1]

    def features(self, z):
        return self.trunk(z)

    def pairs(self, g, rows, bands, readout=None):
        w, b = readout if readout is not None else self.readout()
        return (g[rows] * w[bands]).sum(-1) + b[bands]

    def dense(self, g, bands, readout=None):
        w, b = readout if readout is not None else self.readout()
        return g @ w[bands].T + b[bands]


class _Residual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )

    def forward(self, x):
        return x + self.net(x)


class MSAv11(nn.Module):
    """Encoder + decoder. Inputs are dense ``[N, n_bands]`` physical values and errors
    (NaN absent) and a boolean ``visible`` mask; the robust scaler is a buffer, so the
    same input contract serves pretraining, fine-tuning and inference."""

    def __init__(self, registry: BandRegistry, cfg: ModelConfig):
        super().__init__()
        n = len(registry)
        self.cfg = cfg
        self.band_names = registry.names
        self.register_buffer("center", torch.zeros(n))
        self.register_buffer("scale", torch.ones(n))
        d = cfg.d_model
        self.bands = BandEmbedding(registry, d, cfg.band_frequencies)
        self.tokens = TokenEncoder(d, cfg.value_frequencies)
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, d) * 0.02)
        cross = set(cfg.cross_attn_layers)
        self.blocks = nn.ModuleList(
            Block(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, i in cross)
            for i in range(cfg.n_layers)
        )
        self.ln_out = nn.LayerNorm(d)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = Attention(d, cfg.n_heads)
        self.to_z = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, cfg.latent_dim))
        self.ln_z = nn.LayerNorm(cfg.latent_dim)
        self.decoder = QueryDecoder(
            registry,
            cfg.latent_dim,
            cfg.decoder_hidden,
            cfg.decoder_layers,
            cfg.band_frequencies,
        )

    # ------------------------------------------------------------- scaling
    def set_scaler(self, center, scale) -> None:
        c = torch.as_tensor(center, dtype=torch.float32)
        s = torch.as_tensor(scale, dtype=torch.float32)
        if (
            c.shape != self.center.shape
            or s.shape != self.scale.shape
            or not (s > 0).all()
        ):
            raise ValueError("scaler shape mismatch or non-positive scale")
        self.center.copy_(c)
        self.scale.copy_(s)

    def scale_inputs(self, values, errors):
        return (values - self.center) / self.scale, errors / self.scale

    def unscale(self, x_scaled, bands):
        return x_scaled * self.scale[bands] + self.center[bands]

    # ------------------------------------------------------------- encoder
    def pack(self, v_s, e_s, visible):
        """Left-pack visible entries: ``(band [B,T], value, error, mask)``."""
        t = max(int(visible.sum(1).max()), 1)
        order = torch.argsort((~visible).to(torch.int8), dim=1, stable=True)[:, :t]
        mask = visible.gather(1, order)
        val = torch.where(mask, v_s.gather(1, order), 0.0)
        err = torch.where(mask, e_s.gather(1, order), torch.nan)
        return order, val, err, mask

    def encode_scaled(self, v_s, e_s, visible):
        if bool((visible.sum(1) == 0).any()):
            raise ValueError("a row has no visible token")
        band, val, err, mask = self.pack(v_s, e_s, visible)
        table = self.bands.table()
        tok = self.tokens(val, err, table[band])
        lat = self.latents.unsqueeze(0).expand(tok.shape[0], -1, -1)
        for blk in self.blocks:
            if (
                self.cfg.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                lat = checkpoint(blk, lat, tok, mask, use_reentrant=False)
            else:
                lat = blk(lat, tok, mask)
        lat = self.ln_out(lat)
        pooled = self.pool(self.pool_q.expand(lat.shape[0], -1, -1), lat).squeeze(1)
        return self.ln_z(self.to_z(torch.cat([pooled, lat.mean(1)], dim=-1)))

    def encode(self, values, errors, visible=None):
        """``z`` from physical inputs; ``visible`` defaults to every present value."""
        if visible is None:
            visible = torch.isfinite(values)
        v_s, e_s = self.scale_inputs(values, errors)
        return self.encode_scaled(v_s, e_s, visible & torch.isfinite(values))


def parameter_count(module: nn.Module) -> dict[str, int]:
    out = {"total": sum(p.numel() for p in module.parameters())}
    if isinstance(module, MSAv11):
        dec = sum(p.numel() for p in module.decoder.parameters())
        out["decoder"] = dec
        out["encoder"] = out["total"] - dec
    return out
