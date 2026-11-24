"""Throughput and memory estimate of a v11 configuration (measured FLOPs and saved
activations on CPU, projected to an H100 MIG 1g.12gb slice).

FLOPs: ``torch.utils.flop_counter.FlopCounterMode`` over one forward + backward of a
micro-batch of each row type, with masks drawn by the configured masking (so the token
counts are the real ones after masking). Activation memory: bytes of every tensor
autograd saves for backward (``saved_tensors_hooks``), measured in float32 and halved
for bf16 autocast (parameters, LayerNorm statistics and softmax outputs kept in float32
make this a slight underestimate; a 1.25 safety factor is applied). Parameter memory:
weights, gradients and two AdamW moments in float32.

Projection: the MIG 1g slice of an H100 has 16 of the 132 SMs, so ~1/8.25 of the
989 TFLOPS dense bf16 peak (H100 SXM; ~0.84x that for the NVL clocks) and 1/8 of the
memory bandwidth. Achieved model FLOPs utilisation for d=256 matmuls and short
attention is assumed 25-45 %. Rows/s = MFU x peak / (train FLOPs per row).
"""

from __future__ import annotations

import math

import torch
from torch.utils.flop_counter import FlopCounterMode

from .config import PretrainConfig
from .masking import BandGroups, sample_masks
from .train import Pretrainer

H100_SXM_BF16_DENSE = 989.4e12
MIG_1G_FRACTION = 16 / 132
REC_ROWS = 1_358_782_328
REC_XP_ROWS = 164_400_000


def _synthetic_rows(reg, n: int, kind: str, gen: torch.Generator):
    """Dense rows typical of REC: 'X' = Gaia + XP + optical/NIR/MIR survey bands;
    'P' = Gaia + survey bands without XP (v1a-like coverage); 'Pm' = the same with the
    full-wavelength master's coverage."""
    nb = len(reg)
    v = torch.full((n, nb), float("nan"))
    e = torch.full((n, nb), float("nan"))
    names_v1a = ["gaia_g", "gaia_bp", "gaia_rp", "ps1_g", "ps1_r", "ps1_i", "ps1_z", "ps1_y",
                 "tmass_j", "tmass_h", "tmass_ks", "catwise_w1", "catwise_w2", "skymapper_g",
                 "skymapper_r", "skymapper_i", "parallax", "pmra", "pmdec", "sfd_ebv"]  # fmt: skip
    names_master = names_v1a + ["ls_s_g", "ls_s_r", "ls_s_i", "ls_s_z", "ls_w1", "ls_w2",
                                "vhs_j", "vhs_ks", "uhs_j", "uhs_k", "allwise_w3", "galex_nuv",
                                "des_g", "des_r", "des_i", "des_z", "sdss_u", "sdss_g", "sdss_r"]  # fmt: skip
    names = names_master if kind in ("Pm", "Xm") else names_v1a
    idx = torch.as_tensor([reg.index(x) for x in names])
    v[:, idx] = torch.randn(n, len(idx), generator=gen)
    e[:, idx] = 0.02
    if kind.startswith("X"):
        xi = torch.as_tensor(reg.xp_indices)
        v[:, xi] = torch.randn(n, len(xi), generator=gen)
        e[:, xi] = 0.05
    return v, e


def _saved_bytes(fn):
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, total[0]


def estimate(cfg: PretrainConfig, rows: int = 256, device: str = "cpu") -> dict:
    """FLOP / memory / throughput estimate. Writes nothing: the Pretrainer gets a
    throwaway output directory, never ``run.output_dir`` (an /arc path would otherwise
    be created on the host running the estimate)."""
    import copy
    import tempfile

    with tempfile.TemporaryDirectory(prefix="msa_v11_estimate_") as tmp:
        cfg = copy.deepcopy(cfg)
        cfg.run.output_dir = tmp
        return _estimate(cfg, rows, device)


def _estimate(cfg: PretrainConfig, rows: int, device: str) -> dict:
    tr = Pretrainer(cfg, device=device)
    reg = tr.comp.registry
    groups = BandGroups(reg, tr.device)
    gen = torch.Generator().manual_seed(0)
    mgen = torch.Generator(device=tr.device).manual_seed(0)
    p_model = sum(p.numel() for p in tr.comp.model.parameters())
    p_loss = sum(p.numel() for p in tr.comp.recon.parameters()) + (
        sum(p.numel() for p in tr.comp.fi.parameters()) if tr.comp.fi else 0
    )
    out = {
        "parameters": {"model": p_model, "loss": p_loss},
        "param_optimizer_bytes": 16 * (p_model + p_loss),
        "rows": {},
    }
    peak = H100_SXM_BF16_DENSE * MIG_1G_FRACTION
    for kind in ("X", "P", "Pm", "Xm"):
        v, e = _synthetic_rows(reg, rows, kind, gen)
        v, e = v.to(tr.device), e.to(tr.device)
        k = tr.comp.operator.n_bands + 2 if tr.comp.operator is not None else 0
        xs = torch.full((rows, k), 0.5, device=tr.device)
        mb = {
            "values": v,
            "errors": e,
            "xp_syn": xs,
            "xp_var": xs * 1e-4,
            "source_id": torch.arange(rows),
        }
        plan = sample_masks(v, groups, cfg.masking, mgen)
        norm = tr.counts(mb, plan)
        tr.comp.model.train()
        res = {}
        for ckpt in (False, True):
            tr.comp.model.cfg.gradient_checkpointing = ckpt
            with FlopCounterMode(display=False) as fc:
                loss, _ = tr.micro_loss(mb, plan, norm)
                loss.backward()
            flops = fc.get_total_flops()
            _, saved = _saved_bytes(lambda: tr.micro_loss(mb, plan, norm)[0])
            tr.comp.model.zero_grad(set_to_none=True)
            res[ckpt] = (flops, saved)
        tr.comp.model.cfg.gradient_checkpointing = cfg.model.gradient_checkpointing
        flops = res[False][0] / rows
        tokens = float(plan.visible.sum(1).float().mean())
        targets = float(plan.target.sum(1).float().mean())
        row = {
            "visible_tokens_mean": tokens,
            "targets_mean": targets,
            "train_gflop_per_row": flops / 1e9,
            "train_gflop_per_row_checkpointed": res[True][0] / rows / 1e9,
            "activation_bytes_per_row_fp32": res[False][1] / rows,
            "activation_bytes_per_row_fp32_checkpointed": res[True][1] / rows,
        }
        for mfu in (0.25, 0.45):
            row[f"rows_per_s_mfu{int(mfu * 100)}"] = mfu * peak / flops
        bf16 = 1.25 * 0.5
        budget = 12e9 * 0.85 - out["param_optimizer_bytes"]
        row["max_rows_per_microbatch_bf16"] = int(
            budget / (bf16 * res[False][1] / rows)
        )
        row["max_rows_per_microbatch_bf16_checkpointed"] = int(
            budget / (bf16 * res[True][1] / rows)
        )
        out["rows"][kind] = row
    fx = REC_XP_ROWS / REC_ROWS
    mix = {}
    for tag, xk, pk in (("v1a", "X", "P"), ("master", "Xm", "Pm")):
        gf = (
            fx * out["rows"][xk]["train_gflop_per_row"]
            + (1 - fx) * out["rows"][pk]["train_gflop_per_row"]
        )
        mix[tag] = {
            "train_gflop_per_row": gf,
            "rows_per_s_mfu25": 0.25 * peak / (gf * 1e9),
            "rows_per_s_mfu45": 0.45 * peak / (gf * 1e9),
            "hours_per_rec_pass_mfu25": REC_ROWS / (0.25 * peak / (gf * 1e9)) / 3600,
            "hours_per_rec_pass_mfu45": REC_ROWS / (0.45 * peak / (gf * 1e9)) / 3600,
        }
    out["rec_mix"] = {"xp_fraction": fx, **mix}
    out["assumptions"] = {
        "peak_bf16_tflops_mig_1g": peak / 1e12,
        "mfu_range": [0.25, 0.45],
        "bf16_activation_factor": 0.5,
        "safety": 1.25,
        "memory_budget_bytes": 12e9 * 0.85,
        "steps_per_pass_at_batch": math.ceil(REC_ROWS / cfg.train.batch_rows),
    }
    return out
