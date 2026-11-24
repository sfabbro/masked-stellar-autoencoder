"""v11 pretraining loop (masked heteroscedastic reconstruction + filter-integral term).

One optimiser step accumulates micro-batches until ``train.batch_rows`` rows. Masks for
the whole step are drawn first, so every loss term is normalised by its count over the
step (micro-batching does not change the gradient). bf16 autocast covers the encoder
and decoder on CUDA; the likelihoods run in float32.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import PretrainConfig
from .data import HatsStream, StreamSpec, fit_scalers, scaler_arrays
from .filters import FilterOperator, build_operator, select_fi_bands
from .losses import (
    FilterIntegralLoss,
    ReconstructionLoss,
    calibrate_count,
    observed_xp_valid,
    phot_count,
    sed_count,
)
from .masking import BandGroups, MaskPlan, sample_masks, view
from .model import MSAv11, parameter_count
from .schema import BandRegistry, ContractReader, load_registry

FORMAT = "msa-v11-pretrain-1"
# micro_loss stat -> the count that normalises it
STAT_NORM = {
    "rec_xp": "xp",
    "rec_phot": "phot",
    "fi_calibrate": "calibrate",
    "fi_sed": "sed",
    "fi_phot": "phot_fi",
}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


@dataclass
class Components:
    registry: BandRegistry
    reader: ContractReader
    operator: FilterOperator | None
    model: MSAv11
    recon: ReconstructionLoss
    fi: FilterIntegralLoss | None


def build_components(
    cfg: PretrainConfig, registry: BandRegistry | None = None
) -> Components:
    reg = registry or load_registry()
    reader = ContractReader(
        reg,
        cfg.data.contract,
        survey_tags=cfg.data.survey_tags,
        bands=cfg.data.bands,
        allow_unverified=cfg.data.allow_unverified_columns,
    )
    operator = None
    fi = None
    fic = cfg.filter_integral
    if fic.enabled:
        mapped = {reg.bands[bi].name for bi, *_ in reader.plan}
        cands = fic.bands if fic.bands is not None else sorted(mapped)
        if fic.bands is not None:
            missing = sorted(set(fic.bands) - mapped)
            if missing:
                raise ValueError(
                    f"filter_integral.bands {missing} are not read by {cfg.data.contract}"
                )
        bands = select_fi_bands(
            reg,
            tuple(fic.wavelength_range_nm),
            fic.support_threshold,
            cands,
            strict=fic.bands is not None,
        )
        operator = build_operator(reg, bands, fic.integration_step_nm)
        fi = FilterIntegralLoss(
            len(bands),
            fic.nu,
            fic.sigma_sys_init,
            fic.colour_reference,
            fic.linear_below_fraction,
            fic.max_sigma_xp_mag,
        )
    model = MSAv11(reg, cfg.model)
    groups = BandGroups(reg)
    recon = ReconstructionLoss(
        len(reg),
        groups.is_xp,
        cfg.loss.nu,
        cfg.loss.sigma_model_init,
        cfg.loss.sigma_model_floor,
        cfg.loss.group_weights,
    )
    return Components(reg, reader, operator, model, recon, fi)


def make_stream(
    cfg: PretrainConfig,
    comp: Components,
    partitions,
    *,
    shuffle=True,
    seed=None,
    trees=None,
    assignment=None,
) -> HatsStream:
    d = cfg.data
    spec = StreamSpec(
        trees=trees if trees is not None else [(t.path, t.columns) for t in d.trees],
        assignment=assignment or d.assignment,
        partitions=list(partitions),
        reader=comp.reader,
        registry=comp.registry,
        operator=comp.operator,
        xp_covariance=d.xp_covariance,
        rec_sample_rate=d.rec_sample_rate,
        read_batch_rows=d.read_batch_rows,
        interleave=d.interleave_partitions,
        shuffle_rows=d.shuffle_buffer_rows,
        tokens_per_microbatch=d.tokens_per_microbatch,
        max_rows_per_microbatch=d.max_rows_per_microbatch,
        bucket_edges=d.bucket_edges,
        limit_partitions=d.limit_partitions,
        seed=cfg.run.seed if seed is None else seed,
        shuffle=shuffle,
    )
    return HatsStream(spec)


def param_groups(comp: Components, cfg: PretrainConfig):
    decay, no_decay = [], []
    for name, p in comp.model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not any(k in name for k in ("bands.", "latents", "pool_q")):
            decay.append(p)
        else:
            no_decay.append(p)
    loss_params = list(comp.recon.parameters()) + (
        list(comp.fi.parameters()) if comp.fi else []
    )
    lr = cfg.optim.lr
    return [
        {"params": decay, "weight_decay": cfg.optim.weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {
            "params": loss_params,
            "weight_decay": 0.0,
            "lr": lr * cfg.optim.loss_param_lr_scale,
        },
    ]


def lr_lambda(cfg: PretrainConfig):
    total = cfg.total_steps
    warm = max(cfg.optim.warmup_steps, 1)
    floor = cfg.optim.min_lr_ratio

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        t = min((step - warm) / max(total - warm, 1), 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))

    return f


class Pretrainer:
    def __init__(self, cfg: PretrainConfig, device: str | None = None):
        self.cfg = cfg
        dev = device or cfg.run.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        if cfg.run.num_threads:
            torch.set_num_threads(cfg.run.num_threads)
        torch.manual_seed(cfg.run.seed)
        self.out = Path(cfg.run.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.comp = build_components(cfg)
        self.trees, self.assignment = self._data_paths()
        self.groups = BandGroups(self.comp.registry, self.device)
        self.comp.model.to(self.device)
        self.comp.recon.to(self.device)
        if self.comp.fi is not None:
            self.comp.fi.to(self.device)
            self.A = torch.as_tensor(
                self.comp.operator.matrix, dtype=torch.float32, device=self.device
            )
            self.fi_idx = torch.as_tensor(
                self.comp.operator.band_index, device=self.device
            )
        self.xp_idx = torch.as_tensor(self.comp.registry.xp_indices, device=self.device)
        self.g_idx = self.comp.registry.index("gaia_g")
        self.opt = torch.optim.AdamW(
            param_groups(self.comp, cfg),
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            fused=False,
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda(cfg))
        self.amp = self.device.type == "cuda" and cfg.run.precision == "bf16"
        self.step = 0
        self.rows_seen = 0
        self.epoch = 0
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(cfg.run.seed + 1)
        self.metrics_path = self.out / "metrics.jsonl"

    # ------------------------------------------------------------ setup
    def _data_paths(self):
        d = self.cfg.data
        if d.source == "synthetic":
            from .synthetic import write_synthetic_trees

            root = Path(d.synthetic.directory or self.out / "synthetic")
            if not (root / "assignment" / "partition_info.csv").exists():
                write_synthetic_trees(
                    root,
                    self.comp.registry,
                    n_rows=d.synthetic.n_rows,
                    n_partitions=d.synthetic.n_partitions,
                    xp_fraction=d.synthetic.xp_fraction,
                    seed=d.synthetic.seed,
                )
            return [(str(root / "anchor"), None), (str(root / "xp"), None)], str(
                root / "assignment"
            )
        return [(t.path, t.columns) for t in d.trees], d.assignment

    def stream(self, partitions, shuffle=True, seed=None) -> HatsStream:
        return make_stream(
            self.cfg,
            self.comp,
            partitions,
            shuffle=shuffle,
            seed=seed,
            trees=self.trees,
            assignment=self.assignment,
        )

    def loader(self, stream, workers=None):
        w = self.cfg.data.workers if workers is None else workers
        return torch.utils.data.DataLoader(
            stream,
            batch_size=None,
            num_workers=w,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=4 if w > 0 else None,
            persistent_workers=False,
        )

    def setup_scalers(self) -> dict:
        sc = self.cfg.data.scalers
        path = Path(sc.path) if sc.path else self.out / "scalers.json"
        if path.exists():
            scalers = json.loads(path.read_text())
        else:
            stream = self.stream(
                self.cfg.data.train_partitions, seed=self.cfg.run.seed + 17
            )
            scalers = fit_scalers(
                self.loader(stream), self.comp.registry, sc.fit_rows, sc.min_count
            )
            (self.out / "scalers.json").write_text(json.dumps(scalers, indent=1) + "\n")
        c, s = scaler_arrays(scalers, self.comp.registry)
        self.comp.model.set_scaler(c, s)
        return scalers

    # ------------------------------------------------------------ losses
    def counts(self, mb, plan: MaskPlan) -> dict[str, int]:
        t = plan.target
        xp = self.groups.is_xp[None, :]
        out = {"xp": int((t & xp).sum()), "phot": int((t & ~xp).sum())}
        if self.comp.fi is not None:
            obs = mb["values"][:, self.fi_idx]
            g = mb["values"][:, self.g_idx]
            modes = self.cfg.filter_integral.modes
            ms = self.cfg.filter_integral.max_sigma_xp_mag
            xs, xv = mb["xp_syn"], mb["xp_var"]
            out["calibrate"] = (
                calibrate_count(obs, xs, xv, g, ms) if "calibrate" in modes else 0
            )
            out["sed"] = sed_count(obs, g) if "sed" in modes else 0
            out["phot_fi"] = (
                phot_count(~plan.visible[:, self.fi_idx], xs, xv, g, ms)
                if "phot" in modes
                else 0
            )
        return out

    def micro_loss(self, mb, plan: MaskPlan, norm: dict[str, int]):
        m = self.comp.model
        values, errors = mb["values"], mb["errors"]
        v_s, e_s = m.scale_inputs(values, errors)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            z = m.encode_scaled(v_s, e_s, plan.visible)
            g = m.decoder.features(z)
            readout = m.decoder.readout()
        g = g.float()
        readout = (readout[0].float(), readout[1].float())
        rows, bands = plan.target.nonzero(as_tuple=True)
        pred = m.decoder.pairs(g, rows, bands, readout)
        sums = self.comp.recon(pred, v_s[rows, bands], e_s[rows, bands], bands)
        loss = z.new_zeros((), dtype=torch.float32)
        stats = {}
        for grp in ("xp", "phot"):
            s, _ = sums[grp]
            if norm[grp] > 0:
                loss = loss + self.comp.recon.w[grp] * s / norm[grp]
            stats[f"rec_{grp}"] = float(s.detach())
        fi = self.comp.fi
        if fi is not None:
            lam = self.cfg.filter_integral.weight
            modes = self.cfg.filter_integral.modes
            obs = values[:, self.fi_idx]
            oerr = errors[:, self.fi_idx]
            gobs, gerr = values[:, self.g_idx], errors[:, self.g_idx]
            if "calibrate" in modes and norm["calibrate"]:
                s, _ = fi.calibrate(obs, oerr, mb["xp_syn"], mb["xp_var"], gobs)
                loss = loss + s / norm["calibrate"]
                stats["fi_calibrate"] = float(s.detach())
            if "sed" in modes and norm["sed"]:
                r = (torch.isfinite(obs).any(1) & torch.isfinite(gobs)).nonzero(
                    as_tuple=True
                )[0]
                if len(r):
                    xs = m.decoder.dense(g[r], self.xp_idx, readout)
                    cn = xs * m.scale[self.xp_idx] + m.center[self.xp_idx]
                    syn = cn @ self.A
                    sig = (
                        self.comp.recon.sigma_model()[self.xp_idx]
                        * m.scale[self.xp_idx]
                    ).detach()
                    dvar = (sig * sig)[None, :] @ (self.A * self.A)
                    s, _ = fi.sed(obs[r], oerr[r], syn, dvar, gobs[r], gerr[r])
                    loss = loss + lam * s / norm["sed"]
                    stats["fi_sed"] = float(s.detach())
            if "phot" in modes and norm["phot_fi"]:
                use = ~plan.visible[:, self.fi_idx]
                r = (use.any(1) & torch.isfinite(mb["xp_syn"]).all(1)).nonzero(
                    as_tuple=True
                )[0]
                if len(r):
                    dm = m.decoder.dense(g[r], self.fi_idx, readout)
                    dmag = dm * m.scale[self.fi_idx] + m.center[self.fi_idx]
                    dsig = (
                        self.comp.recon.sigma_model()[self.fi_idx]
                        * m.scale[self.fi_idx]
                    )
                    s, _ = fi.phot(
                        dmag, dsig, use[r], mb["xp_syn"][r], mb["xp_var"][r], gobs[r]
                    )
                    loss = loss + lam * s / norm["phot_fi"]
                    stats["fi_phot"] = float(s.detach())
        return loss, stats

    # ------------------------------------------------------------ loop
    def _to_device(self, mb):
        return {k: v.to(self.device, non_blocking=True) for k, v in mb.items()}

    def _next_step_batches(self, it_holder):
        rows, mbs = 0, []
        empty_epochs = 0
        while rows < self.cfg.train.batch_rows:
            try:
                mb = next(it_holder["it"])
            except StopIteration:
                if it_holder["yielded"] == 0:
                    empty_epochs += 1
                    if empty_epochs > 1:
                        raise RuntimeError(
                            "the training stream yields no rows"
                        ) from None
                self.epoch += 1
                it_holder["stream"].set_epoch(self.epoch)
                it_holder["it"] = iter(self.loader(it_holder["stream"]))
                it_holder["yielded"] = 0
                continue
            it_holder["yielded"] += 1
            mbs.append(self._to_device(mb))
            rows += len(mb["source_id"])
        return mbs, rows

    def train_step(self, mbs) -> dict:
        model = self.comp.model
        model.train()
        plans = [
            sample_masks(mb["values"], self.groups, self.cfg.masking, self.gen)
            for mb in mbs
        ]
        norm: dict[str, int] = {}
        for mb, plan in zip(mbs, plans, strict=True):
            for k, v in self.counts(mb, plan).items():
                norm[k] = norm.get(k, 0) + v
        self.opt.zero_grad(set_to_none=True)
        agg: dict[str, float] = {}
        total = 0.0
        for mb, plan in zip(mbs, plans, strict=True):
            loss, st = self.micro_loss(mb, plan, norm)
            loss.backward()
            total += float(loss.detach())
            for k, v in st.items():
                agg[k] = agg.get(k, 0.0) + v
        params = [p for g in self.opt.param_groups for p in g["params"]]
        gnorm = torch.nn.utils.clip_grad_norm_(params, self.cfg.optim.grad_clip)
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {self.step}")
        self.opt.step()
        self.sched.step()
        out = {
            "loss": total,
            "grad_norm": float(gnorm),
            "lr": self.sched.get_last_lr()[0],
        }
        for k, v in agg.items():
            out[k] = v / max(norm[STAT_NORM[k]], 1)
        out.update({f"n_{k}": v for k, v in norm.items()})
        return out

    @torch.no_grad()
    def evaluate(self) -> dict:
        """REC-DEV: masked NLL with fixed masks, XP-hidden XP reconstruction, and the
        filter-integral calibration residuals."""
        model = self.comp.model
        model.eval()
        stream = self.stream(
            self.cfg.data.eval_partitions,
            shuffle=False,
            seed=self.cfg.masking.eval_seed,
        )
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.cfg.masking.eval_seed)
        sums = {"xp": 0.0, "phot": 0.0, "xp_hidden": 0.0}
        counts = {"xp": 0, "phot": 0, "xp_hidden": 0}
        resid_hidden = []
        fi_res = []
        rows = 0
        for mb in self.loader(stream, workers=0):
            mb = self._to_device(mb)
            plan = sample_masks(mb["values"], self.groups, self.cfg.masking, gen)
            v_s, e_s = model.scale_inputs(mb["values"], mb["errors"])
            z = model.encode_scaled(v_s, e_s, plan.visible).float()
            g = model.decoder.features(z)
            r, b = plan.target.nonzero(as_tuple=True)
            if len(r):
                s = self.comp.recon(
                    model.decoder.pairs(g, r, b), v_s[r, b], e_s[r, b], b
                )
                for k in ("xp", "phot"):
                    sums[k] += float(s[k][0])
                    counts[k] += s[k][1]
            # photometry-only view of XP stars: reconstruct every XP coefficient
            has_xp = torch.isfinite(mb["values"][:, self.xp_idx]).all(1)
            vis = view(mb["values"], self.groups, hide=("xp",)) & has_xp[:, None]
            keep = has_xp & (vis.sum(1) > 0)
            if bool(keep.any()):
                zx = model.encode_scaled(v_s[keep], e_s[keep], vis[keep]).float()
                px = model.decoder.dense(model.decoder.features(zx), self.xp_idx)
                tx = v_s[keep][:, self.xp_idx]
                ex = e_s[keep][:, self.xp_idx]
                sm = self.comp.recon.sigma_model()[self.xp_idx]
                from .losses import student_t_nll

                nll = student_t_nll(tx - px, ex * ex + sm * sm, self.comp.recon.nu)
                sums["xp_hidden"] += float(nll.sum())
                counts["xp_hidden"] += nll.numel()
                resid_hidden.append(
                    ((tx - px) / torch.sqrt(ex * ex + sm * sm)).flatten().cpu()
                )
            if self.comp.fi is not None:
                fi_res.append(self._fi_residuals(mb))
            rows += len(mb["source_id"])
            if rows >= self.cfg.train.eval_rows:
                break
        out = {f"dev_rec_{k}": sums[k] / max(counts[k], 1) for k in sums}
        out.update({f"dev_n_{k}": counts[k] for k in counts})
        out["dev_rows"] = rows
        if resid_hidden:
            rz = torch.cat(resid_hidden)
            out["dev_xp_hidden_nmad_z"] = float(
                1.4826 * (rz - rz.median()).abs().median()
            )
        if fi_res:
            res = torch.cat([x[0] for x in fi_res])
            ok = torch.cat([x[1] for x in fi_res])
            per = {}
            for j, name in enumerate(self.comp.operator.bands):
                v = res[ok[:, j], j]
                if v.numel() >= 5:
                    per[name] = {
                        "n": int(v.numel()),
                        "median": float(v.median()),
                        "nmad": float(1.4826 * (v - v.median()).abs().median()),
                        "zp": float(self.comp.fi.zp[j]),
                        "k": float(self.comp.fi.k[j]),
                        "sigma_sys": float(self.comp.fi.sigma_sys()[j]),
                    }
            out["dev_fi"] = per
        return out

    def _fi_residuals(self, mb):
        fi = self.comp.fi
        n_fi = self.comp.operator.n_bands
        obs = mb["values"][:, self.fi_idx]
        g = mb["values"][:, self.g_idx]
        s = mb["xp_syn"]
        ok = torch.isfinite(obs) & observed_xp_valid(
            s, mb["xp_var"], g, n_fi, fi.max_sigma_xp
        )
        s_safe = torch.where(torch.isfinite(s) & (s > 0), s, 1.0)
        col = -2.5 * torch.log10(s_safe[:, n_fi] / s_safe[:, n_fi + 1])
        m_syn = g[:, None] - 2.5 * torch.log10(s_safe[:, :n_fi])
        r = obs - (m_syn + fi.zp + fi.k * (col[:, None] - fi.c_ref))
        return r.cpu(), ok.cpu()

    # ------------------------------------------------------------ checkpoints
    def state(self) -> dict:
        return {
            "format": FORMAT,
            "step": self.step,
            "rows_seen": self.rows_seen,
            "epoch": self.epoch,
            "config": self.cfg.to_dict(),
            "band_names": list(self.comp.registry.names),
            "contract_version": self.comp.registry.version,
            "fi_operator": None
            if self.comp.operator is None
            else self.comp.operator.to_dict(),
            "model": self.comp.model.state_dict(),
            "recon": self.comp.recon.state_dict(),
            "fi": None if self.comp.fi is None else self.comp.fi.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "generator": self.gen.get_state(),
        }

    def save(self, name: str | None = None) -> Path:
        ck = self.out / "checkpoints"
        ck.mkdir(exist_ok=True)
        path = ck / (name or f"step_{self.step:08d}.pt")
        tmp = path.with_suffix(".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, path)
        latest = ck / "latest.pt"
        tmp = latest.with_suffix(".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, latest)
        return path

    def load(self, path: Path) -> None:
        st = torch.load(path, map_location=self.device, weights_only=False)
        if st.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} checkpoint")
        load_model_state(self.comp.model, st["model"], st["band_names"])
        load_band_state(
            self.comp.recon, st["recon"], st["band_names"], self.comp.registry
        )
        if self.comp.fi is not None and st["fi"] is not None:
            if st["fi_operator"]["bands"] != list(self.comp.operator.bands):
                raise ValueError("filter-integral bands differ from the checkpoint")
            self.comp.fi.load_state_dict(st["fi"])
        if st["band_names"] == list(self.comp.registry.names):
            self.opt.load_state_dict(st["optimizer"])
            self.sched.load_state_dict(st["scheduler"])
            self.gen.set_state(
                st["generator"].cpu()
                if hasattr(st["generator"], "cpu")
                else st["generator"]
            )
        self.step, self.rows_seen, self.epoch = st["step"], st["rows_seen"], st["epoch"]

    # ------------------------------------------------------------ run
    def log(self, rec: dict) -> None:
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def write_manifest(self, scalers: dict) -> None:
        (self.out / "config.yaml").write_text(
            yaml.safe_dump(self.cfg.to_dict(), sort_keys=False)
        )
        man = {
            "format": FORMAT,
            "git_commit": _git_commit(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "device": str(self.device),
            "parameters": parameter_count(self.comp.model),
            "loss_parameters": sum(p.numel() for p in self.comp.recon.parameters())
            + (
                sum(p.numel() for p in self.comp.fi.parameters()) if self.comp.fi else 0
            ),
            "n_bands": len(self.comp.registry),
            "contract": self.cfg.data.contract,
            "contract_version": self.comp.registry.version,
            "fi_operator": None
            if self.comp.operator is None
            else self.comp.operator.to_dict(),
            "scaler_rows": scalers.get("rows"),
            "total_steps": self.cfg.total_steps,
            "trees": self.trees,
            "assignment": self.assignment,
        }
        (self.out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")

    def run(self) -> dict:
        cfg = self.cfg
        latest = self.out / "checkpoints" / "latest.pt"
        scalers = self.setup_scalers()
        if cfg.run.resume and latest.exists():
            self.load(latest)
        self.write_manifest(scalers)
        stop = (
            cfg.total_steps
            if cfg.train.max_steps is None
            else min(cfg.total_steps, cfg.train.max_steps)
        )
        stream = self.stream(cfg.data.train_partitions)
        stream.set_epoch(self.epoch)
        holder = {"stream": stream, "it": iter(self.loader(stream)), "yielded": 0}
        last: dict = {}
        t0 = time.time()
        rows_t0 = self.rows_seen
        while self.step < stop:
            mbs, rows = self._next_step_batches(holder)
            rec = self.train_step(mbs)
            self.step += 1
            self.rows_seen += rows
            rec.update(
                step=self.step,
                rows=self.rows_seen,
                epoch=self.epoch,
                micro_batches=len(mbs),
            )
            if self.step % cfg.train.log_every == 0 or self.step == stop:
                dt = time.time() - t0
                rec["rows_per_s"] = (self.rows_seen - rows_t0) / max(dt, 1e-9)
                self.log(rec)
                print(
                    json.dumps(
                        {
                            k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in rec.items()
                        }
                    ),
                    flush=True,
                )
            if self.step % cfg.train.eval_every == 0 or self.step == stop:
                ev = self.evaluate()
                ev["step"] = self.step
                self.log(ev)
                print(json.dumps(ev), flush=True)
                rec.update({k: v for k, v in ev.items() if k != "dev_fi"})
            if self.step % cfg.train.checkpoint_every == 0 or self.step == stop:
                self.save()
            last = rec
        return last


def load_model_state(model: MSAv11, state: dict, band_names: list[str]) -> None:
    """Load a state dict whose band tables may cover fewer bands (matched by name)."""
    own = model.state_dict()
    names = list(model.band_names)
    pos = {n: i for i, n in enumerate(names)}
    src = [i for i, n in enumerate(band_names) if n in pos]
    dst = [pos[band_names[i]] for i in src]
    fixed = {}
    for k, v in state.items():
        if k not in own:
            raise KeyError(f"unexpected checkpoint key {k}")
        if v.shape == own[k].shape and band_names == names:
            fixed[k] = v
        elif (
            v.ndim >= 1
            and v.shape[0] == len(band_names)
            and own[k].shape[0] == len(names)
        ):
            t = own[k].clone()
            t[dst] = v[src]
            fixed[k] = t
        elif v.shape == own[k].shape:
            fixed[k] = v
        else:
            raise ValueError(f"{k}: shape {tuple(v.shape)} vs {tuple(own[k].shape)}")
    model.load_state_dict(fixed)


def load_band_state(
    module, state: dict, band_names: list[str], registry: BandRegistry
) -> None:
    pos = {n: i for i, n in enumerate(registry.names)}
    src = [i for i, n in enumerate(band_names) if n in pos]
    dst = [pos[band_names[i]] for i in src]
    own = module.state_dict()
    fixed = {}
    for k, v in state.items():
        if (
            v.ndim == 1
            and v.shape[0] == len(band_names)
            and own[k].shape[0] == len(registry)
        ):
            t = own[k].clone()
            t[dst] = v[src]
            fixed[k] = t
        else:
            fixed[k] = v
    module.load_state_dict(fixed)


def encode_views(
    model: MSAv11,
    values,
    errors,
    groups: BandGroups,
    views=((), ("astrometry",), ("xp",), ("xp", "astrometry")),
):
    """Latents of the views downstream heads need (S1/S2: full; S3: astrometry hidden;
    photometry-only: XP hidden). Rows with nothing visible in a view get NaN."""
    out = {}
    model.eval()
    with torch.no_grad():
        for hide in views:
            vis = view(values, groups, hide)
            ok = vis.any(1)
            z = torch.full(
                (values.shape[0], model.cfg.latent_dim),
                float("nan"),
                device=values.device,
            )
            if bool(ok.any()):
                z[ok] = model.encode(values[ok], errors[ok], vis[ok]).float()
            out["+".join(hide) or "full"] = z
    return out


def summarise_scalers(scalers: dict) -> dict:
    c = np.asarray(scalers["count"])
    return {"rows": scalers["rows"], "bands_fitted": int((c > 0).sum())}
