"""Native-CUDA matched seismic supervision/map/noise/dose experiments.

Neutral pre-seismic NSF initialization; same full source-v5 pools, common APOGEE
coordinate convention and exact training schedule across treatments. These are
fixed-budget pilot comparisons, not full FIT epochs or replacements of the large fit.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
ARMS = [
    dict(name="off", kind="identity", weight=0.0, extra_dex=0.0),
    dict(name="identity", kind="identity", weight=1.0, extra_dex=0.0),
    dict(name="affine", kind="affine", weight=1.0, extra_dex=0.0),
    dict(name="identity_floor", kind="identity", weight=1.0, extra_dex=0.02),
    dict(name="affine_floor", kind="affine", weight=1.0, extra_dex=0.02),
    dict(name="identity_lowweight", kind="identity", weight=0.1, extra_dex=0.0),
    dict(name="context", kind="context", weight=1.0, extra_dex=0.0),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", choices=["joint", "frozen"], required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--require-preflight", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.cached_inputs import cache_natural_inputs
    from masked_stellar_autoencoder.training.decision_pdf_metrics import sample_scores
    from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
        _cache_maps,
    )
    from masked_stellar_autoencoder.training.fullfit_records import (
        fixed_coordinates,
        reconstruction_block_start,
    )
    from masked_stellar_autoencoder.training.nsf_confirmation import confirmation_score
    from masked_stellar_autoencoder.training.remediation_energy6 import (
        EnergyJointSixHead,
    )
    from masked_stellar_autoencoder.training.seismic_ablation import (
        SeismicChannel,
        matched_schedule,
    )
    from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap

    cfg = yaml.safe_load(a.config.read_text())
    m = dict(
        status="admitting",
        seed=a.seed,
        mode=a.mode,
        preflight=a.preflight,
        training_performed=False,
        arms={},
        scope="Matched fixed-budget neutral-start supervision controls. All runs use APOGEE as the logg coordinate convention, not an unbiased physical truth. No arm is a pure replica of the production seismic-reference run. CAL not opened. Existing catalogue calibrations/ancestor exposure may remain. Source-level calibration parameters are MAP estimates, not posterior systematic uncertainty.",
    )

    def save():
        prior.atomic(a.output / "manifest.json", m)

    save()
    try:
        if a.require_preflight:
            m["status"] = "waiting_for_preflight"
            save()
            deadline = time.monotonic() + 6 * 3600
            while time.monotonic() < deadline:
                try:
                    pm = json.loads((a.require_preflight / "manifest.json").read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    pm = {}
                if pm.get("status") == "failed":
                    raise RuntimeError("CUDA preflight failed: " + str(pm.get("error")))
                if pm.get("status") == "seismic_experiments_complete":
                    break
                time.sleep(20)
            else:
                raise TimeoutError("CUDA preflight unavailable")
        if not torch.cuda.is_available():
            raise RuntimeError("CANFAR CUDA required")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Library differs")
        source = json.loads(
            (ROOT / cfg["expanded_source_run"] / "manifest.json").read_text()
        )
        for path, expected in [
            (source["prepared_path"], source["prepared_sha256"]),
            (cfg["encoder_parent"], cfg["encoder_parent_sha256"]),
            (cfg["reference_checkpoint"], cfg["reference_checkpoint_sha256"]),
        ]:
            if prior.sha(path) != expected:
                raise ValueError("Source/checkpoint digest differs")
        torch.set_num_threads(4)
        device = torch.device("cuda")
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        reference = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        parent = torch_load_trusted(
            cfg["encoder_parent"], map_location="cpu", weights_only=False
        )
        modelcfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
        model = native.probe.load_model(modelcfg, reference, device)
        model.load_state_dict(parent["model_state_dict"], strict=True)
        head = EnergyJointSixHead(
            latent_dim=modelcfg["model"]["layer_dims"][-1],
            width=cfg["head_width"],
            particles=cfg["particles"],
            seed=cfg["seed"],
            components=cfg["mixture_components"],
            energy_training_draws=32,
            unknown_error_floor_normalized=cfg["unknown_error_floor_normalized"],
        ).to(device)
        head.load_state_dict(parent["head_state_dict"], strict=True)
        center, scale = np.asarray(parent["center"]), np.asarray(parent["scale"])
        initmodel = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        inithead = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        del reference, parent

        def tensor(x, dtype=torch.float32):
            return torch.as_tensor(x, device=device, dtype=dtype)

        refdir = ROOT / "seismic_gravity_qa_20261106_v1"
        rm = json.loads((refdir / "manifest.json").read_text())
        refpath = refdir / "apokasc3-gravity-qa-reference.parquet"
        if prior.sha(refpath) != rm["sources"]["apokasc3"]["reference_sha256"]:
            raise ValueError("Seismic reference differs")
        ref = pd.read_parquet(refpath)
        with h5py.File(source["prepared_path"]) as data:
            ids, roles = data["source_id"][:], data["role"][:].astype("S8")
            eligible = data["primary_input_eligible"][:].astype(bool)
            if ids.dtype.kind != "i" or len(np.unique(ids)) != len(ids):
                raise ValueError("Unique exact signed IDs required")
            x = cache_natural_inputs(
                data, source["feature_profiles"]["phot_xp_measured_parallax"]
            )
            check = np.linspace(0, len(ids) - 1, 128, dtype=int)
            np.testing.assert_array_equal(
                x[check],
                prior.profile_batch(data, check, "phot_xp_measured_parallax", source),
            )
            order = np.argsort(ids)
            sortedids = ids[order]

            def lookup(sids):
                sids = np.asarray(sids)
                if sids.dtype.kind not in "iu":
                    raise ValueError("Floating IDs forbidden")
                pos = np.searchsorted(sortedids, sids)
                good = pos < len(ids)
                good[good] &= sortedids[pos[good]] == sids[good]
                return order[pos[good]], np.flatnonzero(good)

            names = source["source_registry"]
            records = []
            support = []
            for si, name in enumerate(names):
                g = data["native_observations/" + name]
                y, e, obs, known = fixed_coordinates(
                    g["value"][:],
                    g["error"][:],
                    g["observed"][:],
                    g["error_known"][:],
                    center,
                    scale,
                )
                rows = g["star_row"][:]
                support.append(obs.any(0))
                records.append(
                    dict(
                        rows=rows,
                        value=y,
                        error=e,
                        observed=obs,
                        known=known,
                        source=si,
                    )
                )
            sr, ri = lookup(ref.source_id.to_numpy(dtype="i8"))
            v = np.zeros((len(sr), 6), dtype="f4")
            e = v.copy()
            obs = np.zeros(v.shape, bool)
            v[:, 1] = (ref.loggSeis.to_numpy()[ri] - center[1]) / scale[1]
            e[:, 1] = ref.e_loggSeis.to_numpy()[ri] / scale[1]
            obs[:, 1] = True
            if (
                not np.isfinite(v).all()
                or not np.isfinite(e).all()
                or (e[:, 1] <= 0).any()
            ):
                raise ValueError("Bad seismic targets/errors")
            anchor = len(records)
            records.append(
                dict(
                    rows=sr,
                    value=v,
                    error=e,
                    observed=obs,
                    known=obs.copy(),
                    source=anchor,
                )
            )
            support = np.asarray(support)
            # APOGEE g stays identity in every arm: mapping comparisons do not free the gauge.
            maps = SixSurveyMap(support, [0, 0, 0, 1, 3, 3], kind="spline").to(device)
            initmaps = {
                k: v.detach().cpu().clone() for k, v in maps.state_dict().items()
            }
            fit = []
            for b in records:
                keep = (
                    (roles[b["rows"]] == b"fit")
                    & eligible[b["rows"]]
                    & b["observed"].any(1)
                )
                fit.append(
                    {
                        k: (v[keep] if isinstance(v, np.ndarray) else v)
                        for k, v in b.items()
                    }
                )
            lengths = [len(b["rows"]) if i != anchor else 0 for i, b in enumerate(fit)]
            if len(fit[anchor]["rows"]) < 32:
                raise ValueError("Insufficient FIT seismic pool")
            # Fixed TUNE/DEV native-label readouts plus ALL usable seismic DEV and TESS DEV.
            panels = {}
            for role in [b"tune", b"devtest"]:
                for si, b in enumerate(records):
                    pp = np.flatnonzero(
                        (roles[b["rows"]] == role) & eligible[b["rows"]]
                    )
                    take = min(len(pp), cfg["seismic_ablation"]["eval_per_source"])
                    if si == anchor:
                        take = len(pp)
                    if a.preflight:
                        take = min(take, 2)
                    panels[(role, si)] = np.sort(
                        np.random.default_rng(2026110791 + si).choice(
                            pp, take, replace=False
                        )
                    )
            tdir = ROOT / "tess_k2_local_gaia_candidates_20261107_v1"
            tm = json.loads((tdir / "manifest.json").read_text())
            tp = tdir / "seismic-gaia-qa-candidates.parquet"
            if prior.sha(tp) != tm["candidate_sha256"]:
                raise ValueError("TESS source differs")
            tess = pd.read_parquet(tp)
            tess = tess[
                (tess.catalogue == "tess_hon_2021")
                & np.isfinite(tess.logg_nu_scaling)
                & np.isfinite(tess.e_logg_stat)
                & (tess.e_logg_stat > 0)
            ].drop_duplicates("source_id")
            tr, ti = lookup(tess.source_id.to_numpy(dtype="i8"))
            kt = (roles[tr] == b"devtest") & eligible[tr]
            tr, ti = tr[kt], ti[kt]
            if a.preflight:
                tr, ti = tr[:8], ti[:8]
            panel = np.unique(
                np.concatenate(
                    [
                        b["rows"][panels[(r, si)]]
                        for r in [b"tune", b"devtest"]
                        for si, b in enumerate(records)
                    ]
                    + [
                        tr,
                        np.flatnonzero(
                            data["reserved_globular_member"][:].astype(bool) & eligible
                        )[:4]
                        if a.preflight
                        else np.flatnonzero(
                            data["reserved_globular_member"][:].astype(bool) & eligible
                        ),
                        np.flatnonzero(
                            data["reserved_benchmark"][:].astype(bool) & eligible
                        )[:4]
                        if a.preflight
                        else np.flatnonzero(
                            data["reserved_benchmark"][:].astype(bool) & eligible
                        ),
                    ]
                )
            )
            m.update(
                status="training",
                source_sha256=source["prepared_sha256"],
                neutral_parent_sha256=cfg["encoder_parent_sha256"],
                initialization="Earlier pooled NSF before explicit APOKASC supervision; all source maps start identity",
                gravity_reference="APOGEE identity in all arms; seismic identity/map is a separate observation-channel hypothesis",
                fit_seismic_stars=len(fit[anchor]["rows"]),
                source_registry=names + ["APOKASC3_loggSeis"],
                training_records_per_step=320,
                seismic_records_per_step=32,
                panel_stars=len(panel),
                tess_dev_stars=len(tr),
                common_panel_sha256=hashlib.sha256(
                    ids[panel].astype("<i8").tobytes()
                ).hexdigest(),
                seismic_reference_sha256=rm["sources"]["apokasc3"]["reference_sha256"],
                tess_reference_sha256=tm["candidate_sha256"],
            )
            save()
            np.savez_compressed(
                a.output / "panel.npz",
                source_id=ids[panel],
                row=panel,
                role=roles[panel],
                tess_row=tr,
                tess_source_id=ids[tr],
                tess_reference=tess.logg_nu_scaling.to_numpy()[ti],
                tess_error=tess.e_logg_stat.to_numpy()[ti],
            )
            m["panel_sha256"] = prior.sha(a.output / "panel.npz")
            save()
            cache = None
            if a.mode == "joint":
                cache, _, _, _ = _cache_maps(
                    cfg["reconstruction_cache"],
                    data["feature_names"][:].astype("U").tolist(),
                    data["scaler_center"][:],
                    data["scaler_scale"][:],
                    cfg["reconstruction_request_sha256"],
                    cfg["reconstruction_training_rows"],
                )
            features = data["feature_names"][:].astype("U").tolist()
            xp = [j for j, n in enumerate(features) if n.startswith(("bp_", "rp_"))]
            allowed = np.array(
                [
                    n in {"W1", "W2", "G", "BP", "RP"} or j in xp
                    for j, n in enumerate(features)
                ]
            )
            width = len(modelcfg["data"]["recon_cols"])
            if features[:width] != modelcfg["data"]["recon_cols"]:
                raise ValueError("Reconstruction order differs")

            def makebatch(ss, pp, rng):
                pieces = []
                pairs = [(int(si), pp[ss == si]) for si in np.unique(ss)]
                for si, count in [(anchor, 32), (3, 16), (4, 8), (5, 8)]:
                    if not len(fit[si]["rows"]):
                        raise ValueError("Empty auxiliary pool")
                    pairs.append(
                        (si, rng.choice(len(fit[si]["rows"]), count, replace=False))
                    )
                for si, positions in pairs:
                    b = fit[si]
                    pieces.append(
                        dict(
                            source=np.full(len(positions), si, dtype="i8"),
                            **{
                                k: b[k][positions]
                                for k in ["rows", "value", "error", "observed", "known"]
                            },
                        )
                    )
                return {k: np.concatenate([b[k] for b in pieces]) for k in pieces[0]}

            def draw(batch, start, stop, count, seismic, seed=None):
                r = batch["rows"][start:stop]
                z = model.encoder(tensor(x[r]))
                gen = (
                    head.sample_generator
                    if seed is None
                    else torch.Generator(device=device).manual_seed(seed)
                )
                d = head.sample(z, count, gen)
                si = tensor(batch["source"][start:stop], torch.long)
                normal = maps(d, si.clamp(max=anchor - 1))
                return torch.where((si == anchor)[None, :, None], seismic(d), normal)

            arms = (
                ARMS
                if a.mode == "joint" or a.preflight
                else [v for v in ARMS if v["name"] in ["off", "identity", "affine"]]
            )
            steps = 2 if a.preflight else cfg["seismic_ablation"]["updates"]
            baseline_dir = a.output / "baseline"
            baseline_dir.mkdir(exist_ok=False)
            baseline_channel = SeismicChannel(center, scale, "identity").to(device)
            evaluate(
                model,
                head,
                maps,
                baseline_channel,
                ARMS[1],
                panel,
                panels,
                records,
                ids,
                roles,
                x,
                center,
                scale,
                baseline_dir,
                a.seed,
                anchor,
                prior,
                sample_scores,
                torch,
            )
            m["baseline_readout_sha256"] = prior.sha(baseline_dir / "common.npz")
            del baseline_channel
            save()
            for arm in arms:
                d = a.output / arm["name"]
                d.mkdir(exist_ok=False)
                model.load_state_dict(initmodel, strict=True)
                head.load_state_dict(inithead, strict=True)
                maps.load_state_dict(initmaps, strict=True)
                model.requires_grad_(a.mode == "joint")
                head.requires_grad_(True)
                head.gaussian.requires_grad_(False)
                seismic = SeismicChannel(center, scale, arm["kind"]).to(device)
                head.configure_generators(device, a.seed + 301, a.seed + 302)
                torch.manual_seed(a.seed + 303)
                torch.cuda.manual_seed_all(a.seed + 303)
                maskrng = torch.Generator(device=device).manual_seed(a.seed + 304)
                recrng = np.random.default_rng(a.seed + 305)
                optimizer = torch.optim.Adam(
                    [
                        dict(params=head.parameters(), lr=3e-4),
                        dict(params=maps.parameters(), lr=3e-4),
                        dict(params=seismic.parameters(), lr=3e-4),
                    ]
                    + (
                        [dict(params=model.parameters(), lr=cfg["encoder_lr"])]
                        if a.mode == "joint"
                        else []
                    )
                )
                sequence = hashlib.sha256()
                reconstruction_sequence = hashlib.sha256()
                item = dict(
                    status="training",
                    arm=arm,
                    completed_updates=0,
                    source_native_loss_denominator=320,
                    reconstruction_weight=0.03 if a.mode == "joint" else 0.0,
                    calibration_prior_weight=1e-4,
                )
                m["arms"][arm["name"]] = item
                save()
                starttime = time.monotonic()
                for update, (ss, pp, rng) in enumerate(
                    matched_schedule(lengths, steps, a.seed + 201), 1
                ):
                    batch = makebatch(ss, pp, rng)
                    sequence.update(batch["source"].astype("<i8").tobytes())
                    sequence.update(ids[batch["rows"]].astype("<i8").tobytes())
                    model.train(a.mode == "joint")
                    head.train()
                    maps.train()
                    seismic.train()
                    optimizer.zero_grad(set_to_none=True)
                    lossvalue = 0.0
                    for start in range(0, len(batch["rows"]), 16):
                        stop = min(start + 16, len(batch["rows"]))
                        draws = draw(batch, start, stop, 32, seismic)
                        sig = batch["error"][start:stop].copy()
                        isseis = batch["source"][start:stop] == anchor
                        sig[isseis, 1] = np.sqrt(
                            sig[isseis, 1] ** 2 + (arm["extra_dex"] / scale[1]) ** 2
                        )
                        # Pieces are source-homogeneous, with 16-divisible auxiliaries.
                        if isseis.any() and not isseis.all():
                            raise ValueError("Treatment mixed with native chunk")
                        score, _ = confirmation_score(
                            draws,
                            tensor(batch["value"][start:stop]),
                            tensor(batch["observed"][start:stop], torch.bool),
                            tensor(sig),
                            tensor(batch["known"][start:stop], torch.bool),
                            tensor(cfg["unknown_error_floor_normalized"]),
                            head.noise_generator,
                            energy_weight=1.0,
                            crps_weight=0.0,
                        )
                        weight = arm["weight"] if isseis.all() else 1.0
                        loss = score * ((stop - start) / len(batch["rows"])) * weight
                        loss.backward()
                        lossvalue += float(loss.detach())
                    regularizer = 1e-4 * (
                        maps.regularization() + seismic.regularization()
                    )
                    regularizer.backward()
                    reco = torch.zeros((), device=device)
                    if a.mode == "joint":
                        rs = reconstruction_block_start(
                            len(cache), cfg["reconstruction_batch"], recrng
                        )
                        reconstruction_sequence.update(
                            np.asarray([rs], dtype="<i8").tobytes()
                        )
                        raw = np.asarray(
                            cache[rs : rs + cfg["reconstruction_batch"]]
                        ).copy()
                        raw[(~np.isfinite(raw)) | (raw == -9999)] = np.nan
                        raw[:, ~allowed] = np.nan
                        target = tensor(raw)
                        hidden, hm, finite = native.pretrain.training_mask(
                            target, xp, 0.5, 0.2, maskrng
                        )
                        prediction, _ = model(hidden)
                        seen = (hm & finite)[:, :width]
                        reco = native.pretrain.experiment_objective(
                            prediction,
                            target[:, :width],
                            seen,
                            xp,
                            {"loss": "mae"},
                            seen.sum(),
                        )
                        (0.03 * reco).backward()
                    gradients = [
                        p.grad
                        for g in optimizer.param_groups
                        for p in g["params"]
                        if p.grad is not None
                    ]
                    if not gradients or not all(
                        torch.isfinite(g).all() for g in gradients
                    ):
                        raise FloatingPointError("Nonfinite gradients")
                    if update == 1:
                        item["first_step_gradient_l2"] = {
                            name: float(
                                sum(
                                    p.grad.square().sum()
                                    for p in obj.parameters()
                                    if p.grad is not None
                                )
                                ** 0.5
                            )
                            if any(p.grad is not None for p in obj.parameters())
                            else 0.0
                            for name, obj in [
                                ("encoder", model.encoder),
                                ("head", head),
                                ("maps", maps),
                                ("seismic", seismic),
                            ]
                        }
                        if item["first_step_gradient_l2"]["head"] <= 0 or (
                            a.mode == "joint"
                            and item["first_step_gradient_l2"]["encoder"] <= 0
                        ):
                            raise ValueError("Required gradients missing")
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in optimizer.param_groups for p in g["params"]], 10.0
                    )
                    optimizer.step()
                    item.update(
                        completed_updates=update,
                        last_objective=lossvalue,
                        last_reconstruction_mae=float(reco.detach()),
                        elapsed_seconds=time.monotonic() - starttime,
                    )
                    m["training_performed"] = True
                    if update <= 2 or update % 100 == 0:
                        save()
                    if update % 100 == 0:
                        gc.collect()
                item.update(
                    batch_identity_sha256=sequence.hexdigest(),
                    reconstruction_sequence_sha256=reconstruction_sequence.hexdigest(),
                    seismic_coefficients=seismic.coefficients().detach().cpu().tolist(),
                )
                if not a.preflight:
                    torch.save(
                        dict(
                            model_state_dict=model.state_dict(),
                            head_state_dict=head.state_dict(),
                            map_state_dict=maps.state_dict(),
                            seismic_state_dict=seismic.state_dict(),
                            arm=arm,
                            seed=a.seed,
                            mode=a.mode,
                            update=steps,
                            center=center,
                            scale=scale,
                            source_sha256=source["prepared_sha256"],
                            neutral_parent_sha256=cfg["encoder_parent_sha256"],
                        ),
                        d / "terminal-checkpoint.pth",
                    )
                bias, slopes = maps.coefficients()
                np.savez_compressed(
                    d / "map-coefficients.npz",
                    bias=bias.detach().cpu().numpy(),
                    slopes=slopes.detach().cpu().numpy(),
                    center=center,
                    scale=scale,
                    kind=np.array("spline"),
                    source_registry=np.asarray(names),
                    seismic_coefficients=seismic.coefficients().detach().cpu().numpy(),
                )
                item["status"] = "evaluating"
                save()
                evaluate(
                    model,
                    head,
                    maps,
                    seismic,
                    arm,
                    panel,
                    panels,
                    records,
                    ids,
                    roles,
                    x,
                    center,
                    scale,
                    d,
                    a.seed,
                    anchor,
                    prior,
                    sample_scores,
                    torch,
                )
                item["readout_sha256"] = prior.sha(d / "common.npz")
                if not a.preflight:
                    item["checkpoint_sha256"] = prior.sha(d / "terminal-checkpoint.pth")
                item["status"] = "complete"
                save()
                del optimizer, seismic
                gc.collect()
                torch.cuda.empty_cache()
            sequences = [v["batch_identity_sha256"] for v in m["arms"].values()]
            if len(set(sequences)) != 1:
                raise ValueError("Matched source/star schedules differ")
            if (
                len({v["reconstruction_sequence_sha256"] for v in m["arms"].values()})
                != 1
            ):
                raise ValueError("Reconstruction schedules differ")
            m.update(
                status="seismic_experiments_complete",
                completed_arm_count=len(arms),
                completed_updates_per_arm=steps,
            )
            save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


def evaluate(
    model,
    head,
    maps,
    seismic,
    arm,
    panel,
    panels,
    records,
    ids,
    roles,
    x,
    center,
    scale,
    out,
    seed,
    anchor,
    prior,
    sample_scores,
    torch,
):
    model.eval()
    head.eval()
    maps.eval()
    seismic.eval()
    device = next(head.parameters()).device
    common = []
    for start in range(0, len(panel), 32):
        rr = panel[start : start + 32]
        with torch.inference_mode():
            z = model.encoder(
                torch.as_tensor(x[rr], device=device, dtype=torch.float32)
            )
            gen = torch.Generator(device=device).manual_seed(2026110793 + start)
            d = prior.bounded_whole_draws(head, z, 256, gen, 128)
        if not np.isfinite(d).all():
            raise FloatingPointError("Nonfinite eval samples")
        common.append(d.astype("f4"))
        if start % 512 == 0:
            gc.collect()
    # Whole-vector draws retained on a bounded panel for fair rescoring and corners.
    samples = np.concatenate(common, axis=1)
    physical = samples * scale + center
    physical[:, :, 4:] = np.exp(physical[:, :, 4:])
    if not np.isfinite(physical).all():
        raise FloatingPointError("Nonfinite physical draw")
    np.savez_compressed(
        out / "common.npz",
        source_id=ids[panel],
        row=panel,
        role=roles[panel],
        center=center,
        scale=scale,
        median=np.median(physical, axis=0),
        quantiles=np.quantile(physical, [0.05, 0.5, 0.95], axis=0),
        logg_draws=physical[:, :, 1].astype("f4"),
        tail_age_gt20=(physical[:, :, 4] > 20).mean(0),
    )
    reports = []
    for (role, si), pp in panels.items():
        if not len(pp):
            continue
        b = records[si]
        rr = b["rows"][pp]
        pi = np.searchsorted(panel, rr)
        np.testing.assert_array_equal(panel[pi], rr)
        # Bound map application too: no gigantic GPU allocation for a full source.
        pieces = []
        for start in range(0, len(pi), 32):
            dd = torch.as_tensor(samples[:, pi[start : start + 32]], device=device)
            with torch.inference_mode():
                mapped = (
                    seismic(dd)
                    if si == anchor
                    else maps(
                        dd,
                        torch.full((dd.shape[1],), si, device=device, dtype=torch.long),
                    )
                )
            pieces.append(mapped.cpu().numpy())
        draw = np.concatenate(pieces, axis=1)
        sigma = b["error"][pp].copy()
        if si == anchor:
            sigma[:, 1] = np.sqrt(sigma[:, 1] ** 2 + (arm["extra_dex"] / scale[1]) ** 2)
        sigma = np.where(b["observed"][pp], np.where(b["known"][pp], sigma, 0.1), 0.0)
        noisy = (
            draw
            + np.random.default_rng(2026110794 + si).normal(size=draw.shape)
            * sigma[None]
        )
        scores = sample_scores(noisy, b["value"][pp], b["observed"][pp])
        name = "APOKASC3_loggSeis" if si == anchor else str(si)
        np.savez_compressed(
            out / (role.decode() + "-" + name + ".npz"),
            source_id=ids[rr],
            reference=b["value"][pp],
            observed=b["observed"][pp],
            intrinsic_median=np.median(draw, axis=0),
            center=center,
            scale=scale,
            **scores,
        )
        reports.append(
            dict(
                role=role.decode(),
                source=si,
                n=len(pp),
                per_target_crps=[
                    float(np.mean(scores["crps"][b["observed"][pp, j], j]))
                    if b["observed"][pp, j].any()
                    else None
                    for j in range(6)
                ],
            )
        )
    (out / "readout.json").write_text(
        json.dumps(
            dict(
                source_native_scores=reports,
                seismic_coefficients=seismic.coefficients().detach().cpu().tolist(),
                common_samples=256,
                scope="Fixed terminal readout, no checkpoint selection; source CRPS includes declared per-star noise, source calibration parameters are point estimates.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
