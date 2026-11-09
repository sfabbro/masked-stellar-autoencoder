"""CANFAR frozen-encoder six-target NSF/source-map and gravity-gauge controls.

Isolates head and mapping effects. This is not expanded final training and does
not update the MAE; the production reconstruction recipe remains MAE weight .03.
"""

import argparse
import gc
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
SOURCE_NAMES = [
    "apogee",
    "desi",
    "galah",
    "StarHorse_apogee17_posterior_median",
    "StarHorse_galahdr3_posterior_median",
    "StarHorse_catalogue_posterior_median",
    "APOKASC3_loggSeis",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    arm = next(v for v in cfg["mapping_arms"] if v["run_id"] == a.run_id)
    a.output.mkdir(parents=True, exist_ok=False)
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.decision_pdf_metrics import sample_scores
    from masked_stellar_autoencoder.training.nsf_confirmation import confirmation_score
    from masked_stellar_autoencoder.training.remediation_energy6 import (
        EnergyJointSixHead,
    )
    from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap

    m = dict(
        status="admitting",
        run_id=a.run_id,
        arm=arm,
        config_sha256=prior.sha(a.config),
        script_sha256=prior.sha(__file__),
        completed_updates=0,
        training_performed=False,
        encoder_updated=False,
        reconstruction_updated=False,
        scope="Frozen identical six-target NSF encoder; source-mapped sample-energy plus marginal CRPS training. Gauge/complexity controls on historical core source, not expanded final training. Age/distance retain source-specific posterior-teacher estimands. APOKASC is a gravity anchor with catalogue/parent calibration dependencies, not independent absolute physical truth.",
    )

    def save():
        prior.atomic(a.output / "manifest.json", m)

    save()
    try:
        if not torch.cuda.is_available() or not ROOT.is_dir():
            raise RuntimeError("CANFAR CUDA required")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Pinned library differs")
        source, _, _ = prior.admission(cfg, cfg["arms"][0])
        if (
            prior.sha(source["prepared_path"]) != source["prepared_sha256"]
            or prior.sha(cfg["mapping_parent"]) != cfg["mapping_parent_sha256"]
            or prior.sha(cfg["reference_checkpoint"])
            != cfg["reference_checkpoint_sha256"]
        ):
            raise ValueError("Input or parent checkpoint differs")
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        device = torch.device("cuda")
        disk = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        state = torch_load_trusted(
            cfg["mapping_parent"], map_location="cpu", weights_only=False
        )
        if (
            state["prepared_sha256"] != source["prepared_sha256"]
            or state["parent_sha256"] != cfg["checkpoint_sha256"]
            or state["arm"]["kind"] != "nsf_energy"
        ):
            raise ValueError("Warm-start six-target training lineage differs")
        model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
        model = native.probe.load_model(model_cfg, disk, device)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval().requires_grad_(False)
        head = EnergyJointSixHead(
            latent_dim=model_cfg["model"]["layer_dims"][-1],
            width=cfg["head_width"],
            particles=cfg["particles"],
            seed=cfg["seed"],
            components=cfg["mixture_components"],
            energy_training_draws=32,
            unknown_error_floor_normalized=cfg["unknown_error_floor_normalized"],
        ).to(device)
        head.load_state_dict(state["head_state_dict"], strict=True)
        head.configure_generators(device, cfg["seed"] + 100, cfg["seed"] + 101)
        center, scale = np.asarray(state["center"]), np.asarray(state["scale"])
        del state, disk
        refpath = (
            ROOT
            / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
        )
        refmanifest = json.loads((refpath.parent / "manifest.json").read_text())
        if prior.sha(refpath) != refmanifest["sources"]["apokasc3"]["reference_sha256"]:
            raise ValueError("Seismic reference changed")
        ref = pd.read_parquet(refpath)
        refidx = {int(v): i for i, v in enumerate(ref.source_id)}
        support = np.zeros((6, 6), bool)
        support[:3, :3] = True
        support[1, 3] = True
        support[3:, 4:] = True
        reference = [0, 0, 0, 1, 3, 3]
        zero_sum = []
        if arm["gauge"] == "equal_weight":
            reference[:3] = [-1] * 3
            zero_sum = [0, 1, 2]
        elif arm["gauge"] == "seismic":
            reference[1] = -1
        elif arm["gauge"] != "apogee":
            raise ValueError("Unknown reference gauge")
        maps = SixSurveyMap(
            support, reference, zero_sum=zero_sum, kind=arm["map_kind"]
        ).to(device)
        optimizer = torch.optim.Adam(
            [
                dict(params=head.parameters(), lr=3e-4),
                dict(params=maps.parameters(), lr=3e-4),
            ]
        )
        blocked = np.asarray(cfg["benchmark_fit_exclusion_ids"], dtype="i8")
        with h5py.File(source["prepared_path"]) as data:
            ids = data["source_id"][:]
            roles = data["role"][:].astype("U")
            eligible = data["primary_input_eligible"][:].astype(bool)
            values = data["value"][:]
            mask = data["observed"][:].astype(bool) & data["quality"][:].astype(bool)
            label_source = data["label_source"][:].astype("U64")
            y, sigma, valid, c, s, known = prior.normalize_targets(
                values,
                mask,
                data["error"][:],
                data["error_observed"][:],
                roles == "fit",
            )
            np.testing.assert_array_equal(c, center)
            np.testing.assert_array_equal(s, scale)
            records = {}
            for role in ["fit", "tune", "cal", "devtest"]:
                pred = (
                    ROOT
                    / "multisurvey_joint_fullfit_20261105_v1"
                    / ("predictions-" + role + ".h5")
                )
                if prior.sha(pred) != cfg["mapping_native_sha256"][role]:
                    raise ValueError("Native-label export changed: " + role)
                blocks = []
                with h5py.File(pred) as f:
                    native_ids = f["source_id"][:]
                    rows = np.searchsorted(ids, native_ids)
                    if (rows >= len(ids)).any():
                        raise ValueError("Native ID absent from six-target inputs")
                    np.testing.assert_array_equal(ids[rows], native_ids)
                    if not (roles[rows] == role).all():
                        raise ValueError("Source/role assignment differs")
                    nv, ne, nm = (
                        f["native_value"][:],
                        f["native_error"][:],
                        f["native_observed"][:].astype(bool),
                    )
                keep = eligible[rows] & ~np.isin(ids[rows], blocked)
                for si in range(3):
                    v = np.zeros((len(rows), 6))
                    e = v.copy()
                    obs = np.zeros(v.shape, bool)
                    kn = obs.copy()
                    obs[:, :3] = nm[:, si] & np.isfinite(nv[:, si])
                    v[:, :3] = (
                        np.where(obs[:, :3], nv[:, si], center[:3]) - center[:3]
                    ) / scale[:3]
                    kn[:, :3] = obs[:, :3] & np.isfinite(ne[:, si]) & (ne[:, si] > 0)
                    e[:, :3] = np.where(kn[:, :3], ne[:, si] / scale[:3], 0)
                    if si == 1:
                        use = valid[rows, 3] & (
                            label_source[rows, 3] == "DESI_RVSpecFit_alpha_fe"
                        )
                        obs[use, 3], kn[use, 3] = True, known[rows[use], 3]
                        v[use, 3], e[use, 3] = y[rows[use], 3], sigma[rows[use], 3]
                        if arm.get("exclude_desi_logg"):
                            obs[:, 1] = False
                            kn[:, 1] = False
                    take = keep & obs.any(1)
                    blocks.append(
                        dict(
                            rows=rows[take],
                            value=v[take],
                            error=e[take],
                            observed=obs[take],
                            known=kn[take],
                            source=si,
                        )
                    )
                rr = np.flatnonzero((roles == role) & eligible & ~np.isin(ids, blocked))
                for si in range(3, 6):
                    obs = np.zeros((len(rr), 6), bool)
                    obs[:, 4:] = valid[rr, 4:] & (
                        label_source[rr, 4:] == SOURCE_NAMES[si]
                    )
                    take = obs.any(1)
                    blocks.append(
                        dict(
                            rows=rr[take],
                            value=y[rr[take]],
                            error=sigma[rr[take]],
                            observed=obs[take],
                            known=known[rr[take]] & obs[take],
                            source=si,
                        )
                    )
                sr = rr[np.array([int(v) in refidx for v in ids[rr]])]
                ix = np.array([refidx[int(v)] for v in ids[sr]])
                obs = np.zeros((len(sr), 6), bool)
                obs[:, 1] = True
                sv = np.zeros((len(sr), 6))
                se = sv.copy()
                sv[:, 1] = (ref.loggSeis.to_numpy()[ix] - center[1]) / scale[1]
                se[:, 1] = ref.e_loggSeis.to_numpy()[ix] / scale[1]
                if (
                    not np.isfinite(sv).all()
                    or not np.isfinite(se).all()
                    or (se[:, 1] <= 0).any()
                ):
                    raise ValueError("Invalid seismic anchor")
                blocks.append(
                    dict(
                        rows=sr,
                        value=sv,
                        error=se,
                        observed=obs,
                        known=obs.copy(),
                        source=6,
                    )
                )
                records[role] = blocks
            m.update(
                status="training",
                parent_checkpoint_sha256=cfg["mapping_parent_sha256"],
                prepared_sha256=source["prepared_sha256"],
                source_registry=SOURCE_NAMES,
                supported_edges=support,
                reference_source_per_target=reference,
                zero_sum_targets=zero_sum,
                role_source_rows={
                    r: {SOURCE_NAMES[b["source"]]: len(b["rows"]) for b in blocks}
                    for r, blocks in records.items()
                },
                teacher_reference="StarHorse APOGEE posterior scale for age/distance; DESI alpha/Fe scale; APOGEE temperature/iron convention unless equal-weight control",
                benchmark_exclusions=blocked.tolist(),
                ancestor_exposure="Warm-start NSF and its supervised/pretraining ancestors may contain newly reserved benchmark IDs; new ablation FIT excludes them, no fresh absolute benchmark claim",
            )
            save()

            def tensor(x, dtype=torch.float32):
                return torch.as_tensor(x, device=device, dtype=dtype)

            def draw_for(block, positions, n, seed=None):
                rr = block["rows"][positions]
                order = np.argsort(rr)
                with torch.no_grad():
                    z = model.encoder(
                        tensor(
                            prior.profile_batch(
                                data, rr[order], "phot_xp_measured_parallax", source
                            )
                        )
                    )
                z = z[tensor(np.argsort(order), torch.long)]
                gen = (
                    head.sample_generator
                    if seed is None
                    else torch.Generator(device=device).manual_seed(seed)
                )
                draw = head.sample(z, n, gen)
                if block["source"] < 6:
                    draw = maps(
                        draw,
                        torch.full(
                            (len(rr),), block["source"], device=device, dtype=torch.long
                        ),
                        tensor(block["observed"][positions], torch.bool),
                    )
                return draw

            panels = {}
            for role in ["tune", "cal", "devtest"]:
                panels[role] = []
                for b in records[role]:
                    # Reference selection is blind to fitted models and kept identical across arms.
                    q = np.sort(
                        np.random.default_rng(cfg["seed"] + b["source"]).choice(
                            len(b["rows"]),
                            min(cfg.get("mapping_panel_rows", 512), len(b["rows"])),
                            replace=False,
                        )
                    )
                    panels[role].append(q)

            def evaluate(role, export=False):
                head.eval()
                maps.eval()
                output = []
                edge_scores = []
                for b, positions in zip(records[role], panels[role], strict=True):
                    if not len(positions):
                        continue
                    chunks = []
                    for start in range(0, len(positions), 8):
                        pp = positions[start : start + 8]
                        with torch.no_grad():
                            draw = (
                                draw_for(
                                    b,
                                    pp,
                                    cfg.get("mapping_evaluation_draws", 512),
                                    cfg["seed"] + 7000 + b["source"] * 1000 + start,
                                )
                                .cpu()
                                .numpy()
                            )
                        noise = (
                            np.random.default_rng(
                                cfg["seed"] + 8000 + b["source"] * 1000 + start
                            ).standard_normal(draw.shape)
                            * b["error"][pp][None]
                        )
                        scores = sample_scores(
                            draw + noise, b["value"][pp], b["observed"][pp]
                        )
                        chunks.append(
                            dict(
                                source_id=ids[b["rows"][pp]],
                                crps=scores["crps"],
                                pit=scores["pit"],
                                quantiles=scores["quantiles"],
                                reference=b["value"][pp],
                                observed=b["observed"][pp],
                                intrinsic_median=np.median(draw, axis=0),
                            )
                        )
                    arr = {
                        k: np.concatenate(
                            [v[k] for v in chunks], axis=1 if k == "quantiles" else 0
                        )
                        for k in chunks[0]
                    }
                    per = np.nanmean(arr["crps"], axis=0, where=arr["observed"])
                    # Empty source-target edges are absent, not zero scores.
                    edge_scores.extend(per[np.isfinite(per)].tolist())
                    summary = dict(
                        source=SOURCE_NAMES[b["source"]],
                        rows=len(positions),
                        per_target_crps=[
                            float(v) if np.isfinite(v) else None for v in per
                        ],
                    )
                    for j in np.flatnonzero(arr["observed"].any(0)):
                        active = arr["observed"][:, j]
                        summary.setdefault("targets", {})[prior.NAMES[j]] = dict(
                            n=int(active.sum()),
                            mae=float(
                                np.mean(
                                    abs(
                                        arr["intrinsic_median"][active, j]
                                        - arr["reference"][active, j]
                                    )
                                )
                                * scale[j]
                            ),
                            bias=float(
                                np.mean(
                                    arr["intrinsic_median"][active, j]
                                    - arr["reference"][active, j]
                                )
                                * scale[j]
                            ),
                            coverage90=float(
                                (
                                    (arr["pit"][active, j] >= 0.05)
                                    & (arr["pit"][active, j] <= 0.95)
                                ).mean()
                            ),
                        )
                    output.append(summary)
                    if export:
                        np.savez_compressed(
                            a.output
                            / (role + "-" + SOURCE_NAMES[b["source"]] + ".npz"),
                            center=center,
                            scale=scale,
                            **arr,
                        )
                head.train()
                maps.train()
                return float(np.mean(edge_scores)), output

            best = float("inf")
            selected = 0
            fit = [
                b
                for b in records["fit"]
                if len(b["rows"]) and (b["source"] != 6 or arm["gauge"] == "seismic")
            ]
            if arm["gauge"] == "seismic" and not len(records["fit"][6]["rows"]):
                raise ValueError("No FIT seismic anchors")
            starts = time.monotonic()
            for update in range(1, cfg["mapping_updates"] + 1):
                b = fit[(update - 1) % len(fit)]
                # Logg strata remove sample-count domination without restricting eligible rows.
                pool = np.arange(len(b["rows"]))
                if b["source"] < 3 and b["observed"][:, 1].any() and update % 2:
                    physical = b["value"][:, 1] * scale[1] + center[1]
                    bins = np.searchsorted([1, 2, 3, 4, 4.5], physical)
                    bands = np.unique(bins[b["observed"][:, 1]])
                    band = bands[(update // len(fit)) % len(bands)]
                    pool = pool[b["observed"][:, 1] & (bins == band)]
                pp = rng.choice(pool, size=min(128, len(pool)), replace=False)
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for start in range(0, len(pp), 16):
                    chunk = pp[start : start + 16]
                    draw = draw_for(b, chunk, 32)
                    loss, components = confirmation_score(
                        draw,
                        tensor(b["value"][chunk]),
                        tensor(b["observed"][chunk], torch.bool),
                        tensor(b["error"][chunk]),
                        tensor(b["known"][chunk], torch.bool),
                        tensor(cfg["unknown_error_floor_normalized"]),
                        head.noise_generator,
                        energy_weight=1.0,
                        crps_weight=1.0,
                    )
                    fraction = len(chunk) / len(pp)
                    (loss * fraction).backward()
                    total += float(loss.detach()) * fraction
                reg = 1e-3 * maps.regularization()
                reg.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in list(head.parameters()) + list(maps.parameters())
                ):
                    raise FloatingPointError("Nonfinite head/map gradient")
                head_norm = (
                    sum(
                        float(p.grad.detach().square().sum())
                        for p in head.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                if not np.isfinite(head_norm) or head_norm <= 0:
                    raise FloatingPointError(
                        "Source-mapped objective did not reach the NSF"
                    )
                torch.nn.utils.clip_grad_norm_(
                    list(head.parameters()) + list(maps.parameters()), 10.0
                )
                optimizer.step()
                m.update(
                    completed_updates=update,
                    finite_gradient_updates=update,
                    training_performed=True,
                    last_objective=total,
                    last_source=SOURCE_NAMES[b["source"]],
                    elapsed_seconds=time.monotonic() - starts,
                    last_head_gradient_l2=head_norm,
                )
                if update <= 2 or update % 100 == 0:
                    save()
                    print(
                        json.dumps(
                            dict(run_id=a.run_id, update=update, objective=total)
                        ),
                        flush=True,
                    )
                if (
                    update % cfg["mapping_tune_every"] == 0
                    or update == cfg["mapping_updates"]
                ):
                    score, report = evaluate("tune")
                    if not np.isfinite(score):
                        raise FloatingPointError("Nonfinite TUNE proper score")
                    m.setdefault("tune_history", []).append(
                        dict(update=update, macro_edge_crps=score)
                    )
                    if score < best:
                        best, selected = score, update
                        torch.save(
                            dict(
                                run_id=a.run_id,
                                arm=arm,
                                head_state_dict=head.state_dict(),
                                map_state_dict=maps.state_dict(),
                                parent_checkpoint_sha256=cfg["mapping_parent_sha256"],
                                center=center,
                                scale=scale,
                                source_registry=SOURCE_NAMES,
                                support=support,
                                reference=reference,
                                zero_sum=zero_sum,
                                update=update,
                                tune_score=score,
                            ),
                            a.output / "selected-checkpoint.pth.partial",
                        )
                        (a.output / "selected-checkpoint.pth.partial").replace(
                            a.output / "selected-checkpoint.pth"
                        )
                    save()
            state = torch_load_trusted(
                a.output / "selected-checkpoint.pth",
                map_location=device,
                weights_only=False,
            )
            head.load_state_dict(state["head_state_dict"], strict=True)
            maps.load_state_dict(state["map_state_dict"], strict=True)
            m["evaluation"] = {}
            for role in ["tune", "cal", "devtest"]:
                score, report = evaluate(role, True)
                m["evaluation"][role] = dict(macro_edge_crps=score, sources=report)
                save()
            head.eval()
            maps.eval()
            # Common-space populations use model-blind identities, with no survey map.
            for role, filename in [
                ("fit", "training-predictions.npz"),
                ("devtest", "development-predictions.npz"),
            ]:
                candidates = np.flatnonzero(
                    (roles == role) & eligible & ~np.isin(ids, blocked)
                )
                panel = np.sort(
                    np.random.default_rng(cfg["seed"] + 9100).choice(
                        candidates,
                        min(
                            cfg.get("mapping_common_panel_rows", 2048), len(candidates)
                        ),
                        replace=False,
                    )
                )
                medians = []
                for start in range(0, len(panel), 8):
                    rows = panel[start : start + 8]
                    with torch.no_grad():
                        z = model.encoder(
                            tensor(
                                prior.profile_batch(
                                    data, rows, "phot_xp_measured_parallax", source
                                )
                            )
                        )
                        gen = torch.Generator(device=device).manual_seed(
                            cfg["seed"] + 9200 + start
                        )
                        draw = prior.bounded_whole_draws(
                            head, z, cfg.get("mapping_population_draws", 256), gen, 128
                        )
                    native_draw = draw * scale + center
                    native_draw[:, :, 4:] = np.exp(native_draw[:, :, 4:])
                    if not np.isfinite(native_draw).all():
                        raise FloatingPointError(
                            "Nonfinite native common-space population draws"
                        )
                    medians.append(np.median(native_draw, axis=0))
                np.savez_compressed(
                    a.output / filename,
                    source_id=ids[panel],
                    role=np.full(len(panel), role),
                    native_median=np.concatenate(medians),
                    value=values[panel],
                    observed=valid[panel],
                    label_source=label_source[panel],
                )
            corners = prior.heldout_corner_rows(ids, roles, cfg["corner_source_ids"])
            draws = []
            for start in range(0, len(corners), 4):
                rows = corners[start : start + 4]
                with torch.no_grad():
                    z = model.encoder(
                        tensor(
                            prior.profile_batch(
                                data, rows, "phot_xp_measured_parallax", source
                            )
                        )
                    )
                    gen = torch.Generator(device=device).manual_seed(
                        cfg["seed"] + 9300 + start
                    )
                    draws.append(
                        prior.bounded_whole_draws(
                            head, z, cfg.get("mapping_corner_draws", 2048), gen, 128
                        )
                    )
            if draws:
                draws = np.concatenate(draws, axis=1)
                native_draws = draws * scale + center
                native_draws[:, :, 4:] = np.exp(native_draws[:, :, 4:])
                if not np.isfinite(native_draws).all():
                    raise FloatingPointError("Nonfinite native single-star draws")
                np.savez_compressed(
                    a.output / "whole-vector-corners.npz",
                    source_id=ids[corners],
                    standardized_draws=draws,
                    native_draws=native_draws,
                    reference=values[corners],
                    observed=valid[corners],
                )
            bias, slopes = maps.coefficients()
            m.update(
                status="complete",
                selected_update=selected,
                tune_score=best,
                selected_checkpoint_sha256=prior.sha(
                    a.output / "selected-checkpoint.pth"
                ),
                map_bias_normalized=bias.detach().cpu().numpy(),
                map_slopes=slopes.detach().cpu().numpy(),
                selection="First minimum TUNE macro observed source-target marginal CRPS; CAL and DEV opened only after selected state is frozen",
            )
            save()
        del model, head, maps
        gc.collect()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
