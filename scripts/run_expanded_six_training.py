"""Autonomous, gated large six-target NSF training with source-native sample maps.

Complete FIT record passes plus explicit sparse-label/anchor auxiliary streams.
No publication or absolute physical-calibration claim is made by this job.
"""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def prepare_output(output, reuse_waiting=False, retry_failed=False):
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        return
    if not (reuse_waiting or retry_failed):
        raise FileExistsError(
            "Existing experiment output requires explicit pre-admission reconciliation"
        )
    prior = json.loads((output / "manifest.json").read_text())
    if (
        (
            prior.get("status") != "waiting_for_decision"
            and not (retry_failed and prior.get("status") == "failed")
        )
        or prior.get("training_performed")
        or prior.get("completed_updates") != 0
        or list(output.glob("*.pth*"))
        or (output / "architecture-decision.json").exists()
    ):
        raise ValueError(
            "Cannot reuse an admitted, optimizing or checkpointed experiment"
        )
    attempt = 1
    while (output / f"pre-admission-attempt-{attempt}.json").exists():
        attempt += 1
    archive = output / f"pre-admission-attempt-{attempt}.json"
    with archive.open("xb") as f:
        f.write((output / "manifest.json").read_bytes())


def wait_decision(cfg, output, atomic):
    deadline = time.monotonic() + 36 * 3600
    decision = {}
    while time.monotonic() < deadline:
        pending = []
        runs = {}
        for rid in cfg["decision_runs"]:
            try:
                m = json.loads((ROOT / rid / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                m = {}
            if m.get("status") not in [
                "complete",
                "remediation_pilot_complete",
                "failed",
            ]:
                pending.append(rid)
            runs[rid] = m
        if not pending:
            expanded = runs[cfg["expanded_source_run"]]
            if expanded.get("status") != "expanded_six_source_complete":
                # Expanded data has a separate terminal status.
                raise RuntimeError(
                    "Expanded training source did not pass: "
                    + str(expanded.get("error"))
                )
            break
        # The source's status name is deliberately handled explicitly, not as failure.
        source = cfg["expanded_source_run"]
        if runs[source].get("status") == "expanded_six_source_complete":
            pending = [r for r in pending if r != source]
        if not pending:
            break
        atomic(
            output / "decision-wait.json",
            dict(status="waiting_for_science_and_source_gates", pending=pending),
        )
        time.sleep(30)
    else:
        raise TimeoutError("Science/source gates did not complete in 36h")
    expanded = runs[cfg["expanded_source_run"]]
    if (
        expanded.get("status") != "expanded_six_source_complete"
        or expanded.get("training_source_ready") is not True
    ):
        raise RuntimeError("Expanded source failed readiness")
    family_scores = {}
    for family in ["energy", "hybrid"]:
        members = [runs[f"nsfconfirm_{family}_s{i}_20261106_v1"] for i in [1, 2]]
        if not all(m.get("status") == "remediation_pilot_complete" for m in members):
            raise RuntimeError("A required paired NSF objective control failed")
        family_scores[family] = float(np.mean([m["tune_score"] for m in members]))
    objective = min(family_scores, key=family_scores.get)
    noise_choice = dict(
        run_id=f"nsfconfirm_{objective}_s1_20261106_v1",
        correlation=0.0,
        native_student_t=False,
        unknown_floor=0.1,
    )
    # Matched-seed sensitivity screen. A material TUNE gain can choose an
    # operational noise assumption; it does not estimate a physical covariance.
    if objective == "hybrid":
        variants = [
            ("floor003", 0.0, False, 0.03),
            ("corrpos", 0.3, False, 0.1),
            ("corrneg", -0.3, False, 0.1),
            ("studentt", 0.0, True, 0.1),
        ]
        scored = []
        for name, rho, student, floor in variants:
            rid = f"nsfconfirm_hybrid_{name}_20261106_v1"
            item = runs.get(rid, {})
            if item.get("status") == "remediation_pilot_complete":
                scored.append((item["tune_score"], rid, rho, student, floor))
        if scored:
            score, rid, rho, student, floor = min(scored)
            if score < 0.98 * runs["nsfconfirm_hybrid_s1_20261106_v1"]["tune_score"]:
                noise_choice = dict(
                    run_id=rid,
                    correlation=rho,
                    native_student_t=student,
                    unknown_floor=floor,
                )
    candidates = []
    for rid in cfg["mapping_candidates"]:
        m = runs[rid]
        if m.get("status") != "complete":
            continue
        seismic = next(
            v
            for v in m["evaluation"]["tune"]["sources"]
            if v["source"] == "APOKASC3_loggSeis"
        )["targets"]["logg"]
        # This is a large training candidate, not a physical PDF release gate.
        if (
            abs(seismic["bias"]) <= 0.05
            and seismic["mae"] <= 0.1
            and seismic["coverage90"] >= 0.8
        ):
            candidates.append((m["tune_score"], rid, m, seismic))
    if not candidates:
        raise RuntimeError(
            "No anchored mapping meets the TUNE gravity bias/accuracy/coverage candidate gates"
        )
    _, rid, chosen, seismic = min(candidates, key=lambda v: v[0])
    cal = runs[cfg["calibration_run"]]
    if cal.get("status") != "complete":
        raise RuntimeError("Separate CAL/DEV diagnostics did not finish")
    for seed in [1, 2]:
        r = f"nsfconfirm_{objective}_s{seed}_20261106_v1"
        if cal["results"].get(r, {}).get("status") != "complete":
            raise RuntimeError("Selected NSF objective lacks separate CAL diagnostics")
    if cal["results"].get(noise_choice["run_id"], {}).get("status") != "complete":
        raise RuntimeError(
            "Selected operational noise assumption lacks CAL diagnostics"
        )
    decision = dict(
        status="large_candidate_admitted",
        objective=objective,
        noise_choice=noise_choice,
        paired_tune_macro_crps=family_scores,
        mapping_run=rid,
        mapping_checkpoint_sha256=chosen["selected_checkpoint_sha256"],
        mapping_kind=chosen["arm"]["map_kind"],
        mapping_tune_seismic=seismic,
        expanded_source_sha256=expanded["prepared_sha256"],
        noise="TUNE-screened operational noise assumption; correlated/Student-t changes apply only to known-error, complete APOGEE/GALAH atmospheric blocks as in the controls. Other quoted errors use independent Gaussian convolution; unknown-error floors remain illustrative. A single-seed sensitivity gain is provisional, not identified physical covariance. No XP Gaussian NLL reconstruction.",
        coverage_scope="Candidate gates tolerate remaining raw underdispersion. CAL interval diagnostics do not alter or certify a physical joint PDF. Final export needs post-training QA.",
        selection="Paired TUNE objective comparison and same-panel anchored TUNE map comparisons. DEV outcomes do not select this run.",
        all_decision_statuses={r: m.get("status") for r, m in runs.items()},
    )
    atomic(output / "architecture-decision.json", decision)
    return decision, expanded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-waiting-output", action="store_true")
    p.add_argument("--retry-pre-admission-failure", action="store_true")
    a = p.parse_args()
    prepare_output(a.output, a.reuse_waiting_output, a.retry_pre_admission_failure)
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.cached_inputs import cache_natural_inputs
    from masked_stellar_autoencoder.training.decision_pdf_metrics import sample_scores
    from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
        _cache_maps,
    )
    from masked_stellar_autoencoder.training.fullfit_records import (
        epoch_record_batches,
        fixed_coordinates,
        reconstruction_block_start,
        xp_dropout_inputs,
        xp_input_columns,
    )
    from masked_stellar_autoencoder.training.nsf_confirmation import confirmation_score
    from masked_stellar_autoencoder.training.remediation_energy6 import (
        EnergyJointSixHead,
        capture_pilot_rng_state,
    )
    from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap

    cfg = yaml.safe_load(a.config.read_text())
    m = dict(
        status="waiting_for_decision",
        run_id=cfg["run_id"],
        training_performed=False,
        completed_updates=0,
        completed_full_fit_epochs=0,
        script_sha256=prior.sha(__file__),
        config_sha256=prior.sha(a.config),
        scope="Large expanded multi-survey six-target NSF candidate, full FIT record epochs and retained masked MAE reconstruction; age/distance are mapped posterior-teacher estimands. No model publication or physical six-dimensional coverage claim.",
    )

    def save():
        prior.atomic(a.output / "manifest.json", m)

    save()
    try:
        decision, expanded = wait_decision(cfg, a.output, prior.atomic)
        cfg["unknown_error_floor_normalized"] = [
            decision["noise_choice"]["unknown_floor"]
        ] * 6
        m.update(status="admitting", decision=decision)
        save()
        if not torch.cuda.is_available() or not ROOT.is_dir():
            raise RuntimeError("CANFAR CUDA required")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
            raise ValueError("Library revision differs")
        for path, expected in [
            (expanded["prepared_path"], expanded["prepared_sha256"]),
            (cfg["encoder_parent"], cfg["encoder_parent_sha256"]),
            (cfg["reference_checkpoint"], cfg["reference_checkpoint_sha256"]),
        ]:
            if prior.sha(path) != expected:
                raise ValueError("Immutable input/model changed: " + str(path))
        mapping_path = ROOT / decision["mapping_run"] / "selected-checkpoint.pth"
        if prior.sha(mapping_path) != decision["mapping_checkpoint_sha256"]:
            raise ValueError("Selected map checkpoint changed")
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        reconstruction_rng = np.random.default_rng(cfg["seed"] + 1901)
        device = torch.device("cuda")
        reference = torch_load_trusted(
            cfg["reference_checkpoint"], map_location="cpu", weights_only=False
        )
        parent = torch_load_trusted(
            cfg["encoder_parent"], map_location="cpu", weights_only=False
        )
        selected = torch_load_trusted(
            mapping_path, map_location="cpu", weights_only=False
        )
        if (
            selected["parent_checkpoint_sha256"] != cfg["encoder_parent_sha256"]
            or selected["arm"]["gauge"] != "seismic"
        ):
            raise ValueError(
                "Selected mapping has different encoder lineage or gravity convention"
            )
        model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
        model = native.probe.load_model(model_cfg, reference, device)
        model.load_state_dict(parent["model_state_dict"], strict=True)
        prior.validate_microbatch_encoder(model.encoder, finetune=True)
        model.requires_grad_(True)
        head = EnergyJointSixHead(
            latent_dim=model_cfg["model"]["layer_dims"][-1],
            width=cfg["head_width"],
            particles=cfg["particles"],
            seed=cfg["seed"],
            components=cfg["mixture_components"],
            energy_training_draws=32,
            unknown_error_floor_normalized=cfg["unknown_error_floor_normalized"],
        ).to(device)
        head.load_state_dict(selected["head_state_dict"], strict=True)
        head.unknown_error_floor_normalized.copy_(
            torch.as_tensor(cfg["unknown_error_floor_normalized"], device=device)
        )
        head.configure_generators(device, cfg["seed"] + 101, cfg["seed"] + 102)
        center, scale = np.asarray(selected["center"]), np.asarray(selected["scale"])
        source_names = list(expanded["source_registry"])
        if source_names[:6] != selected["source_registry"][:6]:
            raise ValueError("Source registry prefix differs from selected map")
        refs = (
            ROOT
            / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
        )
        rm = json.loads((refs.parent / "manifest.json").read_text())
        if prior.sha(refs) != rm["sources"]["apokasc3"]["reference_sha256"]:
            raise ValueError("Seismic anchor changed")
        ref = pd.read_parquet(refs)
        refidx = {int(v): i for i, v in enumerate(ref.source_id)}
        with h5py.File(expanded["prepared_path"]) as data:
            if data.attrs.get(
                "schema"
            ) != "expanded-sparse-six-target-source-v1" or not data.attrs.get(
                "complete"
            ):
                raise ValueError("Expanded source schema/completion differs")
            ids = data["source_id"][:]
            roles = data["role"][:].astype("S8")
            eligible = data["primary_input_eligible"][:].astype(bool)
            names = data["feature_names"][:].astype("U").tolist()
            if names != reference["run_signature"]["feature_cols"]:
                raise ValueError("Input feature identities differ from encoder")
            for key in ["scaler_center", "scaler_scale"]:
                np.testing.assert_array_equal(
                    data[key][:], reference["run_signature"][key]
                )
            records = {role: [] for role in ["fit", "tune", "cal", "devtest"]}
            support = []
            for si, name in enumerate(source_names):
                g = data["native_observations/" + name]
                rows = g["star_row"][:]
                y, sigma, obs, known = fixed_coordinates(
                    g["value"][:],
                    g["error"][:],
                    g["observed"][:],
                    g["error_known"][:],
                    center,
                    scale,
                )
                support.append(obs.any(0))
                for role in records:
                    take = (roles[rows] == role.encode()) & eligible[rows] & obs.any(1)
                    records[role].append(
                        dict(
                            source=si,
                            rows=rows[take],
                            value=y[take],
                            error=sigma[take],
                            observed=obs[take],
                            known=known[take],
                        )
                    )
            anchor = len(source_names)
            source_names.append("APOKASC3_loggSeis")
            for role in records:
                sr = np.flatnonzero((roles == role.encode()) & eligible)
                sr = sr[np.array([int(v) in refidx for v in ids[sr]])]
                ix = np.array([refidx[int(v)] for v in ids[sr]])
                v = np.zeros((len(sr), 6), dtype="f4")
                e = v.copy()
                obs = np.zeros(v.shape, bool)
                obs[:, 1] = True
                v[:, 1] = (ref.loggSeis.to_numpy()[ix] - center[1]) / scale[1]
                e[:, 1] = ref.e_loggSeis.to_numpy()[ix] / scale[1]
                if (
                    not np.isfinite(v).all()
                    or not np.isfinite(e).all()
                    or (e[:, 1] <= 0).any()
                ):
                    raise ValueError("Invalid seismic anchor values/errors")
                records[role].append(
                    dict(
                        source=anchor,
                        rows=sr,
                        value=v,
                        error=e,
                        observed=obs,
                        known=obs.copy(),
                    )
                )
            support = np.asarray(support)
            # Optional per-source map families (config ``source_map_kinds``):
            # the six selected rows keep the decision family; other sources
            # default to it unless the config names a family for them.
            kinds_cfg = cfg.get("source_map_kinds") or {}
            unknown = set(kinds_cfg) - set(source_names[:anchor])
            if unknown or any(source_names.index(k) < 6 for k in kinds_cfg):
                raise ValueError(
                    "source_map_kinds must name only added sources: "
                    + ", ".join(sorted(unknown) or kinds_cfg)
                )
            source_kinds = [
                decision["mapping_kind"]
                if si < 6
                else kinds_cfg.get(name, decision["mapping_kind"])
                for si, name in enumerate(source_names[:anchor])
            ]
            decision["source_map_kinds"] = source_kinds
            maps = SixSurveyMap(
                support,
                selected["reference"],
                zero_sum=selected["zero_sum"],
                kind=decision["mapping_kind"],
                source_kinds=source_kinds,
            ).to(device)
            with torch.no_grad():
                saved_slope = selected["map_state_dict"]["raw_slope"]
                if saved_slope.shape[-1] == 1 and maps.raw_slope.shape[-1] > 1:
                    # An affine row is the constant-slope spline: exact copy.
                    saved_slope = saved_slope.expand(-1, -1, maps.raw_slope.shape[-1])
                maps.bias[:6].copy_(selected["map_state_dict"]["bias"])
                maps.raw_slope[:6].copy_(saved_slope)
            m.update(source_map_kinds=dict(zip(source_names[:anchor], source_kinds)))
            save()
            del selected, parent, reference
            fit = records["fit"]
            if not len(fit[anchor]["rows"]) or not all(
                len(fit[i]["rows"]) for i in [3, 4, 5]
            ):
                raise ValueError("Sparse teacher/seismic auxiliary streams are empty")
            fit_lengths = [len(b["rows"]) if b["source"] != anchor else 0 for b in fit]
            fit_records = sum(fit_lengths)
            fit_unique = np.unique(
                np.concatenate([b["rows"] for b in fit if len(b["rows"])])
            )
            fit_target_counts = np.sum([b["observed"].sum(0) for b in fit], axis=0)
            if (fit_target_counts == 0).any():
                raise ValueError("A target has no FIT labels")
            cache, _, _, cache_manifest = _cache_maps(
                cfg["reconstruction_cache"],
                names,
                data["scaler_center"][:],
                data["scaler_scale"][:],
                cfg["reconstruction_request_sha256"],
                cfg["reconstruction_training_rows"],
            )
            xp = [j for j, n in enumerate(names) if n.startswith(("bp_", "rp_"))]
            allowed = np.array(
                [
                    n in {"W1", "W2", "G", "BP", "RP"} or j in xp
                    for j, n in enumerate(names)
                ]
            )
            width = len(model_cfg["data"]["recon_cols"])
            if names[:width] != model_cfg["data"]["recon_cols"]:
                raise ValueError("Reconstruction feature order differs")
            mask_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 1701)
            tensor = lambda v, dtype=torch.float32: torch.as_tensor(
                v, device=device, dtype=dtype
            )
            optimizer = torch.optim.Adam(
                [
                    dict(params=head.parameters(), lr=cfg["head_lr"]),
                    dict(params=maps.parameters(), lr=cfg["survey_lr"]),
                    dict(params=model.parameters(), lr=cfg["encoder_lr"]),
                ]
            )
            m.update(
                status="cuda_preflight",
                prepared_sha256=expanded["prepared_sha256"],
                source_registry=source_names,
                supported_edges=support,
                fit_unique_stars=len(fit_unique),
                fit_records_per_epoch=fit_records,
                fit_native_or_teacher_labels_per_target=fit_target_counts,
                fit_source_records={
                    source_names[b["source"]]: len(b["rows"]) for b in fit
                },
                reconstruction_request_sha256=cache_manifest["request_sha256"],
                target_names=list(prior.NAMES),
                target_coordinates=[
                    "K",
                    "dex(cgs)",
                    "dex Fe/H",
                    "dex alpha/Fe",
                    "ln(Gyr posterior median)",
                    "ln(kpc posterior median)",
                ],
                reconstruction_loss="MAE",
                reconstruction_weight=0.03,
                reconstruction_sampling="Independent seed +1901; contiguous random blocks across all 150M cache rows; no fixed prefix. RNG states and epoch-sampler initial state retained in latest checkpoint for recovery.",
                full_pass_contract="Every admitted non-anchor FIT observation record once per epoch; seismic and age/distance teacher auxiliary records are additional explicit repeats, not extra unique stars.",
            )
            save()

            started_loading = time.monotonic()
            natural_inputs = cache_natural_inputs(
                data, expanded["feature_profiles"]["phot_xp_measured_parallax"]
            )
            checked = np.linspace(0, len(ids) - 1, min(128, len(ids)), dtype="i8")
            np.testing.assert_array_equal(
                natural_inputs[checked],
                prior.profile_batch(
                    data, checked, "phot_xp_measured_parallax", expanded
                ),
            )
            m["input_loading"] = dict(
                mode="read_only_RAM",
                bytes=natural_inputs.nbytes,
                source_sha256=expanded["prepared_sha256"],
                exact_profile_rows_checked=len(checked),
                load_seconds=time.monotonic() - started_loading,
                scope="Bit-identical admitted input profile; no change to identities, values, masks or normalizations",
            )
            save()

            def collate(sources, positions, auxiliary=False):
                pieces = []
                for si in np.unique(sources):
                    pp = positions[sources == si]
                    b = fit[int(si)]
                    pieces.append(
                        dict(
                            source=np.full(len(pp), si, dtype="i8"),
                            **{
                                k: b[k][pp]
                                for k in ["rows", "value", "error", "observed", "known"]
                            },
                        )
                    )
                if auxiliary:
                    for si, count in [(anchor, 32), (3, 16), (4, 8), (5, 8)]:
                        b = fit[si]
                        pp = rng.choice(
                            len(b["rows"]),
                            size=min(count, len(b["rows"])),
                            replace=False,
                        )
                        pieces.append(
                            dict(
                                source=np.full(len(pp), si, dtype="i8"),
                                **{
                                    k: b[k][pp]
                                    for k in [
                                        "rows",
                                        "value",
                                        "error",
                                        "observed",
                                        "known",
                                    ]
                                },
                            )
                        )
                return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}

            xp_cols = xp_input_columns(names)
            xp_dropout = float(cfg.get("supervised_xp_dropout", 0.0))
            xp_hidden_weight = float(cfg.get("selection_xp_hidden_weight", 0.0))
            if not (0.0 <= xp_dropout < 1.0 and 0.0 <= xp_hidden_weight <= 1.0):
                raise ValueError("XP dropout/selection weights must lie in [0, 1)")

            def encoder_inputs(unique, *, hide_all=False, rate=0.0, seed=None):
                # Hidden XP uses the encoder's own missing token (= real no-XP).
                return xp_dropout_inputs(
                    natural_inputs[unique],
                    xp_cols,
                    hide_all=hide_all,
                    rate=rate,
                    seed=seed,
                )

            def forward_draws(
                batch,
                start,
                stop,
                count,
                seed=None,
                *,
                hide_xp=False,
                xp_rate=0.0,
                xp_seed=None,
            ):
                rr = batch["rows"][start:stop]
                unique, inverse = np.unique(rr, return_inverse=True)
                z = model.encoder(
                    tensor(
                        encoder_inputs(
                            unique, hide_all=hide_xp, rate=xp_rate, seed=xp_seed
                        )
                    )
                )
                z = z[tensor(inverse, torch.long)]
                gen = (
                    head.sample_generator
                    if seed is None
                    else torch.Generator(device=device).manual_seed(seed)
                )
                draws = head.sample(z, count, gen)
                si = tensor(batch["source"][start:stop], torch.long)
                mapped = maps(draws, si.clamp(max=anchor - 1))
                return torch.where((si == anchor)[None, :, None], draws, mapped)

            def step(batch, update, *, warm=False):
                model.train(not warm)
                head.train(not warm)
                maps.train()
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for start in range(0, len(batch["rows"]), 16):
                    stop = min(start + 16, len(batch["rows"]))
                    draws = forward_draws(
                        batch,
                        start,
                        stop,
                        32,
                        xp_rate=0.0 if warm else xp_dropout,
                        xp_seed=[cfg["seed"], 31, update, start],
                    )
                    loss, _ = confirmation_score(
                        draws,
                        tensor(batch["value"][start:stop]),
                        tensor(batch["observed"][start:stop], torch.bool),
                        tensor(batch["error"][start:stop]),
                        tensor(batch["known"][start:stop], torch.bool),
                        tensor(cfg["unknown_error_floor_normalized"]),
                        head.noise_generator,
                        energy_weight=1.0,
                        crps_weight=1.0 if decision["objective"] == "hybrid" else 0.0,
                        native_rows=tensor(
                            np.isin(batch["source"][start:stop], [0, 2])
                            & batch["observed"][start:stop, :3].all(1)
                            & batch["known"][start:stop, :3].all(1),
                            torch.bool,
                        ),
                        correlation=decision["noise_choice"]["correlation"],
                        native_student_t=decision["noise_choice"]["native_student_t"],
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Nonfinite source-native NSF sample score"
                        )
                    weight = (stop - start) / len(batch["rows"])
                    (loss * weight).backward()
                    total += float(loss.detach()) * weight
                reg = 1e-3 * maps.regularization()
                reg.backward()
                if warm:
                    if maps.bias.grad is not None:
                        maps.bias.grad[:6] = 0
                    if maps.raw_slope.grad is not None:
                        maps.raw_slope.grad[:6] = 0
                supervised = (
                    sum(
                        float(p.grad.detach().square().sum())
                        for p in model.encoder.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                reco = torch.zeros((), device=device)
                cache_start = None
                if not warm:
                    start = reconstruction_block_start(
                        len(cache), cfg["reconstruction_batch"], reconstruction_rng
                    )
                    cache_start = start
                    raw = np.asarray(
                        cache[
                            start : min(start + cfg["reconstruction_batch"], len(cache))
                        ]
                    ).copy()
                    raw[(~np.isfinite(raw)) | (raw == -9999)] = np.nan
                    raw[:, ~allowed] = np.nan
                    target = tensor(raw)
                    hidden, hm, finite = native.pretrain.training_mask(
                        target, xp, 0.5, 0.2, mask_rng
                    )
                    pred, _ = model(hidden)
                    seen = (hm & finite)[:, :width]
                    if not seen.any():
                        raise ValueError("Empty masked reconstruction targets")
                    reco = native.pretrain.experiment_objective(
                        pred, target[:, :width], seen, xp, {"loss": "mae"}, seen.sum()
                    )
                    if (
                        not torch.isfinite(reco)
                        or not np.isfinite(supervised)
                        or supervised <= 0
                    ):
                        raise FloatingPointError(
                            "Supervised encoder gradient or reconstruction is invalid"
                        )
                    if update <= 2:
                        contract = prior.accumulated_encoder_gradient_contract(
                            reco, model.encoder.parameters()
                        )
                        if (
                            contract["supervised_l2"] <= 0
                            or contract["reconstruction_unweighted_l2"] <= 0
                        ):
                            raise FloatingPointError(
                                "Both supervised and reconstruction streams must reach the encoder"
                            )
                        m.setdefault("startup_encoder_gradient_contract", []).append(
                            dict(update=update, **contract)
                        )
                    (0.03 * reco).backward()
                    if update <= 2:
                        encoder_ids = {id(p) for p in model.encoder.parameters()}
                        if not any(
                            p.grad is not None and p.grad.ne(0).any()
                            for p in model.parameters()
                            if id(p) not in encoder_ids
                        ):
                            raise FloatingPointError(
                                "Reconstruction did not reach decoder parameters"
                            )
                gradients = [
                    p.grad
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError(
                        "Missing or nonfinite production gradients"
                    )
                if not warm and not any(
                    p.grad is not None and p.grad.ne(0).any() for p in head.parameters()
                ):
                    raise FloatingPointError("NSF did not receive a gradient")
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    10.0,
                )
                optimizer.step()
                return dict(
                    supervised_objective=total,
                    reconstruction_mae=float(reco.detach()),
                    reconstruction_cache_start=cache_start,
                    supervised_encoder_gradient_l2=supervised,
                )

            # Two full objective steps exercise encoder, decoder, maps, six targets and anchors.
            # Restore parameters and RNGs before any warmup/full epoch; this is not counted as a FIT pass.
            initial = dict(
                model={
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                head={
                    k: v.detach().cpu().clone() for k, v in head.state_dict().items()
                },
                maps={
                    k: v.detach().cpu().clone() for k, v in maps.state_dict().items()
                },
                rng=rng.bit_generator.state,
                reconstruction_rng=reconstruction_rng.bit_generator.state,
                cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
                sample=head.generator_state(),
                mask=mask_rng.get_state(),
            )
            smoke_pairs = []
            for si, block in enumerate(fit):
                for j in range(6):
                    available = np.flatnonzero(block["observed"][:, j])
                    if len(available):
                        smoke_pairs.append((si, int(available[0])))
            smoke_pairs = np.array(sorted(set(smoke_pairs)), dtype="i8")
            smoke = collate(smoke_pairs[:, 0], smoke_pairs[:, 1], True)
            for update in [1, 2]:
                step(smoke, update)
            model.load_state_dict(initial["model"])
            head.load_state_dict(initial["head"])
            maps.load_state_dict(initial["maps"])
            rng.bit_generator.state = initial["rng"]
            reconstruction_rng.bit_generator.state = initial["reconstruction_rng"]
            torch.set_rng_state(initial["cpu"])
            torch.cuda.set_rng_state_all(initial["cuda"])
            head.restore_generator_state(initial["sample"], device=device)
            mask_rng.set_state(initial["mask"])
            optimizer.state.clear()
            del initial
            prior.atomic(
                a.output / "device-preflight.json",
                dict(
                    status="pass",
                    full_objective_steps=2,
                    all_six_targets=bool(smoke["observed"].any(0).all()),
                    encoder_and_decoder_gradients_checked=True,
                    parameters_restored=True,
                ),
            )
            if not smoke["observed"].any(0).all():
                raise ValueError("CUDA smoke test did not exercise all six targets")
            # Warm only newly admitted survey maps against the anchored fixed stellar PDF.
            model.requires_grad_(False)
            head.requires_grad_(False)
            new_sources = [i for i in range(6, anchor) if len(fit[i]["rows"])]
            warm_presentations = np.zeros((anchor + 1, 6), dtype="i8")
            for update in range(1, cfg["warm_map_updates"] + 1):
                si = new_sources[(update - 1) % len(new_sources)]
                b = fit[si]
                pp = rng.choice(
                    len(b["rows"]), size=min(128, len(b["rows"])), replace=False
                )
                batch = collate(np.full(len(pp), si, dtype="i8"), pp)
                progress = step(batch, update, warm=True)
                warm_presentations[si] += batch["observed"].sum(0)
                if update <= 2 or update % 100 == 0:
                    m.update(
                        status="warming_new_survey_maps",
                        training_performed=True,
                        training_scope="Source-map warmup; encoder and NSF frozen",
                        warm_map_updates=update,
                        last_progress=progress,
                        warm_source_target_presentations=warm_presentations,
                    )
                    save()
            model.requires_grad_(True)
            # EnergyJointSixHead intentionally keeps its unused Gaussian proposal frozen.
            head.requires_grad_(True)
            head.gaussian.requires_grad_(False)
            optimizer.state.clear()
            panels = {}
            for role in ["tune", "cal", "devtest"]:
                panels[role] = [
                    np.sort(
                        np.random.default_rng(cfg["seed"] + b["source"]).choice(
                            len(b["rows"]),
                            min(cfg["evaluation_rows_per_source"], len(b["rows"])),
                            replace=False,
                        )
                    )
                    for b in records[role]
                ]

            def evaluate(role, export=False, hide_xp=False):
                model.eval()
                head.eval()
                maps.eval()
                edges = []
                reports = []
                for b, positions in zip(records[role], panels[role], strict=True):
                    if not len(positions):
                        continue
                    blocks = []
                    for start in range(0, len(positions), 8):
                        pp = positions[start : start + 8]
                        batch = dict(
                            source=np.full(len(pp), b["source"], dtype="i8"),
                            **{
                                k: b[k][pp]
                                for k in ["rows", "value", "error", "observed", "known"]
                            },
                        )
                        with torch.no_grad():
                            draw = (
                                forward_draws(
                                    batch,
                                    0,
                                    len(pp),
                                    cfg["evaluation_draws"],
                                    cfg["seed"] + 11000 + b["source"] * 1000 + start,
                                    hide_xp=hide_xp,
                                )
                                .cpu()
                                .numpy()
                            )
                        noisy = (
                            draw
                            + np.random.default_rng(
                                cfg["seed"] + 12000 + b["source"] * 1000 + start
                            ).standard_normal(draw.shape)
                            * batch["error"][None]
                        )
                        scores = sample_scores(noisy, batch["value"], batch["observed"])
                        blocks.append(
                            dict(
                                source_id=ids[batch["rows"]],
                                reference=batch["value"],
                                observed=batch["observed"],
                                crps=scores["crps"],
                                pit=scores["pit"],
                                quantiles=scores["quantiles"],
                                intrinsic_median=np.median(draw, axis=0),
                            )
                        )
                    arrays = {
                        k: np.concatenate(
                            [x[k] for x in blocks], axis=1 if k == "quantiles" else 0
                        )
                        for k in blocks[0]
                    }
                    active = arrays["observed"].any(0)
                    per = [
                        float(np.nanmean(arrays["crps"][:, j])) if active[j] else None
                        for j in range(6)
                    ]
                    edges.extend(v for v in per if v is not None)
                    reports.append(
                        dict(
                            source=source_names[b["source"]],
                            rows=len(positions),
                            per_target_crps=per,
                        )
                    )
                    if export:
                        np.savez_compressed(
                            a.output
                            / (
                                role
                                + ("-xp-hidden" if hide_xp else "")
                                + "-"
                                + source_names[b["source"]]
                                + ".npz"
                            ),
                            center=center,
                            scale=scale,
                            **arrays,
                        )
                return float(np.mean(edges)), reports

            def export_common(tag):
                model.eval()
                head.eval()
                maps.eval()
                for role in ["fit", "devtest"]:
                    available = np.flatnonzero((roles == role.encode()) & eligible)
                    panel = np.sort(
                        np.random.default_rng(cfg["seed"] + 20000).choice(
                            available, min(2048, len(available)), replace=False
                        )
                    )
                    points = []
                    quantiles = []
                    tail = []
                    for start in range(0, len(panel), 8):
                        rr = panel[start : start + 8]
                        with torch.no_grad():
                            z = model.encoder(tensor(natural_inputs[rr]))
                            gen = torch.Generator(device=device).manual_seed(
                                cfg["seed"] + 21000 + start
                            )
                            draw = prior.bounded_whole_draws(head, z, 512, gen, 128)
                        native_draw = draw * scale + center
                        native_draw[:, :, 4:] = np.exp(native_draw[:, :, 4:])
                        if not np.isfinite(native_draw).all():
                            raise FloatingPointError(
                                "Nonfinite physical-coordinate common-space draws"
                            )
                        points.append(np.median(native_draw, axis=0))
                        quantiles.append(
                            np.quantile(
                                native_draw, [0.05, 0.16, 0.5, 0.84, 0.95], axis=0
                            )
                        )
                        tail.append(np.mean(native_draw[:, :, 4] > 20, axis=0))
                    np.savez_compressed(
                        a.output / (tag + "-" + role + "-common.npz"),
                        source_id=ids[panel],
                        native_median=np.concatenate(points),
                        native_quantiles=np.concatenate(quantiles, axis=1),
                        age_probability_above_20_gyr=np.concatenate(tail),
                        scope="Model-blind uniform eligible role panel; common-space coupled PDF draws; not all-Gaia prevalence or physical age truth",
                    )
                corner_rows = prior.heldout_corner_rows(
                    ids, roles.astype("U"), cfg["corner_source_ids"]
                )
                if len(corner_rows):
                    pieces = []
                    for start in range(0, len(corner_rows), 4):
                        rr = corner_rows[start : start + 4]
                        with torch.no_grad():
                            z = model.encoder(tensor(natural_inputs[rr]))
                            gen = torch.Generator(device=device).manual_seed(
                                cfg["seed"] + 22000 + start
                            )
                            pieces.append(
                                prior.bounded_whole_draws(head, z, 2048, gen, 128)
                            )
                    draw = np.concatenate(pieces, axis=1)
                    physical = draw * scale + center
                    physical[:, :, 4:] = np.exp(physical[:, :, 4:])
                    if not np.isfinite(physical).all():
                        raise FloatingPointError(
                            "Nonfinite single-star common PDF draws"
                        )
                    np.savez_compressed(
                        a.output / (tag + "-single-star-corners.npz"),
                        source_id=ids[corner_rows],
                        standardized_draws=draw,
                        native_draws=physical,
                        standardized_covariance=np.asarray(
                            [
                                np.cov(draw[:, i], rowvar=False)
                                for i in range(len(corner_rows))
                            ]
                        ),
                        input_eligible=eligible[corner_rows],
                        parallax_observed=data["measured_parallax/observed"][:][
                            corner_rows
                        ],
                        scope="Predeclared external stress stars, not representative population prevalence; joint samples are coupled, never independent marginal pairings",
                    )
                prior.atomic(
                    a.output / (tag + "-qa-complete.json"),
                    dict(
                        status="common_pdf_arrays_complete",
                        tag=tag,
                        training_update=m["completed_updates"],
                        training_epochs=m["completed_full_fit_epochs"],
                        density_family="NSF",
                        targets=list(prior.NAMES),
                    ),
                )

            update = 0
            best = float("inf")
            presentations = np.zeros((anchor + 1, 6), dtype="i8")
            started = time.monotonic()
            for epoch in range(cfg["epochs"]):
                epoch_sampler_initial_rng_state = rng.bit_generator.state
                epoch_records = 0
                for ss, pp in epoch_record_batches(fit_lengths, cfg["batch_size"], rng):
                    update += 1
                    epoch_records += len(ss)
                    batch = collate(ss, pp, True)
                    progress = step(batch, update)
                    for si in np.unique(batch["source"]):
                        presentations[si] += batch["observed"][
                            batch["source"] == si
                        ].sum(0)
                    m.update(
                        status="training",
                        training_performed=True,
                        training_scope="Joint encoder, decoder, NSF and source-map updates",
                        completed_updates=update,
                        current_epoch=epoch + 1,
                        current_epoch_uniform_records=epoch_records,
                        per_source_target_presentations=presentations,
                        elapsed_training_seconds=time.monotonic() - started,
                        last_progress=progress,
                    )
                    if update <= 2 or update % 100 == 0:
                        save()
                        print(
                            json.dumps(
                                dict(
                                    update=update,
                                    epoch=epoch + 1,
                                    uniform_records=epoch_records,
                                    **progress,
                                )
                            ),
                            flush=True,
                        )
                    if update % cfg["checkpoint_every"] == 0:
                        torch.save(
                            dict(
                                model_state_dict=model.state_dict(),
                                head_state_dict=head.state_dict(),
                                map_state_dict=maps.state_dict(),
                                optimizer_state_dict=optimizer.state_dict(),
                                pilot_rng_state=capture_pilot_rng_state(
                                    head, mask_rng, rng
                                ),
                                reconstruction_rng_state=reconstruction_rng.bit_generator.state,
                                epoch_sampler_initial_rng_state=epoch_sampler_initial_rng_state,
                                source_target_presentations=presentations.copy(),
                                best_complete_epoch_tune_score=best,
                                completed_full_fit_epochs=m[
                                    "completed_full_fit_epochs"
                                ],
                                epoch=epoch,
                                epoch_records=epoch_records,
                                update=update,
                                source_sha256=expanded["prepared_sha256"],
                                decision=decision,
                                center=center,
                                scale=scale,
                                config=cfg,
                            ),
                            a.output / "latest-checkpoint.pth.partial",
                        )
                        (a.output / "latest-checkpoint.pth.partial").replace(
                            a.output / "latest-checkpoint.pth"
                        )
                        if update == cfg["checkpoint_every"]:
                            export_common("first-checkpoint")
                if epoch_records != fit_records:
                    raise ValueError("Full FIT epoch omitted or duplicated a record")
                score, report = evaluate("tune")
                if not np.isfinite(score):
                    raise FloatingPointError("Nonfinite TUNE selection score")
                m.update(
                    completed_full_fit_epochs=epoch + 1,
                    tune_macro_source_target_crps=score,
                    tune_report=report,
                )
                if xp_hidden_weight > 0:
                    # Select on both views of the same TUNE stars: most Gaia
                    # stars have no XP, so an XP-only gain must not win alone.
                    hidden, hidden_report = evaluate("tune", hide_xp=True)
                    if not np.isfinite(hidden):
                        raise FloatingPointError("Nonfinite XP-hidden TUNE score")
                    m.update(
                        tune_macro_source_target_crps_xp_visible=score,
                        tune_macro_source_target_crps_xp_hidden=hidden,
                        tune_report_xp_hidden=hidden_report,
                    )
                    score = (1 - xp_hidden_weight) * score + xp_hidden_weight * hidden
                    m.update(tune_selection_score=score)
                if score < best:
                    best = score
                    torch.save(
                        dict(
                            run_id=cfg["run_id"],
                            model_state_dict=model.state_dict(),
                            head_state_dict=head.state_dict(),
                            map_state_dict=maps.state_dict(),
                            center=center,
                            scale=scale,
                            source_registry=source_names,
                            support=support,
                            reference=maps.reference.cpu(),
                            map_kind=decision["mapping_kind"],
                            map_export_kind=maps.export_kind,
                            source_map_kinds=source_kinds,
                            epoch=epoch + 1,
                            update=update,
                            prepared_sha256=expanded["prepared_sha256"],
                            decision=decision,
                            config=cfg,
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
            model.load_state_dict(state["model_state_dict"])
            head.load_state_dict(state["head_state_dict"])
            maps.load_state_dict(state["map_state_dict"])
            m["evaluation"] = {}
            for role in ["tune", "cal", "devtest"]:
                score, report = evaluate(role, True)
                m["evaluation"][role] = dict(
                    macro_source_target_crps=score, sources=report
                )
                if xp_hidden_weight > 0 or xp_dropout > 0:
                    hidden, hidden_report = evaluate(role, True, hide_xp=True)
                    m["evaluation"][role].update(
                        macro_source_target_crps_xp_hidden=hidden,
                        sources_xp_hidden=hidden_report,
                    )
                save()
            export_common("selected")
            m.update(
                status="large_six_candidate_complete",
                selected_checkpoint_sha256=prior.sha(
                    a.output / "selected-checkpoint.pth"
                ),
                selected_epoch=state["epoch"],
                selected_update=state["update"],
                release_ready=False,
                remaining="Post-training PDF/calibration, native versus common benchmark/cluster and complete per-star corner QA; no automatic catalogue publication.",
            )
            save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
