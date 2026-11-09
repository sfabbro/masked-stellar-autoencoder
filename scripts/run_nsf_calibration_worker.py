"""Unattended CUDA CAL/DEV marginal and seismic interval QA of fixed NSF states.

CAL sets interval thresholds only. The stellar density and its joint draws are
unchanged. No claim of universal conditional or physical joint coverage.
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


def interval_report(cal, dev, names):
    from masked_stellar_autoencoder.training.marginal_calibration import (
        central_bounds,
        central_mass,
    )

    result = []
    for j, name in enumerate(names):
        cm, dm = cal["observed"][:, j], dev["observed"][:, j]
        item = dict(
            target=name, cal_stars=int(cm.sum()), dev_stars=int(dm.sum()), levels=[]
        )
        if not cm.any() or not dm.any():
            item["status"] = "insufficient_reference"
            result.append(item)
            continue
        for coverage in [0.68, 0.9, 0.95]:
            mass = central_mass(cal["pit"][cm, j], coverage)
            lo, hi = central_bounds(dev["quantiles"][:, dm, j], dev["grid"], mass)
            rawlo, rawhi = central_bounds(
                dev["quantiles"][:, dm, j], dev["grid"], coverage
            )
            truth = dev["reference"][dm, j]
            item["levels"].append(
                dict(
                    nominal=coverage,
                    cal_selected_mass=mass if np.isfinite(mass) else None,
                    unbounded=bool(np.isposinf(mass) or mass == 1),
                    raw_coverage=float(((truth >= rawlo) & (truth <= rawhi)).mean()),
                    calibrated_coverage=float(((truth >= lo) & (truth <= hi)).mean()),
                    raw_median_width=float(np.median(rawhi - rawlo)),
                    calibrated_median_width=float(np.median(hi - lo))
                    if np.isfinite(mass) and mass < 1
                    else None,
                )
            )
        item["dev_mean_crps"] = float(np.nanmean(dev["crps"][dm, j]))
        item["dev_mae_median"] = float(
            np.mean(abs(dev["median"][dm, j] - dev["reference"][dm, j]))
        )
        result.append(item)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume-waiting-output", action="store_true")
    a = p.parse_args()
    previous_results = {}
    if a.output.exists() and a.resume_waiting_output:
        previous = json.loads((a.output / "manifest.json").read_text())
        if previous.get("status") != "waiting_or_scoring":
            raise ValueError("Only a reconciled waiting calibration worker can resume")
        previous_results = previous.get("results", {})
        if any(v.get("status") != "complete" for v in previous_results.values()):
            raise ValueError("Investigate incomplete calibration results before resume")
        for rid in previous_results:
            for role in ["cal", "devtest"]:
                for kind in ["marginals", "seismic"]:
                    if not (a.output / f"{rid}-{role}-{kind}.npz").exists():
                        raise ValueError("Completed calibration array missing")
        with (a.output / "pre-cache-resume-1.json").open("xb") as stream:
            stream.write((a.output / "manifest.json").read_bytes())
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    import run_decision_six_pilot as prior
    import run_multilabel_screen as native
    import torch

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.cached_inputs import cache_natural_inputs
    from masked_stellar_autoencoder.training.decision_pdf_metrics import sample_scores

    if not torch.cuda.is_available() or not ROOT.is_dir():
        raise RuntimeError("CANFAR CUDA required")
    cfg = yaml.safe_load(a.config.read_text())
    direct = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]:
        raise ValueError("Pinned library differs")
    source, _, _ = prior.admission(cfg, cfg["arms"][0])
    if (
        prior.sha(source["prepared_path"]) != source["prepared_sha256"]
        or prior.sha(cfg["reference_checkpoint"]) != cfg["reference_checkpoint_sha256"]
    ):
        raise ValueError("Immutable source/reference differs")
    refs = ROOT / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
    sm = json.loads((refs.parent / "manifest.json").read_text())
    if prior.sha(refs) != sm["sources"]["apokasc3"]["reference_sha256"]:
        raise ValueError("Seismic reference changed")
    ref = pd.read_parquet(refs)
    refidx = {int(s): i for i, s in enumerate(ref.source_id)}
    reference = torch_load_trusted(
        cfg["reference_checkpoint"], map_location="cpu", weights_only=False
    )
    model_cfg = yaml.safe_load(Path(cfg["model_config"]).read_text())
    device = torch.device("cuda")
    torch.set_num_threads(4)
    pending = {
        arm["run_id"]: arm
        for arm in cfg["arms"]
        if arm["run_id"] not in previous_results
    }
    results = dict(previous_results)
    deadline = time.monotonic() + 36 * 3600

    def save():
        prior.atomic(
            a.output / "manifest.json",
            dict(
                status="waiting_or_scoring" if pending else "complete",
                pending=sorted(pending),
                results=results,
                training_performed=False,
                density_parameters_changed=False,
                selection_role="TUNE only; CAL never chooses a candidate",
                scope="CAL-fitted central CDF-score intervals; separate devtest evaluation. Sparse noisy/teacher reference coverage is not latent physical PDF calibration; seismic CAL and DEV are scored separately. Parent exposure and survey calibration dependencies remain. Marginal, not simultaneous six-target, coverage.",
            ),
        )

    with h5py.File(source["prepared_path"]) as data:
        profiles = {
            arm.get("input_profile", "phot_xp")
            for arm in cfg["arms"]
            if arm["run_id"] in pending
        }
        if profiles != {"phot_xp_measured_parallax"}:
            raise ValueError(
                "RAM-cache resume requires the same measured-parallax profile for every pending arm"
            )
        cached_inputs = cache_natural_inputs(
            data, source["feature_profiles"]["phot_xp_measured_parallax"]
        )
        checked = np.linspace(0, len(cached_inputs) - 1, 128, dtype="i8")
        np.testing.assert_array_equal(
            cached_inputs[checked],
            prior.profile_batch(data, checked, "phot_xp_measured_parallax", source),
        )
        prior.atomic(
            a.output / "cache-resume-contract.json",
            dict(
                bytes=cached_inputs.nbytes,
                exact_profile_rows=len(checked),
                source_sha256=source["prepared_sha256"],
                preserved_results=list(previous_results),
                preserved_artifact_sha256={
                    p.name: prior.sha(p) for p in a.output.glob("*.npz")
                },
                model_parameters_changed=False,
            ),
        )
        ids = data["source_id"][:]
        roles = data["role"][:].astype("U")
        eligible = data["primary_input_eligible"][:].astype(bool)
        values = data["value"][:]
        quality = data["observed"][:].astype(bool) & data["quality"][:].astype(bool)
        y, sigma, mask, center, scale, known = prior.normalize_targets(
            values, quality, data["error"][:], data["error_observed"][:], roles == "fit"
        )
        panels = {}
        seismic_rows = {}
        for role in ["cal", "devtest"]:
            panels[role] = prior.fixed_role_rows(ids, roles, mask, role, 1024)
            seismic_rows[role] = np.array(
                [
                    i
                    for i, s in enumerate(ids)
                    if eligible[i] and roles[i] == role and int(s) in refidx
                ],
                dtype="i8",
            )
        if np.intersect1d(ids[panels["cal"]], ids[panels["devtest"]]).size:
            raise ValueError("CAL/DEV identity overlap")
        grid = np.linspace(0, 1, 1025)
        save()
        while pending and time.monotonic() < deadline:
            for rid, arm in list(pending.items()):
                path = ROOT / rid / "manifest.json"
                try:
                    manifest = json.loads(path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
                if manifest.get("status") == "failed":
                    results[rid] = dict(
                        status="upstream_failed", error=manifest.get("error")
                    )
                    pending.pop(rid)
                    save()
                    continue
                if manifest.get("status") != "remediation_pilot_complete":
                    continue
                model = head = None
                try:
                    checkpoint = ROOT / rid / "selected-checkpoint.pth"
                    if prior.sha(checkpoint) != manifest["selected_checkpoint_sha256"]:
                        raise ValueError("Selected checkpoint SHA changed")
                    state = torch_load_trusted(
                        checkpoint, map_location="cpu", weights_only=False
                    )
                    if (
                        state["run_id"] != rid
                        or state["arm"] != arm
                        or state["prepared_sha256"] != source["prepared_sha256"]
                        or state["parent_sha256"] != cfg["checkpoint_sha256"]
                    ):
                        raise ValueError("Selected state binding differs")
                    np.testing.assert_array_equal(state["center"], center)
                    np.testing.assert_array_equal(state["scale"], scale)
                    model = native.probe.load_model(model_cfg, reference, device)
                    model.load_state_dict(state["model_state_dict"], strict=True)
                    model.eval().requires_grad_(False)
                    head = prior.build_head(
                        arm["kind"], model_cfg["model"]["layer_dims"][-1], cfg
                    ).to(device)
                    head.load_state_dict(state["head_state_dict"], strict=True)
                    head.eval().requires_grad_(False)
                    del state

                    def score_panel(rows, seismic=False):
                        pieces = []
                        for start in range(0, len(rows), 4):
                            rr = rows[start : start + 4]
                            with torch.no_grad():
                                z = model.encoder(
                                    torch.as_tensor(
                                        cached_inputs[rr],
                                        device=device,
                                        dtype=torch.float32,
                                    )
                                )
                                gen = torch.Generator(device=device).manual_seed(
                                    2026110700 + int(rr[0])
                                )
                                draw = prior.bounded_whole_draws(
                                    head, z, 1024, gen, 128
                                )
                            rng = np.random.default_rng(2026110710 + int(rr[0]))
                            if seismic:
                                ix = np.array([refidx[int(s)] for s in ids[rr]])
                                truth = ref.loggSeis.to_numpy()[ix, None]
                                error = ref.e_loggSeis.to_numpy()[ix, None]
                                if (
                                    not np.isfinite(truth).all()
                                    or not np.isfinite(error).all()
                                    or (error <= 0).any()
                                ):
                                    raise ValueError(
                                        "Invalid independent seismic reference"
                                    )
                                intrinsic = draw[:, :, 1:2] * scale[1] + center[1]
                                observed = (
                                    intrinsic
                                    + rng.standard_normal(intrinsic.shape) * error[None]
                                )
                                seen = np.ones(truth.shape, bool)
                            else:
                                intrinsic = draw * scale + center
                                intrinsic[:, :, 4:] = np.exp(intrinsic[:, :, 4:])
                                observed = (
                                    draw
                                    + rng.standard_normal(draw.shape) * sigma[rr][None]
                                )
                                observed = observed * scale + center
                                observed[:, :, 4:] = np.exp(observed[:, :, 4:])
                                truth = values[rr]
                                seen = mask[rr]
                            scored = sample_scores(observed, truth, seen)
                            pieces.append(
                                dict(
                                    pit=scored["pit"],
                                    crps=scored["crps"],
                                    quantiles=np.quantile(
                                        observed, grid, axis=0
                                    ).astype("f4"),
                                    median=np.median(intrinsic, axis=0),
                                    reference=truth,
                                    observed=seen,
                                )
                            )
                            if start % 512 == 0:
                                prior.atomic(
                                    a.output / (rid + "-progress.json"),
                                    dict(
                                        panel="seismic" if seismic else "six_target",
                                        rows=start + len(rr),
                                        total=len(rows),
                                    ),
                                )
                        if not pieces:
                            raise ValueError("Empty calibration/evaluation panel")
                        arrays = {
                            k: np.concatenate(
                                [b[k] for b in pieces],
                                axis=1 if k == "quantiles" else 0,
                            )
                            for k in pieces[0]
                        }
                        return dict(source_id=ids[rows], grid=grid, **arrays)

                    catalogues = {}
                    seismic = {}
                    for role in ["cal", "devtest"]:
                        catalogues[role] = score_panel(panels[role])
                        seismic[role] = score_panel(seismic_rows[role], True)
                        np.savez_compressed(
                            a.output / (rid + "-" + role + "-marginals.npz"),
                            **catalogues[role],
                        )
                        np.savez_compressed(
                            a.output / (rid + "-" + role + "-seismic.npz"),
                            **seismic[role],
                        )
                    results[rid] = dict(
                        status="complete",
                        selected_update=manifest["selected_update"],
                        checkpoint_sha256=manifest["selected_checkpoint_sha256"],
                        tune_score=manifest["tune_score"],
                        catalogue=interval_report(
                            catalogues["cal"], catalogues["devtest"], prior.NAMES
                        ),
                        seismic=interval_report(
                            seismic["cal"], seismic["devtest"], ["logg"]
                        ),
                    )
                except Exception as exc:
                    results[rid] = dict(status="failed", error=repr(exc))
                finally:
                    del model, head
                    gc.collect()
                    torch.cuda.empty_cache()
                pending.pop(rid)
                save()
            if pending:
                time.sleep(30)
        if pending:
            raise TimeoutError("Confirmation checkpoints did not complete in 36h")
        save()


if __name__ == "__main__":
    main()
