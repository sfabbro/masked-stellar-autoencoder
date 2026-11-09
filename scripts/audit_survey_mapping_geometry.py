"""CANFAR CUDA audit of the frozen survey map, its gauge and nonlinear PDF error.

Uses saved prediction arrays and exact pinned calibration code. No encoder
execution, optimizer, refitting, new identities, or released-output modification.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
RUN = ROOT / "multisurvey_joint_fullfit_20261105_v1"
SELECTED_SHA = "9328ab102a54fae7fb541f08f4b478128f327834c4398bebccfff07112205fd7"
MAPPED = ROOT / "frozen_survey_qa_20261105_v2/mapped-devtest.h5"
PRED_SHA = "b73bd23c939ec8e36a51abcec29a9a306033e07025940c5cfd5702c644c2ccd9"
MAP_SHA = "6dcb955227cf6aff7b89389336a31bd2067650b4f57c8b008aa7dae4fdf890a5"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def quantiles(x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite geometry")
    return dict(
        zip(
            ["min", "p05", "median", "p95", "max"],
            np.quantile(x, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--calibration-source", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="admitting",
        training_performed=False,
        encoder_executed=False,
        gauge_changed=False,
        calibration_source_sha256=sha(a.calibration_source),
        library_source_revision="2777b0de88121768434725ef227893cd35595d2a",
        results={},
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CANFAR CUDA required for trained mapping evaluation")
        device = torch.device("cuda")
        for path, expected in [
            (RUN / "checkpoint-00005088.pth", SELECTED_SHA),
            (RUN / "predictions-devtest.h5", PRED_SHA),
            (MAPPED, MAP_SHA),
        ]:
            if sha(path) != expected:
                raise ValueError("Frozen artifact changed: " + str(path))
        state = torch.load(
            RUN / "checkpoint-00005088.pth", map_location="cpu", weights_only=False
        )["relative"]
        spec = importlib.util.spec_from_file_location(
            "pinned_calibration", a.calibration_source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = (
            module.RelativeSurveySplineHead(
                [state["knots_" + str(j)] for j in range(3)],
                state["weights"],
                input_center=state["input_center"],
                input_scale=state["input_scale"],
                output_scale=state["output_scale"],
            )
            .double()
            .to(device)
        )
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
        source = (
            ROOT
            / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
        )
        sm = json.loads((source.parent / "manifest.json").read_text())
        if sha(source) != sm["sources"]["apokasc3"]["reference_sha256"]:
            raise ValueError("Seismic references differ")
        ref = pd.read_parquet(source)
        with h5py.File(MAPPED) as f:
            ids = f["source_id"][:]
            ci = np.searchsorted(ref.source_id, ids)
            hit = ci < len(ref)
            hit[hit] &= ref.source_id.to_numpy()[ci[hit]] == ids[hit]
            rows = np.flatnonzero(hit)
            rr = f["reference_mean"][:][rows]
            saved = f["component_mean"][:][rows]
        with h5py.File(RUN / "predictions-devtest.h5") as f:
            np.testing.assert_array_equal(f["source_id"][:], ids)
            w = f["weights"][:][rows]
            mu = f["component_mean"][:][rows]
            lower = f["component_lower"][:][rows]
            actual = f["mean"][:][rows]

        def tensor(x):
            return torch.as_tensor(x, device=device, dtype=torch.float64)

        with torch.no_grad():
            delta, deriv = model(tensor(rr))
            jac = deriv + torch.eye(3, device=device, dtype=torch.float64)
        # Simpler independent NumPy reconstruction of the affine map.
        d = delta.cpu().numpy()
        j = jac.cpu().numpy()
        replay = (
            rr[:, None, None]
            + d[:, :, None]
            + np.einsum("bsij,bcj->bsci", j, mu - rr[:, None])
        )
        # Prediction and mapped exports used different CUDA batch shapes.
        # Reuse their original replay's declared physical-unit tolerances.
        drift = np.max(abs(replay - saved), axis=(0, 1, 2))
        m["export_replay_max_abs_by_target"] = drift.tolist()
        m["export_replay_tolerance_by_target"] = [0.02, 2e-5, 2e-5]
        save()
        for target, tolerance in enumerate([0.02, 2e-5, 2e-5]):
            np.testing.assert_allclose(
                replay[..., target], saved[..., target], rtol=0, atol=tolerance
            )
        np.testing.assert_allclose(
            (w[:, :, None] * mu).sum(1), actual, rtol=1e-6, atol=0.02
        )
        mapped_mean = (saved * w[:, None, :, None]).sum(2)
        gauge_mean = mapped_mean.mean(1)
        np.testing.assert_allclose(gauge_mean, actual, rtol=1e-6, atol=0.02)
        m.update(
            status="sampling_nonlinear_mapping",
            stars=len(rows),
            spline_weights=state["weights"].tolist(),
            max_common_vs_average_mapped_abs=np.max(
                abs(gauge_mean - actual), axis=0
            ).tolist(),
            reference_vs_mixture_mean_abs=[
                quantiles(abs(actual[:, t] - rr[:, t])) for t in range(3)
            ],
        )
        save()
        rng = np.random.default_rng(2026110631)
        exact_pieces = []
        affine_pieces = []
        draws_pieces = []
        local_diagonal = []
        local_smin = []
        local_det = []
        for start in range(0, len(rows), 16):
            n = min(16, len(rows) - start)
            ww = w[start : start + n]
            comp = np.stack(
                [rng.choice(ww.shape[1], size=1024, p=v / v.sum()) for v in ww]
            )
            nr = np.arange(n)[:, None]
            draws = mu[start : start + n][nr, comp] + np.einsum(
                "ndij,ndj->ndi",
                lower[start : start + n][nr, comp],
                rng.standard_normal((n, 1024, 3)),
            )
            tt = tensor(draws.reshape(-1, 3))
            with torch.no_grad():
                dd, jj = model(tt)
                exact = (
                    (tt[:, None] + dd)
                    .reshape(n, 1024, 3, 3)
                    .permute(0, 2, 1, 3)
                    .cpu()
                    .numpy()
                )
                # Convert physical-unit Jacobians to standardized coordinates before SVD.
                jj = jj + torch.eye(3, device=device, dtype=torch.float64)
                scale = model.input_scale
                standardized = (
                    jj * scale[None, None, None, :] / scale[None, None, :, None]
                )
                local_smin.append(
                    torch.linalg.svdvals(standardized)[:, :, -1].cpu().numpy()
                )
                local_det.append(torch.linalg.det(standardized).cpu().numpy())
                local_diagonal.append(jj.diagonal(dim1=-2, dim2=-1).cpu().numpy())
            affine = (
                rr[start : start + n, None, None]
                + d[start : start + n, :, None]
                + np.einsum(
                    "nsij,ndj->nsdi",
                    j[start : start + n],
                    draws - rr[start : start + n, None],
                )
            )
            exact_pieces.append(exact)
            affine_pieces.append(affine)
            draws_pieces.append(draws)
        exact = np.concatenate(exact_pieces)
        affine = np.concatenate(affine_pieces)
        draws = np.concatenate(draws_pieces)
        diagonal = np.concatenate(local_diagonal)
        smin = np.concatenate(local_smin)
        det = np.concatenate(local_det)
        # Positive own-coordinate derivative is necessary for coordinatewise
        # monotonicity, not sufficient for global invertibility.
        for s, name in enumerate(["APOGEE", "DESI", "GALAH"]):
            difference = exact[:, s] - affine[:, s]
            m["results"][name] = dict(
                standardized_jacobian_smallest_singular=quantiles(smin[:, s]),
                jacobian_nonpositive_determinant_fraction=float(
                    (det[:, s] <= 0).mean()
                ),
                nonpositive_logg_own_derivative_fraction=float(
                    (diagonal[:, s, 1] <= 0).mean()
                ),
                exact_minus_affine_logg_mean_abs=quantiles(
                    abs(difference[:, :, 1].mean(1))
                ),
                exact_minus_affine_logg_draw_abs=quantiles(
                    abs(difference[:, :, 1]).ravel()
                ),
                exact_logg_mean_bias=float(
                    np.median(
                        exact[:, s, :, 1].mean(1) - ref.loggSeis.to_numpy()[ci[hit]]
                    )
                ),
                affine_logg_mean_bias=float(
                    np.median(
                        affine[:, s, :, 1].mean(1) - ref.loggSeis.to_numpy()[ci[hit]]
                    )
                ),
            )
        np.savez_compressed(
            a.output / "mapping-geometry.npz",
            source_id=ids[rows],
            reference_mean=rr,
            common_mean=actual,
            delta=d,
            jacobian=j,
            mapped_mean=mapped_mean,
            logg_seismic=ref.loggSeis.to_numpy()[ci[hit]],
            common_logg_draws=draws[:, :, 1],
            exact_mapped_logg_draws=exact[:, :, :, 1],
            affine_mapped_logg_draws=affine[:, :, :, 1],
        )
        m.update(
            status="complete",
            artifact_sha256=sha(a.output / "mapping-geometry.npz"),
            scope="Exact spline evaluation is a diagnostic alternative to the trained local-affine observation model, not a corrected release or an exact nonlinear noisy likelihood. Seismic devtest support only; map geometry is not global invertibility proof.",
        )
        save()
    except Exception as e:
        m.update(status="failed", error=repr(e))
        save()
        raise


if __name__ == "__main__":
    main()
