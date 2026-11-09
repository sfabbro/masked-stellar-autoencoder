"""Fit new survey maps to frozen saved common mixtures; encoder/reference never changes."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    marginal_scores,
    summarize_residuals,
    summarize_scores,
)

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
PRED = ROOT + "multisurvey_joint_fullfit_20261105_v1/"
CENTER = np.array([5500.0, 3.0, -0.5])
SCALE = np.array([1000.0, 1.0, 1.0])
FLOOR = np.array([100.0, 0.1, 0.1])


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def project(mean, cov, parameters, target, kind):
    a = np.eye(3)[target].copy()
    if kind == "offset":
        offset = parameters[0]
    else:
        a += parameters[:3]
        offset = parameters[3]
    mu = np.einsum("nkj,j->nk", mean, a) + offset
    var = np.einsum("i,nkij,j->nk", a, cov, a)
    return mu, np.maximum(var, 1e-12), a, offset


def fit_map(data, target, kind):
    w, mean, cov, value, error = data
    initial = np.zeros(1 if kind == "offset" else 4)

    def objective(p):
        mu, var, _, _ = project(mean, cov, p, target, kind)
        total = var + error[:, None] ** 2
        lp = logsumexp(
            np.log(np.maximum(w, 1e-30))
            - 0.5 * ((value[:, None] - mu) ** 2 / total + np.log(2 * np.pi * total)),
            axis=1,
        )
        return float(-lp.mean() + 0.01 * np.dot(p, p))

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        options=dict(maxiter=120, ftol=1e-9),
        bounds=[(-2, 2)] * len(initial),
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise ValueError("Nonfinite map fit")
    return dict(
        parameters=result.x.tolist(),
        fit_objective=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


def load_match(role, sid, observed, values, errors, expected):
    path = PRED + "predictions-" + role + ".h5"
    if sha(path) != expected:
        raise ValueError("Frozen prediction SHA differs")
    with h5py.File(path) as f:
        pid = f["source_id"][:]
        pos = np.searchsorted(pid, sid)
        keep = pos < len(pid)
        keep[keep] &= pid[pos[keep]] == sid[keep]
        ii = np.flatnonzero(keep)
        rr = pos[ii]
        # Bulk HDF reads avoid pathological long fancy-index reads.
        mean = (f["component_mean"][:][rr] - CENTER) / SCALE
        lower = f["component_lower"][:][rr] / SCALE[None, None, :, None]
        cov = lower @ lower.swapaxes(-1, -2)
        return dict(
            source_id=sid[ii],
            weights=f["weights"][:][rr],
            mean=mean,
            cov=cov,
            value=(values[ii] - CENTER) / SCALE,
            error=np.sqrt(
                np.where(np.isfinite(errors[ii]) & (errors[ii] > 0), errors[ii] ** 2, 0)
                + FLOOR**2
            )
            / SCALE,
            observed=observed[ii],
            known_quoted_error=np.isfinite(errors[ii]) & (errors[ii] > 0),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = json.loads(args.config.read_text())
    result = dict(
        status="running",
        encoder_loaded=False,
        reference_gauge_changed=False,
        source_map_training_only=True,
        scope="Fixed common reference; new source offsets/affine maps only; marginal likelihood with quoted errors plus explicit illustrative 100K/.1/.1 floors. Gaia quoted errors unknown: compatibility proxy only. Not expanded six-target training.",
        sources={},
    )

    def save():
        (args.output / "manifest.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )

    save()
    for source in c["sources"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Source sidecar differs")
        with h5py.File(source["path"]) as f:
            g = f["native_observations/" + source["name"]]
            sid = g["source_id"][:]
            value = g["value"][:]
            error = g["error"][:]
            obs = g["observed"][:].astype(bool)
        roles = {
            r: load_match(r, sid, obs, value, error, c["prediction_sha256"][r])
            for r in ["fit", "tune", "devtest"]
        }
        entry = dict(
            roles={r: len(d["source_id"]) for r, d in roles.items()}, targets={}
        )
        for target, name in enumerate(["teff", "logg", "feh"]):
            if not roles["fit"]["observed"][:, target].any():
                continue
            alternatives = {}
            for kind in ["offset", "affine"]:
                d = roles["fit"]
                mask = d["observed"][:, target]
                fit = fit_map(
                    (
                        d["weights"][mask],
                        d["mean"][mask],
                        d["cov"][mask],
                        d["value"][mask, target],
                        d["error"][mask, target],
                    ),
                    target,
                    kind,
                )
                fit["roles"] = {}
                for role in ["tune", "devtest"]:
                    d = roles[role]
                    mask = d["observed"][:, target]
                    mu, var, a, offset = project(
                        d["mean"][mask],
                        d["cov"][mask],
                        np.array(fit["parameters"]),
                        target,
                        kind,
                    )
                    pred = (d["weights"][mask] * mu).sum(1) * SCALE[target] + CENTER[
                        target
                    ]
                    ref = d["value"][mask, target] * SCALE[target] + CENTER[target]
                    scores = marginal_scores(
                        d["weights"][mask],
                        mu * SCALE[target] + CENTER[target],
                        var * SCALE[target] ** 2,
                        ref,
                        d["error"][mask, target] * SCALE[target],
                    )
                    fit["roles"][role] = dict(
                        **summarize_residuals(pred, ref),
                        distribution=summarize_scores(scores),
                        unknown_quoted_error=int(
                            (~d["known_quoted_error"][mask, target]).sum()
                        ),
                    )
                alternatives[kind] = fit

            # Selection uses only the declared TUNE observed-reference NLL.
            def tune_nll(kind):
                d = roles["tune"]
                mask = d["observed"][:, target]
                mu, var, _, _ = project(
                    d["mean"][mask],
                    d["cov"][mask],
                    np.array(alternatives[kind]["parameters"]),
                    target,
                    kind,
                )
                variance = var + d["error"][mask, target, None] ** 2
                return float(
                    -logsumexp(
                        np.log(np.maximum(d["weights"][mask], 1e-30))
                        - 0.5
                        * (
                            (d["value"][mask, target, None] - mu) ** 2 / variance
                            + np.log(2 * np.pi * variance)
                        ),
                        axis=1,
                    ).mean()
                )

            selected = min(alternatives, key=tune_nll)
            entry["targets"][name] = dict(
                selected_by_tune=selected, alternatives=alternatives
            )
        result["sources"][source["name"]] = entry
        save()
        print(
            json.dumps(dict(source=source["name"], status="maps_complete")), flush=True
        )
    result["status"] = "frozen_new_survey_maps_complete"
    save()


if __name__ == "__main__":
    main()
