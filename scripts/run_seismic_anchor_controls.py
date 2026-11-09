"""FIT-only simple anchor calibration of saved common logg PDFs; select on TUNE.

Diagnostic only. Does not alter the checkpoint or the historical survey gauge.
The Gaussian discrepancy kernel is explicitly a calibration model, not a claim
that the intrinsic six-target stellar distribution is Gaussian.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    marginal_scores,
    summarize_residuals,
    summarize_scores,
)

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
SOURCE = ROOT / "seismic_gravity_qa_20261106_v1"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def transform(mu, var, kind, parameters):
    if kind == "identity":
        a, b, sigma = 1.0, 0.0, 0.0
    elif kind == "offset":
        a, b, sigma = 1.0, parameters[0], 0.0
    else:
        a, b = np.exp(parameters[0]), parameters[1]
        sigma = np.exp(parameters[2]) if kind == "affine_scatter" else 0.0
    return (
        a * mu + b,
        a * a * var + sigma * sigma,
        dict(slope=float(a), offset=float(b), extra_logg_scatter=float(sigma)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    data = {}
    for role in ["fit", "tune", "cal", "devtest", "external"]:
        with np.load(SOURCE / (role + "-paired.npz")) as f:
            data[role] = {k: f[k] for k in f.files}
    m = dict(
        status="fitting_controls",
        training_of_encoder_or_head=False,
        original_model_changed=False,
        original_gauge_changed=False,
        reference="APOKASC Gold/Silver loggSeis",
        source_hashes={role: sha(SOURCE / (role + "-paired.npz")) for role in data},
        results={},
    )

    def save():
        t = a.output / "manifest.tmp"
        t.write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
        t.replace(a.output / "manifest.json")

    save()
    x = data["fit"]
    w = x["weights"]
    mu = x["component_mean"]
    var = x["component_variance"]
    y = x["logg_seismic"]
    err = x["error"]
    initial_bias = float(np.median((w * mu).sum(1) - y))
    for kind, initial, bounds in [
        ("identity", [], []),
        ("offset", [-initial_bias], [(-1, 1)]),
        ("affine", [0.0, -initial_bias], [(-1, 1), (-2, 2)]),
        (
            "affine_scatter",
            [0.0, -initial_bias, np.log(0.08)],
            [(-1, 1), (-2, 2), (-8, 0)],
        ),
    ]:

        def objective(parameters):
            mapped, variance, _ = transform(mu, var, kind, parameters)
            total = variance + err[:, None] ** 2
            logp = logsumexp(
                np.log(np.maximum(w, 1e-300))
                - 0.5
                * ((y[:, None] - mapped) ** 2 / total + np.log(2 * np.pi * total)),
                axis=1,
            )
            return float(-logp.mean())

        if initial:
            opt = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
            if not opt.success or not np.isfinite(opt.fun):
                raise ValueError("Anchor fit failed: " + str(opt.message))
            params = opt.x
        else:
            params = []
        result = dict(parameters=list(params), roles={})
        for role, v in data.items():
            mapped, variance, description = transform(
                v["component_mean"], v["component_variance"], kind, params
            )
            result.update(description)
            result["roles"][role] = dict(
                point=summarize_residuals(
                    (v["weights"] * mapped).sum(1), v["logg_seismic"]
                ),
                intrinsic=summarize_scores(
                    marginal_scores(v["weights"], mapped, variance, v["logg_seismic"])
                ),
                convolved=summarize_scores(
                    marginal_scores(
                        v["weights"], mapped, variance, v["logg_seismic"], v["error"]
                    )
                ),
            )
        m["results"][kind] = result
        save()
    selected = min(
        m["results"], key=lambda k: m["results"][k]["roles"]["tune"]["convolved"]["nll"]
    )
    m.update(
        status="complete",
        selected_by_tune_log_score=selected,
        scope="FIT-only scalar anchor calibration controls; no devtest fitting or model changes. Historical teacher exposure and seismic temperature dependencies remain. Affine-only reparameterization can preserve all survey likelihoods with inverse-composed maps; extra discrepancy scatter is a distinct PDF recalibration, not a pure gauge change. This does not calibrate all six targets or establish generalization outside seismic support.",
    )
    save()
    print(json.dumps({k: v for k, v in m.items() if k != "results"}))


if __name__ == "__main__":
    main()
