import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, "scripts")
import run_xp_representation_probes as p
from astropy.io import fits
from run_distance_value import (
    BLOCKED,
    fit_distance_head,
    parallax_metrics,
    positive_distance,
)
from run_native_age import state_digest
from run_pristine_value import replay

root = Path("experiments/xp/results-stellar-improvement")
with h5py.File(root / "distance_feature_cohort_v1/distance-feature-cohort.h5") as f:
    c = {k: f["selected/" + k][:] for k in f["selected"]}
    selection = np.r_[
        np.flatnonzero(c["strict_fit"])[:24], np.flatnonzero(c["identity_test"])[:24]
    ]
config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
features = config["data"]["feature_cols"]
with fits.open("/tmp/msa-labels-complete-20261015.fits", memmap=True) as f:
    raw = f[1].data[c["feature_row"][selection]]
    x = np.column_stack([np.asarray(raw[n], dtype="f4") for n in features])
parent = torch.load(
    "/tmp/msa-standard-checkpoint-57.pth", map_location="cpu", weights_only=False
)
center = np.asarray(parent["run_signature"]["scaler_center"])
scale = np.asarray(parent["run_signature"]["scaler_scale"])
base = p.safe_inputs((x - center) / scale, features, ["distance"], BLOCKED)
model = p.load_model(config, parent, torch.device("cpu"))
model.eval()
before = state_digest(model.encoder)
result = {}
started = time.monotonic()
for mode in ["parallax_free", "parallax_assisted"]:
    inputs = base.copy()
    if mode == "parallax_free":
        inputs = p.safe_inputs(inputs, features, ["distance"], ["PARALLAX"])
    latent = p.encode(model, inputs, torch.device("cpu"))
    prediction, state = fit_distance_head(
        latent,
        c["distance_percentiles"][selection, 2],
        np.arange(24),
        np.arange(24, 48),
    )
    np.testing.assert_array_equal(
        prediction, positive_distance(replay(latent[24:], state))
    )
    # Independently reproduce ridge using augmented least-squares, avoiding normal equations.
    design = np.column_stack([np.ones(48), (latent - state["center"]) / state["scale"]])
    penalty = np.diag(np.r_[0.0, np.full(256, np.sqrt(10.0))])
    target = (
        np.log(c["distance_percentiles"][selection[:24], 2]) - state["ycenter"]
    ) / state["yscale"]
    coef = np.linalg.lstsq(
        np.r_[design[:24], penalty], np.r_[target, np.zeros(257)], rcond=None
    )[0]
    np.testing.assert_allclose(
        prediction,
        np.exp((design[24:] @ coef) * state["yscale"] + state["ycenter"]),
        rtol=1e-6,
        atol=1e-6,
    )
    result[mode] = dict(
        fixture_rows=48,
        train=24,
        test=24,
        head_reload_exact=True,
        independent_augmented_lstsq_passed=True,
    )
assert before == state_digest(model.encoder)
report = dict(
    actual_parent_fixture=True,
    not_scientific_performance=True,
    modes=result,
    encoder_unchanged=True,
    elapsed_seconds=time.monotonic() - started,
    baseline=parallax_metrics(
        np.array([-0.2, 0.5]), np.array([-0.1, 0.4]), np.array([0.1, 0.1])
    ),
)
(
    root / "distance-feature-production-local-v2/actual-parent-runner-smoke.json"
).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
