"""Verify a proposed input policy on preserved paired inputs, without fitting."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from run_xp_representation_probes import safe_inputs  # noqa: E402

source = (
    ROOT
    / "experiments/xp/results-stellar-improvement/distance_value_frozen_v3/paired-cohort.npz"
)
with source.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == "6dbd65b62c1173360b74cce3c9adf131538d09161563fbd98a8b1d2fef8ac23e"
features = yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text())["data"][
    "feature_cols"
]
with np.load(source, allow_pickle=False) as data:
    values = data["input_assisted"].copy()
    ids = data["combined_source_id"].copy()
assert len(ids) == len(np.unique(ids)) == 31550
assert values.shape == (31550, len(features))
# These are archived, standardized safe inputs, with the parent sentinel already
# applied; this check does not substitute for original raw-mask provenance.
observed = np.isfinite(values) & (values != -9999)
safe, visible = spectrophotometric_inputs(
    torch.from_numpy(values), torch.from_numpy(observed), features
)
reference = safe_inputs(values, features, ["distance"], ["PARALLAX"])
np.testing.assert_array_equal(safe.numpy(), reference)
allow = spectrophotometric_inputs(
    torch.ones(1, len(features)),
    torch.ones(1, len(features), dtype=torch.bool),
    features,
)[1][0].numpy()
changed = values.copy()
changed[:, ~allow] = np.float32(1234567)
changed_observed = observed.copy()
changed_observed[:, ~allow] = True
counterfactual, counterfactual_visible = spectrophotometric_inputs(
    torch.from_numpy(changed), torch.from_numpy(changed_observed), features
)
assert torch.equal(safe, counterfactual) and torch.equal(
    visible, counterfactual_visible
)
report = {
    "status": "passed",
    "actual_input_rows": len(ids),
    "parent_feature_width": len(features),
    "source_sha256": digest,
    "byte_exact_to_completed_parallax_free_input_policy": True,
    "withheld_value_and_observation_mask_counterfactual_invariant": True,
    "withheld_named_columns": [name for name, keep in zip(features, allow) if not keep],
    "no_artificial_inference_mask": True,
    "include_reddening": True,
    "training_inference_or_session_submission": False,
    "limitations": "Archived standardized safe-input check, not raw-mask rederivation, model/encoder replay, latest-checkpoint integration, or an independently calibrated distance product. Never establishes absence from pretraining.",
}
output = (
    ROOT
    / "experiments/xp/results-stellar-improvement/joint-continuation-preparation-v1/spectrophotometric-policy-actual-input-verification.json"
)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
