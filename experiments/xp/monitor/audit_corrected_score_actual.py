"""Bounded actual-checkpoint EIV identity audit, separate scalar autograd path."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_xp_representation_probes as probe

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.input_measurement import xp_indices
from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_prediction_head,
)
from masked_stellar_autoencoder.training.prediction_mechanisms import (
    corrected_score_objective,
)


def main():
    torch.set_num_threads(1)
    base = ROOT / "experiments/xp/results-stellar-improvement"
    directory = base / "prediction_e72_corrected_score_xp_10m_v2"
    p = np.load(directory / "paired-cohort.npz", allow_pickle=False)
    with h5py.File(base / "enriched_feature_cohort_v1/enriched-feature-cohort.h5") as f:
        columns = json.loads(f.attrs["feature_columns_json"])
        ids = f["features/source_id"][:]
        order = np.argsort(ids)
        rows = order[np.searchsorted(ids[order], p["fit_source_id"][:8])]
        np.testing.assert_array_equal(ids[rows], p["fit_source_id"][:8])
        errors = f["features/reported_feature_errors"][:][rows]
        raw = f["features/values"][:][rows]
        scale = f["reference_scaler/scale"][:]
    xp = xp_indices(columns)
    g = raw[:, columns.index("G")].astype("f8")
    sigma = np.zeros_like(errors, dtype="f4")
    sigma[:, xp] = (
        errors[:, xp] / 10 ** ((8.5 - g[:, None]) / 2.5) / scale[xp]
    ).astype("f4")
    saved = torch_load_trusted(
        directory / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    parent = torch_load_trusted(
        base / "prediction-comparison-e72/parent/checkpoint-72.pth",
        map_location="cpu",
        weights_only=False,
    )
    model = probe.load_model(
        yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text()),
        parent,
        torch.device("cpu"),
    ).eval()
    model.load_state_dict(saved["model_state_dict"], strict=True)
    head = build_prediction_head("residual").eval()
    head.load_state_dict(saved["head_state_dict"], strict=True)
    x = torch.tensor(p["fit_input"][:8], requires_grad=True)
    y = torch.tensor(
        (
            (p["fit_target"][:8] - float(p["label_center"])) / float(p["label_scale"])
        ).astype("f4")
    )
    values = head(model.encoder(x))[:, 0, 0]
    variances = []
    for i in range(8):
        gradient = torch.autograd.grad(values[i], x, retain_graph=True)[0][i]
        variances.append(
            (gradient.square() * torch.from_numpy(sigma[i]).square()).sum()
        )
    mse = (values - y).square().mean()
    correction = torch.stack(variances).mean()
    reference = mse - correction
    actual = corrected_score_objective(model, head)(
        x, y, sigma_x=torch.diag_embed(torch.from_numpy(sigma).square())
    )
    torch.testing.assert_close(actual, reference, rtol=5e-5, atol=5e-5)
    report = dict(
        status="passed",
        rows=8,
        source_ids=p["fit_source_id"][:8].tolist(),
        mse=float(mse.detach()),
        jacobian_noise_correction=float(correction.detach()),
        scalar_autograd_reference=float(reference.detach()),
        library_objective=float(actual.detach()),
        rtol=5e-5,
        atol=5e-5,
        scope="First8 actual strict-fit stars/final checkpoint; scalar autograd versus library batched Jacobian; not a full production gradient replay or physical-error-model certification",
        library_revision=json.loads(
            __import__("importlib.metadata", fromlist=["distribution"])
            .distribution("torchregress")
            .read_text("direct_url.json")
        )["vcs_info"]["commit_id"],
    )
    (directory / "corrected-score-objective-audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
