"""Compare sentinel gradients using a trusted local legacy pretraining checkpoint."""

import argparse
import json

import torch
import yaml
from rtdl_num_embeddings import PeriodicEmbeddings

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.models.model import EncoderDecoderLoss, make_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", help="Trusted local checkpoint")
parser.add_argument("--config", default="configs/pretrain.canfar.yaml")
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(42)
checkpoint = torch_load_trusted(args.checkpoint, map_location="cpu", weights_only=False)
if any("missing_periodic_encoding" in key for key in checkpoint["model_state_dict"]):
    parser.error("Use a legacy checkpoint from before stable sentinel encoding")
with open(args.config) as stream:
    config = yaml.safe_load(stream)
model_config = config["model"]
features = config["data"]["feature_cols"]
outputs = len(config["data"]["recon_cols"])
model = make_model(
    len(features),
    model_config["layer_dims"],
    outputs,
    model_config["pt_activ_func"],
    model_config["rtdl_embed"],
    model_config["norm"],
    decoder_dims=model_config["decoder_dims"],
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
embedding = model.encoder.encoder.dense_resnet[0]
frequency = embedding.periodic.weight
decay_params = [
    param
    for name, param in model.named_parameters()
    if "bias" not in name and "norm" not in name
]
frequency_index = next(i for i, param in enumerate(decay_params) if param is frequency)
optimizer = checkpoint["optimizer_state_dict"]
frequency_id = optimizer["param_groups"][0]["params"][frequency_index]
squared_moments = {
    key: float(state["exp_avg_sq"].sum()) for key, state in optimizer["state"].items()
}
print(
    json.dumps(
        {
            "checkpoint_epoch": checkpoint["epoch"],
            "squared_gradient_ema": sum(squared_moments.values()),
            "frequency_fraction": squared_moments[frequency_id]
            / sum(squared_moments.values()),
        }
    )
)
targets = torch.randn(128, len(features))
mask = torch.rand_like(targets) < config["training"]["m_masking_ratio"]
xp = [i for i, name in enumerate(features) if name.startswith(("bp_", "rp_"))]
mask[:, xp] = False
xp_rows = torch.randperm(len(targets))[
    : int(config["training"]["xp_masking_ratio"] * len(targets))
]
mask[xp_rows[:, None], xp] = True
inputs = targets.masked_fill(mask, -9999)
criterion = EncoderDecoderLoss(lf="mae")
stable_forward = embedding.forward
predictions = []
norms = []
for stable_missing in (False, True):
    embedding.forward = (
        stable_forward
        if stable_missing
        else lambda values: PeriodicEmbeddings.forward(embedding, values)
    )
    model.zero_grad(set_to_none=True)
    prediction, _ = model(inputs)
    predictions.append(prediction.detach().clone())
    loss = criterion(targets[:, :outputs], prediction, mask[:, :outputs], None)
    loss.backward()
    norm = torch.stack([p.grad.norm() for p in model.parameters()]).norm()
    norms.append(float(norm))
    print(
        json.dumps(
            {
                "stable_missing": stable_missing,
                "synthetic_mae": float(loss.detach()),
                "gradient_norm": float(norm),
                "frequency_gradient_norm": float(frequency.grad.norm()),
            }
        )
    )
torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
assert norms[1] < norms[0], (
    "Stable sentinel should reduce this controlled gradient norm"
)
