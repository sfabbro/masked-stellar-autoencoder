"""Synthetic-only interface check: densities versus prediction-region methods.

Does not load stellar data, encoders or trained checkpoints, or rank methods.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from torchregress.losses import ContrastiveFlowConformal, SLSLoss
from torchregress.losses.nflows import (
    ContrastiveFlowLoss,
    NormalizingFlowLoss,
    create_flow_model,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(2026110632)
    torch.set_num_threads(2)
    context = torch.randn(32, 4)
    target = torch.randn(32, 6)
    flow = create_flow_model(
        6, context_dim=4, flow_type="nsf", n_transforms=3, hidden_features=[32, 32]
    )
    nsf = NormalizingFlowLoss(flow)
    contrastive = ContrastiveFlowLoss(flow, temperature=1.0, margin=0.0)
    logp = nsf.log_prob(context, target)
    draws = nsf.sample(context, n_samples=16)
    assert logp.shape == (32,) and torch.isfinite(logp).all()
    assert draws.shape == (32, 16, 6) and torch.isfinite(draws).all()
    region = ContrastiveFlowConformal(contrastive, alpha=0.1)
    region.calibrate(context, target)
    membership = region.contains(context, target)
    assert membership.shape == (32,)
    sls = SLSLoss(6, context_dim=4, K=2, hidden_dim=64, n_transforms=4)
    loss = sls(context, target, step=1)
    assert torch.isfinite(loss)
    G, volume = sls.evaluate_frontier(target, context, step=1)
    assert G.shape == (32,) and torch.isfinite(G).all()
    objects = dict(
        NSF=nsf,
        ContrastiveFlowLoss=contrastive,
        ContrastiveFlowConformal=region,
        SLSLoss=sls,
    )
    result = dict(
        status="passed",
        synthetic_only=True,
        training_updates=0,
        inference_of_stellar_models=False,
        scope="API and finite-shape proof only, not density fitting, coverage, scientific performance or physical calibration",
        methods={},
    )
    for name, obj in objects.items():
        path = Path(inspect.getfile(type(obj)))
        result["methods"][name] = dict(
            source_file=path.name,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            callable_log_prob=callable(getattr(obj, "log_prob", None)),
            callable_sample=callable(getattr(obj, "sample", None)),
            callable_contains=callable(getattr(obj, "contains", None)),
            callable_evaluate_frontier=callable(
                getattr(obj, "evaluate_frontier", None)
            ),
        )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
