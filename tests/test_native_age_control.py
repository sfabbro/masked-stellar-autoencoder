import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_native_age import fit_control, predict, state_digest


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.decoder = torch.nn.Linear(2, 3)
        self.reconstruction_layer = torch.nn.Linear(3, 3)


def test_matched_streams_frozen_decoder_and_encoder_control():
    torch.manual_seed(42)
    model = Tiny()
    head = torch.nn.Linear(2, 1)
    x = torch.randn(18, 3)
    y = torch.randn(18)
    initial_encoder = state_digest(model.encoder)
    initial_decoder = state_digest(model.decoder) + state_digest(
        model.reconstruction_layer
    )
    results = []
    fitted = []
    for mode in ["frozen", "sft", "recon"]:
        m, h = copy.deepcopy(model), copy.deepcopy(head)
        result = fit_control(m, h, x, y, [0, 1, 2], mode, steps=20, warmup=5, batch=8)
        results.append(result)
        fitted.append((m, h))
        assert (
            state_digest(m.decoder) + state_digest(m.reconstruction_layer)
            == initial_decoder
        )
        assert result["head_updates"] == 20
        assert result["encoder_updates"] == (0 if mode == "frozen" else 15)
    assert len(set(r["actual_row_index_sha256"] for r in results)) == 1
    assert len(set(r["actual_xp_mask_sha256"] for r in results)) == 1
    assert state_digest(fitted[0][0].encoder) == initial_encoder
    assert all(state_digest(m.encoder) != initial_encoder for m, h in fitted[1:])
    assert state_digest(fitted[1][0].encoder) != state_digest(fitted[2][0].encoder)
    m, h = fitted[1]
    reference = predict(m.encoder, h, x, 1.0, 0.4)
    other, oh = Tiny(), torch.nn.Linear(2, 1)
    other.encoder.load_state_dict(m.encoder.state_dict())
    oh.load_state_dict(h.state_dict())
    np.testing.assert_array_equal(predict(other.encoder, oh, x, 1.0, 0.4), reference)
    assert (reference > 0).all() and np.isfinite(reference).all()
