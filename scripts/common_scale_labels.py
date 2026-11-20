"""Common-scale (Teff, log g, [Fe/H]) labels of every fine-tune table row.

usage: common_scale_labels.py --config CFG --checkpoint CKPT --output labels.npz
       [--predictions predictions.npz] [--records REC] [--fits FITS]
       [--core-split ROWS_SPLIT.npz] [--device cuda]

Each uncensored survey record goes through the checkpoint's anchored survey
maps (x + delta_s(x) = y_s solved by Newton from the posterior head's
prediction; error from the quoted error and the survey's learned map noise),
and a star's records are combined per parameter by inverse variance
(``masked_stellar_autoencoder.training.common_scale_labels``).

``labels.npz`` fields (a contract): ``source_id, table_row, teff, e_teff,
logg, e_logg, fe_h, e_fe_h, n_teff, n_logg, n_feh, has_core_feh, split``
(0 train, 1 valid, 2 test, -1 not kept), plus ``fe_h_core``/``e_fe_h_core``
(the frozen high-resolution reference on the common scale), the raw
``fe_h_ref``/``e_fe_h_ref``/``fe_h_ref_survey`` and ``survey_names``.

``--predictions`` is the posterior head's output per table row
(``scripts/s1_likelihood_head.py`` stage ``latents``); without it the encoder
runs first and writes ``latent.npy`` and ``predictions.npz`` next to the output.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--records", type=Path)
    p.add_argument("--fits", type=Path)
    p.add_argument("--core-split", type=Path)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    from masked_stellar_autoencoder.models.checkpoint_load import (
        torch_load_finetune_checkpoint,
    )
    from masked_stellar_autoencoder.training.common_scale_labels import write_labels

    config = yaml.safe_load(a.config.read_text())
    fits = a.fits or Path(config["data"]["ft_datafile"])
    records = a.records or Path(config["multisurvey"]["records"])
    state = torch_load_finetune_checkpoint(a.checkpoint, map_location="cpu")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = a.predictions
    if predictions is None:
        from masked_stellar_autoencoder.training.frozen_latents import encode_table

        _, predictions = encode_table(config, state, fits, a.output.parent,
                                      device=a.device)  # fmt: skip
    diag = write_labels(
        config, state, dict(np.load(records, allow_pickle=False)), np.load(predictions),
        fits, a.output, iterations=a.iterations, device=a.device,
        core_split=a.core_split,
    )  # fmt: skip
    print(json.dumps(diag, indent=2))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
