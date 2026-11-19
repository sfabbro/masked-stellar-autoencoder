"""Table rows with an [Fe/H] core record, with their split (frozen evaluation set).

usage: core_eval_rows.py --config CFG --output DIR

The [Fe/H] core (``multisurvey.fe_h_core`` prefixes) of the table's records,
whatever a variant later excludes or deduplicates, so every checkpoint is
scored on the same stars. Writes ``DIR/rows.npy`` (sorted table rows) and
``DIR/rows_split.npz`` (``table_row``, ``split``: 0 train, 1 valid, 2 test,
-1 not kept).
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from masked_stellar_autoencoder.training.finetune_data import (
        kept_rows,
        split_row_positions,
    )

    config = yaml.safe_load(a.config.read_text())
    ms = config["multisurvey"]
    rec = np.load(ms["records"], allow_pickle=False)
    names = [str(n) for n in rec["survey_names"]]
    core = [i for i, n in enumerate(names) if n.startswith(tuple(ms["fe_h_core"]))]
    fe = list(rec["parameters"]).index("fe_h")
    has = np.isin(rec["survey"], core) & np.isfinite(rec["value"][:, fe])
    rows = np.unique(rec["row"][has]).astype("i8")
    kept = kept_rows(config["data"]["ft_datafile"], config)
    mp = config["finetuning"].get("metal_poor") or {}
    split = np.full(int(max(kept.max(), rows.max())) + 1, -1, "i1")
    for k, pos in enumerate(split_row_positions(len(kept), mp)):
        split[kept[pos]] = k
    a.output.mkdir(parents=True, exist_ok=True)
    np.save(a.output / "rows.npy", rows)
    np.savez(a.output / "rows_split.npz", table_row=rows, split=split[rows])
    print(f"{len(rows)} core rows; per split (train/valid/test/not kept):",
          [int((split[rows] == k).sum()) for k in (0, 1, 2, -1)])  # fmt: skip


if __name__ == "__main__":
    main()
