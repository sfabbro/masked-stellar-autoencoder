#!/usr/bin/env bash
set -euo pipefail

data_file="${MSA_PRETRAIN_DATA:-/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5}"
pixi run python - "$data_file" <<'PYEOF'
import sys

import h5py

with h5py.File(sys.argv[1], "r") as h5:
    keys = list(h5.keys())
    print(f"Top-level groups: {len(keys)}")

    for key in keys[:3]:
        dataset = h5[key]
        print(f"\n  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        if dataset.dtype.names:
            print(f"    columns ({len(dataset.dtype.names)}):")
            for column in dataset.dtype.names:
                print(f"      {column}")

    total = sum(h5[key].shape[0] for key in keys)
    print(f"\nTotal rows across all keys: {total:,}")

    if keys:
        first = h5[keys[0]]
        size_mb = first.dtype.itemsize * first.shape[0] / 1e6
        print(f"\nKey {keys[0]!r}: {first.shape}, {size_mb:.1f} MB")
        if first.dtype.names:
            print(f"  All {len(first.dtype.names)} columns: {list(first.dtype.names)}")
PYEOF
