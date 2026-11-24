"""Entry point of the v11 encoder pretraining (docs/programme/encoder-v11.md).

    python -m masked_stellar_autoencoder.training.pretrain_v11 --config configs/pretrain-v11.yaml
    python -m masked_stellar_autoencoder.training.pretrain_v11 --config configs/pretrain-v11-smoke.yaml
    python -m masked_stellar_autoencoder.training.pretrain_v11 --config ... --describe   # model, bands, FI bands
    python -m masked_stellar_autoencoder.training.pretrain_v11 --config ... --estimate   # FLOPs, memory, MIG rows/s

Everything is read from the one config file (unknown keys fail); the run writes
``config.yaml``, ``manifest.json``, ``scalers.json``, ``metrics.jsonl`` and
``checkpoints/`` under ``run.output_dir`` and resumes from ``checkpoints/latest.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys

from masked_stellar_autoencoder.v11.config import load_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="MSA v11 encoder pretraining")
    p.add_argument(
        "--config",
        required=True,
        help="YAML config (masked_stellar_autoencoder.v11.config)",
    )
    p.add_argument(
        "--describe",
        action="store_true",
        help="print bands, FI bands and parameter counts, then exit",
    )
    p.add_argument(
        "--estimate",
        action="store_true",
        help="print the FLOP/memory/throughput estimate, then exit",
    )
    p.add_argument(
        "--device",
        default=None,
        help="override run.device for --describe/--estimate only",
    )
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.estimate:
        from masked_stellar_autoencoder.v11.estimate import estimate

        print(json.dumps(estimate(cfg, device=args.device or "cpu"), indent=1))
        return 0
    from masked_stellar_autoencoder.v11.model import parameter_count
    from masked_stellar_autoencoder.v11.schema import describe
    from masked_stellar_autoencoder.v11.train import Pretrainer, build_components

    if args.describe:
        comp = build_components(cfg)
        info = {
            "contract": describe(comp.registry),
            "bands_read": [comp.registry.bands[i].name for i, *_ in comp.reader.plan],
            "fi_operator": None if comp.operator is None else comp.operator.to_dict(),
            "parameters": parameter_count(comp.model),
            "total_steps": cfg.total_steps,
        }
        print(json.dumps(info, indent=1))
        return 0
    last = Pretrainer(cfg).run()
    print(
        json.dumps(
            {"final": {k: v for k, v in last.items() if not isinstance(v, dict)}}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
