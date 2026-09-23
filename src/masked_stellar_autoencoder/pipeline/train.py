"""Entry point for the stellar-parameter pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from masked_stellar_autoencoder.pipeline.batch import iter_partitions
from masked_stellar_autoencoder.pipeline.registry import gaia_dr3_registry


def load_config(path: Path) -> dict:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not contain a mapping")
    if "feature_cols" in cfg:
        raise ValueError(
            f"{path} is an old feature_cols config; use configs/pipeline.yaml"
        )
    return cfg


def registry_from_config(cfg: dict):
    return gaia_dr3_registry(
        n_bp=int(cfg["n_bp"]),
        n_rp=int(cfg["n_rp"]),
        g_ref=float(cfg.get("g_ref", 8.5)),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stellar-parameter pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Named HDF table. Absent until a catalogue is mounted.",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    registry = registry_from_config(cfg)
    if args.data is None:
        raise SystemExit(
            f"registry xp_bp={len(registry.names('xp_bp'))} xp_rp={len(registry.names('xp_rp'))}; "
            "pass --data for a named HDF table"
        )
    n_stars = 0
    for batch in iter_partitions(
        str(args.data), registry, flux_state=str(cfg.get("flux_state", "raw"))
    ):
        n_stars += len(batch)
    print(f"read {n_stars} stars, flux_state={cfg.get('flux_state', 'raw')}")


if __name__ == "__main__":
    main()
