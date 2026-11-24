"""Strict configuration schema of the v11 pretraining run.

One YAML file, one section per concern. Every key is declared here; an unknown key
(at any depth, including a misplaced masking ratio such as ``training.xp_masking_ratio``)
raises :class:`ConfigError` with its dotted path. Masking ratios are read from
``masking:`` and nowhere else (the v10 fine-tune read them from the wrong section;
architecture.md section 2.4).

Sections: ``run``, ``data`` (with ``data.synthetic`` and ``data.scalers``),
``masking``, ``model``, ``loss``, ``filter_integral``, ``optim``, ``train``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PARTITION_NAMES = ("FIT", "DEV", "CAL", "TEST")
FI_MODES = ("calibrate", "sed", "phot")


class ConfigError(ValueError):
    pass


@dataclass
class RunConfig:
    name: str = "pretrain-v11"
    output_dir: str = "runs/pretrain_v11"
    seed: int = 20261124
    device: str = "auto"  # auto | cpu | cuda
    precision: str = "bf16"  # bf16 | fp32 (bf16 autocast on CUDA only)
    resume: bool = True
    num_threads: int = 0  # torch CPU threads; 0 = leave torch's default

    def validate(self, path: str) -> None:
        _choice(path + ".device", self.device, ("auto", "cpu", "cuda"))
        _choice(path + ".precision", self.precision, ("bf16", "fp32"))


@dataclass
class TreeConfig:
    path: str = ""
    columns: list[str] | None = None  # None: the columns the contract needs

    def validate(self, path: str) -> None:
        if not self.path:
            raise ConfigError(f"{path}.path is required")


@dataclass
class SyntheticConfig:
    """Tiny synthetic sky for smoke runs and tests (``data.source: synthetic``)."""

    n_rows: int = 4096
    n_partitions: int = 4
    xp_fraction: float = 0.4
    seed: int = 0
    directory: str | None = None  # where the synthetic HATS trees are written

    def validate(self, path: str) -> None:
        _positive(path + ".n_rows", self.n_rows)
        _positive(path + ".n_partitions", self.n_partitions)
        _unit(path + ".xp_fraction", self.xp_fraction)


@dataclass
class ScalerConfig:
    """Per-band robust centre/scale, fitted once from the training stream.

    ``path`` reuses a saved ``scalers.json`` (the same file must serve pretraining and
    fine-tuning); otherwise ``fit_rows`` rows of the shuffled REC stream are used.
    ``fit_rows`` is a statistics sample, not a cap on training data.
    """

    path: str | None = None
    fit_rows: int = 1_000_000
    min_count: int = 50

    def validate(self, path: str) -> None:
        _positive(path + ".fit_rows", self.fit_rows)
        _positive(path + ".min_count", self.min_count)


@dataclass
class DataConfig:
    source: str = "hats"  # hats | synthetic
    contract: str = "gaia_keyed_v1a"
    trees: list[TreeConfig] = field(default_factory=list)  # first = anchor
    assignment: str = "/arc/projects/astroai/msa/runs/msa_split_v1/assignment"
    train_partitions: list[str] = field(default_factory=lambda: ["FIT"])
    eval_partitions: list[str] = field(default_factory=lambda: ["DEV"])
    rec_sample_rate: float = 1.0  # nested REC sample, salt msa-rec-v1 (1 = all)
    bands: list[str] | None = None  # None = every band of the contract
    survey_tags: dict[str, Any] = field(default_factory=dict)
    allow_unverified_columns: bool = False
    xp_covariance: str = "exact"  # exact (packed correlations) | diagonal
    workers: int = 4
    interleave_partitions: int = 4
    read_batch_rows: int = 65536
    shuffle_buffer_rows: int = 262144
    tokens_per_microbatch: int = 1_048_576
    max_rows_per_microbatch: int = 8192
    bucket_edges: list[int] = field(default_factory=lambda: [16, 32, 64, 128, 160, 224])
    limit_partitions: int | None = None  # debugging / smoke only; never rows
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    scalers: ScalerConfig = field(default_factory=ScalerConfig)

    def validate(self, path: str) -> None:
        _choice(path + ".source", self.source, ("hats", "synthetic"))
        _choice(path + ".xp_covariance", self.xp_covariance, ("exact", "diagonal"))
        for name in self.train_partitions + self.eval_partitions:
            _choice(path + ".*_partitions", name, PARTITION_NAMES)
        if set(self.train_partitions) & set(self.eval_partitions):
            raise ConfigError(f"{path}: train and eval partitions overlap")
        if self.source == "hats" and not self.trees:
            raise ConfigError(f"{path}.trees: at least the anchor tree is required")
        if not 0 < self.rec_sample_rate <= 1:
            raise ConfigError(f"{path}.rec_sample_rate must be in (0, 1]")
        for k in ("workers",):
            if getattr(self, k) < 0:
                raise ConfigError(f"{path}.{k} must be >= 0")
        for k in ("interleave_partitions", "read_batch_rows", "shuffle_buffer_rows"):
            _positive(f"{path}.{k}", getattr(self, k))
        _positive(path + ".tokens_per_microbatch", self.tokens_per_microbatch)
        _positive(path + ".max_rows_per_microbatch", self.max_rows_per_microbatch)
        if sorted(self.bucket_edges) != self.bucket_edges or not self.bucket_edges:
            raise ConfigError(f"{path}.bucket_edges must be increasing")
        if self.tokens_per_microbatch < self.bucket_edges[-1]:
            raise ConfigError(f"{path}.tokens_per_microbatch below the largest bucket")
        if self.limit_partitions is not None:
            _positive(path + ".limit_partitions", self.limit_partitions)


@dataclass
class MaskingConfig:
    """The only place masking ratios live.

    * ``token_ratio``: each present magnitude band is masked independently;
    * ``xp_token_ratio``: each coefficient of a visible XP block is masked;
    * ``survey_dropout``: each present survey (all its bands) is dropped whole;
    * ``xp_dropout``: the whole XP block is hidden (photometry-only view);
    * ``astrometry_hide`` / ``sfd_hide``: conditioning groups hidden (never targets);
    * ``min_visible``: rows keep at least this many visible reconstructable tokens.
    """

    token_ratio: float = 0.25
    xp_token_ratio: float = 0.5
    survey_dropout: float = 0.15
    xp_dropout: float = 0.35
    astrometry_hide: float = 0.5
    sfd_hide: float = 0.5
    min_visible: int = 2
    eval_seed: int = 12345

    def validate(self, path: str) -> None:
        for k in (
            "token_ratio",
            "xp_token_ratio",
            "survey_dropout",
            "xp_dropout",
            "astrometry_hide",
            "sfd_hide",
        ):
            v = getattr(self, k)
            if not 0.0 <= v < 1.0:
                raise ConfigError(f"{path}.{k} must be in [0, 1), got {v}")
        _positive(path + ".min_visible", self.min_visible)


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_latents: int = 32
    n_layers: int = 8
    cross_attn_layers: list[int] = field(default_factory=lambda: [0, 4])
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    latent_dim: int = 256
    value_frequencies: int = 16
    band_frequencies: int = 16
    decoder_hidden: int = 512
    decoder_layers: int = 2
    gradient_checkpointing: bool = False

    def validate(self, path: str) -> None:
        for k in ("d_model", "n_heads", "n_latents", "n_layers", "latent_dim"):
            _positive(f"{path}.{k}", getattr(self, k))
        if self.d_model % self.n_heads:
            raise ConfigError(f"{path}: d_model must be divisible by n_heads")
        if not self.cross_attn_layers or self.cross_attn_layers[0] != 0:
            raise ConfigError(f"{path}.cross_attn_layers must start with 0")
        if any(not 0 <= i < self.n_layers for i in self.cross_attn_layers):
            raise ConfigError(f"{path}.cross_attn_layers outside [0, n_layers)")
        if not 0 <= self.dropout < 1:
            raise ConfigError(f"{path}.dropout must be in [0, 1)")


@dataclass
class LossConfig:
    nu: float = 5.0
    sigma_model_init: float = 0.05  # scaled (robust-IQR) units
    sigma_model_floor: float = 1e-3
    group_weights: dict[str, float] = field(
        default_factory=lambda: {"xp": 1.0, "phot": 1.0}
    )

    def validate(self, path: str) -> None:
        if self.nu <= 0:
            raise ConfigError(f"{path}.nu must be > 0")
        if set(self.group_weights) != {"xp", "phot"}:
            raise ConfigError(f"{path}.group_weights needs exactly xp and phot")
        _positive(path + ".sigma_model_init", self.sigma_model_init)


@dataclass
class FilterIntegralConfig:
    enabled: bool = True
    weight: float = 0.1  # lambda_FI on the network-gradient modes (sed, phot)
    bands: list[str] | None = None  # None = every band whose SVO support fits
    wavelength_range_nm: list[float] = field(default_factory=lambda: [336.0, 1020.0])
    support_threshold: float = 0.01  # T > threshold * max(T) defines the support
    modes: list[str] = field(default_factory=lambda: list(FI_MODES))
    nu: float = 5.0
    sigma_sys_init: float = 0.02  # mag
    colour_reference: float = 1.0  # synthetic BP-RP at which zp is quoted
    linear_below_fraction: float = (
        0.1  # sed: magnitude linear in flux below this x F_obs
    )
    max_sigma_xp_mag: float = (
        0.2  # calibrate/phot: observed-XP synthetic mags with larger sigma_XP skipped
    )
    integration_step_nm: float = 0.25

    def validate(self, path: str) -> None:
        bad = sorted(set(self.modes) - set(FI_MODES))
        if bad:
            raise ConfigError(f"{path}.modes: unknown {bad}")
        lo, hi = self.wavelength_range_nm
        if not 0 < lo < hi:
            raise ConfigError(f"{path}.wavelength_range_nm must be increasing")
        if not 0 < self.support_threshold < 1:
            raise ConfigError(f"{path}.support_threshold must be in (0, 1)")
        if self.weight < 0:
            raise ConfigError(f"{path}.weight must be >= 0")
        if not 0 < self.linear_below_fraction < 1:
            raise ConfigError(f"{path}.linear_below_fraction must be in (0, 1)")


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    eps: float = 1e-8
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    loss_param_lr_scale: float = 10.0  # sigma_model and filter-integral nuisances

    def validate(self, path: str) -> None:
        _positive(path + ".lr", self.lr)
        if len(self.betas) != 2:
            raise ConfigError(f"{path}.betas needs two values")


@dataclass
class TrainConfig:
    batch_rows: int = 32768  # rows per optimiser step (accumulated micro-batches)
    presentations: int = 1_360_000_000  # rows seen over the schedule (~1 REC pass)
    max_steps: int | None = None  # stop early (smoke); the schedule keeps its length
    log_every: int = 50
    eval_every: int = 2000
    eval_rows: int = 262_144
    checkpoint_every: int = 2000

    def validate(self, path: str) -> None:
        for k in ("batch_rows", "presentations", "log_every", "eval_every"):
            _positive(f"{path}.{k}", getattr(self, k))
        _positive(path + ".checkpoint_every", self.checkpoint_every)


@dataclass
class PretrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    filter_integral: FilterIntegralConfig = field(default_factory=FilterIntegralConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> PretrainConfig:
        for f in dataclasses.fields(self):
            _validate_tree(getattr(self, f.name), f.name)
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def total_steps(self) -> int:
        return -(-self.train.presentations // self.train.batch_rows)


# ----------------------------------------------------------------- loading


def _validate_tree(obj, path: str) -> None:
    if hasattr(obj, "validate"):
        obj.validate(path)
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if dataclasses.is_dataclass(v):
                _validate_tree(v, f"{path}.{f.name}")
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if dataclasses.is_dataclass(item):
                        _validate_tree(item, f"{path}.{f.name}[{i}]")


def _choice(path: str, value, allowed) -> None:
    if value not in allowed:
        raise ConfigError(f"{path} = {value!r}; allowed: {list(allowed)}")


def _positive(path: str, value) -> None:
    if value is None or value <= 0:
        raise ConfigError(f"{path} must be > 0, got {value}")


def _unit(path: str, value) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{path} must be in [0, 1], got {value}")


def _coerce(tp, value, path: str):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], value, path)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        return _build(tp, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        return [_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        vt = args[1] if args else Any
        return {k: _coerce(vt, v, f"{path}.{k}") for k, v in value.items()}
    if tp is Any:
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string, got {value!r}")
        return value
    return value


def _build(cls, data: dict, path: str):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        where = path or "config"
        raise ConfigError(
            f"{where}: unknown keys {[f'{path}.{k}'.lstrip('.') for k in unknown]}; "
            f"allowed: {sorted(names)}"
        )
    kwargs = {
        k: _coerce(hints[k], v, f"{path}.{k}".lstrip(".")) for k, v in data.items()
    }
    return cls(**kwargs)


def load_config(source: str | Path | dict) -> PretrainConfig:
    """Load a YAML path (or a mapping), reject unknown keys, validate ranges."""
    if isinstance(source, dict):
        data = source
    else:
        data = yaml.safe_load(Path(source).read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return _build(PretrainConfig, data, "").validate()
