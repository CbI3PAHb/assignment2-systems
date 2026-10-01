from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from cs336_basics.model import DEFAULT_ROPE_THETA, BasicsTransformerLM


_PRESETS_DIR = Path(__file__).with_name("presets")


class ModelConfigError(ValueError):
    """Raised when a model configuration is missing or invalid."""


def _require_positive_integer(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ModelConfigError(
            f"{name} must be a positive integer, got {value!r}"
        )


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    """All values needed to construct a ``BasicsTransformerLM``."""

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = DEFAULT_ROPE_THETA

    def __post_init__(self) -> None:
        _require_positive_integer("vocab_size", self.vocab_size)
        _require_positive_integer("context_length", self.context_length)
        _require_positive_integer("d_model", self.d_model)
        _require_positive_integer("num_layers", self.num_layers)
        _require_positive_integer("num_heads", self.num_heads)
        _require_positive_integer("d_ff", self.d_ff)

        if self.d_model % self.num_heads != 0:
            raise ModelConfigError(
                "d_model must be divisible by num_heads, "
                f"got d_model={self.d_model} and num_heads={self.num_heads}"
            )

        head_dimension = self.d_model // self.num_heads
        if head_dimension % 2 != 0:
            raise ModelConfigError(
                "attention head dimension must be even for RoPE, "
                f"got {head_dimension}"
            )

        if (
            not isinstance(self.rope_theta, (int, float))
            or isinstance(self.rope_theta, bool)
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 0
        ):
            raise ModelConfigError(
                f"rope_theta must be finite and positive, got {self.rope_theta!r}"
            )


def list_model_presets(*, presets_dir: Path = _PRESETS_DIR) -> tuple[str, ...]:
    """Return preset names discovered from ``*.yaml`` filenames."""

    return tuple(path.stem for path in sorted(presets_dir.glob("*.yaml")))


def load_model_config(
    preset_name: str,
    *,
    presets_dir: Path = _PRESETS_DIR,
) -> ModelConfig:
    """Load and validate one named YAML preset."""

    available_presets = list_model_presets(presets_dir=presets_dir)
    if preset_name not in available_presets:
        available = ", ".join(available_presets) or "none"
        raise ModelConfigError(
            f"Unknown model preset {preset_name!r}. Available presets: {available}"
        )

    preset_path = presets_dir / f"{preset_name}.yaml"
    try:
        with preset_path.open(encoding="utf-8") as preset_file:
            preset_data = yaml.safe_load(preset_file)
    except yaml.YAMLError as error:
        raise ModelConfigError(f"Invalid YAML in {preset_path}: {error}") from error

    if not isinstance(preset_data, dict):
        raise ModelConfigError(f"{preset_path} must contain a YAML mapping")

    try:
        config = ModelConfig(**preset_data)
    except TypeError as error:
        raise ModelConfigError(
            f"Invalid fields in {preset_path.name}: {error}"
        ) from error
    except ModelConfigError as error:
        raise ModelConfigError(
            f"Invalid values in {preset_path.name}: {error}"
        ) from error

    return config


def build_model(config: ModelConfig) -> BasicsTransformerLM:
    """Construct a model from a validated configuration."""

    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )
