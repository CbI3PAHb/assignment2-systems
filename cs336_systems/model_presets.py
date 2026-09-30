from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import yaml

from cs336_basics.model import BasicsTransformerLM


PRESETS_DIR = Path(__file__).with_name("presets")
DEFAULT_ROPE_THETA = 10_000.0


class PresetConfigError(ValueError):
    """Raised when a model preset does not match the expected schema."""


class ModelPreset(StrEnum):
    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XL = "xl"
    MODEL_2_7B = "2.7b"

    @classmethod
    def parse(cls, value: ModelPreset | str) -> ModelPreset:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower())
        except ValueError as error:
            available = ", ".join(preset.value for preset in cls)
            raise PresetConfigError(
                f"Unknown model preset {value!r}. Available presets: {available}"
            ) from error


@dataclass(frozen=True, slots=True)
class ModelSize:
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int

    def __post_init__(self) -> None:
        for name in ("d_model", "num_layers", "num_heads", "d_ff"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise PresetConfigError(f"model_size.{name} must be a positive integer")
        if self.d_model % self.num_heads != 0:
            raise PresetConfigError(
                "model_size.d_model must be divisible by model_size.num_heads"
            )
        if (self.d_model // self.num_heads) % 2 != 0:
            raise PresetConfigError("model_size head dimension must be even for RoPE")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    model_size: ModelSize
    rope_theta: float = DEFAULT_ROPE_THETA

    def __post_init__(self) -> None:
        if not isinstance(self.model_size, ModelSize):
            raise PresetConfigError("model_size must be a ModelSize")
        for name in ("vocab_size", "context_length"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise PresetConfigError(f"{name} must be a positive integer")
        if (
            not isinstance(self.rope_theta, (int, float))
            or isinstance(self.rope_theta, bool)
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 0
        ):
            raise PresetConfigError("rope_theta must be finite and positive")

    def to_model_kwargs(self) -> dict[str, int | float]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.model_size.d_model,
            "num_layers": self.model_size.num_layers,
            "num_heads": self.model_size.num_heads,
            "d_ff": self.model_size.d_ff,
            "rope_theta": self.rope_theta,
        }


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise PresetConfigError(f"{location} must be a YAML mapping")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: Set[str] = frozenset(),
    location: str,
) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise PresetConfigError(
            f"Missing keys in {location}: {', '.join(sorted(missing))}"
        )
    if extra:
        raise PresetConfigError(
            f"Unknown keys in {location}: {', '.join(sorted(extra))}"
        )


def model_config_from_dict(raw: Mapping[str, Any]) -> ModelConfig:
    data = _require_mapping(raw, "preset")
    _check_keys(
        data,
        required={"vocab_size", "context_length", "model_size"},
        optional={"rope_theta"},
        location="preset",
    )

    size_data = _require_mapping(data["model_size"], "model_size")
    size_fields = {"d_model", "num_layers", "num_heads", "d_ff"}
    _check_keys(size_data, required=size_fields, location="model_size")

    model_size = ModelSize(**{name: size_data[name] for name in size_fields})
    return ModelConfig(
        vocab_size=data["vocab_size"],
        context_length=data["context_length"],
        model_size=model_size,
        rope_theta=data.get("rope_theta", DEFAULT_ROPE_THETA),
    )


def load_model_config(
    preset: ModelPreset | str,
    *,
    presets_dir: Path = PRESETS_DIR,
) -> ModelConfig:
    preset = ModelPreset.parse(preset)
    path = presets_dir / f"{preset.value}.yaml"
    try:
        with path.open(encoding="utf-8") as preset_file:
            raw = yaml.safe_load(preset_file)
    except FileNotFoundError as error:
        raise PresetConfigError(f"Preset file does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise PresetConfigError(f"Invalid YAML in preset {path}: {error}") from error

    return model_config_from_dict(_require_mapping(raw, str(path)))


def build_model(config: ModelConfig) -> BasicsTransformerLM:
    return BasicsTransformerLM(**config.to_model_kwargs())


class PresetTransformerLM(BasicsTransformerLM):
    preset: ClassVar[ModelPreset]

    def __init__(self) -> None:
        config = load_model_config(self.preset)
        super().__init__(**config.to_model_kwargs())
        self.preset_config = config


class ModelNano(PresetTransformerLM):
    preset = ModelPreset.NANO


class ModelSmall(PresetTransformerLM):
    preset = ModelPreset.SMALL


class ModelMedium(PresetTransformerLM):
    preset = ModelPreset.MEDIUM


class ModelLarge(PresetTransformerLM):
    preset = ModelPreset.LARGE


class ModelXL(PresetTransformerLM):
    preset = ModelPreset.XL


class Model2_7B(PresetTransformerLM):
    preset = ModelPreset.MODEL_2_7B


MODEL_CLASSES: dict[ModelPreset, type[PresetTransformerLM]] = {
    model_class.preset: model_class
    for model_class in (
        ModelNano,
        ModelSmall,
        ModelMedium,
        ModelLarge,
        ModelXL,
        Model2_7B,
    )
}


def create_model(preset: ModelPreset | str) -> PresetTransformerLM:
    return MODEL_CLASSES[ModelPreset.parse(preset)]()


__all__ = [
    "MODEL_CLASSES",
    "DEFAULT_ROPE_THETA",
    "Model2_7B",
    "ModelConfig",
    "ModelLarge",
    "ModelMedium",
    "ModelNano",
    "ModelPreset",
    "ModelSize",
    "ModelSmall",
    "ModelXL",
    "PresetConfigError",
    "PresetTransformerLM",
    "build_model",
    "create_model",
    "load_model_config",
    "model_config_from_dict",
]
