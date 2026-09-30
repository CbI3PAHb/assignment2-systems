from pathlib import Path

import pytest

from cs336_systems.model_presets import (
    ModelConfig,
    ModelNano,
    ModelPreset,
    PresetConfigError,
    create_model,
    load_model_config,
)


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (ModelPreset.NANO, (128, 4, 4, 256)),
        (ModelPreset.SMALL, (768, 12, 12, 3072)),
        (ModelPreset.MEDIUM, (1024, 24, 16, 4096)),
        (ModelPreset.LARGE, (1280, 36, 20, 5120)),
        (ModelPreset.XL, (1600, 48, 25, 6400)),
        (ModelPreset.MODEL_2_7B, (2560, 32, 32, 10240)),
    ],
)
def test_load_model_config(
    preset: ModelPreset,
    expected: tuple[int, int, int, int],
) -> None:
    config = load_model_config(preset)

    assert (
        config.model_size.d_model,
        config.model_size.num_layers,
        config.model_size.num_heads,
        config.model_size.d_ff,
    ) == expected


def test_create_model_uses_named_preset_class() -> None:
    model = create_model("nano")

    assert isinstance(model, ModelNano)
    assert model.d_model == 128
    assert model.preset_config == load_model_config(ModelPreset.NANO)


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "nano.yaml").write_text(
        """
vocab_size: 1024
context_length: 128
unknown: true
model_size:
  d_model: 128
  num_layers: 4
  num_heads: 4
  d_ff: 256
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(PresetConfigError, match="Unknown keys in preset: unknown"):
        load_model_config(ModelPreset.NANO, presets_dir=tmp_path)


def test_incompatible_head_count_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "nano.yaml").write_text(
        """
vocab_size: 1024
context_length: 128
model_size:
  d_model: 127
  num_layers: 4
  num_heads: 4
  d_ff: 256
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(PresetConfigError, match="d_model must be divisible"):
        load_model_config(ModelPreset.NANO, presets_dir=tmp_path)


def test_odd_head_dimension_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "nano.yaml").write_text(
        """
vocab_size: 1024
context_length: 128
model_size:
  d_model: 6
  num_layers: 4
  num_heads: 2
  d_ff: 24
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(PresetConfigError, match="head dimension must be even"):
        load_model_config(ModelPreset.NANO, presets_dir=tmp_path)


def test_missing_rope_theta_uses_default(tmp_path: Path) -> None:
    (tmp_path / "nano.yaml").write_text(
        """
vocab_size: 1024
context_length: 128
model_size:
  d_model: 128
  num_layers: 4
  num_heads: 4
  d_ff: 256
""".lstrip(),
        encoding="utf-8",
    )

    assert load_model_config(ModelPreset.NANO, presets_dir=tmp_path).rope_theta == 10_000.0


def test_model_config_rejects_unstructured_model_size() -> None:
    with pytest.raises(PresetConfigError, match="model_size must be a ModelSize"):
        ModelConfig(
            vocab_size=1024,
            context_length=128,
            model_size={
                "d_model": 128,
                "num_layers": 4,
                "num_heads": 4,
                "d_ff": 256,
            },  # type: ignore[arg-type]
        )
