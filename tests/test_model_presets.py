from pathlib import Path

import pytest
import yaml

from cs336_basics.model import DEFAULT_ROPE_THETA, BasicsTransformerLM
from cs336_systems.model_presets import (
    ModelConfig,
    ModelConfigError,
    build_model,
    list_model_presets,
    load_model_config,
)


def _valid_preset_data() -> dict[str, object]:
    return {
        "vocab_size": 1024,
        "context_length": 128,
        "d_model": 128,
        "num_layers": 4,
        "num_heads": 4,
        "d_ff": 256,
        "rope_theta": 10_000.0,
    }


def _write_preset(
    presets_dir: Path,
    preset_data: dict[str, object],
    *,
    name: str = "test",
) -> None:
    (presets_dir / f"{name}.yaml").write_text(
        yaml.safe_dump(preset_data),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("preset_name", "expected"),
    [
        ("nano", (128, 4, 4, 256)),
        ("small", (768, 12, 12, 3072)),
        ("medium", (1024, 24, 16, 4096)),
        ("large", (1280, 36, 20, 5120)),
        ("xl", (1600, 48, 25, 6400)),
        ("2.7b", (2560, 32, 32, 10240)),
    ],
)
def test_load_model_config(
    preset_name: str,
    expected: tuple[int, int, int, int],
) -> None:
    config = load_model_config(preset_name)

    assert (
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
    ) == expected


def test_preset_names_are_discovered_from_yaml_files(tmp_path: Path) -> None:
    _write_preset(tmp_path, _valid_preset_data(), name="toy")

    assert list_model_presets(presets_dir=tmp_path) == ("toy",)
    assert load_model_config("toy", presets_dir=tmp_path).d_model == 128


def test_build_model_uses_every_config_value() -> None:
    config = load_model_config("nano")

    model = build_model(config)

    assert isinstance(model, BasicsTransformerLM)
    assert model.config == {
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "d_model": config.d_model,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "d_ff": config.d_ff,
        "rope_theta": config.rope_theta,
    }


def test_unknown_preset_lists_available_names() -> None:
    with pytest.raises(ModelConfigError, match="Available presets:.*nano"):
        load_model_config("does-not-exist")


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("d_model: [", encoding="utf-8")

    with pytest.raises(ModelConfigError, match="Invalid YAML.*broken.yaml"):
        load_model_config("broken", presets_dir=tmp_path)


def test_yaml_root_must_be_a_mapping(tmp_path: Path) -> None:
    (tmp_path / "list.yaml").write_text("- 128\n- 256\n", encoding="utf-8")

    with pytest.raises(ModelConfigError, match="must contain a YAML mapping"):
        load_model_config("list", presets_dir=tmp_path)


def test_unknown_yaml_field_is_rejected(tmp_path: Path) -> None:
    preset_data = _valid_preset_data()
    preset_data["unknown"] = True
    _write_preset(tmp_path, preset_data)

    with pytest.raises(ModelConfigError, match="unexpected keyword argument 'unknown'"):
        load_model_config("test", presets_dir=tmp_path)


def test_missing_required_yaml_field_is_rejected(tmp_path: Path) -> None:
    preset_data = _valid_preset_data()
    del preset_data["context_length"]
    _write_preset(tmp_path, preset_data)

    with pytest.raises(ModelConfigError, match="missing.*context_length"):
        load_model_config("test", presets_dir=tmp_path)


def test_d_model_must_be_divisible_by_num_heads() -> None:
    with pytest.raises(ModelConfigError, match="d_model must be divisible"):
        ModelConfig(
            vocab_size=1024,
            context_length=128,
            d_model=127,
            num_layers=4,
            num_heads=4,
            d_ff=256,
        )


def test_yaml_value_error_names_the_preset_file(tmp_path: Path) -> None:
    preset_data = _valid_preset_data()
    preset_data["d_model"] = 127
    _write_preset(tmp_path, preset_data)

    with pytest.raises(ModelConfigError, match="test.yaml.*d_model=127"):
        load_model_config("test", presets_dir=tmp_path)


def test_attention_head_dimension_must_be_even() -> None:
    with pytest.raises(ModelConfigError, match="head dimension must be even"):
        ModelConfig(
            vocab_size=1024,
            context_length=128,
            d_model=6,
            num_layers=4,
            num_heads=2,
            d_ff=24,
        )


def test_missing_rope_theta_uses_default(tmp_path: Path) -> None:
    preset_data = _valid_preset_data()
    del preset_data["rope_theta"]
    _write_preset(tmp_path, preset_data)

    config = load_model_config("test", presets_dir=tmp_path)

    assert config.rope_theta == DEFAULT_ROPE_THETA


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("vocab_size", 0),
        ("context_length", -1),
        ("d_model", True),
        ("num_layers", 1.5),
        ("num_heads", "4"),
    ],
)
def test_integer_fields_must_be_positive_integers(
    field_name: str,
    invalid_value: object,
) -> None:
    config_data = _valid_preset_data()
    config_data[field_name] = invalid_value

    with pytest.raises(ModelConfigError, match=field_name):
        ModelConfig(**config_data)


@pytest.mark.parametrize("invalid_value", [0, -1, float("inf"), float("nan"), True])
def test_rope_theta_must_be_finite_and_positive(invalid_value: object) -> None:
    config_data = _valid_preset_data()
    config_data["rope_theta"] = invalid_value

    with pytest.raises(ModelConfigError, match="rope_theta"):
        ModelConfig(**config_data)
