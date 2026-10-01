import math

import pytest

from cs336_basics.model import DEFAULT_ROPE_THETA, BasicsTransformerLM


def _make_model(**overrides) -> BasicsTransformerLM:
    kwargs = {
        "vocab_size": 16,
        "context_length": 8,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 16,
        **overrides,
    }
    return BasicsTransformerLM(**kwargs)


def test_default_rope_theta_is_stored_in_model_config() -> None:
    model = _make_model()

    assert model.config["rope_theta"] == DEFAULT_ROPE_THETA


@pytest.mark.parametrize("rope_theta", [None, True, 0, -1, math.inf, math.nan, "invalid"])
def test_invalid_rope_theta_is_rejected(rope_theta: object) -> None:
    with pytest.raises(ValueError, match="rope_theta must be finite and positive"):
        _make_model(rope_theta=rope_theta)
