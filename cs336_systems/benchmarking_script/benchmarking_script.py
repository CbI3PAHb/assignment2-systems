import argparse

import numpy as np

from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.model_presets import (
    ModelPreset, create_model, load_model_config
)

from cs336_systems.model_presets import (
    ModelPreset,
    create_model,
    load_model_config,
)
N_WARM_UP_STEPS = 4
N_BENCHMARK_S = 4


def main(preset: ModelPreset) -> None:
    config = load_model_config(preset)
    print(f"Loading {preset.value!r} model preset: {config}")
    model = create_model(preset)
    device = "cuda:0"

    optimizer = AdamW(model.parameters())

    model.to(device)
    dataset = np.random.randint(
        0,
        config.vocab_size - 1,
        size=(config.vocab_size,),
    )
    batch_size = 4

    inputs, targets = get_batch(
        dataset=dataset,
        batch_size=batch_size,
        device=device,
        context_length=config.context_length,
    )

    for _ in range(N_WARM_UP_STEPS):
        logits = model.forward(inputs)
        loss = cross_entropy(inputs=logits, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for _ in range(N_BENCHMARK_S):
        logits = model.forward(inputs)
        loss = cross_entropy(inputs=logits, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        type=ModelPreset.parse,
        choices=list(ModelPreset),
        default=ModelPreset.NANO,
    )
    args = parser.parse_args()
    main(args.preset)
