import argparse

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.model_presets import (
    build_model,
    list_model_presets,
    load_model_config,
)

N_WARM_UP_STEPS = 4
N_BENCHMARK_STEPS = 4


def main(preset_name: str) -> None:
    config = load_model_config(preset_name)
    print(f"Loading {preset_name!r} model preset: {config}")
    model = build_model(config)
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

    for _ in range(N_BENCHMARK_STEPS):
        logits = model.forward(inputs)
        loss = cross_entropy(inputs=logits, targets=targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=list_model_presets(),
        default="nano",
    )
    args = parser.parse_args()
    main(args.preset)
