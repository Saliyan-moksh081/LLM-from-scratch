from pathlib import Path

import h5py
import numpy as np
import torch


def load_tokens(path: Path) -> torch.Tensor:
    """Load the HDF5 token stream into a one-dimensional CPU tensor."""
    with h5py.File(path, "r") as data_file:
        tokens = np.asarray(data_file["tokens"], dtype=np.int32)

    if tokens.ndim != 1:
        raise ValueError(f"Expected a one-dimensional token stream in {path}")

    # The TinyStories subset is small enough to keep in CPU memory during training.
    return torch.from_numpy(tokens)


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random input and next-token target windows from a token stream."""
    if batch_size < 1 or context_length < 1:
        raise ValueError("batch_size and context_length must be positive")
    if tokens.ndim != 1 or tokens.numel() <= context_length:
        raise ValueError("tokens must be a one-dimensional stream longer than context_length")

    # Each window has one extra token: inputs are the first part, targets are shifted by one.
    starts = torch.randint(tokens.numel() - context_length, (batch_size,))
    offsets = torch.arange(context_length + 1)
    windows = tokens[starts[:, None] + offsets]

    inputs = windows[:, :-1].long().to(device)
    targets = windows[:, 1:].long().to(device)
    return inputs, targets 